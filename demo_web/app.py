"""Flask + SocketIO server for the RealWonder interactive demo.

Usage:
    python app.py \
        --demo_data demo_data/lamp \
        --checkpoint_path /path/to/model.pt \
        --port 5000

The specified --demo_data case is fully initialized at startup (Genesis scene,
video generator, noise warper). When a client connects, the UI shows the scene
preview and lets the user choose force direction, edit prompt, and click Start.
"""
import os
os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'stdlib'

import argparse
import base64
import io
import threading
import time
from pathlib import Path
from queue import Queue, Full as QueueFull, Empty as QueueEmpty

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from config import (
    FRAMES_PER_BLOCK, FRAMES_PER_BLOCK_PIXEL, FRAMES_FIRST_BLOCK_PIXEL,
    FPS, LATENT_H, LATENT_W, LATENT_C,
    DEFAULT_HEIGHT, DEFAULT_WIDTH, TEMPORAL_FACTOR,
    load_case_sdedit_config,
)
from simulation_engine import InteractiveSimulator
from noise_warper_stream import StreamingNoiseWarper
from video_generator import StreamingVideoGenerator
from case_handlers.base import get_demo_case_handler
import case_handlers  # trigger registration
from gpu_profiler import log_gpu, set_gpu_logging
from simulation.utils import resize_and_crop_pil
from simulation.experiment_logging import ExperimentLogger

app = Flask(__name__)
app.config["SECRET_KEY"] = "realwonder-demo"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global state — all initialized at startup before the server accepts connections
simulator = None
noise_warper = None
generator = None
demo_case_handler = None  # Per-case UI/force handler
preview_b64 = None       # Base64 scene preview rendered once at startup
default_prompt = ""       # Prompt from case config
case_name = ""            # Name of the loaded case
num_blocks = None         # Computed from case config at startup
startup_logger = None

is_generating = False
stop_requested = False


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def on_connect():
    """When a client connects, send the pre-rendered scene preview and config."""
    print("Client connected")
    if simulator is not None and preview_b64 is not None:
        ui_config = demo_case_handler.get_ui_config() if demo_case_handler else {}
        ui_config["allow_change_force"] = simulator.config.get("allow_change_force", False)
        emit("ready", {
            "case_name": case_name,
            "preview": preview_b64,
            "prompt": default_prompt,
            "ui_config": ui_config,
        })
    else:
        emit("error", {"message": "Server not fully initialized. Check startup logs."})


@socketio.on("start_generation")
def on_start_generation(data):
    """User chose direction + prompt and clicked Start."""
    global is_generating, stop_requested
    if simulator is None:
        emit("error", {"message": "Simulator not initialized"})
        return
    if generator is None or not generator.is_setup:
        emit("error", {"message": "Video generator not initialized"})
        return
    if is_generating:
        emit("error", {"message": "Generation already in progress"})
        return

    prompt = data.get("prompt", default_prompt or "A video of physical simulation")
    ui_forces = data.get("forces", [])

    # Convert UI direction strings to 3D vectors and store on handler
    force_configs = demo_case_handler.get_force_config_from_ui(ui_forces)
    demo_case_handler.set_forces(force_configs)

    # Configure simulation state from the main thread (required for cases
    # like santa_cloth where taichi field writes need the creating thread's
    # CUDA context).
    demo_case_handler.configure_simulation(simulator)

    emit("status", {"message": "Forces configured. Starting generation..."})
    stop_requested = False
    socketio.start_background_task(generation_loop, prompt)


@socketio.on("stop_generation")
def on_stop_generation():
    global stop_requested
    stop_requested = True


@socketio.on("update_forces")
def on_update_forces(data):
    """User changed force direction/strength mid-generation.

    Updates the demo handler's wind parameters (plain Python attrs).
    The simulation thread's apply_forces() reads these every step,
    so changes take effect immediately — no CUDA or taichi involved.
    Only works when allow_change_force is enabled in the case config.
    """
    if demo_case_handler is None or simulator is None:
        return
    if not simulator.config.get("allow_change_force", False):
        return
    ui_forces = data.get("forces", [])
    force_configs = demo_case_handler.get_force_config_from_ui(ui_forces)
    demo_case_handler.set_forces(force_configs)
    # Update derived wind params (direction vector, strength scalar)
    demo_case_handler.configure_simulation(simulator)


@socketio.on("reset")
def on_reset():
    global is_generating, stop_requested
    stop_requested = True
    if simulator is not None:
        simulator.reset()
    if noise_warper is not None:
        noise_warper.reset()
    if generator is not None:
        generator.reset()
    is_generating = False
    socketio.emit("status", {"message": "Reset complete"})
    # Re-send the preview so user can start again
    if preview_b64 is not None:
        ui_config = demo_case_handler.get_ui_config() if demo_case_handler else {}
        ui_config["allow_change_force"] = simulator.config.get("allow_change_force", False) if simulator else False
        socketio.emit("ready", {
            "case_name": case_name,
            "preview": preview_b64,
            "prompt": default_prompt,
            "ui_config": ui_config,
        })


def generation_loop(prompt):
    """Main generation loop with 3-stage streaming pipeline.

    Stage 1 (thread): Simulation — produces RGB frames + optical flows per block
    Stage 2 (thread): Noise warping — warps noise using optical flow (lightweight)
    Stage 3 (main):   VAE encoding + mask building + diffusion denoising + streaming

    Each stage runs concurrently: while VGen denoises block N, noise warping
    handles block N+1, and simulation produces block N+2. All heavy GPU work
    (VAE encode + diffusion) is consolidated in Stage 3 to avoid GPU memory
    contention.
    """
    global is_generating, stop_requested
    is_generating = True
    torch.set_grad_enabled(False)  # thread-local: must set in this thread too

    exp_logger = ExperimentLogger(
        experiment_name="interactive_demo_generation",
        run_name=case_name or "demo_case",
        output_dir=Path(simulator.config.get("output_folder", "/tmp/realwonder_demo")) / "experiment_logs",
        metadata={
            "prompt": prompt,
            "num_blocks": num_blocks,
        },
    )

    try:
        socketio.emit("status", {"message": "Preparing video generator..."})

        # Reset noise warper before sim threads start.
        noise_warper.reset()

        frame_steps = simulator.frame_steps

        # --- 4-Stage Pipeline Queues ---
        physics_queue = Queue(maxsize=2)  # Stage 1a → Stage 1b (per pixel frame)
        sim_queue = Queue(maxsize=2)      # Stage 1b → Stage 2  (per block)
        ready_queue = Queue(maxsize=3)    # Stage 2  → Stage 3
        is_debug = simulator.config.get("debug", False)
        all_sim_frames = [] if is_debug else None

        # --- Stage 1a: Physics producer ---
        # Runs Genesis physics steps and puts per-frame point clouds into
        # physics_queue.  Does NOT touch the SVR renderer, so it can run
        # ahead of Stage 1b by up to physics_queue.maxsize frames.
        def physics_producer():
            import time
            try:
                for block_idx in range(num_blocks):
                    if stop_requested:
                        break
                    n_pixel = FRAMES_FIRST_BLOCK_PIXEL if block_idx == 0 else FRAMES_PER_BLOCK_PIXEL
                    for pf_idx in range(n_pixel):
                        if stop_requested:
                            break
                        t0 = time.perf_counter()
                        last_i = frame_steps - 1
                        for i in range(frame_steps):
                            updated_points = simulator.step(extract_points=(i == last_i))
                        t_step = time.perf_counter() - t0
                        # Capture frame_id here: render thread may be behind
                        frame_id = simulator.step_count
                        item = (block_idx, n_pixel, pf_idx,
                                updated_points, frame_id, t_step)
                        # Timed put so stop_requested is checked if render stops consuming
                        while not stop_requested:
                            try:
                                physics_queue.put(item, timeout=0.5)
                                break
                            except QueueFull:
                                pass
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                # Best-effort sentinel — render exits via stop_requested if queue stays full
                for _ in range(20):  # up to 10 s
                    try:
                        physics_queue.put(None, timeout=0.5)
                        break
                    except QueueFull:
                        pass

        # --- Stage 1b: Render + flow producer ---
        # Reads point clouds from physics_queue, runs SVR render + optical
        # flow + resize, accumulates per-block results, then forwards complete
        # blocks to sim_queue (same interface as the old simulation_producer).
        def render_flow_producer():
            import time
            try:
                current_block = -1
                flows, sim_frames, fg_masks, mesh_masks = [], [], [], []
                t_block_start = time.perf_counter()
                t_step_total = t_render_total = t_resize_total = 0.0

                while not stop_requested:
                    try:
                        item = physics_queue.get(timeout=0.5)
                    except QueueEmpty:
                        continue
                    if item is None:
                        break

                    block_idx, n_pixel, pf_idx, updated_points, frame_id, t_step = item

                    if block_idx != current_block:
                        current_block = block_idx
                        flows, sim_frames, fg_masks, mesh_masks = [], [], [], []
                        t_block_start = time.perf_counter()
                        t_step_total = t_render_total = t_resize_total = 0.0

                    t0 = time.perf_counter()
                    frame_pil, flow_2hw, fg_mask, mesh_mask = (
                        simulator.render_and_flow(updated_points, frame_id=frame_id)
                    )
                    t1 = time.perf_counter()
                    frame_pil = resize_and_crop_pil(frame_pil, start_y=simulator.crop_start)
                    t2 = time.perf_counter()

                    sim_frames.append(frame_pil)
                    flows.append(flow_2hw)
                    fg_masks.append(fg_mask)
                    mesh_masks.append(mesh_mask)

                    t_step_total   += t_step
                    t_render_total += t1 - t0
                    t_resize_total += t2 - t1

                    if len(sim_frames) == n_pixel:
                        t_queue_start = time.perf_counter()
                        if all_sim_frames is not None:
                            all_sim_frames.extend(sim_frames)
                        sim_queue.put((block_idx, flows, sim_frames, fg_masks, mesh_masks))
                        t_queue_end = time.perf_counter()
                        total_block = t_queue_end - t_block_start
                        print(f"[TIMING] sim block {block_idx}: "
                              f"physics step = {t_step_total:.3f}s, "
                              f"render+flow = {t_render_total:.3f}s, "
                              f"resize = {t_resize_total:.3f}s, "
                              f"queue put = {t_queue_end - t_queue_start:.3f}s, "
                              f"total = {total_block:.3f}s "
                              f"({n_pixel} frames)")
                        exp_logger.log_event(
                            "demo.stage1_render_flow_block",
                            total_block,
                            block_idx=block_idx,
                            pixel_frames=n_pixel,
                            physics_step_total_sec=t_step_total,
                            render_flow_total_sec=t_render_total,
                            resize_total_sec=t_resize_total,
                            queue_put_sec=t_queue_end - t_queue_start,
                        )
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                sim_queue.put(None)  # Sentinel

        # --- Stage 2: Noise Warping (lightweight, mostly CPU) ---
        def noise_warp_stage():
            import time
            try:
                while not stop_requested:
                    t_wait_start = time.perf_counter()
                    item = sim_queue.get()
                    t_wait_end = time.perf_counter()
                    if item is None:
                        break

                    block_idx, flows, sim_frames, fg_masks, mesh_masks = item

                    # Warp noise incrementally using optical flow
                    t0 = time.perf_counter()
                    for flow in flows:
                        noise_warper.warp_step(flow)
                    t1 = time.perf_counter()
                    structured_noise, sde_noise = noise_warper.get_block_noise(block_idx)
                    t2 = time.perf_counter()

                    ready_queue.put((
                        block_idx,
                        structured_noise,
                        sde_noise,
                        sim_frames, fg_masks, mesh_masks,
                    ))
                    t3 = time.perf_counter()

                    total_block = t3 - t_wait_end
                    print(f"[TIMING] warp block {block_idx}: "
                          f"queue wait = {t_wait_end - t_wait_start:.3f}s, "
                          f"warp steps = {t1 - t0:.3f}s, "
                          f"get_block_noise = {t2 - t1:.3f}s, "
                          f"queue put = {t3 - t2:.3f}s, "
                          f"total = {total_block:.3f}s")
                    exp_logger.log_event(
                        "demo.stage2_noise_warp_block",
                        total_block,
                        block_idx=block_idx,
                        queue_wait_sec=t_wait_end - t_wait_start,
                        warp_steps_sec=t1 - t0,
                        get_block_noise_sec=t2 - t1,
                        queue_put_sec=t3 - t2,
                    )
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                ready_queue.put(None)  # Sentinel

        # Start stages 1a, 1b, and 2 BEFORE prepare_generation so the
        # simulation pipeline (physics → render → warp) runs in parallel
        # with text encoding.  By the time prepare_generation() returns,
        # ready_queue may already contain block 0, eliminating the startup gap.
        physics_thread = threading.Thread(target=physics_producer, daemon=True)
        render_thread = threading.Thread(target=render_flow_producer, daemon=True)
        warp_thread = threading.Thread(target=noise_warp_stage, daemon=True)
        physics_thread.start()
        render_thread.start()
        warp_thread.start()

        # Text encoding (+ conditional dict setup) runs while sim pipeline
        # is already producing frames.
        generator.prepare_generation(prompt)

        # --- Stage 3: VAE Encode + Mask Build + Diffusion ---
        # --- Stage 4: Frame streaming (separate thread, runs concurrently) ---
        import time
        stream_queue = Queue(maxsize=2)  # Stage 3 → Stage 4

        def frame_streamer():
            """Stream frames to browser at FPS rate, decoupled from GPU work."""
            try:
                while not stop_requested:
                    item = stream_queue.get()
                    if item is None:
                        break
                    pixel_frames, blk_idx = item
                    for frame in pixel_frames:
                        if stop_requested:
                            break
                        b64 = base64.b64encode(_encode_jpeg(frame)).decode("ascii")
                        socketio.emit("frame", {"data": b64, "block": blk_idx})
                        socketio.sleep(1.0 / FPS)
            except Exception as e:
                import traceback
                traceback.print_exc()

        stream_thread = threading.Thread(target=frame_streamer, daemon=True)
        stream_thread.start()

        t_block_end = time.perf_counter()

        while not stop_requested:
            t_wait_start = time.perf_counter()
            item = ready_queue.get()
            t_wait_end = time.perf_counter()
            if item is None:
                break

            (block_idx, structured_noise, sde_noise,
             sim_frames, fg_masks, mesh_masks) = item

            print(f"[TIMING] block {block_idx}: queue wait = {t_wait_end - t_wait_start:.3f}s, "
                  f"gap since prev block end = {t_wait_end - t_block_end:.3f}s")

            socketio.emit("status", {
                "message": f"Block {block_idx + 1}/{num_blocks} — Generating...",
                "block": block_idx,
                "total_blocks": num_blocks,
            })

            # 1. Encode simulation frames to latent (GPU)
            t0 = time.perf_counter()
            log_gpu(f"stage3 block {block_idx}: before VAE encode")
            sim_frames_tensor = _frames_to_tensor(sim_frames)
            sim_latent = generator.pipeline.encode_vae.cached_encode_to_latent(
                sim_frames_tensor.to(device=generator.device, dtype=torch.bfloat16),
                is_first=(block_idx == 0),
            )
            if sim_latent.shape[1] > FRAMES_PER_BLOCK:
                sim_latent = sim_latent[:, :FRAMES_PER_BLOCK]
            elif sim_latent.shape[1] < FRAMES_PER_BLOCK:
                pad = FRAMES_PER_BLOCK - sim_latent.shape[1]
                sim_latent = torch.cat(
                    [sim_latent, sim_latent[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1,
                )
            t1 = time.perf_counter()
            log_gpu(f"stage3 block {block_idx}: after VAE encode")

            # 2. Build masks
            sim_mask = _downsample_masks(fg_masks, FRAMES_PER_BLOCK, crop_start=simulator.crop_start, device=generator.device)
            sim_franka_mask = _downsample_masks(mesh_masks, FRAMES_PER_BLOCK, crop_start=simulator.crop_start, device=generator.device)
            t2 = time.perf_counter()
            log_gpu(f"stage3 block {block_idx}: after mask build")

            # 3. Diffusion denoising
            pixel_frames = generator.generate_block(
                block_idx=block_idx,
                structured_noise=structured_noise,
                sim_latent=sim_latent,
                sde_noise=sde_noise,
                sim_mask=sim_mask,
                sim_franka_mask=sim_franka_mask,
            )
            t3 = time.perf_counter()

            # Hand off frames to streaming thread (non-blocking)
            stream_queue.put((pixel_frames, block_idx))

            total_block = t3 - t_wait_end
            print(f"[TIMING] block {block_idx}: VAE encode = {t1 - t0:.3f}s, "
                  f"mask build = {t2 - t1:.3f}s, diffusion = {t3 - t2:.3f}s, "
                  f"total = {total_block:.3f}s")
            exp_logger.log_event(
                "demo.stage3_diffusion_block",
                total_block,
                block_idx=block_idx,
                queue_wait_sec=t_wait_end - t_wait_start,
                vae_encode_sec=t1 - t0,
                mask_build_sec=t2 - t1,
                diffusion_sec=t3 - t2,
            )
            t_block_end = t3

        stream_queue.put(None)  # Sentinel
        physics_thread.join(timeout=10)
        render_thread.join(timeout=10)
        warp_thread.join(timeout=10)
        stream_thread.join(timeout=30)

        # Save debug outputs only if debug mode is on
        if simulator.config.get("debug", False):
            if noise_warper.noise_buffer:
                noise_stack = torch.stack(noise_warper.noise_buffer, dim=0)  # (T, C, H, W)
                downscale_factor = DEFAULT_HEIGHT // LATENT_H  # 480 // 60 = 8
                noise_latent = F.interpolate(
                    noise_stack, size=(LATENT_H, LATENT_W), mode="area",
                ) * downscale_factor  # (T, 32, 60, 104)
                numpy_noises = noise_latent.cpu().permute(0, 2, 3, 1).numpy().astype(np.float16)  # (T, H, W, C)

                debug_dir = Path(simulator.config.get("output_folder", "/tmp/demo_debug"))
                debug_dir.mkdir(parents=True, exist_ok=True)

                noises_path = debug_dir / "noises.npy"
                np.save(noises_path, numpy_noises)

                noise_vis = np.clip(numpy_noises[:, :, :, :3].astype(np.float32) / 4 + 0.5, 0, 1)
                noise_vis = (noise_vis * 255).astype(np.uint8)
                noise_video_tensor = torch.from_numpy(noise_vis)  # (T, H, W, 3) uint8
                from torchvision.io import write_video
                noise_mp4_path = str(debug_dir / "noise_video.mp4")
                write_video(noise_mp4_path, noise_video_tensor, fps=30, video_codec="libx264")
                print(f"Noise saved to: {noises_path}  video: {noise_mp4_path}")

            simulator.save_debug_outputs(sim_frames=all_sim_frames)

        socketio.emit("generation_complete", {})
        socketio.emit("status", {"message": "Generation complete"})
        exp_logger.finalize(status="completed")

    except Exception as e:
        socketio.emit("error", {"message": f"Generation error: {str(e)}"})
        exp_logger.finalize(status="failed", error=str(e))
        import traceback
        traceback.print_exc()
    finally:
        is_generating = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_first_frame():
    """Locate the first-frame image for video generation."""
    case_path = simulator.demo_data_path
    candidate = case_path / "first_frame.png"
    if candidate.exists():
        return str(candidate)
    input_path = Path(simulator.config.get("data_path", "")) / "input.png"
    if input_path.exists():
        return str(input_path)
    return str(candidate)  # fallback, may error later with clear message


def _frames_to_tensor(frames_pil):
    """Convert list of PIL frames (already 480x832) to tensor [1, C, T, H, W] in [-1, 1]."""
    arrays = []
    for f in frames_pil:
        arr = np.array(f.convert("RGB"))
        arr = arr.astype(np.float32) / 127.5 - 1.0
        arrays.append(torch.from_numpy(arr))
    tensor = torch.stack(arrays, dim=0).permute(3, 0, 1, 2).contiguous()
    return tensor.unsqueeze(0)


def _downsample_masks(masks, target_frames, crop_start=176, device="cuda"):
    """Downsample list of mask tensors to target_frames latent frames."""
    if not masks or all(m is None for m in masks):
        return None

    processed = []
    for m in masks:
        if m is None:
            processed.append(torch.zeros(1, 1, LATENT_H, LATENT_W, device=device))
            continue
        if isinstance(m, torch.Tensor):
            m = m.to(device=device)
            if m.dim() == 3:
                m = m.squeeze(-1)
            m_832 = F.interpolate(
                m.float().unsqueeze(0).unsqueeze(0),
                size=(832, 832), mode="bilinear", align_corners=False,
            )
            m_cropped = m_832[:, :, crop_start:crop_start + DEFAULT_HEIGHT, :]
            m_latent = F.interpolate(
                m_cropped, size=(LATENT_H, LATENT_W),
                mode="bilinear", align_corners=False,
            )
            processed.append(m_latent)
        else:
            processed.append(torch.zeros(1, 1, LATENT_H, LATENT_W, device=device))

    stacked = torch.cat(processed, dim=0)
    T = stacked.shape[0]

    time_averaged = []
    for i in range(0, T, TEMPORAL_FACTOR):
        group = stacked[i:i + TEMPORAL_FACTOR]
        time_averaged.append(group.mean(dim=0, keepdim=True))
    stacked = torch.cat(time_averaged, dim=0)

    if stacked.shape[0] > target_frames:
        stacked = stacked[:target_frames]
    elif stacked.shape[0] < target_frames:
        pad = target_frames - stacked.shape[0]
        stacked = torch.cat(
            [stacked, stacked[-1:].repeat(pad, 1, 1, 1)], dim=0,
        )

    result = stacked.squeeze(1).unsqueeze(0)
    return (result > 0.5).bool()


def _encode_jpeg(frame_np, quality=85):
    img = Image.fromarray(frame_np)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _encode_pil_b64(pil_img, fmt="JPEG", quality=85):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Pipeline warmup — compile CUDA kernels before first user request
# ---------------------------------------------------------------------------

def _warmup_pipeline():
    """Run dummy passes through each pipeline stage to trigger CUDA JIT.

    Without this, the first user-facing generation pays ~24s of kernel
    compilation across simulation render, noise warping, and diffusion.
    """
    global startup_logger
    print("[4/5] Warming up CUDA kernels (one-time cost)...")
    torch.set_grad_enabled(False)

    # 1. Warm up simulation render + optical flow
    t0 = time.perf_counter()
    for _pass in range(2):
        for _ in range(simulator.frame_steps):
            updated_points = simulator.step()
        simulator.render_and_flow(updated_points)

    # Reset simulation state (scene.reset restores to built state)
    simulator.scene.reset()
    simulator.case_handler.fix_particles()  # re-pin after reset
    simulator.step_count = 0
    simulator.svr.previous_frame_data = None
    simulator.svr.optical_flow = np.array([])
    simulator.svr._last_optical_flow = None
    simulator.svr._prev_fg_frags_idx = None
    simulator.svr._prev_fg_frags_dists = None
    # Keep cache_bg — background render is reusable
    t1 = time.perf_counter()
    sim_render_warmup = t1 - t0
    print(f"      Sim + render warmup: {sim_render_warmup:.1f}s")
    if startup_logger is not None:
        startup_logger.log_event("demo.startup_warmup.sim_render", sim_render_warmup)

    # 2. Warm up noise warper (grid_sample, meshgrid, interpolate kernels)
    dummy_flow = np.zeros((2, 512, 512), dtype=np.float32)
    noise_warper.warp_step(dummy_flow)
    noise_warper.reset()
    t2 = time.perf_counter()
    noise_warp_warmup = t2 - t1
    print(f"      Noise warp warmup:   {noise_warp_warmup:.1f}s")
    if startup_logger is not None:
        startup_logger.log_event("demo.startup_warmup.noise_warp", noise_warp_warmup)

    # 3. Warm up VAE encode + diffusion (transformer attention kernels)
    generator.prepare_generation(default_prompt)

    # Dummy VAE encode
    dummy_pixel = torch.zeros(
        1, 3, FRAMES_FIRST_BLOCK_PIXEL, DEFAULT_HEIGHT, DEFAULT_WIDTH,
        device=generator.device, dtype=torch.bfloat16,
    )
    sim_latent = generator.pipeline.encode_vae.cached_encode_to_latent(
        dummy_pixel, is_first=True,
    )
    if sim_latent.shape[1] > FRAMES_PER_BLOCK:
        sim_latent = sim_latent[:, :FRAMES_PER_BLOCK]
    elif sim_latent.shape[1] < FRAMES_PER_BLOCK:
        pad = FRAMES_PER_BLOCK - sim_latent.shape[1]
        sim_latent = torch.cat(
            [sim_latent, sim_latent[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1,
        )

    # Dummy diffusion block
    dummy_noise = torch.randn(
        1, FRAMES_PER_BLOCK, LATENT_C, LATENT_H, LATENT_W,
        device=generator.device, dtype=torch.bfloat16,
    )
    generator.generate_block(
        block_idx=0,
        structured_noise=dummy_noise,
        sim_latent=sim_latent,
    )

    # Run two more dummy blocks to warm up the KV-cache-populated code
    # paths (blocks 1+ are structurally different from block 0 because the
    # self-attention KV cache is non-empty).  Without this, real generation
    # blocks 0 and 1 hit slow cuDNN algorithm selection on first use, taking
    # ~4s each instead of ~1s.  The crossattn_cache stays valid across these
    # extra blocks (same prompt), so they run fast (~1s each).
    for _blk in range(1, 3):
        _dummy_latent = torch.zeros(
            1, FRAMES_PER_BLOCK, LATENT_C, LATENT_H, LATENT_W,
            device=generator.device, dtype=torch.bfloat16,
        )
        _dummy_noise = torch.randn_like(_dummy_latent)
        generator.generate_block(
            block_idx=_blk,
            structured_noise=_dummy_noise,
            sim_latent=_dummy_latent,
        )

    # Reset generator state (KV self-attention cache + VAE caches).
    # crossattn_cache is intentionally preserved: it is text-conditioned
    # and stays valid for the default prompt, so real generation blocks 0
    # and 1 skip the expensive cold re-initialization.
    generator.reset()
    generator.pipeline.encode_vae.model.clear_cache()
    t3 = time.perf_counter()
    vae_diffusion_warmup = t3 - t2
    total_warmup = t3 - t0
    print(f"      VAE + diffusion warmup: {vae_diffusion_warmup:.1f}s")
    print(f"      Total warmup: {total_warmup:.1f}s — first generation will be fast.")
    if startup_logger is not None:
        startup_logger.log_event("demo.startup_warmup.vae_diffusion", vae_diffusion_warmup)
        startup_logger.finalize(status="completed", total_warmup_sec=total_warmup)
    log_gpu("after pipeline warmup")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def main():
    global simulator, noise_warper, generator, demo_case_handler
    global preview_b64, default_prompt, case_name, num_blocks, startup_logger

    parser = argparse.ArgumentParser(description="RealWonder Interactive Demo")
    parser.add_argument("--demo_data", type=str, required=True,
                        help="Path to demo data directory (e.g. demo_data/lamp)")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to video generation model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_log", action="store_true",
                        help="Enable GPU memory logging (disabled by default)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug outputs (disabled by default)")
    parser.add_argument("--taehv", action="store_true",
                        help="Use TAEHV tiny VAE decoder (faster but slightly lower quality)")
    args = parser.parse_args()

    if not args.gpu_log:
        set_gpu_logging(False)

    demo_data_path = Path(args.demo_data)
    case_name = demo_data_path.name
    startup_logger = ExperimentLogger(
        experiment_name="interactive_demo_startup",
        run_name=case_name,
        output_dir=demo_data_path / "experiment_logs",
        metadata={
            "checkpoint_path": args.checkpoint_path,
            "demo_data": str(demo_data_path),
        },
    )

    if not demo_data_path.exists() or not (demo_data_path / "config.yaml").exists():
        print(f"ERROR: {demo_data_path} does not exist or has no config.yaml")
        return

    # ---- Load case config and derive SDEdit parameters ----
    import yaml
    with open(demo_data_path / "config.yaml") as f:
        case_config = yaml.safe_load(f)
    sdedit_cfg = load_case_sdedit_config(case_config)
    num_blocks = sdedit_cfg["num_blocks"]
    print(f"Case SDEdit config: {sdedit_cfg}")

    # ---- Step 1: Initialize video generator ----
    print(f"[1/5] Initializing video generator from {args.checkpoint_path} ...")
    log_gpu("before video generator init")
    with startup_logger.time_block("demo.startup.initialize_video_generator"):
        generator = StreamingVideoGenerator(
            checkpoint_path=args.checkpoint_path,
            num_pixel_frames=sdedit_cfg["num_pixel_frames"],
            denoising_steps=sdedit_cfg["denoising_step_list"],
            mask_dropin_step=sdedit_cfg["mask_dropin_step"],
            franka_step=sdedit_cfg["franka_step"],
            use_ema=args.use_ema,
            seed=args.seed,
            enable_taehv=args.taehv,
        )
        generator.setup()
    log_gpu("after video generator setup")
    print("      Video generator ready.")

    # ---- Step 2: Initialize simulator (Genesis scene) ----
    print(f"[2/5] Loading case '{case_name}' and building Genesis scene ...")
    log_gpu("before simulator init")
    # Per-case config overrides (e.g. disable built-in force fields for
    # cases where the demo handler applies forces interactively).
    config_overrides = {}
    if case_name == "santa_cloth":
        config_overrides["skip_force_fields"] = True
    with startup_logger.time_block("demo.startup.initialize_simulator"):
        simulator = InteractiveSimulator(
            str(demo_data_path), config_overrides=config_overrides,
        )
        if not args.debug:
            simulator.config["debug"] = False
    log_gpu("after simulator init")

    # Create per-case demo handler and attach to simulator
    with startup_logger.time_block("demo.startup.initialize_case_handler"):
        demo_case_handler = get_demo_case_handler(case_name, simulator.config)
        demo_case_handler.set_object_masks(simulator.object_masks_b64)
        simulator.set_demo_case_handler(demo_case_handler)
    print(f"      Demo case handler: {type(demo_case_handler).__name__}")

    with startup_logger.time_block("demo.startup.initialize_noise_warper"):
        noise_warper = StreamingNoiseWarper(crop_start=simulator.crop_start)
    log_gpu("after noise warper init")
    print("      Simulator and noise warper ready.")

    # ---- Step 3: Pre-compute first frame encoding + KV cache + default prompt ----
    print("[3/5] Pre-computing first frame encoding + KV cache + default prompt ...")
    with startup_logger.time_block("demo.startup.precompute_first_frame"):
        first_frame_path = _find_first_frame()
        preview_pil = Image.open(first_frame_path).convert("RGB")
        preview_b64 = _encode_pil_b64(preview_pil)
        default_prompt = simulator.config.get("vgen_prompt", "A video of physical simulation")
        generator.precompute_first_frame(first_frame_path, default_prompt=default_prompt)
    log_gpu("after first frame pre-computation")
    print(f"      First frame pre-computed from {first_frame_path}. All components initialized.")

    # ---- Step 4: Warm up CUDA kernels ----
    _warmup_pipeline()

    # ---- Step 5: Start server ----
    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser.\n")
    socketio.run(app, host=args.host, port=args.port, debug=False,
                 allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
