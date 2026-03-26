"""I2V Flow inference with SDEdit from simulation results.

Based on infer_flow.py with SDEdit support added. Loads simulation data
(structured noise, frames, masks) from a simulation output directory and
uses CausalInferencePipelineSDEdit for simulation-guided video generation.

Example usage:
    python infer_sim.py \
        --checkpoint_path /path/to/model.pt \
        --sim_data_path result/tree/25-01_21-35-02/final_sim \
        --output_path ./output.mp4
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
import imageio

from vidgen import (
    WanImageEncoder,
    WanVideoVAE,
    WanVideoUnit_ImageEmbedderCLIP,
    WanVideoUnit_ImageEmbedderVAE,
    set_seed,
    apply_config_overrides,
    gpu,
    get_cuda_free_memory_gb,
    DynamicSwapInstaller,
    load_noise,
    load_first_frame,
    CausalInferencePipelineSDEdit
)
from simulation.experiment_logging import ExperimentLogger

def load_sim_frames(frames_dir, height=480, width=832):
    """Load simulation frames from a directory of PNGs.

    Returns:
        Tensor of shape [1, C, T, H, W] normalized to [-1, 1].
    """
    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frame_*.png files found in {frames_dir}")

    frames = []
    for fp in frame_files:
        img = Image.open(fp).convert("RGB").resize((width, height))
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
        frames.append(torch.from_numpy(arr))

    # [T, H, W, C] -> [C, T, H, W]
    frames_tensor = torch.stack(frames, dim=0).permute(3, 0, 1, 2).contiguous()
    return frames_tensor.unsqueeze(0)  # [1, C, T, H, W]


def load_sim_masks(mask_path, target_frames):
    """Load and temporally resize simulation masks.

    Args:
        mask_path: Path to a .pt mask file of shape [T_mask, H, W].
        target_frames: Target number of latent frames.

    Returns:
        Tensor of shape [1, target_frames, H, W] (bool).
    """
    masks = torch.load(mask_path, map_location="cpu", weights_only=True)  # [T_mask, H, W]
    T_mask = masks.shape[0]
    if T_mask != target_frames:
        # Nearest-neighbor temporal resize
        indices = torch.linspace(0, T_mask - 1, steps=target_frames).round().long().clamp(0, T_mask - 1)
        masks = masks[indices]
    return masks.unsqueeze(0)  # [1, T, H, W]


def main():
    parser = argparse.ArgumentParser(description="I2V Flow Inference with SDEdit from Simulation")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--sim_data_path", type=str, required=True,
                        help="Path to simulation final_sim directory (contains noises.npy, frames/, masks, etc.)")
    parser.add_argument("--output_path", type=str, default="./output_sim.mp4", help="Output video path")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_degradation", type=float, default=0.5,
                        help="Degradation level for noise (0.0 = pure structured noise)")
    parser.add_argument("--local_attn_size", type=int, default=21, help="Local attention size for causal model")

    args, additional_args = parser.parse_known_args()

    device = torch.device("cuda")
    set_seed(args.seed)
    sim_data_path = Path(args.sim_data_path)
    experiment_logger = ExperimentLogger(
        experiment_name="offline_video_generation",
        run_name=sim_data_path.parent.name,
        output_dir=sim_data_path / "experiment_logs",
        metadata={
            "checkpoint_path": args.checkpoint_path,
            "sim_data_path": args.sim_data_path,
            "output_path": args.output_path,
        },
    )

    print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
    low_memory = get_cuda_free_memory_gb(gpu) < 40

    torch.set_grad_enabled(False)

    # -------------------------------------------------------------------------
    # Load simulation config for SDEdit parameters
    # -------------------------------------------------------------------------
    sim_config_path = sim_data_path / "config.yaml"
    with experiment_logger.time_block("infer_sim.load_sim_config"):
        sim_config = OmegaConf.load(sim_config_path)
    denoising_step_list = list(sim_config.denoising_step_list)
    mask_dropin_step = int(sim_config.mask_dropin_step)
    num_output_frames = int(sim_config.num_output_frames)
    print(f"Loaded from config.yaml: denoising_step_list={denoising_step_list}, mask_dropin_step={mask_dropin_step}, num_output_frames={num_output_frames}")

    # -------------------------------------------------------------------------
    # Build config
    # -------------------------------------------------------------------------
    DEFAULT_CONFIG = {
        "independent_first_frame": False,
        "warp_denoising_step": True,
        "context_noise": 0,
        "causal": True,
        "i2v": True,
        "i2v_flow": True,
        "height": 480,
        "width": 832,
        "num_frame_per_block": 3,
        "denoising_step_list": denoising_step_list,
        "mask_dropin_step": mask_dropin_step,
        "model_kwargs": {
            "sink_size": 1,
            "local_attn_size": args.local_attn_size,
            "timestep_shift": 5.0,
        },
    }
    config = OmegaConf.create(DEFAULT_CONFIG)
    config = apply_config_overrides(config, additional_args)

    # Create output directory
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Initialize pipeline (SDEdit version)
    # -------------------------------------------------------------------------
    with experiment_logger.time_block("infer_sim.initialize_pipeline"):
        pipeline = CausalInferencePipelineSDEdit(config, device=device)

    # Load checkpoint
    if args.checkpoint_path:
        with experiment_logger.time_block("infer_sim.load_checkpoint"):
            state_dict = torch.load(args.checkpoint_path, map_location="cpu")
            key = 'generator_ema' if args.use_ema else 'generator'
            gen_state_dict = state_dict[key]
            try:
                pipeline.generator.load_state_dict(gen_state_dict)
            except:
                gen_state_dict = {k.replace('._fsdp_wrapped_module', ''): v for k, v in gen_state_dict.items()}
                pipeline.generator.load_state_dict(gen_state_dict)

    with experiment_logger.time_block("infer_sim.move_models"):
        pipeline = pipeline.to(dtype=torch.bfloat16)
        if low_memory:
            DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
        else:
            pipeline.text_encoder.to(device=gpu)
        pipeline.generator.to(device=gpu)
        pipeline.vae.to(device=gpu)

    # Setup I2V flow processors
    pipeline.processor_dtype = torch.float32
    pipeline.processor_device = gpu
    with experiment_logger.time_block("infer_sim.setup_processors"):
        pipeline.processor_vae = WanVideoVAE().to(device=pipeline.processor_device, dtype=pipeline.processor_dtype)
        pipeline.processor_ienc = WanImageEncoder().to(device=pipeline.processor_device, dtype=pipeline.processor_dtype)

        pipeline.processor_vae.requires_grad_(False)
        pipeline.processor_ienc.requires_grad_(False)

        for p in pipeline.processor_vae.parameters():
            p.data = p.data.to(dtype=pipeline.processor_dtype)
        for b in pipeline.processor_vae.buffers():
            b.data = b.data.to(dtype=pipeline.processor_dtype)

        pipeline.processors = [
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP()
        ]

    # -------------------------------------------------------------------------
    # Load simulation data
    # -------------------------------------------------------------------------
    noise_path = sim_data_path / "noises.npy"
    first_frame_path = sim_data_path / "resized_input_image.png"
    frames_dir = sim_data_path / "frames"

    # 1. Load structured noise
    print(f"Loading noise from: {noise_path}")
    with experiment_logger.time_block("infer_sim.load_structured_noise"):
        noise_data = load_noise(
            noise_path=str(noise_path),
            target_frames=num_output_frames,
            channel_dim=16,
            downsample_mode="nearest",
            eval_degradation=args.eval_degradation,
        )

    # 2. Load prompt from sim_data_path/prompt.txt
    prompt_path = sim_data_path / "prompt.txt"
    with experiment_logger.time_block("infer_sim.load_prompt"):
        with open(prompt_path, 'r') as f:
            prompt = f.read().strip()
    print(f"Prompt: {prompt}")

    # 3. Load first frame
    print(f"Loading first frame from: {first_frame_path}")
    with experiment_logger.time_block("infer_sim.load_first_frame"):
        input_image = load_first_frame(str(first_frame_path), height=480, width=832)  # [C, H, W]

    # 4. Load simulation frames for SDEdit
    print(f"Loading simulation frames from: {frames_dir}")
    with experiment_logger.time_block("infer_sim.load_sim_frames"):
        sim_frames = load_sim_frames(frames_dir, height=480, width=832)  # [1, C, T, H, W]
    print(f"  Loaded {sim_frames.shape[2]} simulation frames")

    # 5. Load object masks (optional, for mask dropin)
    sim_masks = None
    if mask_dropin_step > 0:
        mask_file = str(sim_data_path / "points_masks_downsampled.pt")
        if os.path.exists(mask_file):
            print(f"Loading object masks from: {mask_file}")
            with experiment_logger.time_block("infer_sim.load_object_masks"):
                sim_masks = load_sim_masks(mask_file, target_frames=num_output_frames)
                sim_masks = sim_masks.to(device=device)
            print(f"  Object mask shape: {sim_masks.shape}")
        else:
            print(f"Warning: mask_dropin_step={mask_dropin_step} but mask file not found: {mask_file}")
            print("  Proceeding without mask dropin.")

    # 6. Load franka/mesh masks (for weak sdedit on manipulator region)
    sim_franka_masks = None
    franka_mask_file = sim_data_path / "mesh_masks_downsampled.pt"
    if franka_mask_file.exists():
        with experiment_logger.time_block("infer_sim.load_mesh_masks"):
            franka_masks_raw = load_sim_masks(str(franka_mask_file), target_frames=num_output_frames)
        if franka_masks_raw.any():
            sim_franka_masks = franka_masks_raw.to(device=device)
            print(f"Loading franka masks from: {franka_mask_file}")
            print(f"  Franka mask shape: {sim_franka_masks.shape}")
        else:
            print(f"Franka masks are all False, skipping franka mask sdedit.")

    # -------------------------------------------------------------------------
    # Prepare tensors
    # -------------------------------------------------------------------------
    structured_noise = noise_data['structured_noise'].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    structured_noise_sde = noise_data.get('structured_noise_sde')
    if structured_noise_sde is not None:
        structured_noise_sde = structured_noise_sde.unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    # Encode simulation frames to latent space for SDEdit
    sim_latent = None
    if pipeline.sdedit:
        print("Encoding simulation frames to latent space...")
        with experiment_logger.time_block("infer_sim.encode_sim_frames"):
            sim_frames_device = sim_frames.to(device=device, dtype=torch.bfloat16)  # [1, C, T, H, W]
            sim_latent = pipeline.vae.encode_to_latent(sim_frames_device)  # [1, T_latent, C, H, W]
            sim_latent = sim_latent.to(device=device, dtype=torch.bfloat16)
        print(f"  sim_latent shape: {sim_latent.shape}")

        # Trim or pad sim_latent to match noise frames
        if sim_latent.shape[1] > num_output_frames:
            sim_latent = sim_latent[:, :num_output_frames]
        elif sim_latent.shape[1] < num_output_frames:
            # Pad by repeating last frame
            pad_size = num_output_frames - sim_latent.shape[1]
            sim_latent = torch.cat([
                sim_latent,
                sim_latent[:, -1:].repeat(1, pad_size, 1, 1, 1)
            ], dim=1)
        print(f"  sim_latent shape (after align): {sim_latent.shape}")

    # Prepare batch for I2V processors
    pixel_num_frames = num_output_frames * 4 - 3  # 21 -> 81
    batch = {
        'input_image': input_image.unsqueeze(0),  # [1, C, H, W]
        'end_image': None,
        'height': 480,
        'width': 832,
        'num_frames': pixel_num_frames,
    }

    # -------------------------------------------------------------------------
    # Generate video
    # -------------------------------------------------------------------------
    print("Generating video...")
    with experiment_logger.time_block("infer_sim.diffusion_inference"):
        video, _ = pipeline.inference(
            noise=structured_noise,
            text_prompts=[prompt],
            return_latents=True,
            batch_sample=batch,
            sim_latent=sim_latent,
            sim_masks=sim_masks,
            sim_franka_masks=sim_franka_masks,
            low_memory=low_memory,
            device=device,
            structured_noise_sde=structured_noise_sde,
        )

    # Process output
    with experiment_logger.time_block("infer_sim.postprocess_video_tensor"):
        video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video = (255.0 * video[0]).to(torch.uint8)

    # Clear VAE cache
    with experiment_logger.time_block("infer_sim.clear_vae_cache"):
        pipeline.vae.model.clear_cache()

    # Save the video
    print(f"Saving video to: {args.output_path}")
    with experiment_logger.time_block("infer_sim.save_video"):
        imageio.mimwrite(args.output_path, video.numpy(), fps=10)
    print("Done!")
    experiment_logger.finalize(
        status="completed",
        output_path=args.output_path,
        num_output_frames=num_output_frames,
    )

if __name__ == "__main__":
    main()
