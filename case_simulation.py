import argparse
import torch
import os
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import torch.nn.functional as F
import random
from datetime import datetime
from simulation.genesis_simulator import DiffSim
from simulation.image23D.noise_warp.make_warped_noise import NoiseWarper
from simulation.utils import save_video_from_pil, resize_and_crop_pil, visualize_optical_flow_advanced
from simulation.compat import (
    genesis_flow_to_chw,
    resize_and_crop_flow_chw,
    resolve_torch_device,
    save_json,
)

def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def process_simulated_results(input_image, raw_video_frames, points_masks, mesh_masks, crop_start=176):
    input_image = resize_and_crop_pil(input_image, crop_start)
    raw_video_frames = [resize_and_crop_pil(frame, crop_start) for frame in raw_video_frames]
    points_masks = preprocess_masks_downsample(points_masks, crop_start=crop_start)
    mesh_masks = preprocess_masks_downsample(mesh_masks, crop_start=crop_start)

    return input_image, raw_video_frames, points_masks, mesh_masks

def preprocess_masks_downsample(masks, crop_start=176):
    '''
    input: list of numpy array (512, 512, 1)
    output: 
    '''
    num_masks = len(masks)
    masks = torch.stack(masks, dim=0).squeeze(-1)
    resized_masks = F.interpolate(masks.unsqueeze(1).float(), size=(832, 832), mode='bilinear', align_corners=False)
    crop_height = 480
    crop_width = 832
    start_y = crop_start
    cropped_masks = resized_masks[:, :, start_y:start_y + crop_height, :]
    # assert cropped_masks.shape == (48, 1, 480, 832)
    masks_downsampled = F.interpolate(cropped_masks.float(), size=(60, 104), mode='bilinear', align_corners=False).squeeze(1)
    time_averaged_masks = []
    for i in range(0, num_masks, 4):
        time_averaged_masks.append(masks_downsampled[i : i + 4, :, :].mean(dim=0, keepdim=True))
    masks_downsampled = torch.cat(time_averaged_masks, dim=0)
    masks_downsampled = masks_downsampled > 0.5
    return masks_downsampled # torch.Size([12, 60, 104])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RealWonder Genesis simulation and export training/data streams."
    )
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for reconstruction/rendering: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--allow_cpu_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback to CPU when the CUDA torch build cannot run kernels on this GPU.",
    )
    parser.add_argument(
        "--genesis_backend",
        choices=["gpu", "cpu"],
        default=None,
        help="Genesis backend. Defaults to config.genesis_backend or gpu.",
    )
    parser.add_argument(
        "--noise_flow_source",
        choices=["raft", "genesis", "none"],
        default=None,
        help=(
            "Source used for noises.npy. raft computes RAFT on rendered frames; "
            "genesis uses renderer flow; none exports data without noise warp."
        ),
    )
    parser.add_argument(
        "--skip_noise_warp",
        action="store_true",
        help="Alias for --noise_flow_source none. Useful for data export on unsupported CUDA builds.",
    )
    parser.add_argument(
        "--save_raw_frames",
        action="store_true",
        help="Also save uncropped 512x512 Genesis render frames.",
    )
    parser.add_argument(
        "--raft_version",
        choices=["large", "small"],
        default="large",
        help="Torchvision RAFT variant used when --noise_flow_source raft.",
    )
    parser.add_argument(
        "--flow_viz",
        action="store_true",
        help="Save optical flow visualizations even when config.debug is false.",
    )
    return parser.parse_args()


def save_frames(frames, folder):
    os.makedirs(folder, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(folder, f"frame_{i:04d}.png"))


def main():
    args = parse_args()
    config = OmegaConf.load(args.config_path)

    device = resolve_torch_device(args.device, allow_cpu_fallback=args.allow_cpu_fallback)
    config["device"] = str(device)
    if args.genesis_backend is not None:
        config["genesis_backend"] = args.genesis_backend

    noise_flow_source = args.noise_flow_source or config.get("noise_flow_source", "raft")
    if args.skip_noise_warp:
        noise_flow_source = "none"

    timestamp = datetime.now().strftime("%d-%m_%H-%M-%S")
    output_folder = os.path.join(config["output_folder"], timestamp)
    os.makedirs(output_folder, exist_ok=True)
    config["output_folder"] = output_folder
    debug = config.get("debug", False)

    if debug:
        debug_config_save_path = os.path.join(config["output_folder"], "config.yaml")
        OmegaConf.save(config, debug_config_save_path)

    set_seed(config["seed"])
    torch.set_grad_enabled(False)

    input_image = Image.open(os.path.join(config["data_path"], "input.png")).convert("RGB")

    genesis_simulator = DiffSim(config)
    raw_video_frames, points_masks, mesh_masks = genesis_simulator.simulation_pc_render()

    input_image, video_frames, points_masks_downsampled, mesh_masks_downsampled = process_simulated_results(
        input_image,
        raw_video_frames,
        points_masks,
        mesh_masks,
        crop_start=config["crop_start"],
    )

    final_sim_folder = os.path.join(output_folder, "final_sim")
    os.makedirs(final_sim_folder, exist_ok=True)

    config_save_path = os.path.join(final_sim_folder, "config.yaml")
    OmegaConf.save(config, config_save_path)

    genesis_flows_512 = genesis_flow_to_chw(genesis_simulator.svr.optical_flow)
    genesis_flow_path = os.path.join(final_sim_folder, "genesis_flows_512.npy")
    np.save(genesis_flow_path, genesis_flows_512)

    genesis_flows_video = resize_and_crop_flow_chw(
        genesis_flows_512,
        target_hw=(832, 832),
        crop_start=config["crop_start"],
        crop_height=480,
    )
    genesis_flow_video_path = os.path.join(final_sim_folder, "genesis_flows_480x832.npy")
    np.save(genesis_flow_video_path, genesis_flows_video)

    frame_folder = os.path.join(final_sim_folder, "frames")
    save_frames(video_frames, frame_folder)

    if args.save_raw_frames:
        save_frames(raw_video_frames, os.path.join(final_sim_folder, "raw_frames_512"))

    if debug or args.flow_viz:
        visualize_optical_flow_advanced(
            frame_folder,
            genesis_flow_video_path,
            os.path.join(final_sim_folder, "genesis_optical_flow_viz"),
            arrow_density=30,
        )

    if noise_flow_source != "none":
        noise_warper = NoiseWarper()
        if noise_flow_source == "genesis":
            noise_warper.process(
                genesis_flows_512,
                final_sim_folder,
                crop_start=config["crop_start"],
                input_flow=True,
                debug=debug,
                device=device,
            )
        elif noise_flow_source == "raft":
            noise_warper.process(
                video_frames,
                final_sim_folder,
                crop_start=config["crop_start"],
                input_flow=False,
                debug=debug,
                device=device,
                raft_version=args.raft_version,
            )
        else:
            raise ValueError(f"Unsupported noise_flow_source: {noise_flow_source}")
    else:
        print("[export] skipped noise warp; noises.npy was not generated")

    points_masks_path = os.path.join(final_sim_folder, "points_masks_downsampled.pt")
    torch.save(points_masks_downsampled.cpu(), points_masks_path)
    mesh_masks_path = os.path.join(final_sim_folder, "mesh_masks_downsampled.pt")
    torch.save(mesh_masks_downsampled.cpu(), mesh_masks_path)

    video_path = os.path.join(final_sim_folder, "simulation.mp4")
    save_video_from_pil(video_frames, video_path, fps=10)

    input_image_path = os.path.join(final_sim_folder, "resized_input_image.png")
    input_image.save(input_image_path)

    prompt_txt_path = os.path.join(final_sim_folder, "prompt.txt")
    with open(prompt_txt_path, "w") as f:
        f.write(config["vgen_prompt"])

    save_json(
        os.path.join(final_sim_folder, "metadata.json"),
        {
            "config_path": args.config_path,
            "torch_device": str(device),
            "genesis_backend": config.get("genesis_backend", "gpu"),
            "noise_flow_source": noise_flow_source,
            "raft_version": args.raft_version if noise_flow_source == "raft" else None,
            "frame_count": len(video_frames),
            "raw_frame_count": len(raw_video_frames),
            "genesis_flows_512_shape": list(genesis_flows_512.shape),
            "genesis_flows_480x832_shape": list(genesis_flows_video.shape),
            "points_masks_downsampled_shape": list(points_masks_downsampled.shape),
            "mesh_masks_downsampled_shape": list(mesh_masks_downsampled.shape),
            "outputs": {
                "frames": frame_folder,
                "simulation_video": video_path,
                "genesis_flows_512": genesis_flow_path,
                "genesis_flows_480x832": genesis_flow_video_path,
                "noises": os.path.join(final_sim_folder, "noises.npy")
                if noise_flow_source != "none"
                else None,
            },
        },
    )
    print(f"[export] final_sim written to {final_sim_folder}")


if __name__ == "__main__":
    main()
