import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from simulation.experiment_logging import ExperimentLogger
from simulation.genesis_simulator import DiffSim
from simulation.image23D.noise_warp.make_warped_noise import NoiseWarper
from simulation.utils import resize_and_crop_pil, save_video_from_pil, visualize_optical_flow_advanced


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def process_simulated_results(input_image, raw_video_frames, points_masks, mesh_masks, crop_start=176):
    input_image = resize_and_crop_pil(input_image, crop_start)
    raw_video_frames = [resize_and_crop_pil(frame, crop_start) for frame in raw_video_frames]
    points_masks = preprocess_masks_downsample(points_masks)
    mesh_masks = preprocess_masks_downsample(mesh_masks)
    return input_image, raw_video_frames, points_masks, mesh_masks


def preprocess_masks_downsample(masks):
    num_masks = len(masks)
    masks = torch.stack(masks, dim=0).squeeze(-1)
    resized_masks = F.interpolate(masks.unsqueeze(1).float(), size=(832, 832), mode="bilinear", align_corners=False)
    crop_height = 480
    start_y = (832 - crop_height) // 2
    cropped_masks = resized_masks[:, :, start_y:start_y + crop_height, :]
    masks_downsampled = F.interpolate(cropped_masks.float(), size=(60, 104), mode="bilinear", align_corners=False).squeeze(1)

    time_averaged_masks = []
    for i in range(0, num_masks, 4):
        time_averaged_masks.append(masks_downsampled[i:i + 4, :, :].mean(dim=0, keepdim=True))
    masks_downsampled = torch.cat(time_averaged_masks, dim=0)
    return masks_downsampled > 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    timestamp = datetime.now().strftime("%d-%m_%H-%M-%S")
    output_folder = os.path.join(config["output_folder"], timestamp)
    os.makedirs(output_folder, exist_ok=True)
    config["output_folder"] = output_folder
    debug = config.get("debug", False)

    if debug:
        debug_config_save_path = os.path.join(config["output_folder"], "config.yaml")
        OmegaConf.save(config, debug_config_save_path)

    device = torch.device("cuda")
    set_seed(config["seed"])
    torch.set_grad_enabled(False)

    experiment_logger = ExperimentLogger(
        experiment_name="offline_case_simulation",
        run_name=config.get("example_name", Path(args.config_path).stem),
        output_dir=Path(output_folder) / "experiment_logs",
        metadata={
            "config_path": args.config_path,
            "device": str(device),
            "case_name": config.get("example_name"),
        },
    )

    with experiment_logger.time_block("case_simulation.load_input_image", data_path=config["data_path"]):
        input_image = Image.open(os.path.join(config["data_path"], "input.png")).convert("RGB")

    with experiment_logger.time_block("case_simulation.build_diffsim"):
        genesis_simulator = DiffSim(config, exp_logger=experiment_logger)

    with experiment_logger.time_block("case_simulation.simulation_pc_render"):
        raw_video_frames, points_masks, mesh_masks = genesis_simulator.simulation_pc_render()

    with experiment_logger.time_block("case_simulation.process_simulated_results"):
        input_image, video_frames, points_masks_downsampled, mesh_masks_downsampled = process_simulated_results(
            input_image,
            raw_video_frames,
            points_masks,
            mesh_masks,
            crop_start=config["crop_start"],
        )

    final_sim_folder = os.path.join(output_folder, "final_sim")
    os.makedirs(final_sim_folder, exist_ok=True)

    with experiment_logger.time_block("case_simulation.write_config"):
        config_save_path = os.path.join(final_sim_folder, "config.yaml")
        OmegaConf.save(config, config_save_path)

    noise_warper = NoiseWarper()
    optical_flows = genesis_simulator.svr.optical_flow
    optical_flows = np.array(optical_flows)[..., :2]
    optical_flows = np.transpose(optical_flows, (0, 3, 1, 2))

    if debug:
        with experiment_logger.time_block("case_simulation.save_debug_flows"):
            np.save(os.path.join(final_sim_folder, "flows.npy"), optical_flows)

    with experiment_logger.time_block("case_simulation.save_frames", frame_count=len(video_frames)):
        frame_folder = os.path.join(final_sim_folder, "frames")
        os.makedirs(frame_folder, exist_ok=True)
        for i, frame in enumerate(video_frames):
            frame_path = os.path.join(frame_folder, f"frame_{i:04d}.png")
            frame.save(frame_path)

    if debug:
        with experiment_logger.time_block("case_simulation.visualize_optical_flow"):
            visualize_optical_flow_advanced(
                frame_folder,
                os.path.join(final_sim_folder, "flows.npy"),
                os.path.join(final_sim_folder, "optical_flow_viz"),
                arrow_density=30,
            )

    with experiment_logger.time_block("case_simulation.noise_warp", frame_count=len(video_frames)):
        warped_noise = noise_warper.process(
            video_frames,
            final_sim_folder,
            crop_start=config["crop_start"],
            input_flow=False,
            debug=debug,
            device=device,
        )
    experiment_logger.log_event("case_simulation.noise_warp.result", noise_shape=getattr(warped_noise, "shape", None))

    with experiment_logger.time_block("case_simulation.save_masks"):
        points_masks_path = os.path.join(final_sim_folder, "points_masks_downsampled.pt")
        torch.save(points_masks_downsampled, points_masks_path)
        mesh_masks_path = os.path.join(final_sim_folder, "mesh_masks_downsampled.pt")
        torch.save(mesh_masks_downsampled, mesh_masks_path)

    with experiment_logger.time_block("case_simulation.save_simulation_video"):
        video_path = os.path.join(final_sim_folder, "simulation.mp4")
        save_video_from_pil(video_frames, video_path, fps=10)

    with experiment_logger.time_block("case_simulation.save_resized_input_image"):
        input_image_path = os.path.join(final_sim_folder, "resized_input_image.png")
        input_image.save(input_image_path)

    with experiment_logger.time_block("case_simulation.write_prompt"):
        prompt_txt_path = os.path.join(final_sim_folder, "prompt.txt")
        with open(prompt_txt_path, "w", encoding="utf-8") as f:
            f.write(config["vgen_prompt"])

    experiment_logger.finalize(
        status="completed",
        output_folder=output_folder,
        final_sim_folder=final_sim_folder,
        frame_count=len(video_frames),
    )


if __name__ == "__main__":
    main()
