"""One-time preprocessing: run 3D reconstruction and save results for the demo.

Usage:
    python -m demo_web.preprocess --case_path cases/santa_cloth --output demo_data/santa_cloth
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
import numpy as np
import trimesh
from PIL import Image
from omegaconf import OmegaConf

from simulation.image23D.single_view_reconstructor import SingleViewReconstructor
from simulation.utils import resize_and_crop_pil, pt3d_to_gs


def preprocess(case_path: str, output_dir: str, device: str = "cuda"):
    case_path = Path(case_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load case config
    config_path = case_path / "config.yaml"
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config["device"] = device
    config["data_path"] = str(case_path)
    config["output_folder"] = str(output_dir / "recon_tmp")
    os.makedirs(config["output_folder"], exist_ok=True)

    # Run reconstruction
    svr = SingleViewReconstructor(config)
    fg_pcs, fg_meshes, ground_plane_normal, updated_config = svr.reconstruct()

    # Save foreground meshes
    meshes_dir = output_dir / "fg_meshes"
    meshes_dir.mkdir(exist_ok=True)
    for idx, mesh_info in enumerate(fg_meshes):
        verts = mesh_info["vertices"].cpu().numpy()
        faces = mesh_info["faces"].cpu().numpy()
        colors = mesh_info["colors"].cpu().numpy()
        colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                               vertex_colors=colors_uint8, process=False)
        mesh.export(str(meshes_dir / f"mesh_{idx:02d}.obj"))

    # Save foreground point clouds
    pcs_dir = output_dir / "fg_pcs"
    pcs_dir.mkdir(exist_ok=True)
    for idx, pc_info in enumerate(fg_pcs):
        torch.save({
            "points": pc_info["points"].cpu(),
            "colors": pc_info["colors"].cpu(),
        }, pcs_dir / f"pc_{idx:02d}.pt")

    # Save background points
    torch.save({
        "points": svr.bg_points.cpu(),
        "colors": svr.bg_points_colors.cpu(),
    }, output_dir / "bg_points.pt")

    # Save camera parameters
    camera = svr.current_camera
    torch.save({
        "K": camera.K.cpu(),
        "R": camera.R.cpu(),
        "T": camera.T.cpu(),
        "focal_length": svr.init_focal_length,
    }, output_dir / "camera.pt")

    # Save per-object 2D segmentation masks (resized to 480x832 like first_frame)
    if hasattr(svr, "object_masks") and svr.object_masks:
        masks_dir = output_dir / "fg_masks"
        masks_dir.mkdir(exist_ok=True)
        crop_start = config.get("crop_start", 176)
        for idx, mask_tensor in enumerate(svr.object_masks):
            # mask_tensor is a 512x512 bool tensor on GPU
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask_np, mode="L")
            # Apply same resize+crop as first_frame so masks align pixel-perfectly
            mask_pil_cropped = resize_and_crop_pil(
                mask_pil.convert("RGB"), start_y=crop_start
            ).convert("L")
            mask_pil_cropped.save(masks_dir / f"mask_{idx:02d}.png")
        print(f"  fg_masks: {len(svr.object_masks)} masks saved")

    # Save first frame (resized to 480x832)
    input_img = Image.open(case_path / "input.png").convert("RGB")
    crop_start = config.get("crop_start", 176)
    resized_img = resize_and_crop_pil(input_img, start_y=crop_start)
    resized_img.save(output_dir / "first_frame.png")

    # Save inpainted background
    inpainted_path = case_path / "inpainted.png"
    if inpainted_path.exists():
        shutil.copy2(str(inpainted_path), str(output_dir / "inpainted_bg.png"))

    # Save ground plane normal if estimated
    if ground_plane_normal is not None:
        np.save(output_dir / "ground_plane_normal.npy", ground_plane_normal)

    # Save config (with updated fields like fov)
    OmegaConf.save(OmegaConf.create(updated_config), output_dir / "config.yaml")

    # Cleanup temp
    shutil.rmtree(config["output_folder"], ignore_errors=True)

    print(f"Preprocessing complete. Results saved to: {output_dir}")
    print(f"  fg_meshes: {len(fg_meshes)} meshes")
    print(f"  fg_pcs: {len(fg_pcs)} point clouds")
    print(f"  bg_points: {svr.bg_points.shape[0]} points")
    print(f"  first_frame: {resized_img.size}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess a case for the RealWonder demo")
    parser.add_argument("--case_path", type=str, required=True,
                        help="Path to case directory (e.g. cases/santa_cloth)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory (e.g. demo_data/santa_cloth)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    preprocess(args.case_path, args.output, args.device)


if __name__ == "__main__":
    main()
