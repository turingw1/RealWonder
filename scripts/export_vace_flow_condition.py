#!/usr/bin/env python
"""Export Genesis optical flow as VACE flow-control RGB videos."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.compat import save_json


def make_colorwheel() -> np.ndarray:
    # Same color-wheel convention used by Middlebury/RAFT flow_to_image.
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    ncols = ry + yg + gc + cb + bm + mr
    colorwheel = np.zeros((ncols, 3), dtype=np.float32)
    col = 0

    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.floor(255 * np.arange(0, ry) / ry)
    col += ry

    colorwheel[col:col + yg, 0] = 255 - np.floor(255 * np.arange(0, yg) / yg)
    colorwheel[col:col + yg, 1] = 255
    col += yg

    colorwheel[col:col + gc, 1] = 255
    colorwheel[col:col + gc, 2] = np.floor(255 * np.arange(0, gc) / gc)
    col += gc

    colorwheel[col:col + cb, 1] = 255 - np.floor(255 * np.arange(0, cb) / cb)
    colorwheel[col:col + cb, 2] = 255
    col += cb

    colorwheel[col:col + bm, 2] = 255
    colorwheel[col:col + bm, 0] = np.floor(255 * np.arange(0, bm) / bm)
    col += bm

    colorwheel[col:col + mr, 2] = 255 - np.floor(255 * np.arange(0, mr) / mr)
    colorwheel[col:col + mr, 0] = 255
    return colorwheel


COLORWHEEL = make_colorwheel()


def flow_to_image(flow_hw2: np.ndarray, clip_flow: float | None = None) -> np.ndarray:
    if flow_hw2.ndim != 3 or flow_hw2.shape[2] != 2:
        raise ValueError(f"Expected [H, W, 2] flow, got {flow_hw2.shape}")

    flow = flow_hw2.astype(np.float32).copy()
    if clip_flow is not None:
        flow = np.clip(flow, -clip_flow, clip_flow)

    u = flow[:, :, 0]
    v = flow[:, :, 1]
    unknown = (np.abs(u) > 1e7) | (np.abs(v) > 1e7) | ~np.isfinite(u) | ~np.isfinite(v)
    u[unknown] = 0
    v[unknown] = 0

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(float(np.max(rad)), 1e-5)
    u = u / maxrad
    v = v / maxrad

    ncols = COLORWHEEL.shape[0]
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0

    image = np.zeros((*u.shape, 3), dtype=np.float32)
    for channel in range(3):
        col0 = COLORWHEEL[k0, channel] / 255.0
        col1 = COLORWHEEL[k1, channel] / 255.0
        col = (1 - f) * col0 + f * col1
        col[rad <= 1] = 1 - rad[rad <= 1] * (1 - col[rad <= 1])
        col[rad > 1] *= 0.75
        image[:, :, channel] = col

    image[unknown] = 0
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames to write")
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        path.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def export_one(final_sim: Path, output_dir: Path, fps: int, clip_flow: float | None) -> dict:
    final_sim = final_sim.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    flow_path = final_sim / "genesis_flows_480x832.npy"
    if not flow_path.exists():
        raise FileNotFoundError(flow_path)
    flows = np.load(flow_path).astype(np.float32)
    if flows.ndim != 4 or flows.shape[1] != 2:
        raise ValueError(f"Expected [T-1, 2, H, W] flow, got {flows.shape}")

    frame_dir = output_dir / "flow_vis_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    flow_frames = []
    for i in tqdm(range(flows.shape[0]), desc=f"{final_sim.parent.parent.name}:flow"):
        flow_hw2 = np.transpose(flows[i], (1, 2, 0))
        frame = flow_to_image(flow_hw2, clip_flow=clip_flow)
        flow_frames.append(frame)

    # VACE flow preprocessing returns [first_flow] + flow_list so control length matches video length.
    if flow_frames:
        flow_frames = [flow_frames[0]] + flow_frames

    for i, frame in enumerate(flow_frames):
        Image.fromarray(frame).save(frame_dir / f"frame_{i:04d}.png")

    flow_video = output_dir / "flow_vis.mp4"
    write_video(flow_frames, flow_video, fps=fps)

    prompt_src = final_sim / "prompt.txt"
    prompt_dst = output_dir / "prompt.txt"
    if prompt_src.exists():
        shutil.copy2(prompt_src, prompt_dst)

    first_frame_src = final_sim / "frames" / "frame_0000.png"
    first_frame_dst = output_dir / "first_frame.png"
    if first_frame_src.exists():
        shutil.copy2(first_frame_src, first_frame_dst)

    simulation_src = final_sim / "simulation.mp4"
    simulation_link = output_dir / "simulation_source.mp4"
    if simulation_src.exists() and not simulation_link.exists():
        try:
            simulation_link.symlink_to(simulation_src)
        except OSError:
            shutil.copy2(simulation_src, simulation_link)

    metadata = {
        "schema": "realwonder_vace_flow_condition.v1",
        "source_final_sim": final_sim.as_posix(),
        "source_flow": flow_path.as_posix(),
        "flow_shape": list(flows.shape),
        "flow_frame_count": len(flow_frames),
        "fps": fps,
        "clip_flow": clip_flow,
        "outputs": {
            "flow_video": flow_video.as_posix(),
            "flow_frames": frame_dir.as_posix(),
            "prompt": prompt_dst.as_posix() if prompt_dst.exists() else None,
            "first_frame": first_frame_dst.as_posix() if first_frame_dst.exists() else None,
            "simulation_source": simulation_link.as_posix() if simulation_link.exists() else None,
        },
    }
    save_json(output_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Manifest with cases[].final_sim")
    parser.add_argument("--final_sim", type=Path, help="Single final_sim directory")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--clip_flow", type=float, default=None)
    args = parser.parse_args()
    if args.manifest is None and args.final_sim is None:
        parser.error("Provide --manifest or --final_sim")
    return args


def main() -> None:
    args = parse_args()
    all_metadata = []
    if args.manifest is not None:
        data = json.load(open(args.manifest))
        root = Path(data.get("root", REPO_ROOT))
        for case in data["cases"]:
            final_sim = root / case["final_sim"]
            case_dir = args.output_dir / case["case"]
            metadata = export_one(final_sim, case_dir, args.fps, args.clip_flow)
            metadata["case"] = case["case"]
            all_metadata.append(metadata)
    else:
        metadata = export_one(args.final_sim, args.output_dir, args.fps, args.clip_flow)
        metadata["case"] = args.final_sim.parent.parent.name
        all_metadata.append(metadata)

    save_json(
        args.output_dir / "manifest.json",
        {
            "schema": "realwonder_vace_flow_condition_manifest.v1",
            "cases": all_metadata,
        },
    )
    print(f"[export] wrote VACE flow controls to {args.output_dir}")


if __name__ == "__main__":
    main()
