#!/usr/bin/env python
"""Convert raw optical-flow tensors to VACE RGB flow-control videos."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_vace_flow_condition import flow_to_image
from simulation.compat import save_json


def write_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        raise ValueError("No frames to write")
    height, width = frames[0].shape[:2]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow_npy", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_video", type=Path, default=None)
    parser.add_argument("--prompt_file", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--clip_flow", type=float, default=None)
    parser.add_argument("--save_frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flows = np.load(args.flow_npy).astype(np.float32)
    if flows.ndim != 4 or flows.shape[1] != 2:
        raise ValueError(f"Expected [T-1, 2, H, W], got {flows.shape}")

    frames = [flow_to_image(np.transpose(flow, (1, 2, 0)), clip_flow=args.clip_flow) for flow in flows]
    if frames:
        frames = [frames[0]] + frames

    frame_dir = args.output_dir / "flow_vis_frames"
    if args.save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(frame_dir / f"frame_{i:04d}.png")

    flow_video = args.output_dir / "flow_vis.mp4"
    write_video(frames, flow_video, fps=args.fps)

    if args.source_video is not None:
        shutil.copy2(args.source_video, args.output_dir / "source_video.mp4")
    if args.prompt_file is not None:
        shutil.copy2(args.prompt_file, args.output_dir / "prompt.txt")

    mag = np.linalg.norm(flows, axis=1)
    metadata = {
        "schema": "flow_npy_to_vace_condition.v1",
        "source_flow": args.flow_npy.as_posix(),
        "source_video": args.source_video.as_posix() if args.source_video else None,
        "prompt_file": args.prompt_file.as_posix() if args.prompt_file else None,
        "flow_shape": list(flows.shape),
        "flow_frame_count": len(frames),
        "fps": args.fps,
        "clip_flow": args.clip_flow,
        "flow_magnitude": {
            "mean": float(mag.mean()),
            "p95": float(np.percentile(mag, 95)),
            "p99": float(np.percentile(mag, 99)),
            "max": float(mag.max()),
        },
        "outputs": {
            "flow_video": flow_video.as_posix(),
            "flow_frames": frame_dir.as_posix() if args.save_frames else None,
            "source_video": (args.output_dir / "source_video.mp4").as_posix() if args.source_video else None,
            "prompt": (args.output_dir / "prompt.txt").as_posix() if args.prompt_file else None,
        },
    }
    save_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
