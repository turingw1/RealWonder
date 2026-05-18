#!/usr/bin/env python
"""Export a Genesis-style low-level data demo from an input video.

This is the runnable data contract for downstream training work. It accepts
a normal video and writes the same kind of streams that the Genesis side is
expected to provide later:

    frames / first_frame / coarse_rgb / flow / motion masks / metadata

The default flow backend is OpenCV Farneback so the demo runs without model
weights, Torch CUDA, SAM2, SAM3D, or Helios. RAFT/Genesis exporters can be
plugged into the same output schema later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_video(path: Path, *, max_frames: int | None, stride: int, size_hw: tuple[int, int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path.as_posix())
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    out = []
    index = 0
    height, width = size_hw
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if index % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
            out.append(frame_rgb)
            if max_frames is not None and len(out) >= max_frames:
                break
        index += 1
    cap.release()
    if len(out) < 2:
        raise ValueError("Need at least two frames after stride/max_frames filtering")
    return out


def farneback_flow(frames: list[np.ndarray]) -> np.ndarray:
    flows = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for frame in frames[1:]:
        cur = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow_hw2 = cv2.calcOpticalFlowFarneback(
            prev,
            cur,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        ).astype(np.float32)
        flows.append(np.transpose(flow_hw2, (2, 0, 1)))
        prev = cur
    return np.stack(flows, axis=0)


def coarse_rgb(frames: list[np.ndarray], blur: int = 21, downscale: int = 4) -> list[np.ndarray]:
    out = []
    for frame in frames:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // downscale), max(1, h // downscale)), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        if blur > 1:
            k = blur if blur % 2 == 1 else blur + 1
            restored = cv2.GaussianBlur(restored, (k, k), 0)
        out.append(restored)
    return out


def masks_from_flow(flows: np.ndarray, threshold: float) -> np.ndarray:
    if flows.shape[0] == 0:
        return np.zeros((0, 0, 0), dtype=bool)
    mag = np.sqrt(flows[:, 0] ** 2 + flows[:, 1] ** 2)
    masks = mag > threshold
    # Add one mask for the final frame by repeating the last flow mask.
    return np.concatenate([masks, masks[-1:]], axis=0)


def save_frames(frames: list[np.ndarray], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(folder / f"frame_{i:04d}.png")


def save_flow_preview(flows: np.ndarray, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, flow in enumerate(flows):
        dx, dy = flow
        mag, ang = cv2.cartToPolar(dx, dy)
        hsv = np.zeros((*dx.shape, 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180.0 / np.pi / 2.0).astype(np.uint8)
        hsv[..., 1] = 255
        if mag.max() > 1e-6:
            hsv[..., 2] = np.clip(mag / mag.max() * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)).save(folder / f"flow_{i:04d}.png")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--max_frames", type=int, default=81)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--mask_flow_threshold", type=float, default=0.25)
    parser.add_argument("--flow_preview", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    coarse_dir = output_dir / "coarse_rgb_frames"

    frames = read_video(
        args.video,
        max_frames=args.max_frames,
        stride=args.stride,
        size_hw=(args.height, args.width),
    )
    flows = farneback_flow(frames)
    coarse_frames = coarse_rgb(frames)
    masks = masks_from_flow(flows, args.mask_flow_threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_frames(frames, frames_dir)
    save_frames(coarse_frames, coarse_dir)
    Image.fromarray(frames[0]).save(output_dir / "first_frame.png")
    np.save(output_dir / "flow_fwd.npy", flows)
    np.save(output_dir / "genesis_style_flow.npy", flows)
    np.save(output_dir / "motion_masks.npy", masks.astype(bool))
    imageio.mimsave(output_dir / "coarse_rgb.mp4", coarse_frames, fps=args.fps)
    imageio.mimsave(output_dir / "gt_video.mp4", frames, fps=args.fps)
    (output_dir / "prompt.txt").write_text(args.prompt, encoding="utf-8")
    if args.flow_preview:
        save_flow_preview(flows, output_dir / "flow_preview")

    save_json(
        output_dir / "metadata.json",
        {
            "schema": "genesis_style_video_demo.v1",
            "source_video": args.video.as_posix(),
            "frame_count": len(frames),
            "resolution_hw": [args.height, args.width],
            "flow_shape": list(flows.shape),
            "mask_shape": list(masks.shape),
            "flow_backend": "opencv_farneback",
            "flow_convention": "flow_fwd[t] maps frame_t pixels toward frame_t+1 as dx/dy",
            "coarse_rgb": "downscale-upscale plus gaussian blur; placeholder for Genesis coarse preview",
            "outputs": {
                "frames": frames_dir.as_posix(),
                "coarse_rgb_frames": coarse_dir.as_posix(),
                "first_frame": (output_dir / "first_frame.png").as_posix(),
                "flow_fwd": (output_dir / "flow_fwd.npy").as_posix(),
                "motion_masks": (output_dir / "motion_masks.npy").as_posix(),
                "coarse_rgb_video": (output_dir / "coarse_rgb.mp4").as_posix(),
                "gt_video": (output_dir / "gt_video.mp4").as_posix(),
                "prompt": (output_dir / "prompt.txt").as_posix(),
            },
        },
    )
    print(f"[video-demo] wrote {output_dir}")
    print(f"[video-demo] flow_fwd.npy shape: {flows.shape}")
    print(f"[video-demo] motion_masks.npy shape: {masks.shape}")


if __name__ == "__main__":
    main()
