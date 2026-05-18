#!/usr/bin/env python
"""Export RAFT optical-flow streams from a video, frame directory, or image pair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.compat import resolve_torch_device, save_json
from simulation.image23D.noise_warp.raft import RaftOpticalFlow


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _resize(image: Image.Image, resize_hw: tuple[int, int] | None) -> Image.Image:
    if resize_hw is None:
        return image
    height, width = resize_hw
    return image.resize((width, height), Image.BICUBIC)


def load_video_frames(path: Path, stride: int, max_frames: int | None, resize_hw):
    cap = cv2.VideoCapture(path.as_posix())
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    frames = []
    index = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if index % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(_resize(Image.fromarray(frame_rgb), resize_hw))
            if max_frames is not None and len(frames) >= max_frames:
                break
        index += 1

    cap.release()
    return frames


def load_frame_dir(path: Path, stride: int, max_frames: int | None, resize_hw):
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    files = files[::stride]
    if max_frames is not None:
        files = files[:max_frames]
    return [_resize(Image.open(p).convert("RGB"), resize_hw) for p in files]


def load_image_pair(paths, resize_hw):
    return [_resize(Image.open(Path(p)).convert("RGB"), resize_hw) for p in paths]


def save_frames(frames, output_dir: Path):
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(frame_dir / f"frame_{i:04d}.png")
    return frame_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=str, help="Input video path")
    source.add_argument("--frames_dir", type=str, help="Directory of input frames")
    source.add_argument("--image_pair", nargs=2, help="Two images for one flow field")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--allow_cpu_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback to CPU when the CUDA torch build cannot run kernels on this GPU.",
    )
    parser.add_argument("--raft_version", choices=["large", "small"], default="large")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for video/frame-dir inputs")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of loaded frames")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Resize frames before RAFT. Example: --resize 480 832",
    )
    parser.add_argument("--save_frames", action="store_true", help="Save the frames used for RAFT")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resize_hw = tuple(args.resize) if args.resize is not None else None
    if args.video:
        source_type = "video"
        source_path = Path(args.video)
        frames = load_video_frames(source_path, args.stride, args.max_frames, resize_hw)
    elif args.frames_dir:
        source_type = "frames_dir"
        source_path = Path(args.frames_dir)
        frames = load_frame_dir(source_path, args.stride, args.max_frames, resize_hw)
    else:
        source_type = "image_pair"
        source_path = [Path(p) for p in args.image_pair]
        frames = load_image_pair(source_path, resize_hw)

    if len(frames) < 2:
        raise ValueError("Need at least two frames/images to compute optical flow")

    device = resolve_torch_device(args.device, allow_cpu_fallback=args.allow_cpu_fallback)
    raft = RaftOpticalFlow(str(device), args.raft_version)

    flows = []
    prev = np.asarray(frames[0])
    for frame in tqdm(frames[1:], desc="RAFT"):
        current = np.asarray(frame)
        flow = raft(prev, current).detach().cpu().numpy().astype(np.float32)
        flows.append(flow)
        prev = current

    flows = np.stack(flows, axis=0)
    flows_path = output_dir / "raft_flows.npy"
    np.save(flows_path, flows)

    frame_dir = None
    if args.save_frames:
        frame_dir = save_frames(frames, output_dir)

    metadata = {
        "source_type": source_type,
        "source": source_path,
        "torch_device": str(device),
        "raft_version": args.raft_version,
        "frame_count": len(frames),
        "flow_shape": list(flows.shape),
        "resize_hw": resize_hw,
        "stride": args.stride,
        "outputs": {
            "raft_flows": flows_path,
            "frames": frame_dir,
        },
    }
    save_json(output_dir / "metadata.json", metadata)
    print(f"[export] saved {flows_path} with shape {flows.shape}")


if __name__ == "__main__":
    main()
