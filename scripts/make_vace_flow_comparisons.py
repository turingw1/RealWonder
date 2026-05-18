#!/usr/bin/env python
"""Create side-by-side source/flow/VACE comparison videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frames(path: Path, max_frames: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path.as_posix())
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def put_label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(out, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not write {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def make_one(source: Path, flow: Path, generated: Path, output: Path, fps: float) -> dict:
    src_frames = read_frames(source)
    flow_frames = read_frames(flow)
    gen_frames = read_frames(generated)
    n = min(len(src_frames), len(flow_frames), len(gen_frames))
    if n == 0:
        raise ValueError(f"No comparable frames for {source}")

    out_frames = []
    for i in range(n):
        src = cv2.resize(src_frames[i], (416, 240), interpolation=cv2.INTER_AREA)
        flw = cv2.resize(flow_frames[i], (416, 240), interpolation=cv2.INTER_AREA)
        gen = cv2.resize(gen_frames[i], (416, 240), interpolation=cv2.INTER_AREA)
        row = np.hstack([
            put_label(src, "source"),
            put_label(flw, "flow condition"),
            put_label(gen, "VACE14B"),
        ])
        out_frames.append(row)
    write_video(out_frames, output, fps=fps)
    return {"frames": n, "output": output.as_posix()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vace_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=16.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = []
    for sample in manifest["samples"]:
        sid = sample["id"]
        source = Path(sample["sample_video"])
        flow = args.manifest.parent / "flows" / sid / "flow_vis.mp4"
        runs = sorted((args.vace_root / sid).glob("*/out_video.mp4"))
        if not runs:
            print(f"[skip] missing VACE output for {sid}")
            continue
        generated = runs[-1]
        output = args.output_dir / f"{sid}_source_flow_vace.mp4"
        item = make_one(source, flow, generated, output, fps=args.fps)
        item["id"] = sid
        item["generated"] = generated.as_posix()
        results.append(item)
        print(f"[compare] {sid} -> {output}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparisons_manifest.json").write_text(
        json.dumps({"schema": "vace_flow_comparisons.v1", "results": results}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
