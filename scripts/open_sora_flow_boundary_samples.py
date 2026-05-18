#!/usr/bin/env python
"""Prepare a small Open-Sora/MixKit sample set for VACE flow-boundary tests."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import cv2


DATASET_ROOT = Path("/root/autodl-tmp/Physics_worldmodel/datasets/Open-Sora-Plan-v1.1.0")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/Physics_worldmodel/RealWonder/experiments/vace_flow_boundary_20260511")

SAMPLES = [
    {
        "id": "airplane_takeoff",
        "tar": "Airplane.tar",
        "member": "Airplane/mixkit-airplane-taking-off-2626.mp4",
        "prompt": "A realistic airplane takes off from a runway, viewed from a stable camera.",
    },
    {
        "id": "urban_cyclist",
        "tar": "Bicycle.tar",
        "member": "Bicycle/mixkit-urban-cyclist-riding-in-the-street-1719.mp4",
        "prompt": "A cyclist rides through an urban street with natural camera motion.",
    },
    {
        "id": "bird_flock",
        "tar": "Birds.tar",
        "member": "Birds/mixkit-flock-of-black-birds-flying-in-the-sky-during-sunset-51490.mp4",
        "prompt": "A flock of birds flies across the sky at sunset.",
    },
    {
        "id": "cars_road",
        "tar": "Car.tar",
        "member": "Car/mixkit-cars-driving-by-on-road-2022.mp4",
        "prompt": "Cars drive along a road in a realistic street scene.",
    },
    {
        "id": "black_cat",
        "tar": "Cats.tar",
        "member": "Cats/mixkit-black-cat-with-yellow-eyes-1539.mp4",
        "prompt": "A black cat with yellow eyes moves subtly in a realistic close-up shot.",
    },
    {
        "id": "night_traffic",
        "tar": "Traffic.tar",
        "member": "Traffic/mixkit-aerial-shot-with-frontal-view-to-night-traffic-42046.mp4",
        "prompt": "Night traffic moves through a city street in a realistic aerial shot.",
    },
    {
        "id": "contemporary_dance",
        "tar": "Dance.tar",
        "member": "Dance/mixkit-brunette-woman-dancing-contemporary-dance-43211.mp4",
        "prompt": "A woman performs contemporary dance with smooth body motion.",
    },
    {
        "id": "playful_dog",
        "tar": "Dogs.tar",
        "member": "Dogs/mixkit-a-playful-pitbull-biting-a-teddy-bear-on-the-ground-50675.mp4",
        "prompt": "A playful dog bites and pulls a toy on the ground in a realistic scene.",
    },
    {
        "id": "fish_swimming",
        "tar": "Fish.tar",
        "member": "Fish/mixkit-fish-swimming-in-the-sea-2058.mp4",
        "prompt": "Fish swim through the sea with gentle underwater motion.",
    },
    {
        "id": "waving_fire",
        "tar": "fire.tar",
        "member": "fire/mixkit-waving-fire-closeup-3448.mp4",
        "prompt": "Fire flames wave and flicker in a realistic close-up shot.",
    },
]


def write_sample_video(src_path: Path, dst_path: Path, max_frames: int, size: tuple[int, int]) -> dict:
    width, height = size
    cap = cv2.VideoCapture(src_path.as_posix())
    if not cap.isOpened():
        raise FileNotFoundError(src_path)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        dst_path.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        16.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {dst_path}")

    frame_count = 0
    first_frame_path = dst_path.parent / "first_frame.png"
    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame_count == 0:
            cv2.imwrite(first_frame_path.as_posix(), frame)
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()
    if frame_count < 2:
        raise ValueError(f"Too few frames in {src_path}: {frame_count}")

    return {
        "source_fps": src_fps,
        "source_frames": src_frames,
        "source_size": [src_width, src_height],
        "sample_fps": 16.0,
        "sample_frames": frame_count,
        "sample_size": [width, height],
        "first_frame": first_frame_path.as_posix(),
    }


def extract_member(tar_path: Path, member_name: str, output_path: Path) -> None:
    with tarfile.open(tar_path) as tf:
        member = tf.getmember(member_name)
        extracted = tf.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(member_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            shutil.copyfileobj(extracted, f)


def main() -> None:
    out_root = Path(os.environ.get("FLOW_BOUNDARY_OUTPUT", DEFAULT_OUTPUT)).resolve()
    sample_root = out_root / "samples"
    manifest = {"schema": "open_sora_vace_flow_boundary_samples.v1", "samples": []}

    for sample in SAMPLES:
        sample_dir = sample_root / sample["id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="open_sora_sample_") as tmp:
            raw_path = Path(tmp) / "raw.mp4"
            extract_member(DATASET_ROOT / "all_mixkit" / sample["tar"], sample["member"], raw_path)
            video_path = sample_dir / "source_81f_832x480.mp4"
            meta = write_sample_video(raw_path, video_path, max_frames=81, size=(832, 480))

        prompt_path = sample_dir / "prompt.txt"
        prompt_path.write_text(sample["prompt"] + "\n", encoding="utf-8")

        sample_meta = {
            **sample,
            **meta,
            "sample_video": video_path.as_posix(),
            "prompt_file": prompt_path.as_posix(),
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(sample_meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest["samples"].append(sample_meta)
        print(f"[sample] {sample['id']} -> {video_path}")

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[sample] manifest -> {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
