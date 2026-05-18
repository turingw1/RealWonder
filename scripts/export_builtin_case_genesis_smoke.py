#!/usr/bin/env python
"""Run Genesis-only smoke simulations for the bundled RealWonder cases.

This script does not call SAM2, SAM3D, MoGe, RAFT, or the diffusion model.
It is a runtime check for the Genesis rendering/export stage when the
image-to-3D checkpoints are not available yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

import genesis as gs
from simulation.utils import save_video_from_pil


CASE_ORDER = ["lamp", "persimmon", "sand_house", "santa_cloth", "tree", "two_duck"]


def average_case_color(case_name: str) -> tuple[float, float, float, float]:
    image_path = Path("cases") / case_name / "input.png"
    if not image_path.exists():
        return (0.65, 0.65, 0.65, 1.0)
    image = np.asarray(Image.open(image_path).convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
    color = image.reshape(-1, 3).mean(axis=0)
    color = np.clip(0.35 + 0.65 * color, 0.0, 1.0)
    return (float(color[0]), float(color[1]), float(color[2]), 1.0)


def add_entity(scene, morph, color, *, rho=800.0, friction=0.4, fixed=False):
    material = gs.materials.Rigid(rho=rho, friction=friction)
    if hasattr(morph, "fixed"):
        morph.fixed = fixed
    return scene.add_entity(
        material=material,
        morph=morph,
        surface=gs.surfaces.Default(color=color, vis_mode="visual"),
    )


def track(entity, kind: str, color, *, size=None, radius=None, label=None):
    return {
        "entity": entity,
        "kind": kind,
        "color": tuple(int(np.clip(c, 0.0, 1.0) * 255) for c in color[:3]),
        "size": size,
        "radius": radius,
        "label": label or kind,
    }


def set_entity_pos(entity, pos):
    if hasattr(entity, "set_pos"):
        entity.set_pos(pos)
    elif hasattr(entity, "set_position"):
        entity.set_position(pos)
    else:
        raise AttributeError(f"{entity!r} has neither set_pos nor set_position")


def build_case(scene, case_name: str):
    base = average_case_color(case_name)
    objects = []
    tracked = []
    controls = []

    if case_name == "lamp":
        size = (0.32, 0.32, 0.42)
        obj = add_entity(
            scene,
            gs.morphs.Box(pos=(-0.35, 0.0, 0.22), size=size),
            base,
            friction=0.04,
        )
        objects.append(obj)
        tracked.append(track(obj, "box", base, size=size, label="lamp"))

        def control(frame, n_frames):
            t = frame / max(1, n_frames - 1)
            set_entity_pos(obj, (-0.42 + 0.78 * t, 0.0, 0.22 + 0.025 * np.sin(t * np.pi)))

        controls.append(control)

    elif case_name == "persimmon":
        colors = [(0.95, 0.45, 0.12, 1.0), (0.9, 0.55, 0.16, 1.0)]
        for x, color in [(-0.35, colors[0]), (0.35, colors[1])]:
            obj = add_entity(scene, gs.morphs.Sphere(pos=(x, 0.0, 0.22), radius=0.18), color, friction=0.1)
            objects.append(obj)
            tracked.append(track(obj, "sphere", color, radius=0.18, label="persimmon"))

        def control(frame, _n):
            if frame < 20:
                objects[0].solver.apply_links_external_force(force=np.array([[0.75, 0.0, 0.0]]), links_idx=[objects[0].idx])
                objects[1].solver.apply_links_external_force(force=np.array([[-0.75, 0.0, 0.0]]), links_idx=[objects[1].idx])

        controls.append(control)

    elif case_name == "two_duck":
        colors = [(0.95, 0.78, 0.18, 1.0), (0.95, 0.62, 0.12, 1.0)]
        for x, color in [(-0.42, colors[0]), (0.38, colors[1])]:
            obj = add_entity(scene, gs.morphs.Sphere(pos=(x, 0.0, 0.22), radius=0.17), color, friction=0.05)
            objects.append(obj)
            tracked.append(track(obj, "sphere", color, radius=0.17, label="duck"))

        def control(frame, _n):
            objects[0].solver.apply_links_external_force(force=np.array([[0.65, 0.0, 0.0]]), links_idx=[objects[0].idx])

        controls.append(control)

    elif case_name == "sand_house":
        sand_size = (0.48, 0.34, 0.28)
        sand = add_entity(
            scene,
            gs.morphs.Box(pos=(0.0, 0.0, 0.18), size=sand_size),
            (0.73, 0.55, 0.32, 1.0),
            friction=0.35,
        )
        wall_size = (0.10, 0.50, 0.34)
        left = add_entity(
            scene,
            gs.morphs.Box(pos=(-0.72, 0.0, 0.28), size=wall_size),
            (0.35, 0.42, 0.52, 1.0),
            fixed=True,
        )
        right = add_entity(
            scene,
            gs.morphs.Box(pos=(0.72, 0.0, 0.28), size=wall_size),
            (0.35, 0.42, 0.52, 1.0),
            fixed=True,
        )
        objects.extend([sand, left, right])
        tracked.extend(
            [
                track(sand, "box", (0.73, 0.55, 0.32, 1.0), size=sand_size, label="sand"),
                track(left, "box", (0.35, 0.42, 0.52, 1.0), size=wall_size, label="left_pusher"),
                track(right, "box", (0.35, 0.42, 0.52, 1.0), size=wall_size, label="right_pusher"),
            ]
        )

        def control(frame, n_frames):
            t = min(1.0, frame / max(1, n_frames - 1))
            set_entity_pos(left, (-0.72 + 0.42 * min(t * 2.0, 1.0), 0.0, 0.28))
            set_entity_pos(right, (0.72 - 0.42 * max(0.0, min((t - 0.45) * 2.0, 1.0)), 0.0, 0.28))
            if 8 < frame < 34:
                sand.solver.apply_links_external_force(force=np.array([[0.25, 0.0, 0.0]]), links_idx=[sand.idx])

        controls.append(control)

    elif case_name == "santa_cloth":
        cloth_size = (0.52, 0.045, 0.62)
        cloth = add_entity(
            scene,
            gs.morphs.Box(pos=(0.0, 0.0, 0.42), size=cloth_size),
            base,
            friction=0.25,
        )
        anchor_radius = 0.035
        anchor = add_entity(
            scene,
            gs.morphs.Cylinder(pos=(0.0, 0.0, 0.78), radius=anchor_radius, height=0.72),
            (0.2, 0.2, 0.22, 1.0),
            fixed=True,
        )
        objects.extend([cloth, anchor])
        tracked.extend(
            [
                track(cloth, "box", base, size=cloth_size, label="cloth"),
                track(anchor, "cylinder", (0.2, 0.2, 0.22, 1.0), radius=anchor_radius, size=(0.07, 0.07, 0.72), label="hanger"),
            ]
        )

        def control(frame, n_frames):
            phase = frame / max(1, n_frames - 1)
            cloth.solver.apply_links_external_force(
                force=np.array([[0.6 * np.sin(phase * np.pi * 4.0), 0.0, 0.05]]),
                links_idx=[cloth.idx],
            )

        controls.append(control)

    elif case_name == "tree":
        trunk_radius = 0.07
        trunk = add_entity(
            scene,
            gs.morphs.Cylinder(pos=(0.0, 0.0, 0.32), radius=trunk_radius, height=0.58),
            (0.45, 0.27, 0.14, 1.0),
            fixed=True,
        )
        crown_radius = 0.28
        crown = add_entity(
            scene,
            gs.morphs.Sphere(pos=(0.0, 0.0, 0.78), radius=crown_radius),
            (0.18, 0.55, 0.22, 1.0),
            friction=0.25,
        )
        objects.extend([trunk, crown])
        tracked.extend(
            [
                track(trunk, "cylinder", (0.45, 0.27, 0.14, 1.0), radius=trunk_radius, size=(0.14, 0.14, 0.58), label="trunk"),
                track(crown, "sphere", (0.18, 0.55, 0.22, 1.0), radius=crown_radius, label="crown"),
            ]
        )

        def control(frame, n_frames):
            phase = frame / max(1, n_frames - 1)
            crown.solver.apply_links_external_force(
                force=np.array([[0.35 * (0.5 + np.sin(phase * np.pi * 3.0)), 0.0, 0.0]]),
                links_idx=[crown.idx],
            )

        controls.append(control)

    else:
        raise ValueError(f"Unknown case: {case_name}")

    return controls, tracked


def case_frame_count(case_name: str, override: int) -> int:
    if override > 0:
        return override
    config_path = REPO_ROOT / "cases" / case_name / "config.yaml"
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("simulated_frames_num:"):
                return int(stripped.split(":", 1)[1].strip().split()[0])
    return 48


def background_for_case(case_name: str, resolution: int) -> Image.Image:
    for name in ("inpainted.png", "input.png"):
        path = REPO_ROOT / "cases" / case_name / name
        if path.exists():
            image = Image.open(path).convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
            image = image.filter(ImageFilter.GaussianBlur(radius=max(1, resolution // 80)))
            overlay = Image.new("RGB", image.size, (235, 238, 232))
            return Image.blend(image, overlay, 0.45)
    return Image.new("RGB", (resolution, resolution), (235, 238, 232))


def entity_position(entity) -> np.ndarray:
    pos = entity.get_pos()
    if hasattr(pos, "detach"):
        pos = pos.detach()
    if hasattr(pos, "cpu"):
        pos = pos.cpu()
    return np.asarray(pos, dtype=np.float32).reshape(-1)[:3]


def render_state_frame(case_name: str, tracked: list[dict], resolution: int) -> tuple[Image.Image, list[dict]]:
    image = background_for_case(case_name, resolution)
    draw = ImageDraw.Draw(image, "RGBA")
    scale = resolution / 2.45
    ground_y = int(resolution * 0.80)
    origin_x = int(resolution * 0.50)
    draw.line((0, ground_y, resolution, ground_y), fill=(80, 86, 92, 120), width=max(1, resolution // 180))

    records = []
    ordered = sorted(tracked, key=lambda item: entity_position(item["entity"])[1])
    for item in ordered:
        pos = entity_position(item["entity"])
        x_px = origin_x + int(pos[0] * scale)
        y_px = ground_y - int(pos[2] * scale)
        color = item["color"]
        shadow_w = max(8, int(scale * (item.get("radius") or (item.get("size") or (0.16,))[0]) * 1.8))
        draw.ellipse(
            (x_px - shadow_w, ground_y - shadow_w // 5, x_px + shadow_w, ground_y + shadow_w // 5),
            fill=(25, 28, 30, 45),
        )

        if item["kind"] == "sphere":
            radius = int(scale * float(item["radius"]))
            bbox = (x_px - radius, y_px - radius, x_px + radius, y_px + radius)
            draw.ellipse(bbox, fill=(*color, 235), outline=(20, 22, 24, 120), width=max(1, resolution // 220))
            draw.ellipse(
                (x_px - radius // 3, y_px - radius // 3, x_px, y_px),
                fill=(255, 255, 255, 45),
            )
        else:
            size = item.get("size") or (0.18, 0.18, 0.24)
            width = max(6, int(scale * float(size[0])))
            height = max(6, int(scale * float(size[2])))
            bbox = (x_px - width // 2, y_px - height // 2, x_px + width // 2, y_px + height // 2)
            if item["kind"] == "cylinder":
                draw.rounded_rectangle(bbox, radius=max(2, width // 3), fill=(*color, 230), outline=(20, 22, 24, 120))
            else:
                draw.rectangle(bbox, fill=(*color, 230), outline=(20, 22, 24, 120), width=max(1, resolution // 220))

        records.append(
            {
                "label": item["label"],
                "kind": item["kind"],
                "position_xyz": [float(pos[0]), float(pos[1]), float(pos[2])],
            }
        )
    return image, records


def render_case(case_name: str, output_root: Path, frames: int, fps: int, backend_name: str, resolution: int) -> Path:
    backend = gs.gpu if backend_name == "gpu" else gs.cpu
    gs.init(seed=0, precision="32", backend=backend, logging_level="warning")

    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0 / fps, substeps=8, gravity=(0.0, 0.0, -9.8)),
            rigid_options=gs.options.RigidOptions(dt=1.0 / fps, enable_collision=True),
            show_viewer=False,
            renderer=gs.renderers.Rasterizer(),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                ambient_light=(0.45, 0.45, 0.45),
                lights=[{"type": "directional", "dir": (0, 0, 1), "color": (1.0, 1.0, 1.0), "intensity": 2.0}],
            ),
        )
        scene.add_entity(
            material=gs.materials.Rigid(rho=1000.0, friction=0.7),
            morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)),
            surface=gs.surfaces.Default(color=(0.78, 0.78, 0.74, 1.0), vis_mode="visual"),
        )
        controls, tracked = build_case(scene, case_name)
        scene.build()

        final_sim = output_root / case_name / datetime.now().strftime("%d-%m_%H-%M-%S") / "final_sim"
        frame_dir = final_sim / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        pil_frames = []
        state_tracks = []

        for frame in range(frames):
            for control in controls:
                control(frame, frames)
            scene.step()
            image, records = render_state_frame(case_name, tracked, resolution)
            image.save(frame_dir / f"frame_{frame:04d}.png")
            pil_frames.append(image)
            state_tracks.append({"frame": frame, "objects": records})

        video_path = final_sim / "simulation.mp4"
        save_video_from_pil(pil_frames, video_path.as_posix(), fps=fps)
        (final_sim / "state_tracks.json").write_text(
            json.dumps(state_tracks, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (final_sim / "metadata.json").write_text(
            json.dumps(
                {
                    "schema": "realwonder_builtin_genesis_smoke.v2",
                    "case": case_name,
                    "frames": frames,
                    "fps": fps,
                    "backend": backend_name,
                    "renderer": "state2d",
                    "resolution": [resolution, resolution],
                    "note": (
                        "Genesis scene.step() drives object motion; frames are rendered from entity states "
                        "without SAM2/SAM3D/MoGe/RAFT/diffusion or Genesis OpenGL camera rendering."
                    ),
                    "video": video_path.as_posix(),
                    "frames_dir": frame_dir.as_posix(),
                    "state_tracks": (final_sim / "state_tracks.json").as_posix(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return video_path
    finally:
        gs.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASE_ORDER], default="all")
    parser.add_argument("--output_root", type=Path, default=Path("result/genesis_smoke_cases"))
    parser.add_argument("--frames", type=int, default=0, help="0 means use each case config's simulated_frames_num")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    cases = CASE_ORDER if args.case == "all" else [args.case]
    outputs = {}
    for case_name in cases:
        frames = case_frame_count(case_name, args.frames)
        video_path = render_case(case_name, args.output_root, frames, args.fps, args.backend, args.resolution)
        outputs[case_name] = video_path.as_posix()
        print(f"[genesis-smoke] {case_name}: {video_path}")
    summary = args.output_root / "latest_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[genesis-smoke] summary: {summary}")


if __name__ == "__main__":
    main()
