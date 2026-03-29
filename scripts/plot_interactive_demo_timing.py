#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot interactive demo timing logs as a pipeline timeline.",
    )
    parser.add_argument(
        "log_path",
        help="Run directory under experiment_logs, or the experiment_logs directory itself.",
    )
    parser.add_argument(
        "--output",
        help="Output image path. Defaults to <run_dir>/pipeline_timing.png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI",
    )
    return parser.parse_args()


def resolve_run_dir(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"log path does not exist: {path}")
    if path.is_file():
        path = path.parent

    if (path / "bootstrap.events.jsonl").exists() or (path / "generation.events.jsonl").exists():
        return path

    candidates = sorted(
        [p for p in path.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(f"no run directories found under: {path}")
    return candidates[-1]


def parse_time(value):
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").timestamp()


def load_jsonl(path):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Segment:
    lane: str
    label: str
    start: float
    duration: float
    color: str
    hatch: str = ""
    alpha: float = 1.0
    text: str | None = None
    actual_duration: float | None = None


def segments_from_bootstrap(records):
    segs = []
    for rec in records:
        end = parse_time(rec["timestamp"])
        duration = float(rec.get("duration_sec") or 0.0)
        start = end - duration
        segs.append(
            Segment(
                lane="startup",
                label=rec["stage"],
                start=start,
                duration=duration,
                color="#4C78A8",
                text=_short_stage_label(rec["stage"]),
                actual_duration=duration,
            )
        )
    return segs


def segments_from_startup(records):
    segs = []
    for rec in records:
        end = parse_time(rec["timestamp"])
        duration = float(rec.get("duration_sec") or 0.0)
        start = end - duration
        segs.append(
            Segment(
                lane="warmup",
                label=rec["stage"],
                start=start,
                duration=duration,
                color="#72B7B2",
                text=_short_stage_label(rec["stage"]),
                actual_duration=duration,
            )
        )
    return segs


def segments_from_generation(records):
    segs = []
    for rec in records:
        stage = rec["stage"]
        end = parse_time(rec["timestamp"])
        duration = float(rec.get("duration_sec") or 0.0)
        block_idx = rec.get("block_idx", "?")

        if stage == "demo.stage1_render_flow_block":
            start = end - duration
            cursor = start
            parts = [
                ("sim", f"B{block_idx} physics", float(rec.get("physics_step_total_sec") or 0.0), "#4C78A8"),
                ("sim", f"B{block_idx} render+flow", float(rec.get("render_flow_total_sec") or 0.0), "#2E5EAA"),
                ("sim", f"B{block_idx} resize", float(rec.get("resize_total_sec") or 0.0), "#8FB1E3"),
                ("sim", f"B{block_idx} queue_put", float(rec.get("queue_put_sec") or 0.0), "#C7D5F0"),
            ]
            segs.extend(_split_segments(parts, cursor))

        elif stage == "demo.stage2_noise_warp_block":
            queue_wait = float(rec.get("queue_wait_sec") or 0.0)
            start = end - queue_wait - duration
            if queue_wait > 0:
                segs.append(
                    Segment(
                        lane="warp",
                        label=f"B{block_idx} wait",
                        start=start,
                        duration=queue_wait,
                        color="#BDBDBD",
                        hatch="///",
                        alpha=0.9,
                        text=f"B{block_idx} wait",
                        actual_duration=queue_wait,
                    )
                )
            cursor = start + queue_wait
            parts = [
                ("warp", f"B{block_idx} warp", float(rec.get("warp_steps_sec") or 0.0), "#F58518"),
                ("warp", f"B{block_idx} get_noise", float(rec.get("get_block_noise_sec") or 0.0), "#F2B377"),
                ("warp", f"B{block_idx} queue_put", float(rec.get("queue_put_sec") or 0.0), "#F7D6AE"),
            ]
            segs.extend(_split_segments(parts, cursor))

        elif stage == "demo.stage3_diffusion_block":
            queue_wait = float(rec.get("queue_wait_sec") or 0.0)
            start = end - queue_wait - duration
            if queue_wait > 0:
                segs.append(
                    Segment(
                        lane="diffusion",
                        label=f"B{block_idx} wait",
                        start=start,
                        duration=queue_wait,
                        color="#BDBDBD",
                        hatch="///",
                        alpha=0.9,
                        text=f"B{block_idx} wait",
                        actual_duration=queue_wait,
                    )
                )
            cursor = start + queue_wait
            parts = [
                ("diffusion", f"B{block_idx} VAE", float(rec.get("vae_encode_sec") or 0.0), "#54A24B"),
                ("diffusion", f"B{block_idx} mask", float(rec.get("mask_build_sec") or 0.0), "#A0CF99"),
                ("diffusion", f"B{block_idx} diffusion", float(rec.get("diffusion_sec") or 0.0), "#2E8540"),
            ]
            segs.extend(_split_segments(parts, cursor))

    return segs


def _split_segments(parts, start):
    segs = []
    cursor = start
    for lane, label, duration, color in parts:
        if duration <= 0:
            continue
        segs.append(
            Segment(
                lane=lane,
                label=label,
                start=cursor,
                duration=duration,
                color=color,
                text=label,
                actual_duration=duration,
            )
        )
        cursor += duration
    return segs


def _short_stage_label(stage):
    tail = stage.split(".")[-1]
    mapping = {
        "initialize_video_generator": "video init",
        "initialize_simulator": "sim init",
        "initialize_case_handler": "handler",
        "initialize_noise_warper": "warp init",
        "precompute_first_frame": "first frame",
        "sim_render": "sim warm",
        "noise_warp": "warp warm",
        "vae_diffusion": "diff warm",
    }
    return mapping.get(tail, tail.replace("_", " "))


def _short_generation_label(label):
    parts = label.split()
    if len(parts) < 2:
        return label
    block, stage = parts[0], " ".join(parts[1:])
    mapping = {
        "physics": "phys",
        "render+flow": "render",
        "resize": "resize",
        "queue_put": "put",
        "wait": "wait",
        "warp": "warp",
        "get_noise": "noise",
        "VAE": "vae",
        "mask": "mask",
        "diffusion": "diff",
    }
    return f"{block} {mapping.get(stage, stage)}"


def build_figure(run_dir, output_path, dpi):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    bootstrap_records = load_jsonl(run_dir / "bootstrap.events.jsonl")
    startup_records = load_jsonl(run_dir / "startup.events.jsonl")
    generation_records = load_jsonl(run_dir / "generation.events.jsonl")
    generation_summary = load_json(run_dir / "generation.summary.json")
    bootstrap_summary = load_json(run_dir / "bootstrap.summary.json")
    startup_summary = load_json(run_dir / "startup.summary.json")

    startup_segments = segments_from_bootstrap(bootstrap_records) + segments_from_startup(startup_records)
    generation_segments = segments_from_generation(generation_records)
    all_segments = startup_segments + generation_segments
    if not all_segments:
        raise RuntimeError(f"no timing segments found under {run_dir}")

    gen_t0 = min((seg.start for seg in generation_segments), default=min(seg.start for seg in all_segments))
    startup_t0 = min((seg.start for seg in startup_segments), default=gen_t0)
    startup_end = max((seg.start + seg.duration for seg in startup_segments), default=startup_t0)
    generation_end = max((seg.start + seg.duration for seg in generation_segments), default=startup_end)
    startup_total = max(0.0, startup_end - startup_t0)
    generation_total = max(1.0, generation_end - gen_t0)

    compressed_startup_width = max(6.0, min(18.0, generation_total * 0.16))
    startup_scale = compressed_startup_width / startup_total if startup_total > 0 else 1.0
    startup_gap = 1.2

    lanes = ["startup", "warmup", "sim", "warp", "diffusion"]
    lane_y = {lane: idx for idx, lane in enumerate(reversed(lanes))}

    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)

    for idx, seg in enumerate(all_segments):
        y = lane_y[seg.lane]
        if seg.lane in {"startup", "warmup"}:
            display_start = (seg.start - startup_t0) * startup_scale
            display_duration = max(seg.duration * startup_scale, 0.18)
            text = seg.text or seg.label
        else:
            display_start = compressed_startup_width + startup_gap + (seg.start - gen_t0)
            display_duration = seg.duration
            text = _short_generation_label(seg.text or seg.label)
        ax.barh(
            y=y,
            width=display_duration,
            left=display_start,
            height=0.72,
            color=seg.color,
            alpha=seg.alpha,
            hatch=seg.hatch,
            edgecolor="#444444" if seg.hatch else "white",
            linewidth=0.6,
        )
        actual_duration = seg.actual_duration if seg.actual_duration is not None else seg.duration
        if seg.lane in {"startup", "warmup"}:
            ax.text(
                display_start + display_duration / 2,
                y,
                f"{text}\n{actual_duration:.1f}s",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
        elif display_duration >= 0.12:
            label_y = y + (0.27 if idx % 2 == 0 else -0.27)
            ax.text(
                display_start + display_duration / 2,
                label_y,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    divider_x = compressed_startup_width + startup_gap / 2
    ax.axvline(divider_x, color="#666666", linestyle=":", linewidth=1.0)
    ax.text(compressed_startup_width / 2, max(lane_y.values()) + 0.7, "startup (compressed)", ha="center", va="bottom", fontsize=9)
    ax.text(divider_x + generation_total / 2, max(lane_y.values()) + 0.7, "generation pipeline", ha="center", va="bottom", fontsize=9)

    ax.set_yticks([lane_y[lane] for lane in reversed(lanes)], list(reversed(lanes)))
    ax.set_xlabel("Display Timeline (startup compressed, generation at real scale)")
    ax.set_title(_make_title(run_dir, generation_summary, bootstrap_summary, startup_summary))
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    legend_items = [
        Patch(facecolor="#4C78A8", label="Startup / Physics"),
        Patch(facecolor="#2E5EAA", label="Render + Flow"),
        Patch(facecolor="#F58518", label="Noise Warp"),
        Patch(facecolor="#54A24B", label="VAE / Diffusion"),
        Patch(facecolor="#BDBDBD", hatch="///", label="Queue Wait"),
    ]
    ax.legend(handles=legend_items, loc="upper right")

    _write_summary_text(fig, run_dir, generation_summary, bootstrap_summary, startup_summary)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _make_title(run_dir, generation_summary, bootstrap_summary, startup_summary):
    run_name = generation_summary.get("run_name") or bootstrap_summary.get("run_name") or run_dir.name
    parts = [f"Interactive Demo Timing Pipeline: {run_name}", run_dir.name]
    return "\n".join(parts)


def _write_summary_text(fig, run_dir, generation_summary, bootstrap_summary, startup_summary):
    bootstrap_sec = bootstrap_summary.get("total_duration_sec")
    startup_sec = startup_summary.get("total_duration_sec")
    generation_sec = generation_summary.get("total_duration_sec")
    lines = [
        f"run_dir: {run_dir}",
        f"bootstrap_total_sec: {bootstrap_sec if bootstrap_sec is not None else 'N/A'}",
        f"startup_total_sec: {startup_sec if startup_sec is not None else 'N/A'}",
        f"generation_total_sec: {generation_sec if generation_sec is not None else 'N/A'}",
    ]
    fig.text(0.01, 0.01, "\n".join(lines), ha="left", va="bottom", fontsize=9, family="monospace")


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.log_path)
    output_path = Path(args.output) if args.output else run_dir / "pipeline_timing.png"
    build_figure(run_dir, output_path, args.dpi)
    print(output_path)


if __name__ == "__main__":
    main()
