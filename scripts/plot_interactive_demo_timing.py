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

    candidates = sorted((p for p in path.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(f"no run directories found under: {path}")
    return candidates[-1]


def parse_time(value):
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            pass
    raise ValueError(f"unsupported timestamp format: {value}")


def load_jsonl(path):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def event_bounds(rec):
    if "relative_start_sec" in rec and "relative_end_sec" in rec:
        return float(rec["relative_start_sec"]), float(rec["relative_end_sec"])
    end = parse_time(rec["timestamp"])
    duration = float(rec.get("duration_sec") or 0.0)
    return end - duration, end


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


def segments_from_bootstrap(records, offset=0.0):
    segs = []
    for rec in records:
        start, end = event_bounds(rec)
        start += offset
        end += offset
        duration = end - start
        segs.append(
            Segment(
                lane="startup",
                label=rec["stage"],
                start=start,
                duration=duration,
                color="#4C78A8",
                text=short_stage_label(rec["stage"]),
                actual_duration=duration,
            )
        )
    return segs


def segments_from_startup(records, offset=0.0):
    segs = []
    for rec in records:
        start, end = event_bounds(rec)
        start += offset
        end += offset
        duration = end - start
        segs.append(
            Segment(
                lane="warmup",
                label=rec["stage"],
                start=start,
                duration=duration,
                color="#72B7B2",
                text=short_stage_label(rec["stage"]),
                actual_duration=duration,
            )
        )
    return segs


def segments_from_generation(records, offset=0.0):
    segs = []
    for rec in records:
        stage = rec["stage"]
        start, end = event_bounds(rec)
        start += offset
        end += offset
        duration = end - start
        block_idx = rec.get("block_idx", "?")

        if stage == "demo.prepare_generation":
            segs.append(
                Segment(
                    lane="prepare",
                    label="prepare",
                    start=start,
                    duration=duration,
                    color="#9C755F",
                    text="prepare",
                    actual_duration=duration,
                )
            )
            continue

        if stage == "demo.stage1a_physics_block":
            parts = [
                ("physics", f"B{block_idx} physics", float(rec.get("physics_step_total_sec") or 0.0), "#4C78A8"),
                ("physics", f"B{block_idx} put", float(rec.get("queue_put_sec") or 0.0), "#C7D5F0"),
            ]
            segs.extend(split_segments(parts, start))
            segs.extend(other_segment("physics", block_idx, start, duration, parts, "#9FBCE5"))
            continue

        if stage in {"demo.stage1_render_flow_block", "demo.stage1b_render_flow_block"}:
            queue_wait = float(rec.get("queue_wait_sec") or 0.0)
            if queue_wait > 0:
                segs.append(wait_segment("render", block_idx, start, queue_wait))
            parts = [
                ("render", f"B{block_idx} render+flow", float(rec.get("render_flow_total_sec") or 0.0), "#2E5EAA"),
                ("render", f"B{block_idx} resize", float(rec.get("resize_total_sec") or 0.0), "#8FB1E3"),
                ("render", f"B{block_idx} put", float(rec.get("queue_put_sec") or 0.0), "#C7D5F0"),
            ]
            segs.extend(split_segments(parts, start + queue_wait))
            segs.extend(other_segment("render", block_idx, start + queue_wait, duration - queue_wait, parts, "#DCE7F7"))
            continue

        if stage == "demo.stage2_noise_warp_block":
            queue_wait = float(rec.get("queue_wait_sec") or 0.0)
            if queue_wait > 0:
                segs.append(wait_segment("warp", block_idx, start, queue_wait))
            parts = [
                ("warp", f"B{block_idx} warp", float(rec.get("warp_steps_sec") or 0.0), "#F58518"),
                ("warp", f"B{block_idx} noise", float(rec.get("get_block_noise_sec") or 0.0), "#F2B377"),
                ("warp", f"B{block_idx} put", float(rec.get("queue_put_sec") or 0.0), "#F7D6AE"),
            ]
            segs.extend(split_segments(parts, start + queue_wait))
            segs.extend(other_segment("warp", block_idx, start + queue_wait, duration - queue_wait, parts, "#FBE4C7"))
            continue

        if stage == "demo.stage3_diffusion_block":
            queue_wait = float(rec.get("queue_wait_sec") or 0.0)
            if queue_wait > 0:
                segs.append(wait_segment("diffusion", block_idx, start, queue_wait))
            parts = [
                ("diffusion", f"B{block_idx} vae", float(rec.get("vae_encode_sec") or 0.0), "#54A24B"),
                ("diffusion", f"B{block_idx} mask", float(rec.get("mask_build_sec") or 0.0), "#A0CF99"),
                ("diffusion", f"B{block_idx} diff", float(rec.get("diffusion_sec") or 0.0), "#2E8540"),
            ]
            segs.extend(split_segments(parts, start + queue_wait))
            segs.extend(other_segment("diffusion", block_idx, start + queue_wait, duration - queue_wait, parts, "#C9E3C6"))
            continue

    return segs


def split_segments(parts, start):
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
                text=short_generation_label(label),
                actual_duration=duration,
            )
        )
        cursor += duration
    return segs


def wait_segment(lane, block_idx, start, duration):
    return Segment(
        lane=lane,
        label=f"B{block_idx} wait",
        start=start,
        duration=duration,
        color="#BDBDBD",
        hatch="///",
        alpha=0.9,
        text=f"B{block_idx} wait",
        actual_duration=duration,
    )


def other_segment(lane, block_idx, start, duration, parts, color):
    used = sum(max(0.0, d) for _, _, d, _ in parts)
    remainder = duration - used
    if remainder <= 0.02:
        return []
    return [
        Segment(
            lane=lane,
            label=f"B{block_idx} other",
            start=start + used,
            duration=remainder,
            color=color,
            text=f"B{block_idx} other",
            actual_duration=remainder,
        )
    ]


def short_stage_label(stage):
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


def short_generation_label(label):
    parts = label.split()
    if len(parts) < 2:
        return label
    return f"{parts[0]} {parts[1]}"


def generation_label_offset(seg):
    block_idx = -1
    label = seg.label or ""
    if label.startswith("B"):
        block_token = label.split()[0]
        try:
            block_idx = int(block_token[1:])
        except ValueError:
            block_idx = -1

    # Use four stable vertical slots so adjacent blocks do not all stack onto
    # the same two label positions in dense timeline regions.
    offsets = [0.34, 0.12, -0.12, -0.34]
    if block_idx >= 0:
        return offsets[block_idx % len(offsets)]
    return offsets[0]


def build_figure(run_dir, output_path, dpi):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    bootstrap_records = load_jsonl(run_dir / "bootstrap.events.jsonl")
    startup_records = load_jsonl(run_dir / "startup.events.jsonl")
    generation_records = load_jsonl(run_dir / "generation.events.jsonl")
    generation_summary = load_json(run_dir / "generation.summary.json")
    bootstrap_summary = load_json(run_dir / "bootstrap.summary.json")
    startup_summary = load_json(run_dir / "startup.summary.json")

    bootstrap_total = float(bootstrap_summary.get("total_duration_sec") or 0.0)
    startup_segments = (
        segments_from_bootstrap(bootstrap_records, offset=0.0)
        + segments_from_startup(startup_records, offset=bootstrap_total)
    )
    generation_segments = segments_from_generation(generation_records, offset=0.0)
    all_segments = startup_segments + generation_segments
    if not all_segments:
        raise RuntimeError(f"no timing segments found under {run_dir}")

    gen_t0 = min((seg.start for seg in generation_segments), default=min(seg.start for seg in all_segments))
    startup_t0 = min((seg.start for seg in startup_segments), default=gen_t0)
    startup_end = max((seg.start + seg.duration for seg in startup_segments), default=startup_t0)
    generation_end = max((seg.start + seg.duration for seg in generation_segments), default=gen_t0 + 1.0)
    startup_total = max(0.0, startup_end - startup_t0)
    generation_total = max(1.0, generation_end - gen_t0)

    compressed_startup_width = max(5.0, min(14.0, generation_total * 0.14))
    startup_scale = compressed_startup_width / startup_total if startup_total > 0 else 1.0
    startup_gap = 1.0

    lanes = ["startup", "warmup", "prepare", "physics", "render", "warp", "diffusion"]
    lane_y = {lane: idx for idx, lane in enumerate(reversed(lanes))}

    fig, ax = plt.subplots(figsize=(18, 8.5), constrained_layout=True)

    for idx, seg in enumerate(all_segments):
        y = lane_y[seg.lane]
        if seg.lane in {"startup", "warmup"}:
            display_start = (seg.start - startup_t0) * startup_scale
            display_duration = max(seg.duration * startup_scale, 0.16)
            label_text = f"{seg.text}\n{seg.actual_duration:.1f}s"
        else:
            display_start = compressed_startup_width + startup_gap + (seg.start - gen_t0)
            display_duration = seg.duration
            label_text = seg.text or seg.label

        ax.barh(
            y=y,
            width=display_duration,
            left=display_start,
            height=0.68,
            color=seg.color,
            alpha=seg.alpha,
            hatch=seg.hatch,
            edgecolor="#444444" if seg.hatch else "white",
            linewidth=0.6,
        )

        if seg.lane in {"startup", "warmup"}:
            ax.text(
                display_start + display_duration / 2,
                y,
                label_text,
                ha="center",
                va="center",
                fontsize=7,
            )
        elif display_duration >= 0.08:
            label_y = y + generation_label_offset(seg)
            ax.text(
                display_start + display_duration / 2,
                label_y,
                label_text,
                ha="center",
                va="center",
                fontsize=6.2,
            )

    divider_x = compressed_startup_width + startup_gap / 2
    ax.axvline(divider_x, color="#666666", linestyle=":", linewidth=1.0)
    ax.text(compressed_startup_width / 2, max(lane_y.values()) + 0.72, "startup (compressed)", ha="center", va="bottom", fontsize=9)
    ax.text(divider_x + generation_total / 2, max(lane_y.values()) + 0.72, "generation pipeline", ha="center", va="bottom", fontsize=9)

    ax.set_yticks([lane_y[lane] for lane in reversed(lanes)], list(reversed(lanes)))
    ax.set_xlabel("Display timeline (startup compressed, generation at real scale)")
    ax.set_title(make_title(run_dir, generation_summary, bootstrap_summary, startup_summary))
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    legend_items = [
        Patch(facecolor="#4C78A8", label="Startup / Physics"),
        Patch(facecolor="#2E5EAA", label="Render + Flow"),
        Patch(facecolor="#F58518", label="Noise Warp"),
        Patch(facecolor="#54A24B", label="VAE / Diffusion"),
        Patch(facecolor="#9C755F", label="Prepare"),
        Patch(facecolor="#BDBDBD", hatch="///", label="Queue Wait"),
    ]
    ax.legend(handles=legend_items, loc="upper right")

    write_summary_text(fig, run_dir, generation_summary, bootstrap_summary, startup_summary)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def make_title(run_dir, generation_summary, bootstrap_summary, startup_summary):
    run_name = generation_summary.get("run_name") or bootstrap_summary.get("run_name") or run_dir.name
    return f"Interactive Demo Timing Pipeline: {run_name}\n{run_dir.name}"


def write_summary_text(fig, run_dir, generation_summary, bootstrap_summary, startup_summary):
    lines = [
        f"run_dir: {run_dir}",
        f"bootstrap_total_sec: {bootstrap_summary.get('total_duration_sec', 'N/A')}",
        f"startup_total_sec: {startup_summary.get('total_duration_sec', 'N/A')}",
        f"generation_total_sec: {generation_summary.get('total_duration_sec', 'N/A')}",
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
