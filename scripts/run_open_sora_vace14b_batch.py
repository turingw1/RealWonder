#!/usr/bin/env python
"""Run Wan2.1-VACE-14B on Open-Sora flow-boundary samples."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


RW_ROOT = Path("/root/autodl-tmp/Physics_worldmodel/RealWonder")
VACE_ROOT = Path("/root/autodl-tmp/Physics_worldmodel/VACE")
PYTHON_ENV = Path("/root/autodl-tmp/miniconda3/envs/realwonder_cuda128_test")
DEFAULT_EXP = RW_ROOT / "experiments/vace_flow_boundary_20260511"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, default=DEFAULT_EXP)
    parser.add_argument("--ckpt_dir", type=Path, default=VACE_ROOT / "models/Wan2.1-VACE-14B")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only_missing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def start_monitor(path: Path) -> subprocess.Popen:
    script = (
        "echo 'timestamp,index,utilization.gpu,memory.used,memory.total,power.draw'; "
        "while true; do "
        "nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw "
        "--format=csv,noheader,nounits; "
        "sleep 5; "
        "done"
    )
    return subprocess.Popen(["bash", "-lc", script], stdout=path.open("w"), stderr=subprocess.DEVNULL)


def stop_monitor(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_one(args: argparse.Namespace, sample: dict) -> dict:
    sid = sample["id"]
    flow_video = args.experiment_dir / "flows" / sid / "flow_vis.mp4"
    prompt_file = args.experiment_dir / "flows" / sid / "prompt.txt"
    if not flow_video.exists():
        raise FileNotFoundError(flow_video)
    if not prompt_file.exists():
        raise FileNotFoundError(prompt_file)

    sample_root = args.experiment_dir / "vace_outputs" / sid
    sample_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_dir = sample_root / run_id
    if args.only_missing:
        existing = sorted(sample_root.glob("*/out_video.mp4"))
        if existing:
            return {"id": sid, "status": "skipped", "save_dir": existing[-1].parent.as_posix()}

    save_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    (save_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (save_dir / "run_config.txt").write_text(
        "\n".join(
            [
                f"sample_id={sid}",
                f"size={args.size}",
                f"frame_num={args.frame_num}",
                f"sample_steps={args.sample_steps}",
                "nproc_per_node=1",
                "ulysses_size=1",
                "ring_size=1",
                "dit_fsdp=0",
                "t5_fsdp=0",
                "offload_model=False",
                f"ckpt_dir={args.ckpt_dir}",
                f"flow_video={flow_video}",
                f"save_dir={save_dir}",
                f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
                f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    env["TMPDIR"] = "/root/autodl-tmp/tmp"
    env["PIP_CACHE_DIR"] = "/root/autodl-tmp/pip-cache"
    env["PYTHONPATH"] = f"{VACE_ROOT / 'vace'}{os.pathsep}{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(VACE_ROOT / "vace")
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["SETUPTOOLS_USE_DISTUTILS"] = "local"
    env["LIDRA_SKIP_INIT"] = "1"

    cmd = [
        str(PYTHON_ENV / "bin/torchrun"),
        "--nproc_per_node=1",
        "vace/vace_wan_inference.py",
        "--ulysses_size",
        "1",
        "--ring_size",
        "1",
        "--offload_model",
        "False",
        "--size",
        args.size,
        "--model_name",
        "vace-14B",
        "--ckpt_dir",
        args.ckpt_dir.as_posix(),
        "--src_video",
        flow_video.as_posix(),
        "--prompt",
        prompt,
        "--frame_num",
        str(args.frame_num),
        "--sample_steps",
        str(args.sample_steps),
        "--save_dir",
        save_dir.as_posix(),
    ]

    monitor = start_monitor(save_dir / "gpu_usage.csv")
    start = int(time.time())
    with (save_dir / "vace_stdout_stderr.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=VACE_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    end = int(time.time())
    stop_monitor(monitor)

    status = proc.returncode
    (save_dir / "time.txt").write_text(
        f"start_epoch={start}\nend_epoch={end}\nelapsed_seconds={end - start}\nexit_status={status}\n",
        encoding="utf-8",
    )
    return {"id": sid, "status": "ok" if status == 0 else "failed", "exit_status": status, "elapsed_seconds": end - start, "save_dir": save_dir.as_posix()}


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"][: args.limit]
    results = []
    for sample in samples:
        print(f"[vace] {sample['id']}", flush=True)
        result = run_one(args, sample)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result.get("status") == "failed":
            break

    out = args.experiment_dir / "vace_outputs" / "batch_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "open_sora_vace14b_batch_results.v1", "results": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if any(item.get("status") == "failed" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
