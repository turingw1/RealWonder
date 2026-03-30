import json
import os
import socket
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


class ExperimentLogger:
    """Structured JSONL logger for demo timing experiments."""

    def __init__(self, *, experiment_name, run_name, output_dir, metadata=None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = self.output_dir.name
        self.start_perf = time.perf_counter()
        self.start_time_epoch = time.time()
        self.start_time = _now_local()
        self.metadata = metadata or {}
        self.events = []

        file_stem = _experiment_label(self.experiment_name)
        self.events_path = self.output_dir / f"{file_stem}.events.jsonl"
        self.summary_path = self.output_dir / f"{file_stem}.summary.json"

    def _base(self):
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
        }

    def log_event(self, stage, duration_sec=None, **payload):
        start_perf = payload.pop("start_perf", None)
        end_perf = payload.pop("end_perf", None)
        if end_perf is None:
            end_perf = time.perf_counter()
        if duration_sec is None and start_perf is not None:
            duration_sec = end_perf - start_perf
        if start_perf is None:
            if duration_sec is not None:
                start_perf = end_perf - duration_sec
            else:
                start_perf = end_perf

        end_time = _now_local()
        record = {
            **self._base(),
            "timestamp": end_time,
            "start_time": _format_local_from_offset(self.start_time_epoch, start_perf - self.start_perf),
            "end_time": end_time,
            "stage": stage,
            "duration_sec": duration_sec,
            "relative_start_sec": start_perf - self.start_perf,
            "relative_end_sec": end_perf - self.start_perf,
            **{k: _json_safe(v) for k, v in payload.items()},
        }
        self.events.append(record)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @contextmanager
    def time_block(self, stage, **payload):
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.log_event(stage, end - start, start_perf=start, end_perf=end, **payload)

    def finalize(self, *, status="completed", **payload):
        end_perf = time.perf_counter()
        summary = {
            **self._base(),
            "status": status,
            "start_time": self.start_time,
            "end_time": _now_local(),
            "total_duration_sec": end_perf - self.start_perf,
            "event_count": len(self.events),
            "host": socket.gethostname(),
            "cwd": os.getcwd(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "metadata": _json_safe(self.metadata),
            "events_path": str(self.events_path),
            **{k: _json_safe(v) for k, v in payload.items()},
        }
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


def create_session_dir(base_dir, run_name):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    session_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    session_dir = base_path / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _now_local():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _format_local_from_offset(start_epoch, offset_sec):
    return datetime.fromtimestamp(start_epoch + offset_sec).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _experiment_label(experiment_name):
    short = experiment_name.replace("interactive_demo_", "")
    return short or experiment_name
