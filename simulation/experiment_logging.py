import json
import os
import socket
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
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
    """Structured timing logger for offline and interactive demo runs."""

    def __init__(self, *, experiment_name, run_name, output_dir, metadata=None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}__{run_name}__{timestamp}__{uuid.uuid4().hex[:8]}"
        self.start_perf = time.perf_counter()
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.events = []
        self.metadata = metadata or {}

        self.summary_path = self.output_dir / f"{self.run_id}.summary.json"
        self.events_path = self.output_dir / f"{self.run_id}.events.jsonl"

    def _base_record(self):
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
        }

    def log_event(self, stage, duration_sec=None, **payload):
        record = self._base_record()
        record.update(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "duration_sec": duration_sec,
            }
        )
        record.update({k: _json_safe(v) for k, v in payload.items()})
        self.events.append(record)

        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @contextmanager
    def time_block(self, stage, **payload):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.log_event(stage, time.perf_counter() - start, **payload)

    def finalize(self, *, status="completed", **payload):
        total_duration = time.perf_counter() - self.start_perf
        summary = {
            **self._base_record(),
            "status": status,
            "start_time_utc": self.start_time,
            "end_time_utc": datetime.now(timezone.utc).isoformat(),
            "total_duration_sec": total_duration,
            "event_count": len(self.events),
            "host": socket.gethostname(),
            "cwd": os.getcwd(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "metadata": _json_safe(self.metadata),
            "events_path": str(self.events_path),
        }
        summary.update({k: _json_safe(v) for k, v in payload.items()})

        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary
