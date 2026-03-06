"""GPU memory profiling utility."""

import torch

_enabled = True


def set_gpu_logging(enabled: bool):
    global _enabled
    _enabled = enabled


def log_gpu(label: str):
    """Print GPU memory usage at a labeled checkpoint."""
    if not _enabled or not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    free, total = torch.cuda.mem_get_info()
    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)
    print(
        f"[GPU] {label:<45s} | "
        f"alloc: {allocated:6.2f} GB | "
        f"reserved: {reserved:6.2f} GB | "
        f"peak: {max_allocated:6.2f} GB | "
        f"free: {free_gb:6.2f} GB / {total_gb:.1f} GB"
    )
