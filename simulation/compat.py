"""Small compatibility helpers for data-export workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def _cuda_arch_supported(device_index: int = 0) -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is false"

    try:
        major, minor = torch.cuda.get_device_capability(device_index)
        device_name = torch.cuda.get_device_name(device_index)
    except Exception as exc:
        return False, f"could not query CUDA device capability: {exc}"

    arch = f"sm_{major}{minor}"
    compute = f"compute_{major}{minor}"
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        arch_list = []

    if arch_list and arch not in arch_list and compute not in arch_list:
        return (
            False,
            f"{device_name} requires {arch}, but this torch build only has {arch_list}",
        )

    return True, f"{device_name} capability {arch} is supported by this torch build"


def torch_cuda_is_usable(device: str | torch.device = "cuda") -> tuple[bool, str]:
    """Return whether torch can run kernels on the requested CUDA device."""

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return True, "non-CUDA device requested"

    device_index = torch_device.index or 0
    ok, reason = _cuda_arch_supported(device_index)
    if not ok:
        return False, reason

    try:
        x = torch.ones((1,), device=torch_device)
        y = (x + 1).detach().cpu()
        if int(y.item()) != 2:
            return False, "CUDA kernel smoke test returned an unexpected value"
        torch.cuda.synchronize(torch_device)
    except Exception as exc:
        return False, f"CUDA kernel smoke test failed: {exc}"

    return True, reason


def resolve_torch_device(
    requested: str | torch.device | None = "auto",
    *,
    allow_cpu_fallback: bool = True,
) -> torch.device:
    """Resolve a device string while avoiding unsupported Blackwell CUDA builds."""

    requested = "auto" if requested is None else str(requested)
    if requested == "auto":
        ok, reason = torch_cuda_is_usable("cuda")
        if ok:
            print(f"[device] using cuda ({reason})")
            return torch.device("cuda")
        print(f"[device] falling back to cpu ({reason})")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type != "cuda":
        print(f"[device] using {device}")
        return device

    ok, reason = torch_cuda_is_usable(device)
    if ok:
        print(f"[device] using {device} ({reason})")
        return device

    if not allow_cpu_fallback:
        raise RuntimeError(reason)

    print(f"[device] falling back to cpu ({reason})")
    return torch.device("cpu")


def genesis_flow_to_chw(flows: Any) -> np.ndarray:
    """Normalize RealWonder renderer flows to [T, 2, H, W] float32."""

    arr = np.asarray(flows)
    if arr.size == 0:
        return np.zeros((0, 2, 0, 0), dtype=np.float32)

    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D flow array, got shape {arr.shape}")

    if arr.shape[-1] >= 2:
        arr = np.transpose(arr[..., :2], (0, 3, 1, 2))
    elif arr.shape[1] >= 2:
        arr = arr[:, :2]
    else:
        raise ValueError(f"Could not find two flow channels in shape {arr.shape}")

    return arr.astype(np.float32, copy=False)


def resize_and_crop_flow_chw(
    flows: np.ndarray,
    *,
    target_hw: tuple[int, int] = (832, 832),
    crop_start: int = 176,
    crop_height: int = 480,
) -> np.ndarray:
    """Resize [T, 2, H, W] flow to target size, scale vectors, then crop height."""

    flows = np.asarray(flows)
    if flows.size == 0:
        return np.zeros((0, 2, crop_height, target_hw[1]), dtype=np.float32)
    if flows.ndim != 4 or flows.shape[1] != 2:
        raise ValueError(f"Expected [T, 2, H, W] flow, got {flows.shape}")

    target_h, target_w = target_hw
    if crop_start < 0 or crop_start + crop_height > target_h:
        raise ValueError(
            f"crop_start={crop_start} and crop_height={crop_height} exceed target height {target_h}"
        )

    _, _, old_h, old_w = flows.shape
    scale_x = target_w / old_w
    scale_y = target_h / old_h
    resized = np.empty((flows.shape[0], 2, target_h, target_w), dtype=np.float32)

    for i, flow in enumerate(flows):
        resized[i, 0] = cv2.resize(flow[0], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized[i, 1] = cv2.resize(flow[1], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized[i, 0] *= scale_x
        resized[i, 1] *= scale_y

    return resized[:, :, crop_start : crop_start + crop_height, :]


def jsonable(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")
