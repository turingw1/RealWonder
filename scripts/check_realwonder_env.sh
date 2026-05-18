#!/usr/bin/env bash

set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH"
  exit 1
fi

echo "[Python]"
python -V

echo
echo "[Torch]"
python - <<'PY'
import importlib

try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device 0:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        print("arch list:", torch.cuda.get_arch_list())
        try:
            x = torch.ones((1,), device="cuda")
            y = (x + 1).cpu()
            torch.cuda.synchronize()
            print("cuda kernel smoke:", "OK", y.tolist())
        except Exception as kernel_exc:
            print("cuda kernel smoke: FAIL ->", kernel_exc)
except Exception as exc:
    print("torch check failed:", exc)
PY

echo
echo "[Key Packages]"
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

packages = [
    "torch",
    "torchvision",
    "torchaudio",
    "sam3d_objects",
    "flash_attn",
    "xformers",
    "bitsandbytes",
    "pytorch3d",
    "kaolin",
    "gsplat",
    "SAM-2",
    "genesis-world",
    "diffusers",
    "open_clip_torch",
    "kornia",
    "hatchling",
    "hatch-requirements-txt",
    "editables",
]

for name in packages:
    try:
        print(f"{name}: {version(name)}")
    except PackageNotFoundError:
        print(f"{name}: MISSING")
PY

echo
echo "[Import Checks]"
LIDRA_SKIP_INIT=1 python - <<'PY'
checks = [
    ("sam3d_objects", "sam3d_objects"),
    ("pytorch3d", "pytorch3d"),
    ("kaolin", "kaolin"),
    ("gsplat", "gsplat"),
    ("sam2", "sam2"),
]

for label, module_name in checks:
    try:
        __import__(module_name)
        print(f"{label}: OK")
    except Exception as exc:
        print(f"{label}: FAIL -> {exc}")

optional_checks = [
    ("flash_attn", "flash_attn"),
    ("xformers", "xformers"),
    ("bitsandbytes", "bitsandbytes"),
]

for label, module_name in optional_checks:
    try:
        __import__(module_name)
        print(f"{label}: OPTIONAL PRESENT")
    except Exception as exc:
        print(f"{label}: OPTIONAL MISSING/UNUSABLE -> {exc}")
PY
