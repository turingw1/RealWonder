#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/Physics_worldmodel/RealWonder"

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh

# Large framework/package downloads should use the server default route.
# The 18080 proxy is reserved for small external transfers when the default
# route is unavailable or too slow.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

conda env remove -n simulation -y >/dev/null 2>&1 || true
conda create -n simulation -y \
  --solver=libmamba \
  --override-channels \
  -c conda-forge \
  python=3.11 pip setuptools wheel ffmpeg

source "${ROOT}/scripts/activate_simulation.sh"

python -m pip install -U pip setuptools wheel packaging ninja
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  av==12.0.0 \
  einops \
  fire \
  ffmpeg-python \
  imageio \
  imageio-ffmpeg \
  'numpy<2' \
  opencv-python==4.9.0.80 \
  pillow \
  rp \
  scipy \
  tqdm

python -m pip install -e "${ROOT}/submodules/Genesis"

python -c "import torch, torchvision; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torchvision.__version__)"
python -c "import genesis as gs; print('genesis', gs.__version__)"
