#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-cases/lamp/config.yaml}"

cd "${ROOT_DIR}"

# Use the server default network for model/data paths. The 18080 proxy is only a fallback
# for small external access, so clear inherited proxy variables for the demo run.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

# shellcheck disable=SC1091
source scripts/activate_realwonder.sh

python -u case_simulation.py \
  --config_path "${CONFIG_PATH}" \
  --device cuda \
  --genesis_backend gpu \
  --skip_noise_warp \
  --save_raw_frames
