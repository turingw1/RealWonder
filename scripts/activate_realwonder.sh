#!/usr/bin/env bash
set -eo pipefail

CONDA_ROOT="${CONDA_ROOT:-/root/autodl-tmp/miniconda3}"
REALWONDER_CONDA_ENV="${REALWONDER_CONDA_ENV:-realwonder_cuda128_test}"
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
else
  echo "conda.sh not found under ${CONDA_ROOT}" >&2
  exit 1
fi

set +u
conda activate "${REALWONDER_CONDA_ENV}"
set -u

# Kaolin -> torchvision -> triton import path is stable with the local distutils shim.
export SETUPTOOLS_USE_DISTUTILS=local

# Current sam_3d_objects snapshot imports a missing sam3d_objects.init module.
export LIDRA_SKIP_INIT=1

# Make Python import setuptools early for packages that still touch distutils.
export PYTHONPATH="/root/autodl-tmp/Physics_worldmodel/RealWonder/scripts${PYTHONPATH:+:${PYTHONPATH}}"

echo "Activated conda env: ${REALWONDER_CONDA_ENV}"
echo "SETUPTOOLS_USE_DISTUTILS=${SETUPTOOLS_USE_DISTUTILS}"
echo "LIDRA_SKIP_INIT=${LIDRA_SKIP_INIT}"
echo "PYTHONPATH prepended with RealWonder/scripts"
