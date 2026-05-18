#!/usr/bin/env bash
set -e

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh

__old_nounset="$(set +o | grep nounset || true)"
set +u
conda activate simulation
if [[ -n "${__old_nounset}" ]]; then
  eval "${__old_nounset}"
fi
unset __old_nounset

export SETUPTOOLS_USE_DISTUTILS=local
export PYTHONPATH="/root/autodl-tmp/Physics_worldmodel/RealWonder:/root/autodl-tmp/Physics_worldmodel/RealWonder/submodules/Genesis${PYTHONPATH:+:${PYTHONPATH}}"
