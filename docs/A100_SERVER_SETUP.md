# RealWonder A100 Server Setup Guide

This is the canonical environment setup manual for deploying RealWonder on a shared A100 server.

It is written for this server layout:

- Workspace repo: `~/workspace/Zhengwei/RealWonder`
- Cache root: `/cache/Zhengwei/RealWonder`
- Conda environment prefix: `/cache/Zhengwei/RealWonder/conda_envs/realwonder`

This guide is opinionated. It reflects the real failure cases already seen during setup:

- workspace quota and `.git/modules` growth
- GitHub clone instability
- `hatchling` / `editables` missing for editable installs
- `flash_attn` pretending to build while actually compiling for a long time
- `pytorch3d` and `gsplat` failing in CUDA extension builds
- `nvidia-pyindex` failing because `appdirs` is missing
- gated Hugging Face repos returning `403` on mirrors
- large wheel downloads being too slow to trust in one-shot installs

The priority is:

1. keep the workspace small
2. keep the conda environment and large caches under `/cache`
3. use current-terminal-only environment variables on shared servers
4. prefer explicit installs over optimistic umbrella commands

## 1. One-Time Directory Layout

Create the cache layout first:

```bash
mkdir -p /cache/Zhengwei/RealWonder/{conda_envs,hf,torch,torch_extensions,triton,warp,tmp,logs,wheels,src}
mkdir -p /cache/Zhengwei/RealWonder/{ckpts,wan_models}
mkdir -p /cache/Zhengwei/RealWonder/sam3d_objects/checkpoints
mkdir -p /cache/Zhengwei/RealWonder/sam2/checkpoints
```

Recommended shell variables for the current terminal:

```bash
export RW_ROOT=~/workspace/Zhengwei/RealWonder
export RW_CACHE=/cache/Zhengwei/RealWonder
export RW_ENV_PREFIX=/cache/Zhengwei/RealWonder/conda_envs/realwonder

export HF_HOME=$RW_CACHE/hf
export HUGGINGFACE_HUB_CACHE=$RW_CACHE/hf/hub
export TORCH_HOME=$RW_CACHE/torch
export TORCH_EXTENSIONS_DIR=$RW_CACHE/torch_extensions
export TRITON_CACHE_DIR=$RW_CACHE/triton
export WARP_CACHE_DIR=$RW_CACHE/warp
export XDG_CACHE_HOME=$RW_CACHE/tmp

export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
```

Notes:

- `PIP_INDEX_URL` keeps ordinary Python packages on the Tsinghua mirror.
- `PIP_EXTRA_INDEX_URL` keeps PyTorch and NVIDIA wheels reachable.
- Do not write these into global shell startup files on a shared server unless you control the account.

## 2. Git Clone And Sync

Clone into the workspace:

```bash
mkdir -p ~/workspace/Zhengwei
cd ~/workspace/Zhengwei
git clone --recursive https://github.com/<your-user>/RealWonder.git
cd RealWonder
```

If GitHub is unstable, use current-terminal Git overrides instead of changing global Git config:

```bash
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=url.https://githubfast.com/.insteadOf
export GIT_CONFIG_VALUE_0=https://github.com/
export GIT_CONFIG_KEY_1=http.version
export GIT_CONFIG_VALUE_1=HTTP/1.1
```

Then sync submodules:

```bash
cd $RW_ROOT
git submodule sync --recursive
git submodule update --init --recursive
```

Pin Genesis to the tested commit:

```bash
git -C $RW_ROOT/submodules/Genesis checkout 3aa206cd84729bc7cc14fb4007aeb95a0bead7aa
git -C $RW_ROOT/submodules/Genesis submodule update --init --recursive
```

## 3. Keep Runtime Paths Stable With Symlinks

Large files should live in cache, but runtime code still expects repo-relative paths.

```bash
cd $RW_ROOT

ln -sfn $RW_CACHE/ckpts ckpts
ln -sfn $RW_CACHE/wan_models wan_models
ln -sfn $RW_CACHE/sam3d_objects/checkpoints submodules/sam_3d_objects/checkpoints

ln -sf $RW_CACHE/sam2/checkpoints/sam2.1_hiera_tiny.pt submodules/sam2/checkpoints/sam2.1_hiera_tiny.pt
ln -sf $RW_CACHE/sam2/checkpoints/sam2.1_hiera_small.pt submodules/sam2/checkpoints/sam2.1_hiera_small.pt
ln -sf $RW_CACHE/sam2/checkpoints/sam2.1_hiera_base_plus.pt submodules/sam2/checkpoints/sam2.1_hiera_base_plus.pt
ln -sf $RW_CACHE/sam2/checkpoints/sam2.1_hiera_large.pt submodules/sam2/checkpoints/sam2.1_hiera_large.pt
```

## 4. Conda Environment In Cache

Do not use a named environment under `/home/ma-user/miniconda3/envs/...` for this project. Use a prefix environment in cache.

### 4.1 Create It

```bash
conda create -y -p $RW_ENV_PREFIX python=3.11
conda activate $RW_ENV_PREFIX
```

Optional: if you need a fresh rebuild and the prefix is already corrupted:

```bash
conda deactivate
rm -rf $RW_ENV_PREFIX
conda create -y -p $RW_ENV_PREFIX python=3.11
conda activate $RW_ENV_PREFIX
```

### 4.2 Bootstrap Core Packaging Tools

```bash
python -m pip install -U pip setuptools wheel ninja packaging
python -m pip install hatchling hatch-requirements-txt editables appdirs
```

### 4.3 Install Torch First

```bash
python -m pip install --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.ngc.nvidia.com
```

Validate:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Expected:

- `2.5.1+cu121`
- CUDA available

## 6. Optional: Prefer System CUDA For Extension Builds

If custom CUDA builds fail with:

- `fatbinary died due to signal 11`
- `nvcc` from the conda environment misbehaving

prefer the system CUDA toolchain for that install attempt:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
unset CUDACXX
unset CUDAHOSTCXX

which nvcc
which fatbinary
nvcc --version
```

Expected:

- `which nvcc` -> `/usr/local/cuda/bin/nvcc`
- `which fatbinary` -> `/usr/local/cuda/bin/fatbinary`

Do not enable this blindly for every package; use it when conda CUDA tools fail.

## 7. Install Order

Use this order. Do not jump around.

### 7.1 SAM 3D Objects Base

```bash
cd $RW_ROOT/submodules/sam_3d_objects
conda activate $RW_ENV_PREFIX
python -m pip install -v --no-build-isolation -e '.[dev]'
```

### 7.2 Flash Attention

Preferred: use a wheel, do not compile from source if you already have the matching wheel.

If the wheel is already in cache:

```bash
python -m pip install $RW_CACHE/wheels/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

If you need to download it manually first, place it under:

- `$RW_CACHE/wheels/`

Fallback direct URL:

```bash
python -m pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
```

### 7.3 PyTorch3D

There is no stable direct wheel path in this setup. Use the tested Git commit.

Default install attempt:

```bash
cd $RW_ROOT/submodules/sam_3d_objects
conda activate $RW_ENV_PREFIX
python -m pip install -v --no-build-isolation \
  "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
```

If `pip` fails while cloning into `/tmp/pip-install-*`, for example:

- `fatal: unable to checkout working tree`
- `Clone succeeded, but checkout failed`
- `git clone --filter=blob:none ... did not run successfully`

stop using the direct Git URL through `pip`. Clone once into cache and install from the local checkout instead:

```bash
mkdir -p $RW_CACHE/src
cd $RW_CACHE/src
rm -rf pytorch3d
git clone https://github.com/facebookresearch/pytorch3d.git pytorch3d
cd pytorch3d
git checkout 75ebeeaea0908c5527e7b1e305fbc7681382db47

conda activate $RW_ENV_PREFIX
python -m pip install -v --no-build-isolation .
```

If GitHub is unstable, keep the current-terminal Git overrides from Section 2 enabled before the clone.

If the local install fails with `fatbinary died due to signal 11`, retry the local install after enabling system CUDA as described in Section 6.

### 7.4 SAM 3D Objects Inference Extras

Do not trust a single `-e '.[inference]'` on the first try. Install the Git dependencies explicitly first.

#### 7.4.1 Install MoGe

```bash
mkdir -p $RW_CACHE/src
cd $RW_CACHE/src
git clone https://github.com/microsoft/MoGe.git MoGe
cd MoGe
git checkout a8c37341bc0325ca99b9d57981cc3bb2bd3e255b

conda activate $RW_ENV_PREFIX
python -m pip install -v --no-build-isolation .
```

If GitHub is unstable, keep the current-terminal Git overrides from Section 2 enabled.

#### 7.4.2 Install gsplat

Clone recursively because `glm` is required:

```bash
cd $RW_CACHE/src
git clone --recursive https://github.com/nerfstudio-project/gsplat.git gsplat
cd gsplat
git checkout 2323de5905d5e90e035f792fe65bad0fedd413e7
git submodule update --init --recursive
```

Install:

```bash
conda activate $RW_ENV_PREFIX
python -m pip install -v --no-build-isolation .
```

If it fails with `glm/...: No such file or directory`, the clone was not recursive enough. Re-run:

```bash
git submodule update --init --recursive
```

If it fails with `fatbinary died due to signal 11`, retry after enabling system CUDA from Section 6.

#### 7.4.3 Install remaining inference extras

```bash
cd $RW_ROOT/submodules/sam_3d_objects
conda activate $RW_ENV_PREFIX
python -m pip install -v --no-build-isolation kaolin==0.17.0 seaborn==0.13.2 gradio==5.49.0
python -m pip install -v --no-build-isolation -e '.[inference]' --no-deps
```

### 7.5 Patch Hydra

```bash
cd $RW_ROOT/submodules/sam_3d_objects
./patching/hydra
```

### 7.6 Install SAM2

```bash
cd $RW_ROOT/submodules/sam2
conda activate $RW_ENV_PREFIX
python -m pip install -v --no-build-isolation -e .
```

Validate:

```bash
python -m pip show SAM-2
python -c "import sam2; print('sam2 ok')"
```

### 7.7 Download SAM2 Checkpoints

Store them in cache:

```bash
cd $RW_CACHE/sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Fallbacks:

- download elsewhere and `rsync` into `$RW_CACHE/sam2/checkpoints`
- use `scp`

### 7.8 Install Genesis

```bash
cd $RW_ROOT/submodules/Genesis
conda activate $RW_ENV_PREFIX
git checkout 3aa206cd84729bc7cc14fb4007aeb95a0bead7aa
git submodule update --init --recursive
python -m pip install -v --no-build-isolation -e .
```

Validate:

```bash
python -m pip show genesis-world
python -c "import genesis as gs; print(gs.__version__)"
```

### 7.9 Install Root Requirements

```bash
cd $RW_ROOT
conda activate $RW_ENV_PREFIX
python -m pip install -r requirements.txt
```

If `nvidia-pyindex` fails, `appdirs` is usually missing. It is already preinstalled in Section 4.2. If needed, rerun:

```bash
python -m pip install appdirs
python -m pip install -r requirements.txt
```

For very large wheels such as `bpy` or `open3d`, prefer caching them under `$RW_CACHE/wheels` and reinstalling from there if network is too slow.

### 7.10 Download RealWonder And Wan Checkpoints

Install HF CLI if missing:

```bash
python -m pip install 'huggingface-hub[cli]<1.0'
```

RealWonder checkpoint:

```bash
HF_ENDPOINT="https://hf-mirror.com" \
hf download \
  ziyc/realwonder \
  --include "Realwonder-Distilled-AR-I2V-Flow/*" \
  --local-dir $RW_CACHE/ckpts/
```

Wan checkpoint:

```bash
HF_ENDPOINT="https://hf-mirror.com" \
hf download \
  alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP \
  --local-dir $RW_CACHE/wan_models/Wan2.1-Fun-V1.1-1.3B-InP
```

For `facebook/sam-3d-objects`, first verify the token can access the gated repo. If `hf-mirror.com` still returns `403`, retry the same command against `https://huggingface.co` for that command only.

## 8. Validation

Run these checks:

```bash
cd $RW_ROOT
conda activate $RW_ENV_PREFIX

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
LIDRA_SKIP_INIT=1 python -c "import sam3d_objects; import pytorch3d; import flash_attn; import kaolin; import gsplat; print('sam3d stack ok')"
python -c "import moge; print('moge ok')"
python -c "import sam2; print('sam2 ok')"
python -c "import genesis as gs; print(gs.__version__)"
python -c "import diffusers, open_clip, kornia; print('root deps ok')"
```

Optional demo dependencies:

```bash
python -m pip install -r demo_web/requirements.txt
```

## 9. Known Failure Modes And Exact Repairs

### 9.1 `hatchling.build` missing

```bash
python -m pip install hatchling hatch-requirements-txt
```

### 9.2 `editables` missing

```bash
python -m pip install editables
```

### 9.3 `nvidia-pyindex` wheel build fails

Typical cause:

- `ModuleNotFoundError: No module named 'appdirs'`

Repair:

```bash
python -m pip install appdirs
```

### 9.4 `pip install -e '.[p3d]'` finishes but `flash_attn` / `pytorch3d` are missing

Do not trust the umbrella extra. Install them explicitly as in Sections 7.2 and 7.3.

### 9.5 `pip` fails while cloning `pytorch3d` into `/tmp/pip-install-*`

Typical symptoms:

- `fatal: unable to checkout working tree`
- `Clone succeeded, but checkout failed`
- `git clone --filter=blob:none ... exit code: 128`

Use a persistent local checkout under `$RW_CACHE/src` instead of letting `pip` manage the clone in `/tmp`:

```bash
mkdir -p $RW_CACHE/src
cd $RW_CACHE/src
rm -rf pytorch3d
git clone https://github.com/facebookresearch/pytorch3d.git pytorch3d
cd pytorch3d
git checkout 75ebeeaea0908c5527e7b1e305fbc7681382db47

conda activate $RW_ENV_PREFIX
python -m pip install -v --no-build-isolation .
```

If GitHub is unstable, enable the current-terminal Git overrides from Section 2 first.

### 9.6 Git clone fails with HTTP/2 errors

Use current-terminal Git overrides:

```bash
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=url.https://githubfast.com/.insteadOf
export GIT_CONFIG_VALUE_0=https://github.com/
export GIT_CONFIG_KEY_1=http.version
export GIT_CONFIG_VALUE_1=HTTP/1.1
```

### 9.7 `glm/...: No such file or directory` while building `gsplat`

The repo was cloned without recursive submodules:

```bash
git submodule update --init --recursive
```

### 9.8 `fatbinary died due to signal 11`

This indicates a broken or unstable CUDA toolchain in the current environment. Retry the build with system CUDA from Section 6.

### 9.9 `sam3d_objects.init` missing

This repo snapshot imports `sam3d_objects.init` from `sam3d_objects/__init__.py`, but that file does not exist. Use this for validation:

```bash
LIDRA_SKIP_INIT=1 python -c "import sam3d_objects; print('ok')"
```

### 9.10 `open-clip-torch` requires `timm>=1.0.17`, but `sam3d_objects` installs `timm==0.9.16`

Record the conflict and validate runtime before changing `timm`. Do not blindly upgrade it during base setup.

## 10. Rollback And Cleanup

### 10.1 Remove the cache-based conda environment completely

```bash
conda deactivate
rm -rf $RW_ENV_PREFIX
```

### 10.2 Purge shared caches for this project

```bash
rm -rf $RW_CACHE/torch_extensions
rm -rf $RW_CACHE/triton
rm -rf $RW_CACHE/warp
rm -rf $RW_CACHE/tmp
rm -rf $RW_CACHE/src/*
```

Use with care:

```bash
rm -rf $RW_CACHE/wheels/*
rm -rf $RW_CACHE/hf
rm -rf $RW_CACHE/torch
```

### 10.3 Clean pip and conda package caches

```bash
python -m pip cache purge
conda clean --all -y
```

## 11. Disk Quota And Workspace Pressure

If you see:

- `Disk quota exceeded`
- `project block limit reached`

do not keep installing. Diagnose space first:

```bash
cd $RW_ROOT
du -sh .
du -sh .git
du -sh .git/modules/submodules/* 2>/dev/null
df -h $RW_ROOT $RW_CACHE
```

Known large offenders:

- `.git/modules/submodules/Genesis`
- workspace-local result directories
- accidental wheels or source clones under `~/workspace`

Prefer:

- source clones under `$RW_CACHE/src`
- wheels under `$RW_CACHE/wheels`
- conda environment under `$RW_ENV_PREFIX`

Never rebuild the environment under `/home/ma-user/miniconda3/envs/realwonder` for this project if workspace pressure is already an issue.
