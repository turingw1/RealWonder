# RealWonder A100 Server Setup And Git Migration Guide

This guide is for deploying RealWonder on an A100 server under:

- Workspace: `~/workspace/Zhengwei/RealWonder/`
- Large-file cache root: `/cache/Zhengwei/RealWonder/`

It assumes:

- The server already has a Conda environment named `realwonder`
- The server may access GitHub slowly or intermittently
- Large files must be stored under `/cache/Zhengwei`
- You want to manage the code in your own GitHub repository instead of the upstream repository

## 1. Convert The Current Directory Into Your Own Git Repo

The current local repository still points to upstream:

- upstream repo: `https://github.com/liuwei283/RealWonder.git`

Before pushing to your own GitHub, keep the current upstream as `upstream` and add your own repository as `origin`.

### 1.1 Create your empty GitHub repository

Create an empty repository on GitHub first, for example:

- `git@github.com:<your-user>/RealWonder.git`

### 1.2 Repoint remotes locally

Run locally in this repository:

```bash
cd /home/gzwlinux/vscode/gitProject/RealWonder
git remote rename origin upstream
git remote add origin git@github.com:<your-user>/RealWonder.git
git remote -v
```

Expected result:

- `origin` points to your GitHub repo
- `upstream` points to `liuwei283/RealWonder`

### 1.3 Make sure submodules are tracked correctly

This repository currently has real directories under `submodules/`, but the root repository may not yet have recorded gitlinks for them. Before your first push, verify that the submodule paths are tracked as mode `160000`.

Run:

```bash
git add .gitmodules \
  submodules/sam_3d_objects \
  submodules/sam2 \
  submodules/Genesis \
  submodules/flux_controlnet_inpainting

git ls-files --stage submodules/sam_3d_objects
git ls-files --stage submodules/sam2
git ls-files --stage submodules/Genesis
git ls-files --stage submodules/flux_controlnet_inpainting
```

If the first column is `160000`, they are tracked as submodules correctly.

Then commit and push:

```bash
git commit -m "Track RealWonder submodules and add A100 deployment docs"
git push -u origin main
```

If `git add` reports embedded-repository issues instead of recording gitlinks, re-register them explicitly:

```bash
git submodule add -f https://github.com/facebookresearch/sam-3d-objects.git submodules/sam_3d_objects
git submodule add -f https://github.com/facebookresearch/sam2.git submodules/sam2
git submodule add -f https://github.com/Genesis-Embodied-AI/Genesis.git submodules/Genesis
git submodule add -f https://github.com/alimama-creative/FLUX-Controlnet-Inpainting.git submodules/flux_controlnet_inpainting
```

Then repeat `git add`, `git commit`, and `git push`.

## 2. Clone Your Repo On The Server

Target path:

```bash
mkdir -p ~/workspace/Zhengwei
cd ~/workspace/Zhengwei
```

Preferred:

```bash
git clone --recursive git@github.com:<your-user>/RealWonder.git
cd RealWonder
```

If direct GitHub is slow, try a `githubfast` mirror. The exact mirror format can vary by machine, but the common usable form is:

```bash
git clone --recursive https://githubfast.com/https://github.com/<your-user>/RealWonder.git
```

If the mirror also fails, use one of these fallbacks:

- clone locally and `rsync` the whole repository to the server
- create a tarball locally and upload it
- upload only the repo plus submodules, then run `git remote set-url origin ...` on the server

### 2.1 Sync submodules after clone

Even after `--recursive`, run this once:

```bash
git submodule update --init --recursive
```

Pin Genesis to the tested commit:

```bash
git -C submodules/Genesis checkout 3aa206cd84729bc7cc14fb4007aeb95a0bead7aa
git -C submodules/Genesis submodule update --init --recursive
```

## 3. Cache Layout Under `/cache/Zhengwei`

Do not store large checkpoints or build caches in the workspace.

Create this layout:

```bash
mkdir -p /cache/Zhengwei/RealWonder/{hf,torch,torch_extensions,triton,warp,logs,tmp}
mkdir -p /cache/Zhengwei/RealWonder/{ckpts,wan_models}
mkdir -p /cache/Zhengwei/RealWonder/sam3d_objects/checkpoints
mkdir -p /cache/Zhengwei/RealWonder/sam2/checkpoints
```

Recommended environment variables:

```bash
export HF_HOME=/cache/Zhengwei/RealWonder/hf
export HUGGINGFACE_HUB_CACHE=/cache/Zhengwei/RealWonder/hf/hub
export TORCH_HOME=/cache/Zhengwei/RealWonder/torch
export TORCH_EXTENSIONS_DIR=/cache/Zhengwei/RealWonder/torch_extensions
export TRITON_CACHE_DIR=/cache/Zhengwei/RealWonder/triton
export WARP_CACHE_DIR=/cache/Zhengwei/RealWonder/warp
export XDG_CACHE_HOME=/cache/Zhengwei/RealWonder/tmp
```

For Hugging Face on shared servers, do not rely blindly on a globally exported group token. Prefer overriding credentials only in the current terminal or on a single command:

```bash
export HF_TOKEN="<your-personal-hf-token>"
export HUGGINGFACE_TOKEN="$HF_TOKEN"
```

Mirror usage policy:

- public repos: prefer `https://hf-mirror.com`
- gated repos: test mirror first, but fall back to `https://huggingface.co` if mirror returns `403`
- never change system-wide shell startup files just to switch tokens or endpoints

To keep runtime paths unchanged, use symlinks:

```bash
cd ~/workspace/Zhengwei/RealWonder

rm -rf ckpts
ln -s /cache/Zhengwei/RealWonder/ckpts ckpts

rm -rf wan_models
ln -s /cache/Zhengwei/RealWonder/wan_models wan_models

rm -rf submodules/sam_3d_objects/checkpoints
ln -s /cache/Zhengwei/RealWonder/sam3d_objects/checkpoints submodules/sam_3d_objects/checkpoints
```

For SAM2, keep the script directory itself but place checkpoint files in cache and link them back:

```bash
ln -sf /cache/Zhengwei/RealWonder/sam2/checkpoints/sam2.1_hiera_tiny.pt submodules/sam2/checkpoints/sam2.1_hiera_tiny.pt
ln -sf /cache/Zhengwei/RealWonder/sam2/checkpoints/sam2.1_hiera_small.pt submodules/sam2/checkpoints/sam2.1_hiera_small.pt
ln -sf /cache/Zhengwei/RealWonder/sam2/checkpoints/sam2.1_hiera_base_plus.pt submodules/sam2/checkpoints/sam2.1_hiera_base_plus.pt
ln -sf /cache/Zhengwei/RealWonder/sam2/checkpoints/sam2.1_hiera_large.pt submodules/sam2/checkpoints/sam2.1_hiera_large.pt
```

## 4. Check The Existing `realwonder` Environment Before Reinstalling

Do not recreate the environment first. Repair it only where needed.

Activate:

```bash
conda activate realwonder
```

Run the repo check script:

```bash
bash scripts/check_realwonder_env.sh
```

Minimum expected state:

- Python `3.11.x`
- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `torchaudio==2.5.1+cu121`
- `torch.cuda.is_available() == True`

If the environment is not close to that state, repair it before installing project packages:

```bash
python -m pip install --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.ngc.nvidia.com
```

## 5. A100-Specific Build Settings

For A100:

- GPU architecture: `sm_80`
- recommended `TORCH_CUDA_ARCH_LIST=8.0`

Recommended build settings:

```bash
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
export NINJA_NUM_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
```

If the server still shows high memory pressure during compile, reduce all three values from `4` to `2` or `1`.

## 6. Install Order On The Server

The order below is intentionally different from a naive `README` run. It is based on actual failure cases observed during setup:

- editable backend missing: `hatchling`, `hatch-requirements-txt`, `editables`
- `.[p3d]` can appear to finish while `pytorch3d` and `flash-attn` are still missing
- local CUDA extension builds can saturate CPU and memory

### 6.1 Install build helpers first

```bash
conda activate realwonder
python -m pip install -U pip setuptools wheel ninja packaging
python -m pip install hatchling hatch-requirements-txt editables
```

### 6.2 Install SAM 3D Objects

```bash
cd ~/workspace/Zhengwei/RealWonder/submodules/sam_3d_objects
conda activate realwonder
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
export NINJA_NUM_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
```

Install `dev` first:

```bash
python -m pip install -v --no-build-isolation -e '.[dev]'
```

Install the two `p3d` dependencies explicitly instead of trusting a single editable extra:

```bash
python -m pip install -v --no-build-isolation flash_attn==2.8.3
python -m pip install -v --no-build-isolation \
  "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
```

Then install `inference`:

```bash
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
python -m pip install -v --no-build-isolation -e '.[inference]'
```

Patch hydra:

```bash
./patching/hydra
```

Validate Step 2:

```bash
python -m pip show sam3d_objects flash_attn pytorch3d kaolin gsplat
LIDRA_SKIP_INIT=1 python -c "import sam3d_objects; import flash_attn; import pytorch3d; import kaolin; import gsplat; print('step2 ok')"
```

Notes:

- `sam3d_objects` currently imports `sam3d_objects.init` in `__init__.py`, but that file is absent in this snapshot. Use `LIDRA_SKIP_INIT=1` for lightweight validation.
- You may still see a `timm` conflict with `open_clip_torch`. Keep it noted and validate runtime later.

### 6.3 Download SAM 3D Objects checkpoints into cache

Install HF CLI if missing:

```bash
python -m pip install 'huggingface-hub[cli]<1.0'
```

Preferred:

```bash
cd ~/workspace/Zhengwei/RealWonder/submodules/sam_3d_objects
HF_TOKEN="<your-personal-hf-token>" \
HUGGINGFACE_TOKEN="$HF_TOKEN" \
HF_ENDPOINT="https://hf-mirror.com" \
hf auth whoami
```

If `whoami` succeeds and your personal account can access `facebook/sam-3d-objects`, download with the same per-command override:

```bash
cd ~/workspace/Zhengwei/RealWonder/submodules/sam_3d_objects
HF_TOKEN="<your-personal-hf-token>" \
HUGGINGFACE_TOKEN="$HF_TOKEN" \
HF_ENDPOINT="https://hf-mirror.com" \
hf download \
  --repo-type model \
  --local-dir /cache/Zhengwei/RealWonder/sam3d_objects/checkpoints/hf-download \
  --max-workers 1 \
  facebook/sam-3d-objects

mv /cache/Zhengwei/RealWonder/sam3d_objects/checkpoints/hf-download/checkpoints/* \
   /cache/Zhengwei/RealWonder/sam3d_objects/checkpoints/
rm -rf /cache/Zhengwei/RealWonder/sam3d_objects/checkpoints/hf-download
```

Fallback if HF is slow:

- download on a machine with better network, then `rsync` to `/cache/Zhengwei/RealWonder/sam3d_objects/checkpoints/`
- or download from browser and upload manually

If mirror download returns `403` for this gated repo even though your token can access the model, retry the same command against the official endpoint without changing global environment variables:

```bash
cd ~/workspace/Zhengwei/RealWonder/submodules/sam_3d_objects
HF_TOKEN="<your-personal-hf-token>" \
HUGGINGFACE_TOKEN="$HF_TOKEN" \
HF_ENDPOINT="https://huggingface.co" \
hf download \
  --repo-type model \
  --local-dir /cache/Zhengwei/RealWonder/sam3d_objects/checkpoints/hf-download \
  --max-workers 1 \
  facebook/sam-3d-objects
```

### 6.4 Install SAM2

```bash
cd ~/workspace/Zhengwei/RealWonder/submodules/sam2
conda activate realwonder
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
export NINJA_NUM_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
python -m pip install -v --no-build-isolation -e .
```

The package allows CUDA extension build errors and may still install successfully. Confirm first:

```bash
python -m pip show SAM-2
python -c "import sam2; print('sam2 ok')"
```

Download checkpoints into cache instead of the workspace:

```bash
cd /cache/Zhengwei/RealWonder/sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

If `wget` is too slow, fallback options are:

- use a `githubfast`-style transfer node if you have one
- download locally and `rsync`
- use `scp` from a machine with better bandwidth

### 6.5 Install Genesis

```bash
cd ~/workspace/Zhengwei/RealWonder/submodules/Genesis
conda activate realwonder
git checkout 3aa206cd84729bc7cc14fb4007aeb95a0bead7aa
git submodule update --init --recursive
python -m pip install -v --no-build-isolation -e .
```

Validate:

```bash
python -m pip show genesis-world
python -c "import genesis as gs; print(gs.__version__)"
```

### 6.6 Install root requirements

```bash
cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder
python -m pip install -r requirements.txt
```

### 6.7 Download RealWonder and Wan checkpoints into cache

```bash
cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder
python -m pip install 'huggingface-hub[cli]<1.0'
```

RealWonder checkpoint:

```bash
HF_TOKEN="<your-personal-hf-token>" \
HUGGINGFACE_TOKEN="$HF_TOKEN" \
HF_ENDPOINT="https://hf-mirror.com" \
hf download \
  ziyc/realwonder \
  --include "Realwonder-Distilled-AR-I2V-Flow/*" \
  --local-dir /cache/Zhengwei/RealWonder/ckpts/
```

Wan checkpoint:

```bash
HF_TOKEN="<your-personal-hf-token>" \
HUGGINGFACE_TOKEN="$HF_TOKEN" \
HF_ENDPOINT="https://hf-mirror.com" \
hf download \
  alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP \
  --local-dir /cache/Zhengwei/RealWonder/wan_models/Wan2.1-Fun-V1.1-1.3B-InP
```

If HF download is too slow:

- download locally and `rsync` to the same cache path
- avoid storing these models under the git workspace

## 7. Final Validation

Run these checks from `~/workspace/Zhengwei/RealWonder`:

```bash
conda activate realwonder

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
LIDRA_SKIP_INIT=1 python -c "import sam3d_objects; import pytorch3d; import flash_attn; import kaolin; import gsplat; print('sam3d stack ok')"
python -c "import sam2; print('sam2 ok')"
python -c "import genesis as gs; print(gs.__version__)"
python -c "import diffusers, open_clip, kornia; print('root deps ok')"
```

Optional demo dependencies:

```bash
python -m pip install -r demo_web/requirements.txt
```

## 8. Known Issues And Repairs

### 8.1 `Cannot import 'hatchling.build'`

Install:

```bash
python -m pip install hatchling hatch-requirements-txt
```

### 8.2 `No module named 'editables'`

Install:

```bash
python -m pip install editables
```

### 8.3 `pip install -e '.[p3d]'` completes but `pytorch3d` and `flash-attn` are still missing

Do not trust the single command. Install these two explicitly:

```bash
python -m pip install -v --no-build-isolation flash_attn==2.8.3
python -m pip install -v --no-build-isolation \
  "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
```

### 8.4 Build saturates server CPU or RAM

Reduce:

```bash
export MAX_JOBS=1
export NINJA_NUM_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
```

### 8.5 `open-clip-torch` requires `timm>=1.0.17`, but `sam3d_objects` installs `timm==0.9.16`

This conflict was observed during setup. Keep it recorded and validate runtime paths that use `open_clip_torch`. Do not blindly upgrade `timm` without retesting `sam3d_objects`.

### 8.6 `sam3d_objects.init` missing

The current code snapshot imports `sam3d_objects.init` from `sam3d_objects/__init__.py`, but the file is absent. For environment validation:

```bash
LIDRA_SKIP_INIT=1 python -c "import sam3d_objects; print('ok')"
```

If full runtime later requires that module, patch the package or pin a corrected upstream snapshot.

### 8.7 Gated Hugging Face repo returns `403` from `hf-mirror.com`

First verify that the personal token in the current terminal can really access the gated model:

```bash
HF_TOKEN="<your-personal-hf-token>" \
HUGGINGFACE_TOKEN="$HF_TOKEN" \
HF_ENDPOINT="https://hf-mirror.com" \
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
print(api.whoami())
print(api.model_info("facebook/sam-3d-objects").id)
PY
```

If model metadata is accessible but `hf download` still returns `403`, retry the exact download with the official endpoint for that command only:

```bash
HF_TOKEN="<your-personal-hf-token>" \
HUGGINGFACE_TOKEN="$HF_TOKEN" \
HF_ENDPOINT="https://huggingface.co" \
hf download --repo-type model facebook/sam-3d-objects
```

Do not rewrite global server environment variables just to switch endpoints. Keep the override local to the active terminal or to a single command line.
