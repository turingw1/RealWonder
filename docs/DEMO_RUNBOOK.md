# RealWonder Demo Runbook

This document explains how to run a RealWonder demo after the environment has already been installed.

It is written for the server layout used in deployment:

- Workspace: `~/workspace/Zhengwei/RealWonder`
- Cache root: `/cache/Zhengwei/RealWonder`
- Conda environment: `realwonder`

## 1. Goal And Recommended Order

Do not start with the web UI.

Use this order:

1. check environment and checkpoints
2. run offline physics simulation on the `lamp` case
3. run offline video generation from the simulation output
4. run the web demo only after the offline path works

This gives the shortest debug loop and separates:

- simulation errors
- model loading errors
- checkpoint path errors
- frontend errors

## 2. Preflight Check

Activate the environment:

```bash
cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder
```

Check GPU visibility:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

Check installed packages:

```bash
bash scripts/check_realwonder_env.sh
```

Expected:

- `torch==2.5.1+cu121`
- `sam3d_objects`, `flash_attn`, `pytorch3d`, `kaolin`, `gsplat`, `sam2`, `genesis-world` available

## 3. Checkpoint Layout

The runtime expects these large files to exist, but they should physically live under `/cache/Zhengwei/RealWonder`.

Check these locations:

```bash
find /cache/Zhengwei/RealWonder/sam3d_objects/checkpoints -maxdepth 3 | sed -n '1,120p'
find /cache/Zhengwei/RealWonder/sam2/checkpoints -maxdepth 2 | sed -n '1,120p'
find /cache/Zhengwei/RealWonder/ckpts -maxdepth 5 | sed -n '1,120p'
find /cache/Zhengwei/RealWonder/wan_models -maxdepth 4 | sed -n '1,120p'
```

Recommended links inside the workspace:

```bash
cd ~/workspace/Zhengwei/RealWonder

ln -sfn /cache/Zhengwei/RealWonder/ckpts ckpts
ln -sfn /cache/Zhengwei/RealWonder/wan_models wan_models
ln -sfn /cache/Zhengwei/RealWonder/sam3d_objects/checkpoints submodules/sam_3d_objects/checkpoints

ln -sf /cache/Zhengwei/RealWonder/sam2/checkpoints/sam2.1_hiera_tiny.pt submodules/sam2/checkpoints/sam2.1_hiera_tiny.pt
ln -sf /cache/Zhengwei/RealWonder/sam2/checkpoints/sam2.1_hiera_small.pt submodules/sam2/checkpoints/sam2.1_hiera_small.pt
ln -sf /cache/Zhengwei/RealWonder/sam2/checkpoints/sam2.1_hiera_base_plus.pt submodules/sam2/checkpoints/sam2.1_hiera_base_plus.pt
ln -sf /cache/Zhengwei/RealWonder/sam2/checkpoints/sam2.1_hiera_large.pt submodules/sam2/checkpoints/sam2.1_hiera_large.pt
```

Minimum files to verify before a demo run:

- `ckpts/Realwonder-Distilled-AR-I2V-Flow/.../step=000800.pt`
- `wan_models/Wan2.1-Fun-V1.1-1.3B-InP/...`
- `submodules/sam2/checkpoints/sam2.1_hiera_large.pt` or another required SAM2 checkpoint
- `submodules/sam_3d_objects/checkpoints/...`

## 4. Offline Demo: Shortest Working Path

Use the `lamp` case first because it is already referenced in the project README and in `demo_web/demo_data/lamp`.

### 4.1 Run physics simulation

```bash
cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder
python case_simulation.py --config_path demo_web/demo_data/lamp/config.yaml
```

Expected output location:

- `result/lamp/final_sim`

Quick check:

```bash
find result/lamp -maxdepth 3 | sed -n '1,120p'
```

### 4.2 Run video generation from simulation

```bash
cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder
python infer_sim.py \
  --checkpoint_path ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt \
  --sim_data_path result/lamp/final_sim \
  --output_path result/lamp/final_sim/final.mp4
```

Expected result:

- `result/lamp/final_sim/final.mp4`

Verify:

```bash
ls -lh result/lamp/final_sim/final.mp4
```

If the offline path works, then the core stack is usable.

## 5. Web Demo

Install demo-only dependencies if not already installed:

```bash
cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder
python -m pip install -r demo_web/requirements.txt
```

Start the web app:

```bash
cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder
python demo_web/app.py \
  --demo_data demo_web/demo_data/lamp \
  --checkpoint_path "/cache/Zhengwei/RealWonder/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt"
```

Notes:

- always launch the app from the repository root
- use `--demo_data demo_web/demo_data/lamp` when running from the repository root
- keep the checkpoint path quoted to avoid shell line-wrap mistakes on long paths
- use the absolute checkpoint path from `/cache/Zhengwei/RealWonder/ckpts/...`
- if running on a remote server, expose the corresponding port through your SSH tunnel or server policy

## 6. Suggested Validation Commands

Before blaming the app, check imports first:

```bash
cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
LIDRA_SKIP_INIT=1 python -c "import sam3d_objects; import pytorch3d; import flash_attn; import kaolin; import gsplat; print('sam3d stack ok')"
python -c "import sam2; print('sam2 ok')"
python -c "import genesis as gs; print(gs.__version__)"
python -c "import diffusers, open_clip, kornia; print('root deps ok')"
```

## 7. Common Failure Points

### 7.1 Checkpoint path is wrong

Typical symptom:

- `FileNotFoundError`
- `torch.load` cannot find checkpoint

Action:

- use absolute cache paths
- check the symlink targets
- confirm the expected file exists with `ls -lh`

### 7.2 `sam3d_objects` import fails because `sam3d_objects.init` is missing

For lightweight validation use:

```bash
LIDRA_SKIP_INIT=1 python -c "import sam3d_objects; print('ok')"
```

This is enough for environment sanity checks.

### 7.3 GPU OOM during demo run

Action:

- make sure no old Python job is still using GPU memory
- check `nvidia-smi`
- if needed, kill stale processes before rerunning

### 7.4 Web demo starts but model loading hangs

Action:

- verify the `--checkpoint_path`
- verify `wan_models` exists and is linked to `/cache/Zhengwei/RealWonder/wan_models`
- verify `sam2` checkpoints exist

### 7.5 Hugging Face gated download still fails

The environment may be correct while the checkpoint is still unavailable.

For `facebook/sam-3d-objects`:

- access must be manually approved on Hugging Face
- a mirror cannot bypass gated access
- use your personal token in the current terminal only

## 8. Recommended Run Sequence For A Fresh Server Session

```bash
cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder

bash scripts/check_realwonder_env.sh
nvidia-smi

find /cache/Zhengwei/RealWonder/ckpts -maxdepth 5 | sed -n '1,80p'
find /cache/Zhengwei/RealWonder/wan_models -maxdepth 4 | sed -n '1,80p'
find /cache/Zhengwei/RealWonder/sam2/checkpoints -maxdepth 2 | sed -n '1,80p'
find /cache/Zhengwei/RealWonder/sam3d_objects/checkpoints -maxdepth 3 | sed -n '1,80p'

python case_simulation.py --config_path demo_web/demo_data/lamp/config.yaml

python infer_sim.py \
  --checkpoint_path ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt \
  --sim_data_path result/lamp/final_sim \
  --output_path result/lamp/final_sim/final.mp4

python demo_web/app.py \
  --demo_data demo_web/demo_data/lamp \
  --checkpoint_path "/cache/Zhengwei/RealWonder/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt"
```
