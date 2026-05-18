# RealWonder Data Export on the Current Server

This server has Blackwell GPUs (`sm_120`). The current lightweight
RealWonder environment imports successfully, but its Torch build is
`cu121` and does not include `sm_120` kernels. Use the data-export modes
below until the environment is upgraded to a Blackwell-compatible Torch
stack such as `cu128`.

## Network Rule

Use the server default network for datasets, model weights, pip/conda
packages, and other large downloads. Use the `18080` proxy only for small
external code/file operations when the default route fails or is too slow.

## CUDA/cuDNN Judgement

The server dependency note is consistent with the current failure mode:
`nvidia-smi` reports the maximum CUDA version supported by the driver, not
the CUDA runtime embedded in the active framework. For normal PyTorch /
TorchVision usage, do not install standalone CUDA/cuDNN first. Pick a
framework build that already carries a CUDA runtime and kernels compatible
with the GPU.

On this server the important mismatch is:

```text
GPU:   Blackwell, sm_120
Torch: 2.5.1+cu121 in the current realwonder env
Issue: Torch CUDA kernels are not built for sm_120
```

Installing system CUDA/cuDNN alone will not make the existing `cu121`
PyTorch wheel run CUDA kernels on Blackwell. The target GPU environment is
therefore the `simulation` conda env with a Blackwell-compatible PyTorch
stack, for example a `cu128` wheel. The helper script
`scripts/setup_simulation_env.sh` intentionally unsets proxy variables
before large conda/pip downloads so those transfers use the server default
network.

## Runnable Video-In Data Demo

Before the full Genesis/SAM/RAFT stack is available on Blackwell, use this
demo to validate the downstream data contract from an ordinary video. It
does not require Torch CUDA, Genesis rendering, RAFT weights, SAM2, SAM3D,
or Helios.

```bash
cd /root/autodl-tmp/Physics_worldmodel/RealWonder
source scripts/activate_realwonder.sh

python scripts/export_genesis_style_video_demo.py \
  --video /path/to/input.mp4 \
  --output_dir /root/autodl-tmp/Physics_worldmodel/RealWonder/data_demos/genesis_style_video_demo \
  --height 480 \
  --width 832 \
  --max_frames 81 \
  --stride 1 \
  --fps 12 \
  --prompt "optional prompt text" \
  --flow_preview
```

Outputs:

- `frames/`: resized input RGB frames.
- `coarse_rgb_frames/` and `coarse_rgb.mp4`: blurred/downsampled coarse RGB
  placeholder, matching the role of a later Genesis coarse preview.
- `first_frame.png`: first conditioning frame.
- `flow_fwd.npy` and `genesis_style_flow.npy`: `[T-1, 2, H, W]`, float32,
  where channel 0/1 are `dx/dy`.
- `motion_masks.npy`: `[T, H, W]`, boolean motion masks from flow magnitude.
- `gt_video.mp4`, `prompt.txt`, `metadata.json`.

This is not a physics simulation yet; it is the runnable schema bridge.
Once the Blackwell-compatible Torch/Genesis environment is ready, the real
Genesis exporter can replace the placeholder `coarse_rgb` and Farneback
flow while keeping the same output layout.

## Genesis Simulation Export

This path runs the RealWonder image-to-simulation pipeline and writes the
intermediate streams used by later training code.

```bash
cd /root/autodl-tmp/Physics_worldmodel/RealWonder
source scripts/activate_realwonder.sh

python case_simulation.py \
  --config_path cases/lamp/config.yaml \
  --device auto \
  --skip_noise_warp \
  --save_raw_frames
```

`--device auto` checks whether Torch can actually launch CUDA kernels. On
the current `cu121` build it falls back to CPU instead of failing on
Blackwell. `--skip_noise_warp` avoids RAFT/noise generation and exports
Genesis data only.

This mode still needs the RealWonder reconstruction checkpoints. On this
machine, the SAM2 and SAM3D checkpoint directories are currently empty
except for download scripts/placeholders, so a first full run will need
those model weights through the server default network.

Main outputs under `result/<case>/<timestamp>/final_sim/`:

- `frames/`: cropped 480x832 simulation preview frames.
- `raw_frames_512/`: uncropped Genesis render frames, when
  `--save_raw_frames` is set.
- `genesis_flows_512.npy`: renderer optical flow, `[T-1, 2, 512, 512]`,
  float32.
- `genesis_flows_480x832.npy`: resized/cropped Genesis flow aligned with
  `frames/`, `[T-1, 2, 480, 832]`, float32.
- `points_masks_downsampled.pt` and `mesh_masks_downsampled.pt`: masks at
  latent-ish resolution for later conditioning.
- `simulation.mp4`, `resized_input_image.png`, `prompt.txt`,
  `metadata.json`.

To also build `noises.npy`, use one of:

```bash
# Use RealWonder renderer flow as the motion field.
python case_simulation.py --config_path cases/lamp/config.yaml --noise_flow_source genesis

# Recompute RAFT flow on rendered RGB frames.
python case_simulation.py --config_path cases/lamp/config.yaml --noise_flow_source raft
```

These modes are much more likely to need a Blackwell-compatible Torch CUDA
environment. CPU fallback is available but may be slow.

## RAFT Flow Export From Ordinary Video Or Images

Use this for OpenVid/custom clips or quick image-pair checks, independent
of Genesis:

```bash
cd /root/autodl-tmp/Physics_worldmodel/RealWonder
source scripts/activate_realwonder.sh

python scripts/export_video_raft_flows.py \
  --video /path/to/input.mp4 \
  --output_dir /root/autodl-tmp/Physics_worldmodel/data/raft/example \
  --device auto \
  --raft_version small \
  --resize 480 832 \
  --save_frames
```

For two images:

```bash
python scripts/export_video_raft_flows.py \
  --image_pair frame0.png frame1.png \
  --output_dir /root/autodl-tmp/Physics_worldmodel/data/raft/pair_test \
  --device auto
```

Outputs:

- `raft_flows.npy`: `[T-1, 2, H, W]`, float32.
- `frames/`: optional frame dump when `--save_frames` is set.
- `metadata.json`: source, resize, device, RAFT variant, and shapes.

## Practical Interpretation

Use Genesis export for physics-derived `F` and coarse RGB previews. Use
RAFT export for real/custom videos when constructing broader video-flow
training pairs. The two streams can be kept separate at first:

```text
image + Genesis action -> Genesis frames + Genesis flow + masks
real video / image pair -> RAFT flow
```

Later training code can decide whether to use `F-only`, `F + coarse RGB`,
or flow-warped noise.
