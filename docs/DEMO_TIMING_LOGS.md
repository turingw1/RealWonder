# Demo Timing Logs

This document describes the structured timing logs generated for the server-side interactive demo.

## Scope

The timing instrumentation is intentionally limited to:

- `demo_web/app.py`
- `demo_web/experiment_logging.py`

It does not modify:

- `demo_web/vidgen/*`
- `demo_web/simulation/*`
- core model code
- simulation kernels

This keeps the demo behavior unchanged while still capturing timing data.

## Output Directory

Logs are written under:

```text
demo_web/demo_data/<case>/experiment_logs/
```

For example:

```text
demo_web/demo_data/lamp/experiment_logs/
```

## Log Files

Each run writes:

- `*.events.jsonl`
- `*.summary.json`

## Experiment Types

### 1. `interactive_demo_bootstrap`

Captures startup phases before warmup:

- `demo.startup.initialize_video_generator`
- `demo.startup.initialize_simulator`
- `demo.startup.initialize_case_handler`
- `demo.startup.initialize_noise_warper`
- `demo.startup.precompute_first_frame`

### 2. `interactive_demo_startup`

Captures warmup phases:

- `demo.startup_warmup.sim_render`
- `demo.startup_warmup.noise_warp`
- `demo.startup_warmup.vae_diffusion`

### 3. `interactive_demo_generation`

Captures one interactive generation request:

- `demo.stage1_render_flow_block`
- `demo.stage2_noise_warp_block`
- `demo.stage3_diffusion_block`

Each block event includes breakdown fields where available.

## Key Fields

All events contain:

- `run_id`
- `experiment_name`
- `run_name`
- `timestamp_utc`
- `stage`
- `duration_sec`

Common additional fields:

- `block_idx`
- `queue_wait_sec`
- `physics_step_total_sec`
- `render_flow_total_sec`
- `resize_total_sec`
- `warp_steps_sec`
- `get_block_noise_sec`
- `vae_encode_sec`
- `mask_build_sec`
- `diffusion_sec`

## Recommended Analysis

The first useful comparisons are:

1. Startup cost vs steady-state generation cost
2. Stage 1 / Stage 2 / Stage 3 block time proportions
3. Per-block diffusion variance across requests
4. Whether simulation or diffusion is the current bottleneck

## Recommended Server Workflow

Run the demo normally:

```bash
cd ~/workspace/Zhengwei/RealWonder
python demo_web/app.py \
  --demo_data demo_web/demo_data/lamp \
  --checkpoint_path /cache/Zhengwei/RealWonder/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt
```

Then inspect:

```bash
ls demo_web/demo_data/lamp/experiment_logs
tail -n 20 demo_web/demo_data/lamp/experiment_logs/*.events.jsonl
cat demo_web/demo_data/lamp/experiment_logs/*.summary.json
```
