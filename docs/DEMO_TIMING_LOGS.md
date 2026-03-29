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

Each startup creates one run directory under that path, for example:

```text
demo_web/demo_data/lamp/experiment_logs/lamp_20260330_153000_a1b2c3/
```

This keeps logs from different runs grouped together cleanly.

## Log Files

Inside each run directory, the demo writes:

- `bootstrap.events.jsonl`
- `bootstrap.summary.json`
- `startup.events.jsonl`
- `startup.summary.json`
- `generation.events.jsonl`
- `generation.summary.json`

Not every file appears immediately:

- startup creates `bootstrap.*` and `startup.*`
- actual interaction/generation creates `generation.*`

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
- `timestamp`
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
ls demo_web/demo_data/lamp/experiment_logs/*/
tail -n 20 demo_web/demo_data/lamp/experiment_logs/*/*.events.jsonl
cat demo_web/demo_data/lamp/experiment_logs/*/*.summary.json
```

## Visualize The Pipeline

Use the standalone plotting tool to draw a pipeline-style timeline from one interactive demo run.

If you pass the `experiment_logs` directory, the tool automatically picks the latest run subdirectory.

```bash
cd ~/workspace/Zhengwei/RealWonder
python scripts/plot_interactive_demo_timing.py demo_web/demo_data/lamp/experiment_logs
```

This writes:

```text
demo_web/demo_data/lamp/experiment_logs/<latest_run>/pipeline_timing.png
```

You can also target a specific run directory and choose the output path:

```bash
python scripts/plot_interactive_demo_timing.py \
  demo_web/demo_data/lamp/experiment_logs/lamp_20260330_153000_a1b2c3 \
  --output demo_web/demo_data/lamp/experiment_logs/lamp_20260330_153000_a1b2c3/pipeline_timing.png
```

The generated figure is designed to match the article's pipeline logic:

- startup and warmup are shown as early sequential phases
- Stage 1 (`sim`) shows physics, render+flow, and resize time
- Stage 2 (`warp`) shows queue wait and noise warp time
- Stage 3 (`diffusion`) shows queue wait, VAE encode, mask build, and diffusion time
- queue wait is rendered as a gray hatched segment so block overlap is easy to see
