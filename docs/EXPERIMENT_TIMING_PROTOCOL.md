# Experiment Timing Protocol

This document defines the long-term timing and throughput experiment format for RealWonder.

## Goal

Timing should be:

- reproducible
- machine-readable
- comparable across cases, GPUs, and code versions

The codebase now writes structured timing logs for:

- offline simulation runs via `case_simulation.py`
- offline video generation runs via `infer_sim.py`
- interactive web demo startup via `demo_web/app.py`
- interactive web demo per-block generation via `demo_web/app.py`

## Experiment Names

Current experiment types:

- `offline_case_simulation`
- `offline_video_generation`
- `interactive_demo_startup`
- `interactive_demo_generation`

## Output Layout

### Offline simulation

```text
<case_output>/<timestamp>/experiment_logs/
```

Example:

```text
result/lamp/27-03_12-00-00/experiment_logs/
```

### Offline video generation

```text
<final_sim_dir>/experiment_logs/
```

Example:

```text
result/lamp/27-03_12-00-00/final_sim/experiment_logs/
```

### Interactive demo

Startup:

```text
demo_web/demo_data/<case>/experiment_logs/
```

Per-generation:

```text
<demo output folder>/experiment_logs/
```

## File Format

Each run writes:

1. `*.events.jsonl`
2. `*.summary.json`

### `events.jsonl`

Each line is one timing event.

Core fields:

- `run_id`
- `experiment_name`
- `run_name`
- `timestamp_utc`
- `stage`
- `duration_sec`

Optional fields depend on stage:

- `block_idx`
- `frame_count`
- `object_count`
- `simulation_steps`
- `rendered_frames`
- `mesh_vertices`
- `mesh_faces`
- `point_count`
- `queue_wait_sec`
- `physics_step_total_sec`
- `render_flow_total_sec`
- `warp_steps_sec`
- `vae_encode_sec`
- `mask_build_sec`
- `diffusion_sec`

### `summary.json`

Core fields:

- `run_id`
- `experiment_name`
- `run_name`
- `status`
- `start_time_utc`
- `end_time_utc`
- `total_duration_sec`
- `event_count`
- `host`
- `cwd`
- `cuda_visible_devices`
- `metadata`
- `events_path`

Optional fields:

- `output_folder`
- `final_sim_folder`
- `frame_count`
- `output_path`
- `num_output_frames`
- `total_warmup_sec`
- `error`

## Recommended Long-Term Analysis Keys

For downstream analysis tables, keep these keys:

- `experiment_name`
- `run_id`
- `case_name`
- `gpu_type`
- `cuda_visible_devices`
- `total_duration_sec`
- `status`

## First Benchmark Plan

Recommended initial matrix:

1. Run all current demo cases through `case_simulation.py`
2. Run all current demo cases through `infer_sim.py`
3. Run one startup pass of `demo_web/app.py` per case
4. Repeat at least 3 runs for `lamp` to estimate variance

This separates:

- one-time initialization cost
- per-run offline simulation cost
- per-run video generation cost
- startup warmup cost
- steady-state interactive block cost

## Why This Format

This format is designed for later aggregation into:

- CSV / pandas
- plotting notebooks
- latency decomposition tables
- scaling studies against:
  - frame count
  - object count
  - simulation steps
  - denoising steps
  - GPU model
