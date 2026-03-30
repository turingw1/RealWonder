# RealWonder A100 80GB High-Efficiency Workflow Playbook

Date: 2026-03-30

## 0. This Document Is A Living Playbook

This is not a static setup note. It is a living operations and experiment manual for running RealWonder efficiently on an A100 80GB server.

You should use it in this order:

1. fill in the parameter block once for your server
2. finish Stage S0 to S2 to establish a stable baseline
3. only then start optimization experiments O1+
4. after each experiment, send me the feedback template from Section 10
5. I will update the direction and this playbook based on your real results

The intended loop is:

`baseline -> design -> run -> analyze -> update next step`

## 1. Scope And Current Strategy

This playbook is built specifically for:

- current RealWonder repo
- A100 80GB server
- single-GPU work mode
- preferred runtime mask `CUDA_VISIBLE_DEVICES=1`
- large-file cache stored outside the git workspace
- potentially slow GitHub and Hugging Face networking

It integrates:

- `docs/A100_SERVER_SETUP.md`
- `docs/SERVER_GIT_WORKFLOW.md`
- `docs/DEMO_RUNBOOK.md`
- `docs/INTERACTIVE_VIDEO_SPEEDUP_SURVEY_2026-03-30.md`

The immediate goal is not to jump into model retraining. The immediate goal is to make the current repo stable, measurable, and efficient on your A100 server first.

## 2. Parameter Block

Edit and use this block at the beginning of every server session.

Do not write these blindly into global shell startup files on a shared server. Prefer copying them into the current terminal, a `tmux` pane, or a user-local session script.

```bash
export RW_ROOT=~/workspace/Zhengwei/RealWonder
export RW_CACHE=/cache/Zhengwei/RealWonder
export RW_ENV=realwonder

export RW_GPU_PHYSICAL=1
export CUDA_VISIBLE_DEVICES=${RW_GPU_PHYSICAL}

export HF_HOME=${RW_CACHE}/hf
export HUGGINGFACE_HUB_CACHE=${RW_CACHE}/hf/hub
export TORCH_HOME=${RW_CACHE}/torch
export TORCH_EXTENSIONS_DIR=${RW_CACHE}/torch_extensions
export TRITON_CACHE_DIR=${RW_CACHE}/triton
export WARP_CACHE_DIR=${RW_CACHE}/warp
export XDG_CACHE_HOME=${RW_CACHE}/tmp

export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
export NINJA_NUM_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
```

Important note:

- after `CUDA_VISIBLE_DEVICES=1`, Python will still usually see this GPU as `cuda:0`
- so `torch.cuda.get_device_name(0)` is expected and correct

Recommended quick validation:

```bash
echo "RW_ROOT=${RW_ROOT}"
echo "RW_CACHE=${RW_CACHE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -i ${RW_GPU_PHYSICAL}
```

## 3. Network And Repository Workflow

## 3.1 Git network policy

If GitHub is slow, use a shell-local mirror rewrite instead of changing global git config:

```bash
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=url.https://githubfast.com/.insteadOf
export GIT_CONFIG_VALUE_0=https://github.com/
```

Unset it after the session if needed:

```bash
unset GIT_CONFIG_COUNT
unset GIT_CONFIG_KEY_0
unset GIT_CONFIG_VALUE_0
```

## 3.2 Hugging Face network policy

For public downloads:

- prefer `HF_ENDPOINT="https://hf-mirror.com"` per command

For gated repos that return `403` on mirror:

- retry the exact command with `HF_ENDPOINT="https://huggingface.co"`

Do not switch global shell state just to test one endpoint.

## 3.3 Safe server update workflow

At the beginning of each workday:

```bash
cd ${RW_ROOT}
conda activate ${RW_ENV}

git status --short --branch
git pull
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive

bash scripts/check_realwonder_env.sh
```

If dependency files changed, reinstall only the affected layer.

## 4. One-Time Server Layout And Environment Rules

## 4.1 Required cache layout

```bash
mkdir -p ${RW_CACHE}/{hf,torch,torch_extensions,triton,warp,tmp,logs}
mkdir -p ${RW_CACHE}/{ckpts,wan_models}
mkdir -p ${RW_CACHE}/sam3d_objects/checkpoints
mkdir -p ${RW_CACHE}/sam2/checkpoints
```

## 4.2 Required symlink layout inside repo

```bash
cd ${RW_ROOT}

ln -sfn ${RW_CACHE}/ckpts ckpts
ln -sfn ${RW_CACHE}/wan_models wan_models
ln -sfn ${RW_CACHE}/sam3d_objects/checkpoints submodules/sam_3d_objects/checkpoints

ln -sf ${RW_CACHE}/sam2/checkpoints/sam2.1_hiera_tiny.pt submodules/sam2/checkpoints/sam2.1_hiera_tiny.pt
ln -sf ${RW_CACHE}/sam2/checkpoints/sam2.1_hiera_small.pt submodules/sam2/checkpoints/sam2.1_hiera_small.pt
ln -sf ${RW_CACHE}/sam2/checkpoints/sam2.1_hiera_base_plus.pt submodules/sam2/checkpoints/sam2.1_hiera_base_plus.pt
ln -sf ${RW_CACHE}/sam2/checkpoints/sam2.1_hiera_large.pt submodules/sam2/checkpoints/sam2.1_hiera_large.pt
```

## 4.3 Build policy on A100

Use:

- single visible GPU
- `bf16` default as already used by repo
- `TORCH_CUDA_ARCH_LIST=8.0`

If package builds saturate server CPU or RAM, reduce:

```bash
export MAX_JOBS=2
export NINJA_NUM_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
```

If the server is busy or unstable during compile, drop to `1`.

## 5. Daily Session Bootstrap

Run this at the start of every server session.

```bash
export RW_ROOT=~/workspace/Zhengwei/RealWonder
export RW_CACHE=/cache/Zhengwei/RealWonder
export RW_ENV=realwonder
export RW_GPU_PHYSICAL=1
export CUDA_VISIBLE_DEVICES=${RW_GPU_PHYSICAL}
export HF_HOME=${RW_CACHE}/hf
export HUGGINGFACE_HUB_CACHE=${RW_CACHE}/hf/hub
export TORCH_HOME=${RW_CACHE}/torch
export TORCH_EXTENSIONS_DIR=${RW_CACHE}/torch_extensions
export TRITON_CACHE_DIR=${RW_CACHE}/triton
export WARP_CACHE_DIR=${RW_CACHE}/warp
export XDG_CACHE_HOME=${RW_CACHE}/tmp
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
export NINJA_NUM_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4

cd ${RW_ROOT}
conda activate ${RW_ENV}

nvidia-smi -i ${RW_GPU_PHYSICAL}
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
bash scripts/check_realwonder_env.sh
```

Pass condition:

- `torch.cuda.is_available()` is `True`
- `torch.cuda.get_device_name(0)` reports A100
- env check script finishes without missing critical packages

If this fails, do not start experiments yet. Repair environment first.

## 6. Scientific Experiment Stages

Each stage has:

- objective
- commands
- pass criteria
- artifacts to save
- next-step decision

Do not skip stages.

## 6.1 Stage S0: Environment And Asset Sanity

Objective:

- confirm current repo can run on the target A100 server
- confirm checkpoints and dependencies are visible from the expected paths

Commands:

```bash
cd ${RW_ROOT}
conda activate ${RW_ENV}

find ${RW_CACHE}/ckpts -maxdepth 5 | sed -n '1,80p'
find ${RW_CACHE}/wan_models -maxdepth 4 | sed -n '1,80p'
find ${RW_CACHE}/sam2/checkpoints -maxdepth 2 | sed -n '1,80p'
find ${RW_CACHE}/sam3d_objects/checkpoints -maxdepth 3 | sed -n '1,80p'

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
LIDRA_SKIP_INIT=1 python -c "import sam3d_objects; import pytorch3d; import flash_attn; import kaolin; import gsplat; print('sam3d stack ok')"
python -c "import sam2; print('sam2 ok')"
python -c "import genesis as gs; print(gs.__version__)"
python -c "import diffusers, open_clip, kornia; print('root deps ok')"
```

Pass criteria:

- all imports succeed
- checkpoints are present

Artifacts:

- terminal output log
- screenshot or plain-text note of `nvidia-smi`

Decision:

- pass -> go to S1
- fail -> stop and repair setup before doing anything else

## 6.2 Stage S1: Offline Physics Baseline

Objective:

- validate simulation path independently of video generation

Recommended first case:

- `lamp`

Commands:

```bash
cd ${RW_ROOT}
conda activate ${RW_ENV}
CUDA_VISIBLE_DEVICES=${RW_GPU_PHYSICAL} python case_simulation.py --config_path demo_web/demo_data/lamp/config.yaml
```

Important note:

- `case_simulation.py` writes timestamped output directories
- do not assume the output is always exactly `result/lamp/final_sim`

Find the latest output:

```bash
cd ${RW_ROOT}
find result -maxdepth 3 -type d | sort | tail -n 20
```

Pass criteria:

- a new timestamped run directory appears
- the directory contains `final_sim/frames`, masks, config, and prompt

Artifacts:

- latest run path
- frame count
- whether `simulation.mp4` was produced

Decision:

- pass -> go to S2
- fail -> focus on Genesis / case config / rendering path

## 6.3 Stage S2: Offline Video Generation Baseline

Objective:

- validate the video generator independently of the web UI

Commands:

```bash
cd ${RW_ROOT}
conda activate ${RW_ENV}

LATEST_SIM=$(find result -path '*/final_sim' -type d | sort | tail -n 1)
echo ${LATEST_SIM}

CUDA_VISIBLE_DEVICES=${RW_GPU_PHYSICAL} python infer_sim.py \
  --checkpoint_path ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt \
  --sim_data_path "${LATEST_SIM}" \
  --output_path "${LATEST_SIM}/final.mp4"
```

Pass criteria:

- `final.mp4` is generated
- no missing-checkpoint or CUDA OOM errors

Artifacts:

- output video path
- wall-clock runtime
- any printed profiling information

Decision:

- pass -> go to S3
- fail -> this is still a model/runtime issue, not a web issue

## 6.4 Stage S3: Interactive Demo Baseline With Timing Logs

Objective:

- validate the real-time path
- collect structured timing logs

Install demo dependencies once if needed:

```bash
cd ${RW_ROOT}
conda activate ${RW_ENV}
python -m pip install -r demo_web/requirements.txt
```

Run demo:

```bash
cd ${RW_ROOT}
conda activate ${RW_ENV}
CUDA_VISIBLE_DEVICES=${RW_GPU_PHYSICAL} python demo_web/app.py \
  --demo_data demo_web/demo_data/lamp \
  --checkpoint_path "${RW_CACHE}/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt"
```

If you want GPU memory logging:

```bash
CUDA_VISIBLE_DEVICES=${RW_GPU_PHYSICAL} python demo_web/app.py \
  --demo_data demo_web/demo_data/lamp \
  --checkpoint_path "${RW_CACHE}/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt" \
  --gpu_log
```

After one full interaction, inspect logs:

```bash
cd ${RW_ROOT}
find demo_web/demo_data/lamp/experiment_logs -maxdepth 2 -type d | sort | tail -n 5

LATEST_RUN=$(find demo_web/demo_data/lamp/experiment_logs -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
echo ${LATEST_RUN}

find "${LATEST_RUN}" -maxdepth 1 -type f | sort
cat "${LATEST_RUN}/startup.summary.json"
cat "${LATEST_RUN}/generation.summary.json"

python scripts/plot_interactive_demo_timing.py "${LATEST_RUN}"
```

Pass criteria:

- browser can start generation
- `generation.summary.json` exists
- `pipeline_timing.png` is produced

Artifacts:

- `startup.summary.json`
- `generation.summary.json`
- `pipeline_timing.png`
- case name and prompt used

Decision:

- pass -> start optimization experiments
- fail -> fix web path before attempting research changes

## 7. Immediate Optimization Experiments On Current Repo

These experiments require no training and should be completed before architecture changes.

## 7.1 Experiment O1: TAEHV Decoder On vs Off

Objective:

- check whether the optional fast decoder improves end-to-end responsiveness on your A100

Commands:

Baseline:

```bash
CUDA_VISIBLE_DEVICES=${RW_GPU_PHYSICAL} python demo_web/app.py \
  --demo_data demo_web/demo_data/lamp \
  --checkpoint_path "${RW_CACHE}/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt"
```

TAEHV:

```bash
CUDA_VISIBLE_DEVICES=${RW_GPU_PHYSICAL} python demo_web/app.py \
  --demo_data demo_web/demo_data/lamp \
  --checkpoint_path "${RW_CACHE}/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt" \
  --taehv
```

Pass criteria:

- same case and same interaction can run stably in both settings

What to compare:

- wall-clock from button click to first visible block
- wall-clock to full sequence completion
- generated log summaries
- subjective quality drop

Decision gate:

- if speed gain is clear and quality loss is acceptable, keep `--taehv` in your daily demo workflow
- otherwise keep it off and focus elsewhere

## 7.2 Experiment O2: Case-Wise Bottleneck Map

Objective:

- determine whether the bottleneck is stable across materials

Cases:

- `lamp`
- `tree`
- `persimmon`
- `santa_cloth`

Procedure:

1. run S3 once per case
2. save the four `generation.summary.json`
3. compare Stage 1 vs Stage 2 vs Stage 3 share

Decision gate:

- if diffusion dominates in all four cases, prioritize model-side speedup
- if simulation/render dominates in deformable cases, prioritize simulator proxy or renderer changes for those cases

## 7.3 Experiment O3: Cold-Start vs Warm-Start Verification

Objective:

- verify that warmup is actually buying you large savings on your A100 server

Procedure:

1. start the demo clean once
2. record first interaction latency
3. restart and run again after warmup path finishes
4. compare block-0 and total latency

Expected based on code comments:

- warmup should strongly reduce first-request latency

Decision gate:

- if warmup effect is weak on your server, inspect whether kernel caches are being reused from the intended cache paths

## 8. Research Experiment Roadmap After Baseline

Do not begin these until S0-S3 and O1-O3 are complete.

## 8.1 Research R1: Distill 4-step To 2-step

Priority: highest

Why:

- most direct speed path
- best fit with current code and current survey conclusions

Read first:

- `docs/INTERACTIVE_VIDEO_SPEEDUP_SURVEY_2026-03-30.md`
- CausVid
- ASD one-step causal video generation

Stage goal:

- preserve RealWonder's physics-conditioned bridge
- reduce denoising steps in Stage 3

Minimum success criterion:

- at least one working 2-step checkpoint or ablation path
- clear latency reduction against the 4-step baseline
- no catastrophic loss of temporal coherence

## 8.2 Research R2: Memory Upgrade For Streaming

Priority: second

Why:

- once steps drop, long-horizon consistency becomes more fragile

Read first:

- WorldPlay
- StreamDiT

Stage goal:

- keep current causal streaming design
- improve context handling beyond plain KV-cache reuse

Minimum success criterion:

- better long-horizon consistency at equal or lower latency

## 8.3 Research R3: Fast Path For Simple Rigid Dynamics

Priority: third

Why:

- current repo still pays full Genesis cost even for easy rigid cases

Read first:

- Goal Force
- NewtonGen

Stage goal:

- build a learned fast path for simple force-conditioned rigid interactions
- keep Genesis only for harder materials or hard-contact cases

Minimum success criterion:

- measurable speed gain on easy rigid cases
- acceptable motion plausibility

## 9. Artifact And Logging Discipline

Every experiment must leave artifacts that let us decide the next step scientifically.

For each run, save:

1. command line
2. git commit hash
3. case name
4. whether `--taehv` was used
5. visible GPU mapping
6. output path
7. timing summaries
8. qualitative note

Minimal capture commands:

```bash
cd ${RW_ROOT}
git rev-parse --short HEAD
echo ${CUDA_VISIBLE_DEVICES}
nvidia-smi -i ${RW_GPU_PHYSICAL}
```

Recommended experiment directory convention:

```text
${RW_CACHE}/logs/realwonder/
  2026-03-30/
    S3_lamp_baseline/
    O1_lamp_taehv/
    O2_tree_baseline/
```

If you want, you can manually copy:

- `generation.summary.json`
- `startup.summary.json`
- `pipeline_timing.png`
- final output video

into those folders after each run.

## 10. Feedback Template For Updating This Playbook

After each completed stage or experiment, send me a message using this template.

```text
[Stage]
S3 / O1 / O2 / R1 ...

[Server]
A100 80GB

[Visible GPU]
CUDA_VISIBLE_DEVICES=1

[Commit]
<git rev-parse --short HEAD>

[Case]
lamp / tree / persimmon / santa_cloth

[Command]
<full command>

[Result]
success / fail

[Artifacts]
<absolute paths to summary json, png, mp4>

[Key Metrics]
startup total:
generation total:
stage1 total:
stage2 total:
stage3 total:
first block latency:

[Quality Observation]
<one short paragraph>

[Failure Or Constraint]
<OOM / slow warmup / missing checkpoint / browser lag / etc.>

[Your Current Guess]
<what you think is the current bottleneck>
```

Once you send that back, I can do one of three things:

1. update the next experiment priority
2. refine the command sequence
3. update this playbook with a new branch or ablation plan

## 11. Decision Tree

Use this after S3.

### Case A: Stage 3 dominates clearly

Action:

- prioritize R1 first
- then R2

### Case B: Stage 1 dominates on deformable materials

Action:

- prioritize simulator/render-side work
- consider PhysTalk-style or hybrid proxy ideas earlier

### Case C: TAEHV gives immediate benefit with acceptable quality

Action:

- adopt it for daily demo work
- keep baseline runs without TAEHV for fair comparison

### Case D: warmup does not help much

Action:

- inspect cache path reuse
- check whether the session actually reuses the same cache directories
- verify the server is not clearing build caches unexpectedly

## 12. Recommended First Week Plan

Day 1:

- finish S0, S1, S2

Day 2:

- finish S3 on `lamp`
- collect first timing plot

Day 3:

- finish O1 with and without `--taehv`

Day 4:

- finish O2 across four cases

Day 5:

- summarize bottleneck distribution
- choose between R1-first and simulator-first

## 13. Bottom Line

Your current best operating strategy on the A100 80GB server is:

1. always run single-GPU with `CUDA_VISIBLE_DEVICES=1`
2. keep checkpoints and caches under `/cache`
3. establish a reproducible baseline before any research changes
4. use timing logs to decide whether the next step is model-side or simulator-side
5. send results back in the structured format so the experiment plan can be updated in real time

This is how we keep the work both efficient and scientifically correct.

