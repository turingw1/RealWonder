# RealWonder + Genesis Flow + VACE 14B 运行记忆

本文档从 2026-05-04 开始的 Codex 旧 session 中提取，记录 RealWonder 原流程、Genesis 光流导出、VACE/Wan2.1-VACE-14B flow-control demo 的可复用运行信息。

## 1. 总体目标

目标链路：

```text
RealWonder case
  -> SAM2 / SAM3D / MoGe reconstruction
  -> Genesis simulation
  -> Genesis optical flow tensor
  -> VACE flow-control RGB video
  -> Wan2.1-VACE-14B generated video
```

关键约束：

- 不改 RealWonder / VACE 核心模型算法。
- 不用低质量 fallback 替代 SAM3D、Genesis、VACE 14B。
- 大模型和数据下载走服务器默认网络。
- 只有默认网络失败或非常慢的小文件，才用 `18080` 代理。

## 2. 环境记忆

推荐环境：

```text
conda env: realwonder_cuda128_test
torch: 2.7.1+cu128
cuda runtime: 12.8
GPU: 4 x NVIDIA RTX PRO 6000 Blackwell, 97887 MiB each
```

激活方式：

```bash
cd /root/autodl-tmp/Physics_worldmodel/RealWonder
source scripts/activate_realwonder.sh
```

原因：

- 原 README 默认 `torch 2.5.1 + cu121` 在 Blackwell 上会触发 `no kernel image is available for execution on the device`。
- 仅 Genesis / RealWonder simulation 链路可以迁到 cu128 环境，不需要走 Wan 生成器。
- VACE 14B 运行也复用了这套 cu128 环境。

## 3. RealWonder Genesis Case 复跑

已跑通 6 个内置 case 的 RealWonder 原流程：

```text
lamp
tree
santa_cloth
persimmon
two_duck
sand_house
```

单 case 复跑：

```bash
cd /root/autodl-tmp/Physics_worldmodel/RealWonder
source scripts/activate_realwonder.sh

env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
  python -u case_simulation.py \
    --config_path cases/lamp/config.yaml \
    --device cuda \
    --genesis_backend gpu \
    --skip_noise_warp \
    --save_raw_frames
```

封装脚本：

```bash
cd /root/autodl-tmp/Physics_worldmodel/RealWonder
bash scripts/run_realwonder_genesis_demo.sh cases/lamp/config.yaml
```

说明：

- `--skip_noise_warp` 只跳过后续 diffusion/noise-warp，不跳过 SAM2/SAM3D/MoGe/Genesis simulation。
- 当前有效输出在 `result/<case>/<timestamp>/final_sim/`。
- 已有 manifest：`manifests/realwonder_genesis_cases.json`。

标准输出结构：

```text
final_sim/
  config.yaml
  simulation.mp4
  frames/frame_0000.png ...
  raw_frames_512/frame_0000.png ...
  genesis_flows_512.npy
  genesis_flows_480x832.npy
  points_masks_downsampled.pt
  mesh_masks_downsampled.pt
  resized_input_image.png
  prompt.txt
  metadata.json
```

已验证 shape：

```text
lamp/tree/santa_cloth/persimmon/two_duck:
  frames: 81
  genesis_flows_512.npy: (80, 2, 512, 512)
  genesis_flows_480x832.npy: (80, 2, 480, 832)

sand_house:
  frames: 165
  genesis_flows_512.npy: (164, 2, 512, 512)
  genesis_flows_480x832.npy: (164, 2, 480, 832)
```

## 4. Genesis Flow 导出为 VACE 输入

VACE flow-control 使用 RGB 光流可视化视频作为 `--src_video`。RealWonder 原始 flow tensor 仍保留 `.npy`，额外导出 `flow_vis.mp4` 给 VACE。

导出命令：

```bash
cd /root/autodl-tmp/Physics_worldmodel/RealWonder
source scripts/activate_realwonder.sh

python scripts/export_vace_flow_condition.py \
  --manifest manifests/realwonder_genesis_cases.json \
  --output_dir data_exports/vace_flow \
  --fps 10
```

导出目录：

```text
RealWonder/data_exports/vace_flow/<case>/
  flow_vis.mp4
  flow_vis_frames/frame_0000.png ...
  first_frame.png
  prompt.txt
  simulation_source.mp4
  metadata.json

RealWonder/data_exports/vace_flow/manifest.json
```

已导出 case：

```text
lamp
tree
santa_cloth
persimmon
two_duck
sand_house
```

注意：

- `flow_vis.mp4` 是给 VACE 的控制视频。
- `genesis_flows_*.npy` 才是原始 dense flow tensor。
- 导出脚本使用 Middlebury/RAFT 风格 color wheel。
- 为了 VACE 控制长度匹配，导出时会把第一帧 flow 复制一次，使 `flow_vis.mp4` 帧数等于 simulation 帧数。

## 5. VACE / Wan2.1-VACE-14B 权重与依赖

VACE 项目目录：

```text
/root/autodl-tmp/Physics_worldmodel/VACE
```

权重目录：

```text
/root/autodl-tmp/Physics_worldmodel/VACE/models/Wan2.1-VACE-14B
```

权重状态：

```text
size: about 70G
diffusion shards: 7 safetensors
text encoder: models_t5_umt5-xxl-enc-bf16.pth
vae: Wan2.1_VAE.pth
tokenizer: google/umt5-xxl
```

ModelScope 下载命令：

```bash
cd /root/autodl-tmp/Physics_worldmodel/VACE
source /root/autodl-tmp/Physics_worldmodel/RealWonder/scripts/activate_realwonder.sh

env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
  TMPDIR=/root/autodl-tmp/tmp \
  MODELSCOPE_CACHE=/root/autodl-tmp/modelscope-cache \
  modelscope download \
    --model Wan-AI/Wan2.1-VACE-14B \
    --local_dir /root/autodl-tmp/Physics_worldmodel/VACE/models/Wan2.1-VACE-14B \
    --max-workers 4
```

旧 session 记录的下载耗时约 `1:09:43`。

关键依赖：

```text
wan==2.1.0
xfuser==0.4.5
onnxruntime-gpu==1.25.1
transformers==4.x
tokenizers
peft
diffusers
flash-attn==2.8.3
```

重要点：

- VACE/Wan2.1 14B 的 attention 路径实际硬依赖 `flash_attn`。
- 旧 session 中 `pip install flash-attn` 没拿到可用 wheel 并尝试源码编译；当前机器没有独立 `nvcc`。
- 最终安装的是匹配 `torch2.7/cu12/cp311` 的预编译 wheel。
- 默认网络下载该 244MB wheel 卡住，才按规则使用了 `18080` fallback。

## 6. VACE Flow-Control 复跑

Runner：

```text
/root/autodl-tmp/Physics_worldmodel/VACE/run_realwonder_flow_vace14b.sh
```

默认命令：

```bash
cd /root/autodl-tmp/Physics_worldmodel/VACE
bash run_realwonder_flow_vace14b.sh lamp
```

默认参数：

```text
model_name: vace-14B
size: 832*480
frame_num: 81
sample_steps: 50
nproc_per_node: 4
ulysses_size: 4
ring_size: 1
dit_fsdp: true
t5_fsdp: true
```

旧 session 中真正成功的命令是单卡路径：

```bash
cd /root/autodl-tmp/Physics_worldmodel/VACE

CUDA_VISIBLE_DEVICES=0 \
VACE_NPROC=1 \
VACE_ULYSSES_SIZE=1 \
VACE_DIT_FSDP=0 \
VACE_T5_FSDP=0 \
VACE_OFFLOAD_MODEL=False \
bash run_realwonder_flow_vace14b.sh lamp
```

成功输出：

```text
VACE/results/realwonder_flow/lamp/20260509_044113/out_video.mp4
```

验收结果：

```text
out_video.mp4: 81 frames, 832x480, 16 fps, about 4.6MB
src_video.mp4: 81 frames, 832x480, 16 fps
source flow_vis.mp4: 81 frames, 832x480, 10 fps
```

性能记录：

```text
total elapsed: 881 s
sampling: 50 steps, about 15.73 s/step
GPU0 max memory: 96629 MiB / 97887 MiB
GPU0 average memory: 88431 MiB
GPU0 max power: 607 W
GPU0 average power: 533 W
```

结论：

- 当前服务器可以单卡完整跑通 `VACE 14B + Genesis flow`。
- 单卡 98GB 显存基本贴满，余量很小。
- 常规 4090 24GB 不适合这条完整 14B 路径。

### 6.1 2026-05-10 续跑结果

本 session 继续使用同一条单卡成功路径，跑完 `RealWonder/data_exports/vace_flow` 中除
`lamp` 外剩余的 5 个 case：

```bash
cd /root/autodl-tmp/Physics_worldmodel/VACE

CUDA_VISIBLE_DEVICES=0 \
VACE_NPROC=1 \
VACE_ULYSSES_SIZE=1 \
VACE_DIT_FSDP=0 \
VACE_T5_FSDP=0 \
VACE_OFFLOAD_MODEL=False \
bash run_realwonder_flow_vace14b.sh <case>
```

完成输出：

```text
case         output_dir                                                        elapsed_s  frames  size     fps  gpu0_max_mem_mib
lamp         VACE/results/realwonder_flow/lamp/20260509_044113                 881        81      832x480  16   96629
tree         VACE/results/realwonder_flow/tree/20260510_152748                 898        81      832x480  16   96769
santa_cloth  VACE/results/realwonder_flow/santa_cloth/20260510_154316          887        81      832x480  16   96629
persimmon    VACE/results/realwonder_flow/persimmon/20260510_155823            883        81      832x480  16   96629
two_duck     VACE/results/realwonder_flow/two_duck/20260510_161324             881        81      832x480  16   96629
sand_house   VACE/results/realwonder_flow/sand_house/20260510_162839           884        81      832x480  16   96769
```

核验方式：

```text
time.txt: all exit_status=0
OpenCV video metadata: all out_video.mp4 are 81 frames, 832x480, 16 fps
nvidia-smi after all runs: no running GPU processes, GPU0 memory 0 MiB
```

`sand_house` 的源 `flow_vis.mp4` 是 165 帧。VACE 在 `832*480` 下会设置
`seq_len=32760`，配合 `keep_last=True` 和 latent downsample 容量，实际从长视频中采样
81 帧，覆盖完整时长并保留末尾；因此没有尝试单卡 165 帧生成。按 `lamp` 的显存记录，
81 帧已经接近 98GB 上限。

## 7. 多卡问题

官方风格 4 卡配置：

```text
torchrun --nproc_per_node=4
--dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 1
```

失败模式：

```text
torch.distributed.fsdp -> FlatParamHandle -> torch.cat
CUDA error: an illegal memory access was encountered
```

4 卡非 FSDP + USP 在安装 `flash_attn` 后能进入采样，但第 0 step 仍触发 NCCL/CUDA illegal memory access。

当前判断：

- 问题不在 flow input。
- 问题不在权重缺失。
- 问题不在 prompt。
- 更可能是 Blackwell + torch2.7/cu128 + xFuser/NCCL/FSDP/USP 的兼容性矩阵问题。
- 不建议通过改 VACE 模型算法或降模型质量绕过；应单独排查分布式依赖版本和 NCCL/P2P 配置。

## 8. 已知风险与注意事项

RealWonder：

- `sand_house` 和 `two_duck` 当前使用 RealWonder demo web fallback `fov=60`，不是精确相机标定。
- `sand_house` 有 Genesis `substep_dt` 稳定性 warning。
- `two_duck` 有 PyTorch3D rasterization warning。

VACE：

- 单卡 14B 可以跑通，但显存贴近 98GB 上限。
- 多卡高效路径暂不可用。
- `flash_attn` 是必需依赖，不应删。
- VACE runner 会写 `gpu_usage.csv`、`run_config.txt`、`time.txt`，这些是性能验收依据。

磁盘：

- `VACE/models/Wan2.1-VACE-14B` 约 70G，不要误删。
- `RealWonder/result/*/final_sim` 中两份 float32 flow 很占空间，但它们是训练/分析真值，不建议只保留 `simulation.mp4`。

## 9. 相关文档

详细报告：

```text
RealWonder/docs/GENESIS_DATAFLOW_REPORT.md
RealWonder/docs/VACE14B_FLOW_PIPELINE_REPORT.md
RealWonder/docs/GENESIS_BUILTIN_CASE_TEST_REPORT.md
```

关键脚本：

```text
RealWonder/scripts/run_realwonder_genesis_demo.sh
RealWonder/scripts/export_vace_flow_condition.py
VACE/run_realwonder_flow_vace14b.sh
```
