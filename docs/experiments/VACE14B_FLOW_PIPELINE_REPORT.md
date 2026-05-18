# RealWonder -> Genesis Flow -> VACE 14B 工作报告

## 目标

本轮目标是把当前 RealWonder 原流程继续推进到 VACE flow-control 视频生成：

```text
RealWonder case
  -> SAM2 / SAM3D / MoGe reconstruction
  -> Genesis simulation
  -> Genesis optical flow tensor
  -> VACE flow-control RGB video
  -> Wan2.1-VACE-14B generated video
```

约束：

- 不改 RealWonder / VACE 的核心模型算法。
- 不用低质量 fallback 替代 SAM3D、Genesis、VACE 14B。
- 大模型和数据下载使用服务器默认网络，不走 `18080`。
- 需要记录当前服务器上真正影响耗时和算力的环节。

## 当前机器与环境

GPU：

```text
4 x NVIDIA RTX PRO 6000 Blackwell Server Edition
显存: 97887 MiB / GPU
```

当前复用 conda 环境：

```text
realwonder_cuda128_test
torch 2.7.1+cu128
cuda runtime 12.8
```

VACE 额外依赖状态：

```text
wan==2.1.0
xfuser==0.4.5
onnxruntime-gpu==1.25.1
transformers==4.57.3
tokenizers==0.22.2
peft==0.19.1
diffusers==0.38.0
flash-attn==2.8.3
```

VACE / Wan2.1 14B 的当前 attention 路径实际硬依赖 `flash_attn`。直接 `pip install flash-attn` 没有拿到 wheel，会进入源码构建；当前机器没有 `nvcc`，所以改为下载官方匹配 wheel：

```text
flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

默认网络能访问 GitHub release header，但正文传输长时间 0 bytes；该 wheel 约 244MB，因此按“默认不可用时才用 18080”的原则，用 `18080` fallback 下载，随后本地安装。

## RealWonder 耗时拆解

下面的耗时基于已跑通的 6 个内置 case，采用输出目录创建时间、mesh 文件时间、`final_sim/config.yaml`、`simulation.mp4`、`metadata.json` 的文件时间戳估算。它不是逐函数 profiler，但足够说明瓶颈在哪里。

```text
case         total_s  to_mesh_s  mesh_to_config_s  export_after_config_s
lamp         116.0    51.6       31.9              32.6
tree         175.7    65.0       81.8              28.9
santa_cloth   84.7    39.7       15.3              29.8
persimmon    149.1    67.9       56.7              24.5
two_duck     256.8    50.3       174.5             32.1
sand_house   333.8    51.2       229.4             53.2
```

阶段含义：

- `to_mesh_s`：从 case 启动到 SAM3D/MoGe 物体 mesh 产出。主要包括模型加载、SAM3D sparse structure / sparse latent sampling、mesh decode、MoGe 深度对齐。
- `mesh_to_config_s`：mesh 产出后到 `final_sim/config.yaml` 写出。这里基本覆盖 Genesis scene build、物理仿真、renderer 光流生成。
- `export_after_config_s`：保存 `.npy` 光流、PNG frames、MP4、metadata。这个阶段看起来也不短，因为每个 case 会落盘两份大 float32 光流。

为什么只是 simulation video 也慢：

1. **真正耗时不只是渲染视频。** RealWonder 的 simulation video 前面还有 SAM2/SAM3D/MoGe 单图 3D 重建，SAM3D 权重约 11GB，首次加载和多对象重建会占明显时间。
2. **Genesis 同时在导出物理数据流。** 当前不是只编码 `simulation.mp4`，还保存 `genesis_flows_512.npy` 和 `genesis_flows_480x832.npy`。单个 81 帧 case 两份 flow 加起来约 400MB 级别，`sand_house` 接近 900MB 级别。
3. **物理类型差异很大。** `lamp/santa_cloth/persimmon/tree` 的 simulation 本身只有约 6-10 秒；`two_duck` rigid collision 用原参数跑了约 117 秒；`sand_house` MPM sand 约 9.4 万粒子，simulation + rendering 约 156 秒。
4. **多对象会重复重建。** `persimmon` 和 `two_duck` 每个对象都会走 SAM3D reconstruction，耗时随 object 数量增加。
5. **当前没有为性能牺牲质量。** 我没有降低 `simulated_frames_num`、`substeps`、`target_faces` 或替换物理/重建算法，只做路径、权重、本地 cache、headless 参数适配。

已观察到的 warning：

- `two_duck`：PyTorch3D coarse rasterization bin warning，提示 rasterization 可能不完整。
- `sand_house`：Genesis 提示 `substep_dt=0.001` 大于基于 grid density 的建议值 `0.0003125`，可能有数值稳定风险。
- `sand_house` 和 `two_duck` 当前使用 RealWonder `demo_web` fallback `fov_x_input=60.0`，不是 case-specific calibrated fov。

## Genesis Flow 输出

所有 case 的原始 Genesis flow 已在 `final_sim` 下：

```text
result/<case>/<timestamp>/final_sim/genesis_flows_512.npy
result/<case>/<timestamp>/final_sim/genesis_flows_480x832.npy
```

数学形式：

```text
shape = [T-1, 2, H, W]
flow[t, 0, y, x] = dx
flow[t, 1, y, x] = dy
```

VACE flow control 需要 RGB flow visualization 视频，因此已额外导出：

```text
data_exports/vace_flow/<case>/
  flow_vis.mp4
  flow_vis_frames/frame_0000.png ...
  first_frame.png
  prompt.txt
  simulation_source.mp4
  metadata.json

data_exports/vace_flow/manifest.json
```

核验结果：

```text
case         frames  size     fps   flow_vis.mp4 bytes
lamp         81      832x480  10.0  319690
tree         81      832x480  10.0  2232479
santa_cloth  81      832x480  10.0  1086884
persimmon    81      832x480  10.0  687703
two_duck     81      832x480  10.0  150927
sand_house   165     832x480  10.0  2616495
```

导出脚本：

```bash
cd /root/autodl-tmp/Physics_worldmodel/RealWonder
source scripts/activate_realwonder.sh
python scripts/export_vace_flow_condition.py \
  --manifest manifests/realwonder_genesis_cases.json \
  --output_dir data_exports/vace_flow \
  --fps 10
```

## VACE 14B 配置

VACE 项目目录：

```text
/root/autodl-tmp/Physics_worldmodel/VACE
```

权重目录：

```text
/root/autodl-tmp/Physics_worldmodel/VACE/models/Wan2.1-VACE-14B
```

权重校验：

```text
目录大小: 70G
diffusion_pytorch_model-00001-of-00007.safetensors
diffusion_pytorch_model-00002-of-00007.safetensors
diffusion_pytorch_model-00003-of-00007.safetensors
diffusion_pytorch_model-00004-of-00007.safetensors
diffusion_pytorch_model-00005-of-00007.safetensors
diffusion_pytorch_model-00006-of-00007.safetensors
diffusion_pytorch_model-00007-of-00007.safetensors
models_t5_umt5-xxl-enc-bf16.pth
Wan2.1_VAE.pth
google/umt5-xxl tokenizer files
```

ModelScope 下载耗时：

```text
1:09:43
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

VACE flow-control runner：

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

说明：

- 先用 `832*480` 是为了和 RealWonder / VACE flow input 当前分辨率对齐，仍使用 Wan2.1-VACE-14B 权重。
- 若 4 卡 98GB 在 `832*480` 能稳定跑通，再测试 `720p`；VACE README 对 14B 720p 官方示例是 8 卡，因此 4 卡 720p 需要实际验证。

## VACE Demo 结果

成功输出：

```text
/root/autodl-tmp/Physics_worldmodel/VACE/results/realwonder_flow/lamp/20260509_044113/out_video.mp4
```

成功命令：

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

运行参数：

```text
case: lamp
model: Wan2.1-VACE-14B
input: RealWonder Genesis flow visualization video
size: 832*480
frame_num: 81
sample_steps: 50
sample_fps: 16
execution: single GPU, no FSDP, no USP, flash-attn enabled
```

视频核验：

```text
out_video.mp4: 81 frames, 832x480, 16 fps, 4.6MB
src_video.mp4: 81 frames, 832x480, 16 fps
source flow_vis.mp4: 81 frames, 832x480, 10 fps
```

耗时：

```text
total elapsed: 881 s
sampling: 50 steps, about 15.73 s/step, about 786 s
```

GPU：

```text
GPU0 max memory: 96629 MiB / 97887 MiB
GPU0 average memory: 88431 MiB
GPU0 max power: 607 W
GPU0 average power: 533 W
```

说明：成功路径只使用 GPU0。`nvidia-smi` 在本机对空闲 GPU 的 utilization 读数显示异常偏高，但显存为 0 MiB；因此报告里以显存占用判断实际使用 GPU。

## VACE 多卡问题

官方 4 卡配置：

```text
torchrun --nproc_per_node=4
--dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 1
```

失败位置：

```text
torch.distributed.fsdp -> FlatParamHandle -> torch.cat
CUDA error: an illegal memory access was encountered
```

4 卡非 FSDP + USP 配置在安装 `flash_attn` 后能进入采样，但第 0 step 仍触发 NCCL/CUDA illegal memory access。当前判断是 Blackwell + torch 2.7.1/cu128 + xFuser/NCCL/USP/FSDP 的兼容性问题，不是权重缺失、flow 输入错误或 VACE prompt 问题。

当前可用结论：

- 单卡 98GB 可以完整跑通 14B、81 帧、832x480、50 steps，但显存余量很小，速度约 15.7 秒/step。
- 多卡高效路径还不能认为可用，需要单独处理 Blackwell 分布式兼容性。优先方向是测试 VACE/Wan 官方推荐 torch/CUDA 组合、xfuser/yunchang/flash-attn 版本矩阵，以及 NCCL/P2P 配置，而不是修改 VACE 模型算法。

## 当前状态

- RealWonder 6 个 case 已完成 simulation + Genesis flow。
- VACE flow-control RGB videos 已生成。
- `Wan-AI/Wan2.1-VACE-14B` 已通过 ModelScope 下载并校验。
- `flash-attn` 已安装并被 Wan / yunchang 识别。
- `lamp` 已完成 VACE 14B flow-control demo 输出。
- 多卡高效 VACE 路径暂时被 Blackwell 分布式兼容性阻塞，已记录失败模式。
