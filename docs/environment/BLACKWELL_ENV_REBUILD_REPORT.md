# Blackwell 环境重建与 RealWonder 运行检查报告

日期：2026-05-08

## 结论

已在不破坏原 `realwonder` conda 环境的前提下，克隆并重建了一个 Blackwell 可用测试环境：

```bash
conda activate realwonder_cuda128_test
```

`scripts/activate_realwonder.sh` 已默认指向该环境，也可以通过 `REALWONDER_CONDA_ENV=...` 覆盖。

当前确认：

- Torch CUDA 可用，支持 `sm_120`。
- Genesis GPU 后端可运行最小场景。
- RealWonder 主入口 `case_simulation.py --help` 可正常加载。
- RealWonder 内部视频数据 demo 可运行，并能输出帧、光流、motion mask、metadata。
- 完整 `cases/lamp/config.yaml` 运行目前停在缺少 SAM2 checkpoint，不再是 CUDA/ABI 问题。

## 关键环境版本

```text
torch       2.7.1+cu128
torchvision 0.22.1+cu128
torchaudio  2.7.1+cu128
triton      3.3.1
pytorch3d   0.7.9
kaolin      0.18.0
genesis     0.3.1
```

处理过的兼容性问题：

- 原 `torch 2.5.1+cu121` 不包含 `sm_120` kernel，Blackwell 上 CUDA kernel smoke test 失败。
- 新环境升级到 `torch 2.7.1+cu128` 后，CUDA kernel smoke test 通过。
- `pytorch3d` 旧 wheel 与新 torch ABI 不兼容，已用 CUDA 12.8 / `TORCH_CUDA_ARCH_LIST=12.0` 从源码重编。
- `kaolin 0.17.0` 旧 wheel 与新 torch ABI 不兼容，已改用官方 `kaolin 0.18.0` 的 `torch-2.7.1_cu128` wheel。
- 旧 `flash-attn`、`xformers`、`bitsandbytes` 会阻塞导入；在 Blackwell 上 SAM3D 已走 `sdpa`，因此测试环境中移除了这三个旧包。

## 已验证命令

激活与环境检查：

```bash
cd /root/autodl-tmp/Physics_worldmodel/RealWonder
source scripts/activate_realwonder.sh
scripts/check_realwonder_env.sh
```

核心结果：

```text
cuda available: True
device 0: NVIDIA RTX PRO 6000 Blackwell Server Edition
capability: (12, 0)
arch list: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120', 'compute_120']
cuda kernel smoke: OK [2.0]
sam3d_objects: OK
pytorch3d: OK
kaolin: OK
gsplat: OK
sam2: OK
```

RealWonder 主入口：

```bash
PYTHONPATH=/root/autodl-tmp/Physics_worldmodel/RealWonder:/root/autodl-tmp/Physics_worldmodel/RealWonder/submodules/Genesis \
LIDRA_SKIP_INIT=1 \
python case_simulation.py --help
```

结果：正常输出 CLI 参数。

Torch + Genesis GPU smoke test：

```text
torch 2.7.1+cu128 cuda 12.8 available True
device NVIDIA RTX PRO 6000 Blackwell Server Edition
torch cuda smoke OK
genesis gpu smoke OK
```

## 数据 demo

可运行 demo 命令：

```bash
cd /root/autodl-tmp/Physics_worldmodel/RealWonder
source scripts/activate_realwonder.sh
python scripts/export_genesis_style_video_demo.py \
  --video data_demos/genesis_style_video_demo/gt_video.mp4 \
  --output_dir data_demos/env_rebuild_smoke/video_demo \
  --height 128 \
  --width 192 \
  --max_frames 12 \
  --fps 8 \
  --flow_preview \
  --prompt "environment rebuild smoke test"
```

输出位置：

```text
data_demos/env_rebuild_smoke/video_demo/
```

关键输出：

```text
flow_fwd.npy       float32, shape (11, 2, 128, 192)
motion_masks.npy   bool,    shape (12, 128, 192)
frames/            输入帧
coarse_rgb.mp4     coarse RGB 视频
gt_video.mp4       标准化后视频
metadata.json      数据契约说明
```

## 完整 case 当前阻塞

已运行：

```bash
PYTHONPATH=/root/autodl-tmp/Physics_worldmodel/RealWonder:/root/autodl-tmp/Physics_worldmodel/RealWonder/submodules/Genesis \
LIDRA_SKIP_INIT=1 \
python case_simulation.py \
  --config_path cases/lamp/config.yaml \
  --device cuda \
  --no-allow_cpu_fallback \
  --genesis_backend gpu \
  --skip_noise_warp \
  --save_raw_frames
```

结果停在缺少 SAM2 checkpoint：

```text
FileNotFoundError:
/root/autodl-tmp/Physics_worldmodel/RealWonder/submodules/sam2/checkpoints/sam2.1_hiera_large.pt
```

也就是说，环境/CUDA/Kaolin/PyTorch3D 的主阻塞已经解除；下一步需要准备模型权重。

## 下一步

完整 RealWonder case 需要补齐：

- `submodules/sam2/checkpoints/sam2.1_hiera_large.pt`
- `submodules/sam_3d_objects/checkpoints/hf/pipeline.yaml`
- SAM3D 对应的 checkpoint 文件

权重属于大文件，应按服务器原则走默认网络或手动上传；`18080` 只用于默认网络失败或很慢时的小文件/小 wheel。
