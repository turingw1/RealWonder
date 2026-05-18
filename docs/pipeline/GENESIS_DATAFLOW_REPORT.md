# RealWonder Genesis 数据流工作汇报

## 当前原则

本阶段只打通 RealWonder 原始范式的数据流：

```text
case image + prompt/config
  -> SAM2 segmentation
  -> SAM3D Objects / MoGe single-view reconstruction
  -> Genesis physical simulation + rendering
  -> simulation video + Genesis optical flow tensors + frames
```

代码适配范围限制为环境、路径、缓存、本地权重、导出参数和 case 配置。没有用简化几何、OpenCV 光流或手写 toy renderer 代替 RealWonder 的重建/仿真算法。旧的 smoke/fallback 输出不再作为有效结果。

网络和权重策略：

- 大模型、pip 包、数据集下载使用服务器默认网络。
- `18080` 代理只作为小文件外网访问失败时的备用路径。
- HF 不默认假设需要 token；遇到 gated repo 或直连不稳定时，优先在 ModelScope 查找同名/相近模型。
- 如果权重缺失或 Blackwell 适配失败，记录问题，不改核心模型算法去伪造输出。

## 已跑通的真实 case

已用 RealWonder 原流程跑通 6 个内置 case。所有 case 都产出了 `final_sim/simulation.mp4`、PNG frames、raw frames、两份 Genesis optical flow `.npy` 和 `metadata.json`。
已同步写入 manifest：

```text
manifests/realwonder_genesis_cases.json
```

```text
lamp         result/lamp/09-05_00-55-11/final_sim
tree         result/tree/09-05_01-04-56/final_sim
santa_cloth  result/santa_cloth/09-05_01-08-18/final_sim
persimmon    result/persimmon/09-05_01-10-15/final_sim
two_duck     result/two_duck/09-05_01-14-00/final_sim
sand_house   result/sand_house/09-05_01-18-45/final_sim
```

单个 case 的运行命令：

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

也可以用封装脚本：

```bash
cd /root/autodl-tmp/Physics_worldmodel/RealWonder
bash scripts/run_realwonder_genesis_demo.sh cases/lamp/config.yaml
```

`--skip_noise_warp` 只跳过后续 diffusion/noise-warp 训练辅助数据，不跳过 SAM3D/MoGe 重建和 Genesis simulation。当前目标是先确认 simulation video 与 Genesis optical flow 能稳定输出。

## 输出内容

当前有效输出：

```text
final_sim/
  config.yaml
  simulation.mp4
  frames/frame_0000.png ... frame_0080.png
  raw_frames_512/frame_0000.png ... frame_0080.png
  genesis_flows_512.npy
  genesis_flows_480x832.npy
  points_masks_downsampled.pt
  mesh_masks_downsampled.pt
  resized_input_image.png
  prompt.txt
  metadata.json
```

已验证的 shape / 数量：

```text
case         frames  frame size  raw frames  mp4 bytes  flow512 shape          flow480x832 shape
lamp         81      832x480     81          224579     (80,2,512,512)        (80,2,480,832)
tree         81      832x480     81          907865     (80,2,512,512)        (80,2,480,832)
santa_cloth  81      832x480     81          468582     (80,2,512,512)        (80,2,480,832)
persimmon    81      832x480     81          317881     (80,2,512,512)        (80,2,480,832)
two_duck     81      832x480     81          161254     (80,2,512,512)        (80,2,480,832)
sand_house   165     832x480     165         1362268    (164,2,512,512)       (164,2,480,832)
```

光流数学形式仍是逐像素位移场：

```text
flow[t, 0, y, x] = dx
flow[t, 1, y, x] = dy
```

也就是第 `t` 帧像素 `(x, y)` 到第 `t+1` 帧位置 `(x + dx, y + dy)` 的运动估计。对训练而言，这类“视频形式”的 dense tensor 可以作为 motion condition、noise warp、mask supervision 或物理控制信号，连接 Genesis 的可控仿真和后续视频生成模型。

## 权重状态

SAM3D Objects 已按 ModelScope 方法下载：

```bash
modelscope download \
  --model facebook/sam-3d-objects \
  --local_dir submodules/sam_3d_objects/checkpoints/hf-modelscope-download \
  --include 'checkpoints/*' \
  --max-workers 1
```

并建立本地兼容路径：

```text
submodules/sam_3d_objects/checkpoints/hf
  -> hf-modelscope-download/checkpoints
```

关键大权重已存在：

```text
slat_generator.ckpt  4.9G
ss_generator.ckpt    6.7G
MoGe model.pt        1.2G
DINOv2 vitl14        1.2G
```

MoGe 情况：

- RealWonder/SAM3D 默认请求 `Ruicheng/moge-vitl`。
- ModelScope 未找到同名模型。
- HF 直连不稳定；最终通过 `hf-mirror.com` 下载 `model.pt`，使用服务器默认网络，不走 `18080`。
- 本地路径为：

```text
submodules/sam_3d_objects/checkpoints/moge-vitl/model.pt
```

DINOv2 情况：

- SAM3D 内部通过 `torch.hub.load("facebookresearch/dinov2", ...)` 加载。
- 即使源码和 checkpoint 已在本地 cache，`source="github"` 仍会尝试访问 GitHub。
- 当前只做缓存路径适配：本地存在 `/root/.cache/torch/hub/facebookresearch_dinov2_main` 时优先 `source="local"`，避免运行时再次联网。

注意：ModelScope 下载出的 `ss_encoder.safetensors` 和 `ss_encoder.yaml` 当前是 0 bytes。`lamp` 路径没有因此失败；如果后续 case 或功能调用到该 encoder，需要重新获取完整文件并记录来源。

## 最小代码/配置适配

case config：

- `lamp`、`tree`、`santa_cloth`、`persimmon` 的 `fov_x_input` 来自 RealWonder `demo_web/demo_data/*/config.yaml`。
- `sand_house`、`two_duck` 没有对应 demo fov，当前使用 RealWonder `demo_web/simulation_engine.py` 的 fallback `fov=60` 作为运行测试配置；这不是 case-specific calibrated fov。
- 所有 case 增加 `moge_model_path` 指向本地 MoGe 权重。
- 将需要 headless 运行的 case 设置为 `debug: false`。

`simulation/image23D/single_view_reconstructor.py`：

- 将硬编码 `MoGeModel.from_pretrained("Ruicheng/moge-vitl")` 改成优先读取 `config["moge_model_path"]`，没有配置时仍回退原 repo id。

`submodules/sam_3d_objects/.../dino.py`：

- 加本地 DINOv2 torch hub cache 优先逻辑，避免每次运行访问 GitHub。

`case_simulation.py`：

- 增加数据导出用 CLI 参数：`--device`、`--genesis_backend`、`--skip_noise_warp`、`--save_raw_frames`。
- 保留 RealWonder 主流程，只是在 simulation 完成后把视频帧、Genesis flow 和 metadata 结构化落盘。

## 仍需处理的问题

1. `sand_house`、`two_duck` 当前用的是 RealWonder web fallback `fov=60`，不是精确标定；如果要作为高质量训练数据，建议补 case-specific fov 或相机内参。
2. `sand_house` 运行时 Genesis 提示 `substep_dt=0.001` 大于基于 grid density 的建议值 `0.0003125`，可能有数值稳定性风险；我没有改 `dt/substeps`，保持原 case 参数。
3. `two_duck` 重建/渲染阶段出现 PyTorch3D coarse rasterization bin warning，提示输出可能不完整；当前没有改 rasterizer 参数，已保留问题记录。
4. `ffprobe` 当前不可用，所以视频 metadata 主要通过 PNG 帧数、文件存在性和 numpy shape 验证。
5. 后续如果要生成 diffusion/noise-warp 训练数据，需要再打开 `--noise_flow_source genesis` 或 `raft` 路径，并确认 RAFT 权重/torchvision 版本在 Blackwell 环境可用。

## 下一步建议

当前底层 simulation 数据流已经打通。下一步建议：

```text
1. 补齐 sand_house / two_duck 的真实 fov 或相机内参。
2. 抽查每个 simulation.mp4 的视觉质量，特别是 two_duck 的 rasterization warning 和 sand_house 的稳定性 warning。
3. 再开启 --noise_flow_source genesis 或 raft，生成后续训练用 noise-warp/RAFT 对齐数据。
4. 将这些 final_sim 目录注册为后续训练数据 manifest。
```

每个 case 的合格标准统一为：

```text
result/<case>/<timestamp>/final_sim/simulation.mp4
result/<case>/<timestamp>/final_sim/frames/*.png
result/<case>/<timestamp>/final_sim/genesis_flows_512.npy
result/<case>/<timestamp>/final_sim/genesis_flows_480x832.npy
result/<case>/<timestamp>/final_sim/metadata.json
```
