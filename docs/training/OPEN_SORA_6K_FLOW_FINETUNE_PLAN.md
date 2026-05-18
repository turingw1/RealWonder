# Open-Sora 6k 视频光流控制微调方案

日期：2026-05-11

基础依据：

- `docs/training/训练方案508.md`
- `docs/experiments/OPEN_SORA_VACE_FLOW_BOUNDARY_REPORT.md`

## 判断

10 个 Open-Sora/MixKit 样本的 VACE14B zero-shot 实验说明：

```text
prompt + RGB flow video -> VACE14B
```

可以生成语义正确的视频，但不能严格遵守光流和原始外观。  
所以 6k 视频训练的目标不应该是“继续直接喂 flow_vis 期待模型自己懂”，而应该是：

```text
真实视频自监督重建：
source video
  -> first frame + raw flow + flow_vis debug + coarse_rgb/depth + prompt
  -> model
  -> reconstruct source video
```

训练完成后，再把输入源从真实视频 RAFT flow 换成 Genesis/RealWonder flow。

## 数据结构

建议在 `datasets/Open-Sora-Plan-v1.1.0` 基础上生成一个独立训练缓存：

```text
datasets/realwonder_flow_6k/
  manifest.jsonl
  clips/<video_id>/
    source.mp4              # 81f 或 33f clip，保留原始重建目标
    first_frame.png
    flow_raw.npy            # [T-1, 2, H, W]，主条件
    flow_vis.mp4            # debug / VACE 兼容输入
    coarse_rgb.mp4          # blur/downsample 或 first-frame warp
    mask.mp4                # 可选，前景/主体区域
    depth.mp4               # 可选，后续接 Genesis / point cloud
    prompt.txt
    meta.json               # fps, crop, resize, flow scale, category
```

原则：

- `flow_raw.npy` 是训练主输入，不要只保留 RGB flow video。
- `flow_vis.mp4` 只作为可视化和 VACE 兼容层。
- `source.mp4` 必须保留，方便以后换 VAE/latent/chunk 配置。
- prompt 第一版可从文件名生成；后续可换自动 caption。

## 训练阶段

### Stage 0：数据和 baseline

规模：10 条样本。

任务：

1. 复用本次 `vace_flow_boundary_20260511` 的 10 条样本；
2. 跑 `flow-only`、`first-frame + prompt`、`first-frame + flow` 三组 baseline；
3. 记录 generated RAFT flow 与 input flow 的 EPE/cos；
4. 肉眼检查 source/flow/output 对比。

验收：

- pipeline 可重复；
- 指标和对比视频自动生成；
- 明确知道 prompt 主导还是 flow 主导。

### Stage 1：10 条视频过拟合

目标：证明模型真的能学会听 flow。

推荐设置：

- 冻结 VACE/Wan14B backbone；
- 训练 LoRA + 小型 flow/coarse adapter；
- 输入：`first_frame + raw_flow + coarse_rgb + prompt`；
- 输出：重建 source video；
- 帧数先用 `33f`，成功后扩到 `81f`；
- 分辨率先对齐 `832x480`。

必须做 ablation：

| ablation | 目的 |
| --- | --- |
| no-flow | 确认模型不是纯靠 prompt |
| reversed-flow | 确认运动方向由 flow 控制 |
| flow-only | 测试 flow 本身上限 |
| first-frame + flow | 测试外观锁定能力 |
| first-frame + flow + coarse_rgb | 测试遮挡/结构改善 |

### Stage 2：扩大到 500 条

目标：学会常见真实视频运动。

采样策略：

- 从 MixKit/Open-Sora 类别中均匀抽样；
- 覆盖车辆、人物、动物、流体、火焰、天空、城市、室内；
- 过滤极低运动和镜头切换过猛的视频；
- 每类保留 flow 统计，避免数据全是某一类运动。

评估：

- held-out 50 条；
- generated RAFT vs input flow；
- source/output LPIPS 或 VAE latent MSE；
- contact sheet 人工分级。

### Stage 3：扩大到 6k 条

目标：训练通用 flow-conditioned 真实视频生成能力。

建议配比：

| 类别 | 比例 |
| --- | ---: |
| 人体/动作 | 20% |
| 车辆/交通/刚体 | 20% |
| 动物/多目标 | 15% |
| 自然/水/云/火 | 20% |
| 城市/室内/相机运动 | 15% |
| 低运动/细节变化 | 10% |

训练策略：

- 先只训 LoRA + adapter；
- 主干冻结，避免 6k 小数据破坏 Wan/VACE 先验；
- 对 flow magnitude 做全局或分位数归一化；
- 保存 raw flow scale，训练时随机 scale jitter；
- 加 `flow dropout`，让模型不要过拟合光流颜色；
- 加 `prompt dropout`，避免 prompt 压过 flow。

### Stage 4：Genesis / RealWonder domain adaptation

真实视频 RAFT flow 和 Genesis flow 分布不同。接 Genesis 前需要加一个 domain bridge：

1. 用已有 RealWonder case 导出：
   - `genesis_flows_480x832.npy`
   - `simulation.mp4`
   - `first_frame.png`
   - masks
2. 将 Genesis flow 转成训练时相同的 raw-flow schema；
3. 做少量 physics-flow finetune 或 adapter calibration；
4. 训练时混入 synthetic/Genesis flow augmentation：
   - 稀疏 foreground flow；
   - 大面积 zero background；
   - rigid translation；
   - collision/stop；
   - non-rigid deformation。

## Loss 和评测

核心 loss：

- diffusion / flow-matching 原任务 loss；
- source video reconstruction loss；
- VAE latent reconstruction loss；
- temporal consistency loss。

辅助评测：

```text
input flow F
generated video -> RAFT -> F_pred
EPE(F, F_pred)
cosine(F, F_pred)
```

对 Genesis case 额外评测：

- object mask 内 flow EPE；
- keypoint / tracker 轨迹误差；
- 碰撞、落地、倒塌等 event frame 是否对齐；
- prompt 与 physics 冲突时，是否遵守 physics。

## 预期结论路径

如果 10 条 overfit 失败：

```text
先查 adapter 注入位置、flow scale、mask、loss，不扩大数据。
```

如果 10 条 overfit 成功但 500 条泛化差：

```text
增加 coarse_rgb/depth，做更严格的 flow normalization 和类别均衡。
```

如果 6k 真实视频成功但 Genesis flow 失败：

```text
做 Genesis flow domain adaptation，不要直接责怪生成 backbone。
```

最终目标不是让模型“看懂 RGB flow 图”，而是让它学到：

```text
首帧给外观，
raw flow 给运动，
coarse/depth/mask 给结构和遮挡，
prompt 给语义边界。
```
