# RealWonder Wan Pipeline Notes

本文记录 RealWonder 当前视频生成后端实际喂给 Wan 基座的输入、张量形状、condition 列表，以及这些 condition 在 diffusion / flow-matching 采样过程中的注入方式。目的是给后续讨论“把 RealWonder 的视频生成基座换成 Helios”提供接口对照。

## 0. 先澄清模型名

代码和 README 里实际使用的是：

- base: `Wan2.1-Fun-V1.1-1.3B-InP`
- RealWonder checkpoint: `Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt`

所以如果讨论里说“Wan 1.4B”，在本笔记中对应的是这个约 1.3B 的 Wan InP/I2V 基座加 RealWonder distilled causal checkpoint。

## 1. 已有相关笔记

已有文档里有几份可以直接复用：

- `RealWonder/docs/GENESIS_DATAFLOW_REPORT.md`: Genesis simulation 输出、flow tensor 语义、`genesis_flows_480x832.npy` 等。
- `RealWonder/docs/训练方案508.md`: 已经开始讨论把 RealWonder 后端换成 Helios 的训练路线。
- `Helios/docs/training.md`: Helios Stage-1 latent dataset、history latent、flow matching 训练入口。
- `Helios/docs/LORA_training.md`: Helios 上做 RealWonder-style LoRA / flow-conditioned 训练的初步方案。

本文件补齐的是 RealWonder 现有 Wan 生成接口的“实际输入侧”。

## 2. 分辨率和基本维度

RealWonder interactive demo 默认：

```text
pixel video: 480 x 832
VAE stride: 4 x 8 x 8
latent H, W: 60 x 104
latent C: 16
latent frames per block: 3
tokens per latent frame after Wan patchify: 30 x 52 = 1560
tokens per block: 3 x 1560 = 4680
```

对于一个完整 request：

```text
T_lat = num_output_frames
T_pix = (T_lat - 1) * 4 + 1
```

常见 case 是 `T_lat=21`，对应 `T_pix=81`、7 个 block。`sand_house` 这类配置也可能用 `T_lat=42`。

Interactive streaming 的像素帧输入略有区别：

```text
block 0 pixel frames: (3 - 1) * 4 + 1 = 9
block 1+ pixel frames: 3 * 4 = 12
```

这是为了配合 Wan VAE 的 temporal cache，让每个 block 都编码出 3 个 latent frames。

## 3. Wan 生成器收到的 condition

RealWonder 默认走 `i2v_flow=True`，因此 pipeline 使用 `WanI2VDiffusionWrapper`。生成器一次 block call 的主接口近似是：

```python
pipeline.generator(
    noisy_image_or_video=noisy_input,   # [B, F, 16, 60, 104]
    conditional_dict={
        "prompt_embeds": ...,
        "clip_feature": ...,
        "y": ...,
    },
    curr_y=curr_y,                      # [B, F, 20, 60, 104]
    timestep=timestep,                  # [B, F]
    kv_cache=...,
    crossattn_cache=...,
    current_start=block_start * 1560,
)
```

其中 `F=3` 是 streaming block 的 latent frame 数。完整离线 `infer_sim.py` 会把 `F` 换成当前循环切出的 block length。

| 输入 | 来源 | 完整形状 | block 形状 | 注入方式 |
| --- | --- | --- | --- | --- |
| `prompt_embeds` | prompt text -> UMT5 encoder | `[B, 512, 4096]` | 同完整 | Wan 内部 `text_embedding` 映射到 hidden dim，作为 cross-attention text context |
| `clip_feature` | first frame -> Wan CLIP image encoder | `[B, 257, 1280]` | 同完整 | Wan 内部 `img_emb` 映射到 hidden dim，作为 I2V image context；I2V cross-attn 对 image tokens 和 text tokens 分别 attention 后相加 |
| `y` / `curr_y` | first frame -> Wan VAE image embedder + mask | `[B, 20, T_lat, 60, 104]` | 传入 wrapper 前常为 `[B, F, 20, 60, 104]` | 在 Wan model 入口与 noisy latent 沿 channel concat，变成 `[B, 36, F, 60, 104]` 后再 patch embedding |
| `noisy_input` | SDEdit 初始化得到的 latent | `[B, T_lat, 16, 60, 104]` | `[B, F, 16, 60, 104]` | Wan diffusion backbone 的主输入 `x_t` |
| `timestep` | case `denoising_step_list` 经 FlowMatch scheduler 映射 | `[B, T_lat]` 或 `[B, F]` | `[B, F]` | sinusoidal time embedding -> time projection，作为每个 transformer block 和 head 的 modulation |
| `sim_latent` | Genesis RGB simulation frames -> Wan VAE encode | `[B, T_lat, 16, 60, 104]` | `[B, F, 16, 60, 104]` | 不直接进 Wan condition dict；用于 SDEdit 生成 `noisy_input`，并可在中途 mask drop-in |
| `structured_noise` | Genesis/RAFT flow-warped noise 前 16 channels | `[B, T_lat, 16, 60, 104]` | `[B, F, 16, 60, 104]` | SDEdit 的 noise source，决定初始 `x_t` 的 motion prior |
| `sde_noise` | flow-warped noise 后 16 channels，可选 | `[B, T_lat, 16, 60, 104]` | `[B, F, 16, 60, 104]` | denoise step 之间重新加噪时替代 `torch.randn_like` |
| `sim_mask` | point / foreground masks 下采样 | `[B, T_lat, 60, 104]` | `[B, F, 60, 104]` | 在 `mask_dropin_step` 用 `torch.where` 把背景替换成 sim-noised latent |
| `sim_franka_mask` | mesh / manipulator masks 下采样 | `[B, T_lat, 60, 104]` | `[B, F, 60, 104]` | 在 `franka_step` 对 manipulator 区域做较弱 SDEdit 约束 |
| `kv_cache` | 前序 block 的 clean generated latents | 每层 `k/v: [B, 32760, 12, 128]`，默认 local window 21 latent frames | 持续更新 | Causal self-attention 缓存，让后续 block 看到过去生成内容 |

注意：Genesis optical flow 本身没有作为显式 flow map / control token 送进 Wan。它主要通过 `structured_noise` 和 `sde_noise` 进入 diffusion 过程。

## 4. `y` 的具体结构

`WanVideoUnit_ImageEmbedderVAE` 会把 first frame 构造成一个 Wan I2V 条件视频：

```text
input_image: [B, 3, 480, 832]
vae_input:   [3, T_pix, 480, 832]     # 第 1 帧是真图，后面是 0
vae latent:  [16, T_lat, 60, 104]
mask:        [4, T_lat, 60, 104]
y:           [1, 20, T_lat, 60, 104]
```

这 20 个 channel = 4 个 mask channel + 16 个 first-frame VAE latent channel。

在 block generation 里：

```text
full_y = y.permute(0, 2, 1, 3, 4)       # [B, T_lat, 20, 60, 104]
curr_y = full_y[:, start:start + 3]      # [B, 3, 20, 60, 104]
```

`WanI2VDiffusionWrapper` 再把它 permute 回 `[B, 20, F, 60, 104]` 交给 Wan model。Wan model 内部执行：

```python
x = torch.cat([noisy_latent, y], dim=channel)
```

因此 I2V 图像条件的一部分是以 channel concat 方式进入 patch embedding，而不是 cross-attention。

## 5. SDEdit / flow-warp noise 注入方式

RealWonder 当前物理控制的核心不是“把 flow tensor 喂给 transformer”，而是：

```text
Genesis simulation / RAFT flow
  -> warp Gaussian noise over time
  -> structured_noise, sde_noise
Genesis simulation RGB
  -> Wan VAE encode
  -> sim_latent
sim_latent + structured_noise + timestep
  -> SDEdit initial noisy_input
```

FlowMatch scheduler 的加噪公式是：

```text
x_t = (1 - sigma_t) * z_sim + sigma_t * eps_struct
```

其中：

- `z_sim = sim_latent`
- `eps_struct = structured_noise`
- `sigma_t` 由当前 `denoising_step_list[0]` 映射得到

每个 denoise step 中，Wan 预测 flow / velocity：

```text
flow_pred = Wan(x_t, timestep, prompt, first-frame conditions, KV cache)
x0_pred = x_t - sigma_t * flow_pred
```

如果还没有到最后一步，pipeline 会重新加噪到下一个 timestep：

```text
x_next = add_noise(x0_pred, sde_noise or random_noise, next_timestep)
```

这意味着 flow-warped noise 有两个入口：

1. 初始 `x_t` 的 motion prior。
2. denoise step 之间的 SDE noise source。

## 6. Mask drop-in 的含义

如果 `mask_dropin_step > 0`，pipeline 会提前准备一个背景用的 `bg_noise`：

```text
bg_noise = add_noise(sim_latent, noisy_input, mask_dropin_timestep)
```

到指定 denoise index 时：

```python
noisy_input = torch.where(sim_mask.unsqueeze(2), noisy_input, bg_noise)
```

代码注释里 `sim_mask=True` 表示 object region 保持生成结果，`False` 表示 background 被替换成 simulation-noised latent。因此 background 更贴近模拟帧，object 区域保留模型生成自由度。

`sim_franka_mask` 类似，但作用在 manipulator / mesh 区域，通常用较晚 step 做弱 SDEdit。

## 7. Causal block 和 KV cache

Wan latent 被 patchify：

```text
[B, 36, F, 60, 104]
  -- Conv3d patch_size=(1,2,2), stride=(1,2,2) -->
[B, hidden_dim, F, 30, 52]
  -- flatten -->
[B, F * 1560, hidden_dim]
```

RealWonder streaming 每次生成 3 个 latent frames，然后用预测出的 clean latent 再调用一次 generator，`timestep=context_noise`，把 clean context 写入 KV cache。后续 block 的 causal self-attention 通过 `current_start = current_start_frame * 1560` 接上时间位置。

默认 `local_attn_size=21`，所以 local KV window 正好覆盖 21 个 latent frames：

```text
21 * 1560 = 32760 tokens
```

## 8. 对替换成 Helios 的接口启示

直接替换不是“把 Wan 类名换成 Helios 类名”这么简单。RealWonder 依赖 Wan 的几个接口事实：

1. Wan VAE latent 布局是 `[B, T, C, H, W]`，但 Wan transformer 内部吃 `[B, C, T, H, W]`。
2. I2V first-frame condition 分成两路：CLIP image tokens 走 cross-attention，VAE+mask `y` 走 channel concat。
3. 物理 flow 不显式进 transformer，而是先变成 flow-warped noise，再通过 SDEdit 初始化和 step 间 re-noise 控制采样。
4. Streaming 依赖 3 latent frames / block、Wan VAE temporal cache、以及 causal KV cache。

Helios 文档显示它的 Stage-1 更偏 offloaded latent dataset：

```text
x0_latents
history_latents
target_latents
prompt_embeds
```

因此替换时最需要对齐的是：

- Helios 是否支持等价的 first-frame image condition；如果没有，需要把 RealWonder 的 `clip_feature + y` 映射成 Helios 的 `x0_latents/history_latents`。
- Helios 的 denoising / flow matching 入口是否允许自定义 noise；如果允许，RealWonder 的 `structured_noise` 可以作为 `custom_noise` 注入。
- Helios 是否有 block-causal KV / memory 接口；如果没有，RealWonder interactive streaming 的低延迟结构需要重做。
- Helios latent temporal grid 与 Wan VAE 是否一致；如果 Helios 仍用 Wan VAE，`T_pix -> T_lat` 对齐成本会低很多。

一个保守迁移顺序：

```text
Phase 1: 先离线 batch 对齐 prompt + first frame + sim_latent，不做交互 streaming。
Phase 2: 加 flow-warped custom_noise，看 Helios flow matching 是否能复现 RealWonder motion control。
Phase 3: 再做 block streaming / history memory，对齐 RealWonder 的 3-latent-frame block。
```

## 9. 代码入口索引

- `RealWonder/demo_web/video_generator.py`: interactive streaming 的 setup、precompute first frame、per-block generation。
- `RealWonder/vidgen/pipeline_sdedit.py`: 完整 SDEdit pipeline，包括 full-video causal loop。
- `RealWonder/vidgen/models.py`: Wan text/image/VAE/diffusion wrapper。
- `RealWonder/vidgen/utils.py`: `load_noise`、first-frame preprocess、Wan I2V processor units、FlowMatch scheduler。
- `RealWonder/demo_web/noise_warper_stream.py`: streaming flow-warped noise 的 block 输出。
- `RealWonder/simulation/image23D/noise_warp/noise_warp.py`: offline noise-warp 输出 `noises.npy` / `flows.npy`。
- `RealWonder/wan/modules/causal_model.py`: Causal Wan patchify、time modulation、KV cache、causal attention。
- `RealWonder/wan/modules/model.py`: Wan I2V/T2V cross-attention 和 `y` channel concat 逻辑。
