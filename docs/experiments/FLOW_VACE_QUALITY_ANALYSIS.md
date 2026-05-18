# FlowVideo + VACE14B 质量诊断

日期：2026-05-11

## 已检查产物

RealWonder 导出的 VACE flow 条件视频：

- `data_exports/vace_flow/lamp/flow_vis.mp4`
- `data_exports/vace_flow/tree/flow_vis.mp4`
- `data_exports/vace_flow/santa_cloth/flow_vis.mp4`
- `data_exports/vace_flow/persimmon/flow_vis.mp4`
- `data_exports/vace_flow/two_duck/flow_vis.mp4`
- `data_exports/vace_flow/sand_house/flow_vis.mp4`

VACE14B 输出：

- `../VACE/results/realwonder_flow/lamp/20260509_044113/out_video.mp4`
- `../VACE/results/realwonder_flow/tree/20260510_152748/out_video.mp4`
- `../VACE/results/realwonder_flow/santa_cloth/20260510_154316/out_video.mp4`
- `../VACE/results/realwonder_flow/persimmon/20260510_155823/out_video.mp4`
- `../VACE/results/realwonder_flow/two_duck/20260510_161324/out_video.mp4`
- `../VACE/results/realwonder_flow/sand_house/20260510_162839/out_video.mp4`

成功的 VACE run 均为 `Wan2.1-VACE-14B`，`832*480`，`81` 帧，`sample_steps=50`，单进程运行，耗时约 `881-898s/case`。成功目录中的 `gpu_monitor.err` 为空，未看到 VACE 运行时错误。

## 关键结论

当前结果不好，不是单一模型失败，而是条件信号链路不够完整：

1. `two_duck` 的后半段静止已经存在于 RealWonder/Genesis 源模拟和源 flow 中，不是 VACE 才造成的。
2. `tree` 的模拟画面持续变化，但导出的 Genesis flow 有效区域很小，传给 VACE 的控制信号弱且容易被 RGB flow 可视化放大/扭曲。
3. VACE 官方 flow control 接口吃的是经过 RAFT annotator 生成的 RGB 光流可视化视频；当前输入是 Genesis 几何光流转 RGB，可用作接口输入，但分布和官方预期不同。
4. 当前未提供 `src_ref_images`，`src_mask` 也是全白全图 mask。VACE 只看到光流颜色视频和 prompt，没有真实外观参考，也没有前景/背景区域约束。
5. RealWonder 原流程里 `NoiseWarper` / `noises.npy` 是后续扩散一致性的重要中间量；当前为导出数据设置了 `noise_flow_source=none`，这会跳过原 RealWonder 的噪声 warp 环节。

## Flow 源数据诊断

读取 `genesis_flows_480x832.npy` 后得到：

| case | flow shape | mean_mag 走势 | 备注 |
| --- | --- | --- | --- |
| `two_duck` | `(80, 2, 480, 832)` | 三段均值约 `[0.2361, 0.0001, 0.0001]` | 第 30 帧后几乎全零；`mag>0.05` 的像素比例后两段为 `0` |
| `tree` | `(80, 2, 480, 832)` | 三段均值约 `[0.0450, 0.0947, 0.0382]` | 运动幅度整体小，`mag>0.05` 像素比例约 `4.9%` |
| `sand_house` | `(164, 2, 480, 832)` | 三段均值约 `[2.9588, 1.3161, 0.0125]` | RealWonder 有 165 帧，但 VACE 结果只生成 81 帧 |

渲染帧差异进一步确认：

- `two_duck` 的 `simulation.mp4` 第三段帧差均值约 `0.0034`，已经接近静止。
- `tree` 的 `simulation.mp4` 第三段帧差均值约 `2.1617`，说明画面还在变，但导出的 flow 只覆盖小区域。

## 接口和改动检查

VACE 官方 flow 预处理逻辑在 `vace/vace_preproccess.py` 中调用 `FlowVisAnnotator`，由 RAFT 对 RGB 视频逐帧估计光流，再保存 `src_video-flow.mp4`。当前导出脚本 `scripts/export_vace_flow_condition.py` 是把 Genesis raw flow 直接可视化成 RGB flow video，并补一帧首帧以匹配 VACE 长度习惯。

这个接口方向没有接错：VACE inference 最终确实接收 `src_video`。但质量上有三个缺口：

- 当前 `flow_to_image()` 按每帧自身最大 flow 做归一化，绝对速度信息被破坏，弱运动可能被放大，强运动和弱运动之间的时序尺度不稳定。
- 没有传入对象 mask；`WanVace.prepare_source()` 在无 `src_mask` 时会生成全图 mask，保存的 `src_mask.mp4` 也确认为全白。
- 没有外观参考图；这使 VACE 只根据 prompt 合成物体和背景，Genesis simulation 的具体物体外观没有被锁住。

RealWonder 当前代码改动主要是 Blackwell/本地权重适配和数据导出：`case_simulation.py` 增加了设备、backend、flow/noise 导出参数；`genesis_simulator.py` 增加 CPU/GPU backend 选择；`single_view_reconstructor.py` 增加本地 MoGe 权重路径和 device 修正。`two_duck.py` 和 `tree.py` case handler 未改动。

## 当前欠缺

要让 “Genesis flow -> VACE14B 真实视频” 稳定可用，缺的不是再降低质量的 fallback，而是以下补齐：

1. 需要先验证每个 case 的 Genesis simulation 本身是否满足 motion 目标；`two_duck` 目前源模拟后半段静止，VACE 不可能凭 flow 生成持续碰撞运动。
2. 需要一个更严谨的 Genesis flow 到 VACE flow RGB 的规范：固定或分位数全局尺度、可选 clip、记录 scale，避免逐帧归一化导致控制信号漂移。
3. 需要前景 mask 或区域条件，避免把物体光流条件作为全图条件喂给 VACE。
4. 需要给 VACE 提供外观参考，例如 RealWonder 的 `resized_input_image.png` 或 `first_frame.png` 作为 `src_ref_images`，否则只靠 flow 和 prompt 生成真实视频会欠约束。
5. 需要恢复或并行测试 RealWonder 原流程里的 `NoiseWarper`/RAFT/noises 数据流，判断它和 VACE flow control 谁更适合作为后续训练/生成接口。
