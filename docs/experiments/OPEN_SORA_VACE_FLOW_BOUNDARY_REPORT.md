# Open-Sora FlowVideo + VACE14B 性能边界实验

日期：2026-05-11

## 目标

用 `Open-Sora-Plan-v1.1.0/all_mixkit` 中 10 个不同运动类型的视频，测试：

```text
真实视频 -> RAFT flow video -> VACE14B -> 预测视频
```

核心问题不是看 VACE 能否生成“好看视频”，而是判断它在 **只给 prompt + 光流可视化视频** 时，是否能可靠遵守光流运动。

## 实验目录

```text
experiments/vace_flow_boundary_20260511/
  manifest.json
  samples/<id>/source_81f_832x480.mp4
  raft_raw/<id>/raft_flows.npy
  flows/<id>/flow_vis.mp4
  vace_outputs/<id>/<timestamp>/out_video.mp4
  comparisons/<id>_source_flow_vace.mp4
  comparisons/sheets/contact_sheet_mid.png
  comparisons/sheets/contact_sheet_start_mid_end.png
  metrics_summary.json
```

对比视频是三列布局：

```text
source | flow condition | VACE14B
```

## 样本

| id | 运动类型 | 输入 flow 特征 |
| --- | --- | --- |
| `airplane_takeoff` | 弱运动/远景飞机 | flow 极弱，几乎只有小区域运动 |
| `urban_cyclist` | 人/车局部大运动 | 前景人体/车辆运动明显 |
| `bird_flock` | 小目标群体运动 | 全局天空 + 小目标群体 |
| `cars_road` | 俯视车辆高速运动 | 大位移、清晰方向 |
| `black_cat` | 近景低纹理动物 | 细节/眼部局部变化 |
| `night_traffic` | 夜景车流 | 中低幅度、全局稳定 |
| `contemporary_dance` | 人体非刚体 | 关节/轮廓变化复杂 |
| `playful_dog` | 动物 + 玩具交互 | 大幅局部非刚体/遮挡 |
| `fish_swimming` | 水下动物 | 多目标/流体背景 |
| `waving_fire` | 高频火焰 | 最大幅度、高频非刚体 |

## 运行配置

- 模型：`Wan-AI/Wan2.1-VACE-14B`
- checkpoint：`/root/autodl-tmp/Physics_worldmodel/VACE/models/Wan2.1-VACE-14B`
- 分辨率：`832*480`
- 帧数：`81`
- FPS：`16`
- steps：`50`
- 并行：单进程、单卡、无 FSDP
- GPU：物理 GPU1，`CUDA_VISIBLE_DEVICES=1`
- 单样本耗时：`884-903s`
- 峰值显存：约 `96629 MiB`

## 结果汇总

| id | input mean | input p95 | gen mean | active EPE | active cos | gen/input | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| airplane_takeoff | 0.018 | 0.054 | 0.029 | 0.579 | 0.096 | 1.60 | 889 |
| urban_cyclist | 2.538 | 13.094 | 2.753 | 1.462 | 0.903 | 1.08 | 889 |
| bird_flock | 5.025 | 5.964 | 1.165 | 4.072 | 0.846 | 0.23 | 888 |
| cars_road | 5.282 | 40.656 | 5.485 | 7.618 | 0.869 | 1.04 | 889 |
| black_cat | 2.688 | 3.567 | 2.129 | 1.853 | 0.937 | 0.79 | 888 |
| night_traffic | 0.712 | 1.728 | 0.896 | 0.736 | 0.961 | 1.26 | 889 |
| contemporary_dance | 0.955 | 5.582 | 1.018 | 5.185 | 0.682 | 1.07 | 903 |
| playful_dog | 6.582 | 31.164 | 6.608 | 10.403 | 0.794 | 1.00 | 884 |
| fish_swimming | 1.864 | 7.188 | 1.597 | 2.852 | 0.532 | 0.86 | 892 |
| waving_fire | 50.788 | 103.238 | 90.887 | 107.069 | 0.098 | 1.79 | 892 |

说明：

- `active cos` 是在输入 flow 有效区域上的方向余弦均值，越接近 1 表示方向越一致。
- `active EPE` 不是严格评测真值，因为 VACE 没有被要求重建 source；它只是反映“输出视频重新提取的 RAFT flow 是否和输入 flow 同向/同量级”。

## 观察

### VACE 能做到的部分

1. **接口稳定**：10/10 成功，无 VACE 崩溃。
2. **能按 prompt 生成同类视频**：飞机、鸟群、车流、猫、舞蹈、狗、鱼、火焰都生成了对应语义。
3. **中等结构化运动有一定跟随能力**：`urban_cyclist`、`cars_road`、`black_cat`、`night_traffic` 的 active cos 较高，说明输出运动方向和输入 flow 有一定相关性。

### 明显边界

1. **flow-only 严重欠约束**  
   VACE 只拿到 RGB flow video + prompt，不拿原视频首帧、不拿外观参考、不拿 mask，因此它会重新想象场景。结果常常“像这个类别”，但不是原视频的延续。

2. **弱 flow 会被 prompt 主导**  
   `airplane_takeoff` 输入 flow 几乎空白，VACE 仍生成了机场跑道和飞机，说明弱运动条件下模型主要靠 prompt 和先验。

3. **群体小目标/非刚体难以精确遵守**  
   `bird_flock` 生成了鸟群但运动幅度被压低；`fish_swimming` active cos 只有 `0.532`，水下小目标和背景都被重新组织。

4. **高频非刚体是明显失败边界**  
   `waving_fire` active cos 只有 `0.098`，gen/input 幅度比 `1.79`，说明 VACE 会生成“像火焰的动图”，但没有可靠遵守输入火焰 flow。

5. **人体/动物交互仍不可靠**  
   `contemporary_dance` 和 `playful_dog` 看起来语义正确，但外观、姿态、接触关系和原视频不同。对后续 Genesis 物理交互来说，单靠 flow video 不够。

## 结论

VACE14B zero-shot flow control 的边界很明确：

```text
它能把 flow video 当成“运动风格/方向提示”，
但不能把它当作严格的物理运动约束。
```

因此，如果目标是：

```text
Genesis 物理 flow + prompt -> 高质量真实视频
```

当前 zero-shot VACE 只能作为 baseline，不能直接作为最终方案。必须补充：

1. 首帧或参考图，锁定物体外观；
2. foreground/object mask，限制 flow 作用区域；
3. coarse RGB/depth/preview，提供遮挡和空间结构；
4. raw flow scale normalization，避免 RGB flow 可视化丢失绝对幅度；
5. 用真实视频自监督微调，让模型学会从 flow 重建真实视频。

## 可复现命令

样本抽取：

```bash
python scripts/open_sora_flow_boundary_samples.py
```

RAFT flow 导出：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/export_video_raft_flows.py \
  --video experiments/vace_flow_boundary_20260511/samples/<id>/source_81f_832x480.mp4 \
  --output_dir experiments/vace_flow_boundary_20260511/raft_raw/<id> \
  --device cuda:0 --raft_version large --max_frames 81 --resize 480 832
```

VACE 条件视频：

```bash
python scripts/flow_npy_to_vace_condition.py \
  --flow_npy experiments/vace_flow_boundary_20260511/raft_raw/<id>/raft_flows.npy \
  --output_dir experiments/vace_flow_boundary_20260511/flows/<id> \
  --source_video experiments/vace_flow_boundary_20260511/samples/<id>/source_81f_832x480.mp4 \
  --prompt_file experiments/vace_flow_boundary_20260511/samples/<id>/prompt.txt \
  --fps 16
```

VACE14B 批处理：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_open_sora_vace14b_batch.py
```

对比视频：

```bash
python scripts/make_vace_flow_comparisons.py \
  --manifest experiments/vace_flow_boundary_20260511/manifest.json \
  --vace_root experiments/vace_flow_boundary_20260511/vace_outputs \
  --output_dir experiments/vace_flow_boundary_20260511/comparisons
```
