# RealWonder 内置 Case 的 Genesis Simulation 测试报告

日期：2026-05-08

## 结论

当前已经在 RealWonder 仓库内跑通六个内置 case 的 Genesis simulation smoke 输出，并生成可打开的 `simulation.mp4`。

本次输出路径：

```text
result/genesis_builtin_case_outputs/
```

合并索引：

```text
result/genesis_builtin_case_outputs/all_cases_latest_summary.json
```

## 已生成视频

| case | 帧数 | 分辨率 | simulation.mp4 |
| --- | ---: | --- | --- |
| lamp | 81 | 512x512 | `result/genesis_builtin_case_outputs/lamp/08-05_22-51-50/final_sim/simulation.mp4` |
| persimmon | 81 | 512x512 | `result/genesis_builtin_case_outputs/persimmon/08-05_22-42-43/final_sim/simulation.mp4` |
| sand_house | 165 | 512x512 | `result/genesis_builtin_case_outputs/sand_house/08-05_22-43-35/final_sim/simulation.mp4` |
| santa_cloth | 81 | 512x512 | `result/genesis_builtin_case_outputs/santa_cloth/08-05_22-44-41/final_sim/simulation.mp4` |
| tree | 81 | 512x512 | `result/genesis_builtin_case_outputs/tree/08-05_22-46-19/final_sim/simulation.mp4` |
| two_duck | 81 | 512x512 | `result/genesis_builtin_case_outputs/two_duck/08-05_22-47-02/final_sim/simulation.mp4` |

每个 `final_sim/` 下还包含：

```text
frames/
metadata.json
state_tracks.json
simulation.mp4
```

其中 `state_tracks.json` 记录每帧 Genesis entity 的位置，用于确认视频确实由 Genesis `scene.step()` 后的状态驱动。

## 运行命令

激活环境：

```bash
source scripts/activate_realwonder.sh
```

生成全部内置 case：

```bash
python -u scripts/export_builtin_case_genesis_smoke.py \
  --case all \
  --fps 10 \
  --backend gpu \
  --resolution 512 \
  --output_root result/genesis_builtin_case_outputs
```

如果只跑单个 case：

```bash
python -u scripts/export_builtin_case_genesis_smoke.py \
  --case lamp \
  --fps 10 \
  --backend gpu \
  --resolution 512 \
  --output_root result/genesis_builtin_case_outputs
```

## 当前模式说明

本次跑通的是 `Genesis scene.step() + state2d renderer` smoke 路径：

- 使用 Genesis GPU 后端构建场景、施加 case motion、执行 `scene.step()`。
- 不调用 SAM2/SAM3D/MoGe/RAFT/diffusion。
- 不调用 Genesis OpenGL camera `cam.render()`。
- 视频帧由 Genesis entity 位姿投影为轻量 2D 状态画面。

这样可以验证底层 Genesis simulation loop、case 批处理、视频输出结构和后续数据落盘路径。

## 原始 RealWonder 路径的当前阻塞

原始命令：

```bash
python -u case_simulation.py \
  --config_path cases/lamp/config.yaml \
  --device cuda \
  --genesis_backend gpu \
  --skip_noise_warp
```

当前会在 image-to-3D 阶段停止：

```text
FileNotFoundError:
submodules/sam_3d_objects/checkpoints/hf/pipeline.yaml
```

原因是 SAM3D Objects checkpoint 属于 gated HuggingFace 资源，本机目前没有可用 token/授权权重。SAM2 large checkpoint 已经存在：

```text
submodules/sam2/checkpoints/sam2.1_hiera_large.pt
```

另外，Genesis Rasterizer 的离屏相机渲染 `cam.render()` 在当前 Blackwell/EGL 环境会 segfault；RayTracer 后端也因为未编译 `LuisaRenderPy` 不可用。因此本次 smoke 路径刻意避开 Genesis camera renderer，只验证 simulation 状态与结构化输出。

## 后续建议

1. 如果要跑原始 RealWonder image-to-3D，需要提供 SAM3D gated checkpoint，并放到 `submodules/sam_3d_objects/checkpoints/hf/`。
2. 如果要拿真实 Genesis RGB camera render，需要修复当前 EGL Rasterizer segfault，或编译 LuisaRender/LuisaRenderPy 后切 RayTracer。
3. 在这两个外部条件齐之前，`export_builtin_case_genesis_smoke.py` 可作为后续训练数据流的本地可运行 demo 框架。
