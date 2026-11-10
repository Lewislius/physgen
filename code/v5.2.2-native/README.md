# V5.2.2 Native

当前使用 **v25–v28：H / Q128 × 冻结 verifier / verifier LoRA**。先按下面四行找入口和权重；旧实验统一放在 [archive/20261027_cleanup](archive/20261027_cleanup)。

## 四套当前方案与 checkpoint

2026-10-27 核对了 Determined 状态，并实际读取 `latest.pt` 的版本和步数：

| 方案 | 训练脚本 | 主线 checkpoint | 保存步数 / 总预算 | 任务状态 |
|---|---|---|---|---|
| v25 · H + 冻结 verifier | [H frozen](train/train_h_content_message_verifier_frozen_blk96g.sh) | [66356/latest.pt](checkpoints/v522-h-content-message-verifier-frozen-simple_66356/latest.pt) | 224 / 224 | 66356 已完成 |
| v26 · H + verifier LoRA | [H LoRA](train/train_h_content_message_verifier_lora_blk96g.sh) | [66355/latest.pt](checkpoints/v522-h-content-message-verifier-lora-simple_66355/latest.pt) | 已验证保存 66 / 256，继续训练 | Search 66419 / Run 67799，RTX 单卡，继续写 66355 目录 |
| v27 · Q128 + 冻结 verifier | [Q frozen](train/train_q_seed128_message_verifier_frozen_blk96g.sh) | [66353/latest.pt](checkpoints/v522-q-seed128-message-verifier-frozen-simple_66353/latest.pt) | 224 / 224 | 66353 已完成 |
| v28 · Q128 + verifier LoRA | [Q LoRA](train/train_q_seed128_message_verifier_lora_blk96g.sh) | [66384/latest.pt](checkpoints/v522-q-seed128-message-verifier-lora-simple_66384/latest.pt) | 已验证保存 66 / 256，继续训练 | Search 66418 / Run 67798，RTX 单卡，继续写 66384 目录 |

H 保留完整内容长度，Q 固定 128 个槽位。四套都有 verifier。冻结版为 64 interface + 160 joint；LoRA 版另有 32 calibration；全部 accumulation=4。完成训练不代表已经证明生成质量或物理效果改善。

**恢复任务沿用原 checkpoint 目录，所以任务 ID 不一定等于目录后缀。** H 的 66385、66419 均继续写 66355；Q 的 66418 继续写 66384。`latest.pt` 包含模型、optimizer、scheduler、RNG 和配置；同目录的 `metrics.jsonl` 是日志，`config.yaml` 是这次运行实际使用的配置。[本次恢复记录](analysis/recovery_lora_20261027/current_jobs.json)。

本次对照确认：两个 frozen 任务在 RTX PRO 6000 上完成 224 步；失败的 66384/66385 的六次分配均为同一 H100，退出码 137、cgroup OOM 计数为 0，信号发送者未确定。Q 首次失败发生在保存 interface 第 32 步后，尚未开始 verifier LoRA 校准。恢复任务保留原训练配方，拒绝已记录失败的 GPU UUID，每步保存并开启执行跟踪。两者已完成至少两次真实校准更新：仅 verifier 梯度非零，96 个 adapter 张量均已更新，checkpoint 中 Adam 步数为 2；峰值分别约 Q 42.35 / H 41.60 GiB。[实际 GPU 与 checkpoint 验证](analysis/recovery_lora_20261027/verification.json)。24 项针对性测试通过，包含新增 H/Q 四微批累积、连续两次更新与关闭 checkpoint 的数值对照；这些测试不替代完整 256 步训练。[适配、视觉输入、梯度归属与重算代码审查](analysis/recovery_lora_20261027/code_audit.json)。

新训练共同使用 [checkpoints/initialization/generator.pt](checkpoints/initialization/generator.pt) 初始化。这是迁移起点，不能当作上表四套训练后的成品。

## 训练、恢复和推理

在本目录运行；四个脚本默认只做资源预检，传入 `--submit` 才提交 blk-96g 单卡作业：

```bash
bash train/train_h_content_message_verifier_frozen_blk96g.sh --check
bash train/train_h_content_message_verifier_lora_blk96g.sh --check
bash train/train_q_seed128_message_verifier_frozen_blk96g.sh --check
bash train/train_q_seed128_message_verifier_lora_blk96g.sh --check

# 新建训练，以 H frozen 为例。
bash train/train_h_content_message_verifier_frozen_blk96g.sh --submit
```

`configs/*_simple_wisa3857.yaml` 定义模型、损失、学习率；`train/*_simple_1x96g.yaml` 定义 Determined 资源和启动命令。共享训练实现是 [train/train.py](train/train.py)。

当天正在使用的恢复入口是 [H 从 64 步恢复](train/native_h_content_message_verifier_lora_simple_resume64_20261027_1x96g.yaml) 和 [Q 从 64 步恢复](train/native_q_seed128_message_verifier_lora_simple_resume64_20261027_1x96g.yaml)。另保留两份 10 月 26 日恢复 YAML 供已有记录引用，其中 Q 的 `*_recovery_1x96g.yaml` 只会在同一实验重启时自动恢复，重新提交成新实验则从初始化开始。后续如需接续已保存的 66384，可在该运行停止后、已分配 GPU 的环境中运行：

```bash
bash train/train_q_seed128_message_verifier_lora_blk96g.sh --local \
  --resume checkpoints/v522-q-seed128-message-verifier-lora-simple_66384
```

生成视频时显式指定上表的 checkpoint，例如：

```bash
bash tools/runtime_python.sh inference/infer_heldout.py \
  --checkpoint checkpoints/v522-h-content-message-verifier-frozen-simple_66356/latest.pt \
  --samples 1 --seed 42
```

[完整模型、梯度、参数与评估说明](docs/message_reasoning.md)。

H frozen（66356）和 Q frozen（66353）最终第 224 步权重的 **21 个 demo 推理入口**见 [inference/README.md](inference/README.md)，其中提供各自的 SH 和 amp-48g / blk-96g 单卡 YAML。当前按用户选择使用 amp-48g；按训练设置使用 H 1 轮、Q 2 轮，全部样例 seed 固定为 checkpoint 的训练 seed 42。

## 目录用途

| 目录 | 放什么 |
|---|---|
| `train/` | 四套当前脚本、四份提交 YAML、四份恢复 YAML（含当天新任务）、共享脚本和 train.py，共 14 个文件 |
| `checkpoints/` | 上表四组主线权重、`initialization/` 初始化权重、`recovery_logs/` 恢复日志 |
| `analysis/recovery_simple_blk96g_20261026/` | v25–v28 阶段验证、故障与恢复记录；其中状态是当时的快照，以本页核对日期为准 |
| `analysis/simple_verifier_v25_v28_20261026/` | 当前四套配方的资源预检记录 |
| `analysis/videophy40_20261026/` | VideoPhy40 评测清单和准备记录 |
| `analysis/recovery_lora_20261027/` | 当天 LoRA 恢复诊断、64 步备份和 66418/66419 任务记录，保留原路径 |
| `analysis/demo21_final_20261027/`、`analysis/output_review_20261027/` | 并行工作新生成的推理与结果检查资料 |
| `analysis/queue_20261026_fullwidth_attention` | 指向归档的兼容链接，保证旧任务记录中的初始化路径仍可读取 |
| `inference/`、`inference_outputs/` | 推理/评估代码与生成的视频 |
| `physgen_native/`、`tests/` | 训练推理实现和已有回归测试 |
| `configs/` | 当前四份 simple 配置，以及旧 checkpoint 和回归测试仍使用的版本配置 |
| `tools/` | 运行环境、恢复、诊断与评估工具；历史输入路径已指向归档 |
| `docs/`、`fig/` | 设计说明和图示；旧版本文档按其版本号阅读 |
| `determined_checkpoints/` | Determined 配置使用的存储目录，目前为空；实际模型权重在 checkpoints |
| `archive/20261027_cleanup/` | 历史 analysis、36 份旧训练入口、33 个旧 checkpoint/失败记录目录，以及整理记录 |

## 这次整理了什么

- 历史实验移入 archive；主线代码、四组权重、生成视频与外部数据缓存保持原样。
- 删除早期 smoke 任务 **65825、65828、65830、66341、66342、66343、66344** 的 19 个 `.pt` 文件，共 **12.93 GiB**。这些包括短测权重、对齐初始化和短测 evidence 张量；配置、日志与 teacher JSON 保留。后续四套成功阶段测试 66348/66350/66349/66351 的权重仍在归档。
- 初始化权重移到 `checkpoints/initialization/`，更新当前启动路径，并保留旧路径兼容链接。归档目录已加入 `.detignore`，提交训练时不会上传历史产物。

[删除及移动清单](archive/20261027_cleanup/cleanup_manifest.json) 记录每项旧路径和新路径；历史日志和不可变代码快照保留当时的路径，可按清单定位。[整理前的完整 README](archive/20261027_cleanup/README_before_cleanup.md) 保留历史论证与实验说明。修改前的文本源文件另存为 [sources_before_cleanup.tar.gz](archive/20261027_cleanup/sources_before_cleanup.tar.gz)；该备份不包含已删除的 smoke 张量。
