# R18A / R18B / R18C / R18D

官方 fresh 初始化、**400 次 optimizer update**、有效 batch 4 的四个独立实现。

| R18 | v9 方案 | 主线 |
|---|---|---|
| R18A | PLAN A | independent joint spatio-temporal expert |
| R18B | PLAN B | in-block LoRA cooperative execution |
| R18C | PLAN C | semantic-enhanced physical memory then physical-value executor |
| R18D | PLAN D | native text update guides physical propagation/transformation |

模型结构、站点、rank、初始化、损失函数形式、推理协议严格沿用 v9 ABCD
设计；只按 2026-11-11 的 400 步要求替换训练日程和 raw-route 系数。

## 400 步唯一训练日程

所有允许参数从 update 1 到 400 一直可训练；没有阶段解冻、后训练、ST Loss 或额外 view。

| 阶段 | update | 学习率乘子 | raw-route λ |
|---|---:|---|---:|
| S1 联合起步 | 1–32 | s/32 线性 warmup | 0.03·s/32 |
| S2 物理条件学习 | 33–128 | 1.0 | 0.03 |
| S3 物理—语义联合生成 | 129–320 | 0.2+0.8·cosine((s−128)/192)/2 → 0.2 | 0.03−0.02·(s−128)/192 → 0.01 |
| S4 最终联合优化 | 321–400 | 0.2 | 0.01 |

损失始终为：

```text
L(s) = 0.8 FM + 0.2 tail + lambda_r(s) * L_route(raw Cr)
```

`L_route` 只约束语义增强前的 raw `Cr`，不约束 `Ce`、`C_tilde_execute`
或 `C_tilde_route`。总样本曝光为 400×4 = 1600。

## 实现内容

- `physical_content/base_bank.py`: 在任何 main LoRA 安装前引用官方冻结
  Wan Linear；Expert 只共享冻结 base，不共享 adapter。
- `physical_content/lora.py`: 严格六个视频目标、30 层；B 的十个注入层
  rank64，其余层 rank32；原生文本 Q/K/V/O 从不安装 LoRA。
- `physical_content/content.py`: fresh 256-slot R10A 结构；C 的两轮共享
  semantic-physics memory（官方 layer14 冻结 base + 独立 rank32 adapter，
  identity `Wr`）。
- `physical_content/readers.py`: 公共全宽物理 reader；官方冻结 cross Q/norm_q，
  新 dense K/V 从官方 Wk/Wv 初始化，等关系 prior，官方 Wo 权重无 bias，
  没有 null reference slot、没有旧 state_value。A/B 的 independent semantic
  reader 使用官方 cross base + 独立 rank32 adapter，仅 mask 人工 padding。
- `physical_content/experts.py`: A joint SA/FFN expert；B in-block writer；
  C zero-preserving physical-value SA/FFN executor；D zero-preserving guided
  executor，Uq/Uk/G 全宽零初始化。
- `physical_content/model.py`: 主干每层只遍历一次。A/C 先完整 native block，
  再调用 expert；B/D 显式执行 native SA → frozen text CA → fusion/injection
  → native FFN 一次；D 使用同一层、同一 CFG 臂的 text_update，不重复做
  text attention。
- `physical_content/training.py`: 单一 400-update 循环，micro1 × accum4，
  condition dropout 10%，按角色 clip，AdamW 矩阵 decay 0.01 / bias-norm 0。
- `physical_content/checkpointing.py`: fresh-only 与同 run resume；不接受任何
  旧项目 checkpoint。
- `fit_interface.py`: 只用 train 重新拟合 closed-form Qwen4096→3072 interface。
- `infer.py`, `run_candidate.py`, `prepare_inputs.py`, `reports.py`: 固定
  50-step、shift5、CFG5、seed42；40 与额外 20 分开生成，再合并 60。

## 可用训练数据说明

400×4=1600 次样本曝光使用已审计的 `qwen38_wisa_reviewed_v1` 400-step
data overlay：`data_revision_qwen38/runs/20261103_dataset/encoded`。该 overlay
选择 1600 个 train 样本，并同时提供 positive/null Qwen words、native targets 和
first-image/request evidence；因此 1600 次采样正好无放回覆盖该训练子集一次。
manifest 仍保留 3857 条记录，`TrainingData` 只使用拥有完整条件的 1600 个 train
行，不会用 null 或缺失条件冒充训练样本。报告会记录 `train_records` 与
`unavailable_train_records`。

## 运行

```bash
export WAN_CODE=/path/to/Wan2.2
export WAN_CKPT=/path/to/Wan2.2-TI2V-5B
export QWEN_CKPT=/path/to/Qwen3-VL-8B-Instruct
export TRAIN_CACHE=/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/data_revision_qwen38/runs/20261103_dataset/encoded
export RUN_ROOT=/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physics_expert_abcd_fresh1000/runs
```

```bash
python fit_interface.py --config configs/r18c.yaml
python train.py --config configs/r18c.yaml --check-only
python -m unittest discover -s tests -v
python train.py --config configs/r18c.yaml
python run_candidate.py --config configs/r18c.yaml
```

`RUN_ROOT` 必须解析到本实验目录内。训练不会自动提交；每个 YAML 是一个独立
候选。

## 明确限制

- 本实现使用默认单 GPU micro1 × accumulation4。v9 契约允许 2/4 GPU 累积，
  这里未实现分布式路径。
- 源码交付不等同于已训练或已达标；没有执行正式 400-step GPU 训练、完整视频生成或 VideoPhy2
  评分；本 README 不声称这些结果。
- R18A-D 不是 R17 改名：原生文本 LoRA 已移除；官方冻结 expert base 共享；
  A/C/D 有完整额外时空执行器。
- 本目录只实现固定 VideoPhy2 40/60 的生成与评分入口；WMB、Physics-IQ、
  VBench 的独立 runner 不在本次代码范围内，需要按其官方协议另行运行。
