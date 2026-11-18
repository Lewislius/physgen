# R19CTN：每注入层独立、同深度初始化的物理状态专家

本目录是 R19CT 的独立修订版。核心变化只有一类：**每个注入层都有自己的
PhysicalStateExpert、video_value、read_output 和 feature_output，注入层之间不共享
任何物理参数；所有 Wan 复制都来自同编号的 Wan block。**

- 注入层：`[0, 7, 14, 21, 28]`（参考 ProPhy 的五处分布式注入，相邻注入层间隔 6 层）
- 每层 state expert：同层 Wan block 的 cross_attn / self_attn / FFN / norms
- 每层 video_value：默认同层 `cross_attn.v`；`identity` 只与 `read_output=identity` 成对使用
- 每层 read_output：默认同层 `cross_attn.o`；`identity` 只与 `video_value=identity` 成对使用
- 每层 feature_output：独立、零初始化（初始精确走原生路径）
- 初始化不一致会直接抛错，不会静默训练

## 快速检查

```bash
cd /home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_r19ct_new
PY=/home/liuzhirui/miniconda3/envs/moviestory/bin/python
$PY -m compileall -q .
$PY -m unittest discover -s tests -v
```

CPU 测试覆盖：逐层 Q/K/V/O/FFN 复制来源、不同 site 参数不共享、feature_output 零初始化、
真实生成梯度进入每条新路径、检查点保存/恢复。

## 提交前

```bash
# 1) 全尺寸梯度/显存 + 初始化审计（必须通过）
$PY check_full_model.py --config configs/r19ctn.yaml \
  --output reports/full_model_check.json

# 2) 正式 400 步 + 40/60 + 配对16
$PY run_candidate.py --config configs/r19ctn.yaml
```

`check_full_model.py` 会检查：
- 每个 site 的 Q/K/V/O、norms、state expert 的 attention/FFN 是否来自同号 Wan block；
- 不同 site 的物理参数是否零共享；
- feature_output / noise_modulation 是否零初始化；
- 每条参数组是否收到真实生成梯度；
- 实际显存峰值。

预计可训练参数约 2.371B，AdamW fp32 状态估算约 35.3 GiB；估算峰值约 52.3 GiB，96 GiB 单卡有较充足余量。
如果 OOM，不要静默改回共享 expert；先按 DESIGN.zh.txt 的资源章节处理。

## 评测

`run_candidate.py` 会依次：
1. 准备/校验缓存和 interface；
2. 训练 400 步（官方 Wan 初始化，无项目权重）；
3. 生成并评分固定 40 + 补充20 / 完整60；
4. 生成并评分配对 16 的 7 组对照；
5. 生成实际视频审查材料，停在人工审查，不自动提交 WMB/Physics-IQ。

## 训练后逐层审计

```bash
$PY analysis/audit_checkpoint_initialization.py \
  --checkpoint runs/r19ctn_prophy5_state_400/step0400.pt
```

该脚本直接在真实 Wan safetensors 上比较 checkpoint 的每个 site 与同号/异号 block，
输出 best-match 和余弦；用于确认训练后的权重仍能追溯到正确深度。
