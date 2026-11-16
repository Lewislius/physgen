# R20FT on Wan2.1-T2V-1.3B

本目录把 R20FT 的完整设计适配到官方 **Wan2.1-T2V-1.3B**，所有 Wan2.2 的
3072 宽度、14336 FFN、24 头、48 通道 latent 和 16x16 VAE 假设均已删除。
详细逐项清单见 `DESIGN.zh.txt`。

## 关键参数

| 项目 | Wan2.1-T2V-1.3B |
|---|---|
| 宽度 / FFN / 头 / 层 | 1536 / 8960 / 12 / 30 |
| VAE | Wan2.1_VAE.pth，z=16，stride 4x8x8 |
| 81 帧 832x480 latent | `[1, 16, 21, 60, 104]` |
| 文本 | UMT5-XXL 4096 宽，text_len 512 |
| 采样 | 50 步、shift 5、CFG 5、24 fps |
| 注入层 | 0 / 7 / 14 / 21 / 28 |
| 视频 LoRA | 180 个 rank32/alpha32 目标 |

## 目录

```
configs/r20ft_wan21.yaml       唯一正式配置
prepare_wan21_cache.py         生成 Wan2.1 latent / native_words / inference
physical_content/              R20FT 架构（Wan2.1 参数化）
fit_interface.py               训练输入归一化（CPU）
check_full_model.py            一次性全尺寸梯度/显存预检
train.py                       400 步训练入口
run_candidate.py               训练 + 40/60 + 配对 16 总入口
infer.py                       fixed 40/60 生成
tests/                         CPU 形状/配置检查
submissions/                   Determined 提交 YAML
```

## 准备数据

```bash
cd /home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_r20f_wan2.1
PY=/home/liuzhirui/miniconda3/envs/moviestory/bin/python

# 1) base links + Wan2.1 VAE latent + 1536 native words + inference
$PY prepare_wan21_cache.py --config configs/r20ft_wan21.yaml --stage all

# 小样本冒烟（先确认 8 条 latent）
$PY prepare_wan21_cache.py --config configs/r20ft_wan21.yaml --stage vae --limit 8
```

`prepare_wan21_cache.py` 会复用现有 5B 缓存中的 `manifest.json`、4096 维
UMT5 `text` 和 `null_text`（Wan2.1 与 Wan2.2 的 T5 checkpoint sha256 相同），
但 **不会** 复用 5B 的 `base/vae` 和 3072 维 `native_words`。

## 资源分工

- **ada-48g**：缓存准备、训练输入归一化、准备 60 例 inference、全尺寸
  21×60×104 梯度/显存预检。对应 `--stage prepare-check --gpu-policy ada48`。
- **blk-96g**：正式 400 步训练、fixed 40/60 生成与评分、三组配对 16。
  对应 `--stage train-eval --gpu-policy rtx6000`。

预检成功后会把 `preparation/full_model_check.json` 写成设计绑定的 marker；
blk-96g 的 `train` 阶段会校验 marker 的 `design_hash`、模型名和 latent 形状，
不匹配或缺失时拒绝开始正式训练。

Determined 提交：

```bash
# 1) 先在 ada-48g 跑准备与全尺寸预检
det experiment create submissions/r20ft_wan21_check_ada48.yaml .

# 2) marker 通过后，再在 blk-96g 跑正式训练和评测
det experiment create submissions/r20ft_wan21_train_blk96g.yaml .
```

也可以手动分阶段执行：

```bash
$PY run_candidate.py --config configs/r20ft_wan21.yaml \
    --stage prepare-check --gpu-policy ada48
$PY run_candidate.py --config configs/r20ft_wan21.yaml \
    --stage train-eval --gpu-policy rtx6000
```

其他可用阶段：`prepare`、`check`、`train`、`eval`、`all`。

## 检查

```bash
$PY -m compileall -q .
$PY -m unittest discover -s tests -v
```

上述检查只证明配置、形状和接口一致，不证明生成质量；最终仍以正式 400 步后的
fixed 40/60 SA/PC/Joint 和实际视频审查为准。
