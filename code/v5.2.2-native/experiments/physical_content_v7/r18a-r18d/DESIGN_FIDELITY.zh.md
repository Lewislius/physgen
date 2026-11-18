# R18A-D 设计—代码逐项核对

本文件只核对代码与 v9 四份设计文档、以及 2026-11-11 的 400 步修订。
它不是训练结果或效果声明。

## 1. 四版共同部分

| 设计 | 代码 |
|---|---|
| 官方 Wan/Qwen/T5/VAE 初始化；禁止旧项目 checkpoint | `checkpointing.py` 只接受 `REVISION`/同 run；`common.py` 强制 `official_fresh` |
| Wan 原生 base 全冻结 | `OfficialBaseBank.assert_frozen`；optimizer 不包含 base |
| 原生文本 Q/K/V/O 不安装 LoRA | `lora.py` 只替换 `self_attn` 与 `ffn`；`assert_native_text_frozen` |
| 30 层、六目标、每个目标唯一 main LoRA | `install_video_lora` 校验 180 个唯一目标 |
| A/C/D main rank32；B 注入 10 层 rank64、其余 rank32 | `install_video_lora` 使用 `rank64_layers` |
| 400 updates、有效 batch 4 | `configs/r18*.yaml`；`training.py` micro1×accum4 |
| 所有允许参数从 update 1 全程可训练 | `set_phase` 不做解冻；`optim_groups` 只包含 learned parameters |
| 无后训练、无额外 view、无 ST loss | `common.py` 拒绝 `post_training`/`extra_training_views`；`loss` 只含 FM/tail/route |
| 10% 条件 dropout 同时 null 文本与 Qwen C | `training.py` 的 `drop` 同时选择 null evidence 与 null text |
| Writer 零初始化、未来 token only、倍率 1.0 | `experts.Writer`；所有 variant 只在 future 位置写 |
| 条件 dropout 不跨 optimizer update 缓存 | 每个 micro 重新 `Teacher.input`；C 只在推理缓存 |
| 50-step shift5 CFG5 seed42 | `infer.py` |

## 2. 400 步 schedule

`physical_content/schedule.py`：

| update | LR 乘子 | λ_route |
|---:|---|---:|
| 1–32 | s/32 | 0.03·s/32 |
| 33–128 | 1.0 | 0.03 |
| 129–320 | 0.2+0.4(1+cos(π(s−128)/192)) | 0.03−0.02(s−128)/192 |
| 321–400 | 0.2 | 0.01 |

总损失始终 `0.8 FM + 0.2 tail + λ_r * L_route(raw Cr)`。

## 3. 公共物理 reader

`readers.PhysicalReader`：

- Q：当前 block 官方冻结 cross `q` + `norm_q`，输入独立非仿射 LayerNorm，
  **覆盖全部 N 个视频 token（含首帧）**。
- K：新全宽 dense，初值复制官方 cross Wk，bias=False。
- V：新全宽 dense，初值复制官方 cross Wv，bias=False。
- K norm：复制官方 `norm_k`，作为一个新可训练 norm。
- 无 null reference slot；仅有 `prior[j] = -log(count[group[j]])`。
- attention 输出经官方冻结 cross Wo 的**weight、无 bias**转到 residual 坐标。
- 没有 `state_value`、time projection 或旧的零 reference。
- 最终 writer 是另一个独立 3072×3072 零初始化 dense。

## 4. 各版专属

### R18A

`experts.JointPhysicsExecutor`：

```text
S = SemanticReader(H, T)
P = PhysicalReader(H + S, Cr, Ce)
U = H + S + P
A = g_sa * SA_E(LN(U) * (1+s_sa) + b_sa)
F = g_ff * FFN_E(LN(U + A) * (1+s_ff) + b_ff)
delta = Writer(S + P + A + F)
```

- semantic reader：官方 cross base（Q/K/V/O）加独立 rank32 LoRA。
- video SA/FFN：官方本层 self/ffn base 加独立 rank32 LoRA；完整 3072/14336。
- writer 首帧到未来 mask；payload 不含 H；没有把 H 再写一遍。

### R18B

`experts.InBlockInjector` + `model.velocity` 的显式 B 分段：

```text
H_sa = native_self_with_video_lora(H)
T_update = frozen_native_text_cross(Norm3(H_sa), T)
H_t = H_sa + T_update
S = SemanticReader(H_t, T)
P = PhysicalReader(H_t + S, Cr, Ce)
H_mix = H_t + future_mask(Writer(S + P))
H_out = native_ffn_with_video_lora(H_mix)
```

- 10 个注入站点 [2,5,...,29]。
- main LoRA rank64 只在 10 个注入层，其余 20 层 rank32。
- 没有额外 video SA / FFN 或 source-conditioned expert。
- 原生 SA、text CA、FFN 各执行一次。

### R18C

`content.SemanticPhysicsMemoryEnhancer`：

```text
X = Ce
repeat 2:
  X = X + CA_mem(LN0(X), T)
  X = X + SA_mem(LN0(X))
  X = X + FFN_mem(LN0(X))
C_tilde_e = X
C_tilde_r = Cr + Wr(X - Ce),  Wr = identity
```

- 源 layer 14 官方 cross/self/FFN base 冻结，独立 rank32 LoRA。
- CA 使用合法 text mask；SA 只处理 256 个物理槽，无 video RoPE；FFN 保留
  官方 14336 宽度和官方 bias。
- raw Cr/Ce 保留；route loss 只作用 raw Cr。
- `PhysicalExecutor`：V 来自 P；V/O/FFN 省略官方 bias；Z 不加 time shift；
  `P=0 -> delta=0`。
- 源码中 `normalize` 使用非仿射 LN0；执行器 Q/K 使用官方时间 scale/shift。

### R18D

`experts.GuidedPhysicsExecutor`：

```text
S = current-layer native text_update (no detach, no second text attention)
P = PhysicalReader(H_native_out, Cr, Ce)
X = LN0(H+P) * (1+s_sa) + b_sa
Q = Q_E(X) + Uq LN0(S)
K = K_E(X) + Uk LN0(S)
V = V_E(P; bias=0)
Q/K = official norm_q/norm_k after the guide addition
Q/K = native-grid RoPE
P1 = P + g_sa * O_E(attn; bias=0)
Z = LN0(P1) * (1+s_ff) * (1 + G LN0(S))
P2 = P1 + g_ff * W2E GELU(W1E Z; bias=0)
delta = Writer(P2)
```

- Uq/Uk/G 为每站点独立、全宽、bias=False、零初始化；无 sigmoid。
- `P=0` 时 V=0、attention=0、P1=0、LN0(0)=0、所有 value 路径无 bias，
  因此 delta 严格为 0（CPU toy 已断言精确 0）。
- text_update 来自当前层、当前 sigma、当前 CFG 臂和当前样本；不复制上一层
  或上一去噪步。

## 5. 已明确不符合“原 v9 默认配置”的地方

1. 训练从 1000 改为 **400**，并按用户给出的四阶段 LR/route schedule 实现；
   这不是原 v9 YAML 的 1000-step 配置。
2. 训练使用 v7 `qwen38_wisa_reviewed_v1` 的 400-step data overlay，实际可用
   train 条件为 1600 个；400×4=1600 正好覆盖一次。manifest 仍是 3857。
3. 单 GPU micro1×accum4 已实现；2/4 GPU 分布式未实现。
4. 新增物理 reader 的可训练 `key_norm` 带来每站点 3072 个参数（C/D/A 六站点
   共 18,432；B 十站点共 30,720），设计文档的参数预算忽略了该小项。
5. 源码交付没有执行正式 400-step 优化训练、完整视频生成或 VideoPhy2 评分；
   只做过真实 Wan GPU 小尺寸前向/反向与 CPU toy 检查。
