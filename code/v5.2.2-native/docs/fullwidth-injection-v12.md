# V12：对应 C 注入维度图的实际修正

这是历史 v12 图文。当前默认 **v13 已删除 a，并加入全维 C 归一化/逐通道仿射及首阶段 Wan 冻结**，见 [当前实现说明](../README.md)。下图不表示 v13。

![v12 全维 C 注入及 a 的完整维度流程](../fig/v522_native_v12_03_C_injection_dimensions_20261026.png)

上图由内置 imagegen 绘制并对照当前代码核对，包含强度 a 的逐层维度、query 广播调制与 native block 内残差路径。[绘图提示词](../fig/v522_native_v12_03_C_injection_dimensions_20261026.prompt.txt)、[核对记录与修订提示词索引](../fig/v522_native_v12_03_C_injection_dimensions_20261026.json)。

本文件描述 `v522_native_fullwidth_attention_v12`，不是对 v9 图片文字的重新解释。v9–v11 的紧凑接口保留用于读取旧 checkpoint；当前新训练默认选用 v13。原图 (c) 的 rank-64 状态算子与输出门控不再存在于 v12 主路径。

| 用户问题 | 原实现 | 新实现与判断 |
|---|---|---|
| 为什么 3072→768？ | 新建紧凑 Q/K/V，12×64 heads，输出再映射回 3072 | 删除该读取器，复用原生 3072→3072 投影，24×128 heads。没有充分证据为 768 容量限制辩护 |
| 小写 q 为什么变为大写 Q？ | q 是状态；Q 是投影后的检索向量，同时被降到 768 | 保留原生方形 Wq，Q 仍为 3072。学习匹配坐标与压缩宽度是两个不同问题；Q 只计算一次并由文本/C 两路复用 |
| m 为什么还经过 64 维算子？ | `W_up[SiLU(W_h Norm(q)) ⊙ W_c m]`，每个 site 输出处于 W_up 的 ≤64 维列空间 | 整个额外状态算子移除。attention 本身已按当前状态选取 C，随后由当前 native FFN 做非线性处理 |
| 是否直接把 rank 改成 3072？ | 会变成每站新增多套大矩阵，仍重复 native attention/FFN 的职责 | 没有这样改。消息保持全维，但不增加这套冗余大网络 |
| 输出标量门控 g 是否需要？ | `sigmoid(Linear[Norm(q),σ,a])` 再乘整个消息 | 删除额外 g。Wan 原生 self-attention/FFN 的去噪调制保留 |
| 连续变化强度 a 是否也删除？ | `3072→192→1` 读出，参与输出门控 | 保留相同辅助读出和首阶段 RGB 弱监督；改为 Q 的软偏移 `(a−0.5)r`，r 为 3072 维参数。C/value 不经过标量读出，a≈0 也可读取 C |
| H+ΔH 是否不如注意力？ | 已先读 attention，再经额外低秩算子，最后在整个 block 之后加残差 | 注意力与残差并非二选一。新 C 消息位于 native cross-attention 内部，由原生残差加入 H，再经过本 block FFN；不丢弃当前视频状态 |
| 注入是否能到后段？ | 十处 block 后处理未来 20 个 latent 时间位置 | 仍覆盖全部 20 个未来位置，在第 3、6、…、30 block 内读取 C。首帧没有直接 C 残差；后续时空 self-attention 仍可交换隐藏状态 |

## 实际计算

令 X 为 native cross-attention 的输入 `norm3(H)`。原生 self-attention 已处理完整视频的时空状态，native block 的去噪调制保持。去掉旧额外 Fourier query 偏移，使用原生状态作为读 C 的依据。

```
Q  = norm_q(Wq X + bq)                    # [B,8190,24,128]
Kt = norm_k(Wk T + bk); Vt = Wv T + bv    # T 为 native T5 context
Kc = norm_k(Wk C + bk); Vc = Wv C         # C 值分支不重复增加 bv
Qc = Q_future + reshape((a−0.5) r)
Mt = Attention(Q, Kt, Vt)
Mc = Attention(Qc, Kc, Vc)               # [B,7800,24,128]
M  = concat_time(Mt_first, Mt_future + strength Mc)
Y  = Wo flatten_heads(M) + bo            # [B,8190,3072]
H1 = H + Y
H2 = native_FFN_residual(H1, σ)
```

以上 W 包含对应的 native base 与已有 LoRA 增量。原生矩阵在 3072 维工作，LoRA 的低秩增量与完整 base 并行，并不是 C 的必经低维瓶颈。Wq/Wk 学习 query-key 的匹配空间；Wv/Wo负责内容表示和输出坐标对接。24 个头只是把 3072 重排为 24×128，拼接后的总宽度不减少。

文本与 C 的 softmax 分开，使新 C 不直接夺走原生文本的归一化权重；在原生 O 前合并，只执行一次 O，也只计入一次输出 bias。V 的 bias 已存在于文本分支，C 分支减去其重复项，使零 C 对应零 C 消息。`strength=1` 是新默认值；`strength=0` 是干预开关，不是学习到的额外门控。普通 Tensor context 完全走原生 forward，便于与真实 native Wan 对比。

方形投影不保证数学上永远满秩，也不保证所有物理信息被保留；本修正的确切结论是取消人为指定的 768 和 64 维上限。每一注入点复用自己的原生投影，没有引入第二套大 Q/K/V/O。额外接口参数从旧结构的 15,994,783 降为 596,353，其中大部分是原有强度读出。

## 权重与训练

Qwen 输出宽度 4096 与 Wan 宽度 3072 不同，因此 `KnowledgeCondition` 的 `4096→3072` 对接仍保留。它发生在 C 构造处，与用户质疑的注入后第二次降到 64 再升回 3072 是不同操作。连续 reasoning、条件内容/来源编码、verifier 回传 C、单轨迹推理均保留，VJEPA2 未重新引入。

读取器从随机紧凑投影换成 native 投影后，旧 C 与新读出空间的语义对齐不能视为已经成立。`initialize_generator` 显式继承 Qwen、C、Wan-LoRA 和强度读出，弃用旧紧凑 reader、旧 Fourier 相关归一化及各 rank-64 site；新 `intensity_query` 初始化。所有未被明确替换的张量必须同名同形，LoRA scale 与基础模型必须一致。普通 resume 仍要求同一配置，不把旧 optimizer 带入新结构。

新训练默认是 `configs/native_fullwidth_attention_wisa3857.yaml`。完整配方仍为 interface / calibration / joint，目标数量 4 / 1 / 4；FM、跨视频时间 FM 误差、低噪声全过程 RGB 变化与后续 verifier→C 的职责不变。为控制时间，准备的单卡 YAML 只执行前 32 次新增适配更新，尚未提交。不存在额外 S rollout 或另一套未来视频预测器。

## 当前证据边界

91 项 CPU 检查通过，其中 8 项针对新接口：原生旁路与零 C 一致性、Q 仅计算一次、首帧直接注入排除、低 a 仍能读取 C、没有强制低秩 value 导数、checkpoint 重算梯度一致、生成/verifier 的参数梯度归属、显式旧权重迁移和新 checkpoint 恢复。小模型的满秩 Jacobian 只证明代码没有强制窄通路，不证明真实 Wan 的有效信息秩或物理效果。

真实 v11 step16 首个 site 的 `W_c`、`W_up` 秩均为 64。一个 RMS=1、位于 W_c 零空间的消息扰动，映射后 RMS 为约 4.80e-16。这具体证明该旧 site 无法响应这类消息变化，但不能把扰动叫做真实物理语义，也不能推导整个生成模型只有 64 维。

Run66281 已于 07:21:39 UTC 在 BLK-96G 单卡完成，固定 169 文件快照，脚本耗时 245.10 秒。真实模型关闭 C 的原生一致性误差为零；生成反传和 16 步完整 verifier→C 反传均到达 C。随机评分头只用于导数检查。真实生成是否改善、长程是否稳定、verifier 的方向是否正确，都尚未得到 v12 质量结论。

后续复核发现，迁移后未适配样本的 C/text 消息 RMS 比达 12.27–383.80（原生 O 投影前），整个 FM 输出改变量与 native 输出本身相当。宽度一致不等于语义与尺度对齐。a 的单样本强度误差高于零变化基线，且当前只校准十层平均 a。`ΔQ=(a−0.5)r` 在所有位置构成秩至多为一的附加偏移；24×128 重排不会增加其自由度，但原始全维 Q 和 C 消息没有因此被压成秩一。这些发现要求优先处理 C 的原生读取适配，以及 a 的监督语义与实际使用的一致性。详细数字与检索来源见 [复算证据](../archive/20261027_cleanup/analysis/query_intensity_review_20261026/evidence.json) 和 [文献记录](../archive/20261027_cleanup/analysis/query_intensity_review_20261026/literature.json)。本图仍准确描述当前代码，不表示 a 调制已经获得效果证明。

Run66272 使用固定 v11 旧结构继续完成当天 40 例 VideoPhy2 比较；不把其结果混称为 v12。技术来源上，独立条件 attention 不是新发明；物理因果贡献仍需来自 C 的可验证知识作用、过程反馈与生成改善，而不是“换成全维 attention”这一点本身。

实现：`physgen_native/native_attention.py`、`generator.py`、`runtime.py`、`config.py`。证据：`archive/20261027_cleanup/analysis/queue_20261026_fullwidth_attention/`。
