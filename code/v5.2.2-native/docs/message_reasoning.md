**H_content_message / Q_seed128_message：实现与训推入口**

**当前训练方案：v25–v28，按用户最终要求保留 224／256 步、accumulation=4。**

本节覆盖下方旧版本说明。四个新配置使用 `*_simple_wisa3857.yaml`，对应 `train/*_simple_1x96g.yaml`；旧配置和 checkpoint 的行为保留。

| 版本 | blk-96g 提交 YAML | 阶段与步数 |
|---|---|---|
| v25：H＋冻结 verifier | [H frozen](../train/native_h_content_message_verifier_frozen_simple_1x96g.yaml) | interface 64 → joint 160，共 224 |
| v26：H＋verifier LoRA | [H LoRA](../train/native_h_content_message_verifier_lora_simple_1x96g.yaml) | interface 64 → calibration 32 → joint 160，共 256 |
| v27：Q128＋冻结 verifier | [Q frozen](../train/native_q_seed128_message_verifier_frozen_simple_1x96g.yaml) | interface 64 → joint 160，共 224 |
| v28：Q128＋verifier LoRA | [Q LoRA](../train/native_q_seed128_message_verifier_lora_simple_1x96g.yaml) | interface 64 → calibration 32 → joint 160，共 256 |

每个微批仍为一个视频，累计四次梯度后更新。四版都有 verifier；冻结版仅不安装 verifier LoRA。单个作业使用一个 blk-96g slot，复用既有镜像、挂载和运行环境。

**表示与观察。** memory、H 状态、Q 工作区和全部 G 更新保持 4096 维；G 为 32 heads × 128、FFN 4096→8192→4096。最后执行一次 P：4096→3072，再用既有对齐仿射接入 Wan。Wan 内部仍为 3072 维、24 heads，不改其预训练矩阵。H 使用完整内容长度，固定出口 1；Q 保持 128 个槽位，固定出口 2（包含初次 grounded read，共执行三次 G）。没有新增模块、未来预测器或额外训练阶段。

前置投影确实可能消除后续 message 更新需要的方向；移动到末端让 G 先利用全部 Qwen 特征。最终接入 Wan 的降维和 Q 的固定 token 容量仍是结构取舍，不能宣称全部信息可逆保留。初始化时只复用兼容的 Wan LoRA、输入 norm、P 和输出仿射；旧 3072 维 G／query 不迁移到新 4096 维工作区。同一新架构之间可以完整迁移 C；精确恢复使用保存的配置和 optimizer 状态。

verifier、teacher 和校准均按原视频每隔 3 帧取一次，并补上末帧。81 帧的实际索引为 `0,3,...,78,80`，共 28 帧；原生视频 temporal patch 对相邻采样帧分组，视觉 token 数为 `14×390=5460`。空间尺寸保持 832×480，不缩放、裁剪或改变宽高比。VAE 保留完整时间历史，仅保留所选 RGB 输出用于判别。

**达到分析长度上限时继续训练。** Qwen 提示词要求 OBSERVED／RESPONSE／PERSISTENCE／CHANGE 四行，每行最多 25 词，总计最多 100 词；原始 caption 和 clauses 单独保留，不让分析重复它们。生成预算仍为 512，必要时重试至 1024 tokens。`analysis_on_token_limit: use_partial` 保留未出现 EOS 的非空分析并正常构造训练条件，日志记录 `analysis_truncated`；不增加 Q 的 128 个槽位，也不将自生成分析当作新的 SFT 正确标签。teacher 观察要求 EARLY／MIDDLE／LATE 三行、总计最多 75 词；达到 384-token 上限时保留非空观察，完整性与可用性分别记录，原有置信度／顺序一致性筛选继续执行。

**真实 GPU 显存修正。** verifier 的整次读取 checkpoint 之外，对 Qwen vision blocks 和 language layers 逐层 checkpoint。原生 Transformers 的 checkpoint 依赖 train 模式，单纯冻结并保持 eval 会跳过它；因此显式在 eval 下启用逐层重算，保留 dropout 状态、LoRA 角色及 RGB 梯度。既不降低观察帧数、分辨率，也不截断生成反馈梯度。

**训练项。**

| 阶段 | 更新参数 | 损失及系数 |
|---|---|---|
| interface | C 的 norm、G、P、对齐参数；Q 版包含 query seeds | FM 1 + temporal FM 0.1 |
| calibration，仅 LoRA 版 | verifier LoRA；生成器固定 | teacher soft-CE 1 |
| joint | C 与 Wan LoRA；verifier 固定 | FM 1 + temporal FM 0.1 + verifier support NLL 0.05 |

RGB process loss 已从新版本删除，不是仅将权重设成零：普通训练不再解码参考未来及单步重建视频，不调用 RGB 差分损失。它是参考视频重建约束而非独立过程判断，并且参考变化量归一化会改变样本权重；原生全分辨率 VAE 反传成本较高。FM 已监督同一 clean latent，temporal FM 补充误差的时间结构，verifier 负责实际采样视频的过程关系。验证中的 RGB 重建指标保留为只读诊断，不参加训练梯度；verifier 必需的视频解码和 RGB 输入梯度保留。

verifier 保留四个固定维度：要求完成、条件响应、持续终止、材料与来源连续性。原始 caption／clauses 直接参与判断，不再添加动态检查表、排序损失、周期性重校准或第四个训练阶段。三类为 supported／violated／unknown，unknown 仅表示证据不足；正常静止或缺少不相关触发不自动判 unknown。训练损失为四维度、两种标签顺序分别计算 `-log_softmax(logits/T)[supported]` 后取均值，T=1；旧 `p_supported-p_violated` 仅保留为易读诊断。

NLL 在明确错误端仍有有效的 logit 梯度，所以不直接沿用旧概率奖励的大系数。初值 0.05 对应每次反馈的权重；每 4 个联合更新反馈一次，不再除以 accumulation=4。共有 40 次完整反馈，按更新平均的损失系数为 0.0125；这不是实际梯度大小的估计。FM 权重 1 作为锚点，temporal 0.1 是低成本的误差度量修正，不把 FM velocity 当视频运动速度。没有实测证据支持进一步增加这些系数。

反馈从纯噪声开始，固定 16 个 Euler 步，CFG=5、shift=5；调用与推理同一采样函数和正负 C 构造，两条分支均完整求导。普通推理仍 50 步。verifier 只更新 C；Wan LoRA 只接收 FM／temporal 梯度。逐步 adjoint 和逐分支 checkpoint 控制图内存。日志额外记录配对监督与 verifier 的 C 参数梯度范数及比值，不增加额外前后向，也不自动调权。

**学习率与收敛。**

| 参数 | interface 峰值 | joint 峰值 | 校准峰值 |
|---|---:|---:|---:|
| P、G、Q query seeds | 1e-4 | 5e-5 | 0 |
| 输入 RMSNorm、输出 gain／bias | 1e-5 | 5e-6 | 0 |
| Wan LoRA | 0 | 1e-5 | 0 |
| verifier LoRA | 0 | 0 | 1e-5 |

C 前 8 步 warmup，随后在接口阶段平滑降到 joint 的峰值；校准期间暂停，joint 内 cosine 衰减至峰值的 20%。Wan 在 joint 解冻后单独 warmup 8 步，随后衰减到 2e-6。verifier 的 32 步校准 warmup 4 步，随后衰减至 2e-6。短预算内不再设额外奖励课程。新 G 使用较高学习率，对齐尺度较慢调整，预训练模型 LoRA 较低；不再为每个 attention／FFN 单独设学习率。

AdamW 的矩阵 weight decay=0.01；norm、bias、query seeds 为 0。C、Wan、verifier 分别按范数 1 裁剪，避免 verifier 的大梯度通过全局 clipping 改变 Wan 的监督更新。条件 dropout 互斥：10% 关闭两路文本条件，15% 只关闭 Wan 原生文本并保留完整 C，75% 正常；独立 requirements dropout=0。这些设置是可执行的默认起点，质量收益以真实采样验证为准。

可微采样奖励的依据见 [DRaFT](https://arxiv.org/abs/2309.17400)；按梯度而非仅 loss 数值衡量任务权重的依据见 [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)。这里未引入 GradNorm 动态调权。

**提交方式。** 四个独立 shell 默认只做资源预检；显式 `--submit` 才提交相应 YAML，不进行 hash／SHA 校验。

```bash
bash train/train_h_content_message_verifier_frozen_blk96g.sh --submit
bash train/train_h_content_message_verifier_lora_blk96g.sh --submit
bash train/train_q_seed128_message_verifier_frozen_blk96g.sh --submit
bash train/train_q_seed128_message_verifier_lora_blk96g.sh --submit
```

也可使用 `det -m http://10.130.130.5:4096 experiment create train/<对应的 simple_1x96g.yaml>`。`--local` 入口用于已经分配的单卡环境，支持 `--smoke`、`--stop-after`、`--resume <run目录>`；恢复时自动去掉新任务初始化参数。147 项 CPU 回归及四套真实 GPU 阶段测试均通过；阶段测试保持完整模型、视频尺寸、累计梯度和反馈链路，每阶段执行一个更新，峰值 63.6–64.6 GiB。正式任务已按用户授权重提：H frozen 66356／H LoRA 66355／Q frozen 66353／Q LoRA 66357。运行记录见 `analysis/recovery_simple_blk96g_20261026/`。

**最新几何修正与维度审计：v21–v24。**

原问题确实是宽高比被改变，而不是宽高字段的书写顺序。生成画面的比例为 `832/480=26/15≈1.7333`；旧 verifier 先解码缩到 `320/192≈1.6667`，再因为 `_inputs(video=False)` 读取 image_size 而放大到 `512/288≈1.7778`。这会改变几何证据且丢失细节；第二次放大无法恢复第一次丢失的信息。旧文档把 192×320 当作最终 Qwen 输入尺寸是不准确的，实际应以此审计为准。

修正后，四个新 revision 中的 `geometry.rgb_size`、`knowledge.image_size/video_size`、`verifier.size`、`sequence.size` 均为 `[480,832]`，即生成画面的 832×480。RGB 监督只用 `scales: [1]`，不再包含 208×120 输入或内部 2/4 倍空间池化。`geometry.py`、配置校验和实际输入检查共同拒绝非等比拉伸、尺寸错配及悄悄恢复低分辨率监督；原生尺寸相同的 VAE 输出也不执行无意义的 interpolate。

覆盖的执行路径包括：首帧文字分析和 memory 编码、teacher 观察/打分、LoRA 校准、生成训练反馈、在线 C 更新、独立 heldout 评价和 RGB 过程损失。后续入口为 `configs/native_{h_content_message|q_seed128_message}_verifier_{frozen|lora}_geometry_wisa3857.yaml`，对应四份 `train/*_geometry_1x96g.yaml`；[README](../README.md) 提供逐一链接。旧配置的预处理不会被静默改写，新训练请明确使用修正版。

为控制额外输出缓存，verifier 通过 `decode_qwen_view` 只保留选定的 8 张原生 RGB 帧；VAE 仍按时间顺序处理此前所有 chunk 和因果缓存，训练生成仍包含 81 帧，不重置中间历史。该优化不缩小空间尺寸，也不截断回到 C 的梯度。RGB 过程监督仍比较整段视频的原尺寸输出。8 帧抽样本身是剩余的时间覆盖限制，没有宣称观察了全部 81 帧。

**逐项维度审计。** 依据真实 checkpoint 配置及本机原生 Qwen3-VL/Wan 类实例化结果，不只根据图或注释判断。

| 部位 | 实际变化 | 是否原生/必要 | 审计结论 |
|---|---|---|---|
| message 前置 P | 每个 memory token `4096→3072` | 自定义 Qwen/Wan 适配 | **真实降维，非无损**。当前原生 Wan K/V 输入为 3072，需匹配接口；但不必在 reasoning 更新之前做，当前时机是设计选择。 |
| H_content 初态 | 从 L 个上下文位置选择 T 个完整内容位置 | 自定义读出 | 没有按段平均池化；但只保留文字位置为输出状态，不等于所有视觉位置都直接出现在 C。每轮仍读取完整投影后 memory。 |
| Q_seed128 工作区 | 变长 memory 由 128 个工作 token 读取 | 自定义固定容量 | **存在 token 容量瓶颈**，不是 Wan 原生要求；全宽 3072 不能保证任意长证据无损装入 128 个位置。 |
| message Q/K/V/O | `3072→3072`，24 heads×128 | 自定义满宽更新模块 | 没有额外紧凑通道；拆成 24×128 只是 reshape，不是将整向量压成 128。 |
| message FFN | `3072→6144→3072`，带残差 | 自定义标准扩展 FFN | 隐层扩展，未经过低于 3072 的狭窄层。 |
| C 对齐 | 3072 维 LayerNorm、逐通道 gain/bias | 自定义对齐 | 通道数不减，但归一化不是严格无损；最终 LayerNorm 不保留每 token 的原始绝对均值/尺度。 |
| Wan 注入 Q/K/V/O | `3072→3072`，24×128 | 复用预训练原生矩阵 | 当前注入没有 `3072→768` 或 `3072→64→3072`。attention 仍会按 query 聚合 token，满宽不等于完全无损。 |
| Wan 原生 FFN | `3072→14336→3072` | 原生 | 扩展 FFN，不是 64 维条件瓶颈。 |
| Wan 原生文本适配 | `4096→3072→3072` | 原生 T5 context 路径 | 与新增条件 P 是两个不同映射，不能把搜索到的两个 4096→3072 混为一处。 |
| Wan 时间调制 | `256→3072→3072→18432`，拆成 6×3072 | 原生 | timestep 编码及调制，不是把 C 压成 6 维。 |
| Wan patch/head | 每个 latent patch `48×1×2×2=192→3072`；输出 `3072→192` 后 unpatchify | 原生输入/输出头 | **确实有 3072→192**，用于预测最终 latent 的 48×2×2 个数；不在中间 C 消息上再降维回写。 |
| Qwen 语言 Q/O | `4096→4096` | 原生 | 全宽。 |
| Qwen 语言 K/V | `4096→1024`，8 KV heads×128；32 Q heads 共享 KV | 原生 GQA | 确实减少 KV 容量，不能称作纯 reshape；但来自预训练结构，残差 hidden 仍为 4096，没有新增 64 维中间瓶颈。 |
| Qwen 语言 FFN | 两个 `4096→12288` 分支，`12288→4096` | 原生 | 扩展/门控 FFN。 |
| Qwen 视觉编码 | patch `3×2×16×16=1536→1152`；注意力 `1152→3456` 拆 Q/K/V，O `1152→1152`；FFN `1152→4304→1152` | 原生 | 原生视觉特征编码，不能承诺无损，但不是额外添加的低秩 C 算子。图片模式按模型要求复制时间 patch。 |
| Qwen 视觉 merger/DeepStack | 四个 1152 维 patch 拼接为 4608，`4608→4608→4096`；4 个空间 patch 合并为 1 token | 原生 | 空间与通道均有原生压缩；原生 832×480 对应 390 个视觉 token/帧。 |
| verifier 读答案 | 最后位置 `[B,4096]` 与原 LM head 的 A/B/C 三行相乘，得到 `[B,3]` | 原生权重，自定义答案协议 | 没有新建低维评分 MLP；这是终端判别读出，不会把三个数升维回 C。四项×两次标签分配为 `[B,4,2,3]`，最终归约 loss。 |
| Wan LoRA | 增量支路 `3072→32→3072` | 附加适配器 | 必须承认该支路低秩；完整输出是 `W_base x + B A x`，C 仍经过全维 base 矩阵，不是全部消息只经过 rank 32。 |
| verifier LoRA | Q/O 增量 `4096→16→4096`，K/V 增量 `4096→16→1024` | 附加适配器，仅 LoRA 版 | 同样保留完整原生 base 路径；约束的是可学习权重增量的秩，而不是把全部 hidden 压成 16 维。 |

新注入器和新评分读出自身都没有独立可训练线性参数；它们分别调用 Wan 原生投影和 Qwen 原生答案权重。旧 `CompactAttention`、`StateConditionedSite`、`InjectionSite`、强度头和 `QwenProcessCritic` 的代码为旧 checkpoint 保留，但四个新配置实例中均未出现这些类。旧配置或旧任务快照仍可能调用它们，不能用当前审计宣称历史任务已经被替换。

`4096→3072` 线性映射的秩最多为 3072，存在至少 1024 维零空间，因此不能保证任意输入可逆；这不等于已经测得丢失 25% 的物理信息。当前投影在 message 更新前执行，后面的多轮更新不能凭空恢复被它完全消掉的区别。若后续要专门减少这种风险，可把工作区保持为 4096、将适配延后到最终 Wan 接口；这属于另一项架构修改，本次几何修正没有悄悄实施。H 模式避免固定 128 个输出槽位的限制；Q128 应保留为有明确容量取舍的对照，不能称作无压缩版本。

全分辨率后，Qwen 每帧视觉 token 从原实际 144 增至 390，8 帧合计从 1152 增至 3120，视觉计算会增加；RGB loss 的比较与缓存也增加。没有通过降低分辨率回避这些成本，也没有据 CPU 测试声称真实 8B/5B 模型的 GPU 峰值已验证。

[审计及验证记录](../archive/20261027_cleanup/analysis/native_geometry_dimension_audit_20261026/verification.json) 保存生产配置真实类的完整线性层清单、四版资源预检和测试日志。135 项全量回归通过；最终取消 RGB 降采样后的 5 项专项检查通过（22.306 秒），验证原图 patch 坐标逐像素一致、尺寸错误被拒绝、四版反馈梯度和在线推理、兼容参数迁移。

```bash
# 新四版均替换成 *_geometry_wisa3857.yaml；完整训练/恢复/推理命令与下方相同。
bash tools/runtime_python.sh train/train.py \
  --config configs/native_h_content_message_verifier_lora_geometry_wisa3857.yaml --check-only
```

**新增四套 verifier 全流程（v17–v20，2026-10-26）。**

以下四版均使用 verifier。“无 LoRA”指 verifier 不安装 LoRA；生成侧 Qwen 在四版中都冻结，既有 Wan LoRA 的训练安排仍保留。v15/v16 的旧协议在本文后半部保留；它们不等同于本次的四项评价协议。

| 版本 | 训练配置（`configs/`） | C | verifier | optimizer steps |
|---|---|---|---|---|
| v17 / H + 冻结 verifier | `native_h_content_message_verifier_frozen_wisa3857.yaml` | `[B,T,3072]` | 无适配器 | interface 64 → joint 160，共 224 |
| v18 / H + verifier LoRA | `native_h_content_message_verifier_lora_wisa3857.yaml` | `[B,T,3072]` | 独立 LoRA | interface 64 → calibration 32 → joint 160，共 256 |
| v19 / Q128 + 冻结 verifier | `native_q_seed128_message_verifier_frozen_wisa3857.yaml` | `[B,128,3072]` | 无适配器 | interface 64 → joint 160，共 224 |
| v20 / Q128 + verifier LoRA | `native_q_seed128_message_verifier_lora_wisa3857.yaml` | `[B,128,3072]` | 独立 LoRA | interface 64 → calibration 32 → joint 160，共 256 |

每份配置都有同名 `train/*_1x96g.yaml` 单卡 Determined 入口，共享完整实现，不复制四份易于分叉的训练循环。既有 v15/v16 配置和默认训练入口未替换。这里的步数是配置预算，不是任务已运行步数，也不是效果验证。

**生成侧保持两种 C 架构。** 输入仍只含首帧、原始 caption、合法 clauses 和时长；Qwen 生成分析并编码后输出冻结 memory `[B,L,4096]`。P 投影到 3072 后，H 模式保留全部选中内容位置 T 并做 U1/U2 更新；Q 模式由 128 个种子读完整 memory，保留 R0/R1/R2。训练随机出口、各轮独立推理以及现有 C 对齐均保留。C 接入原生满宽 Wan 读出，不增加 3072→64 算子，也不加载 VJEPA2。

**评价对象是实际视频。** `criterion_verifier.py` 的 judge 接口仅接收 `(frames, times, caption, clauses)`，不接收 C、H_content、Q_seed、生成侧 analysis、参考视频身份或目标未来状态。训练反馈从纯噪声进行既有 16 步、81 帧完整生成；VAE 解码后均匀取 8 帧，含首尾，按真实时间戳送入 Qwen。v17–v20 实际为先解码到 192×320，再因图片模式放大到 288×512；此几何问题已在上方 v21–v24 修正。仍不是观察全部 81 帧，不能声称完整识别所有短暂物理违例。

四项评价分别为：

1. `requirements`：视频中有无实现原始要求的可见证据。
2. `response`：可见前提、交互或来源与响应是否一致；不凭空推断隐藏原因。
3. `persistence`：早、中、晚阶段的持续、延迟、停止是否合理；源停止后既有烟雾仍可扩散。
4. `continuity`：材料、属性和来源能否在形变、扩散、遮挡、分裂、合并及合理出现/消失中保持一致。

每项输出三类：有证据支持、存在可见违例、证据不足/不适用。读取原有 LM head 的 A/B/C 三个 logits，没有新评分头。两次标签分配分别为 `A=支持/B=违例/C=未知` 和 `A=未知/B=支持/C=违例`；映射回统一语义后得到 `[B,4,2,3]`。这些是三个候选答案内部的归一化值，**不是已校准的真实物理正确率**。交换标签有助于暴露字母偏置，不能保证消除偏置。

令两种标签分配的平均概率为 `p_i`，当前反馈目标为：

```text
score = mean_i [p_i(支持) - p_i(违例)]
L_verifier = 1 - score
```

纯支持、纯未知、纯违例的 loss 分别为 0、1、2。未知不给正奖励，也不直接被标为违例；不使用可学习置信度门来关闭反馈。joint 中该项权重 0.02，每 4 个 optimizer step 执行一次。日志保存四项各自的三类概率、未知比例、标签交换差异，以及真实反向得到的 `C_gradient_rms`。该目标仍可能被生成器利用，必须通过独立视频评估检查。

**verifier LoRA 与生成侧 Qwen 隔离。** v18/v20 在 Qwen 最后 12 个语言层的 q/k/v/o 安装 rank=16、alpha=16 的 `VerifierLoRALinear`；参数名字为 `verifier_lora_a/b`，优化器组为 `verifier`，学习率 1e-5。Qwen 基础参数、视觉塔和 LM head 冻结。LoRA 的低秩是权重增量参数化，**不是把 C 从 3072 压缩为 16 或 64 维**；Qwen 自身的 GQA 维度也按原模型保留。

角色通过 `verifier_role` 显式切换，checkpoint 重算捕获同一角色：

- 生成侧分析与 memory 编码：verifier LoRA OFF，Qwen `no_grad`，因此训练 judge 不改变条件编码器。
- 固定 observation teacher：LoRA OFF，`no_grad`，不随 judge 学习改变。
- 学生 judge 校准：LoRA ON，只允许它的参数梯度；视频及 teacher 标签 detach。
- 生成训练与可选在线 C 更新：LoRA ON，但参数冻结，保留对输入视频的导数。
- v17/v19：根本不安装 verifier LoRA；采用完全相同的评价提示、标签交换和生成损失。

**校准阶段的数据与标签。** 只使用现有训练 split 的缓存参考视频与当前模型生成视频，按 micro-step 交替。当前视频由 interface 阶段学到的 C 和冻结生成器完成，绝不把参考未来传入 C；没有额外准备数据集，也不使用 heldout 更新权重。每条视频独立评价，不自动赋予“参考正确／生成错误／静止错误”的标签。

冻结 teacher 先在**不提供要求**的情况下描述实际视频的早、中、晚状态，再把这段观察和原始要求一起提供给四项 judge。学生仅见实际视频和要求。这样不是将同一个模型同一输入产生的概率直接作为自身标签，避免初始蒸馏梯度为零。teacher 与学生仍来自同一基础模型，因此这种监督只能称为弱监督，不能证明获得了新的物理真值。

teacher 的两种标签分配先对齐并取平均；默认只有最大类概率 ≥0.55、两分配的总变差距离 ≤0.15、且观察文字到 EOS 完整结束的项参与 soft cross-entropy。观察预算为 384 token；未完成不会被当成完整标签。置信度高的“未知”也是合法校准目标。阈值是初始配置，尚未经过真实模型标定。所有项被拒绝时跳过适配器梯度与 AdamW 衰减；若整个校准阶段没有接受任何标签，后续反馈明确报错，避免把零监督适配器宣称为已校准。

每个校准 step 的首个 micro-step 导出真实观察帧及 JSON，包括观察文字、是否完整、三类标签分布、接受 mask、学生分布与来源；验证阶段使用同一流程但不反传、不累计训练接受计数。checkpoint 保存 adapter、独立 optimizer 状态以及 `verifier_state` 接受/总数，支持跨 calibration 边界精确恢复。接受计数只证明存在校准监督，不代表评分可靠性。

**梯度与训练阶段。**

| 阶段/目标 | P、message block、Q_seed/对齐 | Wan LoRA | verifier LoRA | Qwen/Wan 基础权重 |
|---|---|---|---|---|
| interface：FM + temporal FM + RGB process | 更新 | 冻结 | 冻结/OFF | 冻结 |
| calibration：软标签交叉熵，仅 v18/v20 | 冻结 | 冻结 | 更新/ON | 冻结 |
| joint：三个生成目标 | 更新 | 更新 | 冻结 | 冻结 |
| joint：verifier 反馈 | 更新 | 此目标不更新 | 冻结/ON（如有） | 冻结，保留输入导数 |
| 普通推理 | 不更新 | 不更新 | 不调用 judge | 不更新 |
| `--refine-c` | 仅临时 C 数值更新 | 不更新 | 冻结/ON（如有） | 不更新 |

反馈链为 `L_verifier → Qwen视频输入 → VAE解码 → 完整16步Wan生成 → C → P/message/Q_seed`。生成损失与 verifier 损失分别反传，verifier 目标不能经由更新 Wan reader 绕开 C。Q_seed128 的梯度到达种子与共享更新模块；H_content 的梯度到达全部有效内容状态及共享更新模块。Qwen judge 的每次读出使用非重入 checkpoint，以保留 LoRA/RGB 梯度并减少多个评价前向的存储；仍然需要实际大模型显存和耗时验证。

**四套完整入口。** 在项目目录选择任意一个配置即可自动走完对应阶段，无需另写校准脚本。例如：

```bash
# 将配置名换成表格中任一版本，先检查现有资源。
bash tools/runtime_python.sh train/train.py \
  --config configs/native_h_content_message_verifier_lora_wisa3857.yaml --check-only

# 新训练。四份 Determined YAML 已填写相同既有 generator 初始化路径；未提交任务。
bash tools/runtime_python.sh train/train.py \
  --config configs/native_h_content_message_verifier_lora_wisa3857.yaml \
  --initialize-generator checkpoints/initialization/generator.pt

# 分阶段小规模 GPU 检查，配置内部每阶段一更新；不代替完整训练或效果验证。
bash tools/runtime_python.sh train/train.py \
  --config configs/native_q_seed128_message_verifier_lora_wisa3857.yaml --smoke

# 精确恢复：必须使用该次 run 保存的配置，不用初始化参数。
bash tools/runtime_python.sh train/train.py \
  --config /path/to/run/config.yaml --resume /path/to/run

# 普通推理；0/1/2 仍可分别比较，H 的 U0 与 Q 的 R0 计算含义不同。
bash tools/runtime_python.sh inference/infer_heldout.py \
  --checkpoint /path/to/run/latest.pt --samples 4 --reasoning-round 1

# 可选同一采样轨迹内更新 C；默认关闭，不更新任何模型权重。
bash tools/runtime_python.sh inference/infer_heldout.py \
  --checkpoint /path/to/run/latest.pt --samples 4 --refine-c --output /path/to/refined_outputs

# 独立 heldout 评估：导出实际生成视频、参考/生成各项概率与标签偏置。
# LoRA 版本同时在相同帧上比较 base/adapter；可选导出弱 teacher。
bash tools/runtime_python.sh inference/evaluate_message_verifier.py \
  --checkpoint /path/to/run/latest.pt --samples 4 --output /path/to/new_evaluation --with-teacher
```

`--initialize-generator` 不转移 verifier、optimizer 或校准计数。相同 message 架构跨版本时，完整继承 P、message block、Q_seed（如有）、C 对齐和 Wan LoRA；跨 H/Q 或从旧版迁移时只继承兼容的投影/对齐/Wan 参数，重新初始化不同的工作区。精确 `--resume` 则恢复本版本的所有学习参数与校准计数。普通推理仍装载 checkpoint 中的 verifier adapter（若有）以保证参数集合一致，但其角色关闭且不执行视频评分。

**如何判断是否值得采用 LoRA 版本。** 两版本使用相同生成训练步数、随机种子、出口、采样配置与 heldout；另行报告 32 步校准的额外成本。先看人工复核的支持/违例/未知分类、标签交换偏差、相同实际视频上 base 与 adapter 的差异，再看独立生成视频的早中晚过程、多轮出口差异和多 seed 稳定性。不可只用新 judge 给自己的生成打高分来宣布提升。固定 8 帧、同源弱 teacher、没有人工物理标签、可微奖励被利用的可能性，都是当前实现的明确限制。

代码与测试：`criterion_verifier.py`、`criterion_training.py`、`lora.py`，并接入 `knowledge.py`、`system.py`、`operator_training.py`、`runtime.py`、`data.py` 和既有训推脚本。`tests/test_criterion_verifier.py` 覆盖四版原生小 Qwen/Wan 训练、隔离校准、视频导数、C/种子梯度、角色重算、checkpoint/迁移及普通/在线反馈推理。CPU 检查不证明真实 8B/5B 模型的显存、速度或视频质量。

[实际验证记录及日志](../archive/20261027_cleanup/analysis/implementation_message_verifier_v17_v20_20261026/verification.json)：129 项全量 CPU 回归通过（104.974 秒）；补充真实 teacher 生成与角色隔离检查后，9 项专项测试通过（51.018 秒，与全量有重叠）。四版均通过资源预检：3,857 条记录，训练 3,666、验证 191；初始化 checkpoint 存在。评估 CLI 和实际训练入口的 `--check-only` 可执行。没有启动真实大模型训练或提交 Determined 任务。

---

以下为原有 v15/v16 的设计和入口，保留用于旧 checkpoint 复现。

两版已实现为独立 checkpoint revision，共用现有 `aligned_native_cross_attention` Wan 注入。新条件模块位于冻结 Qwen 之后；Qwen 维度为 4096，条件模块及 C 维度为 3072。没有末端的 `concat(H_content, R)`。

| 版本 | 配置 | revision | 默认推理出口 | C 有效长度 | 条件模块参数量 |
|---|---|---|---|---|---:|
| H_content_message | `configs/native_h_content_message_wisa3857.yaml` | `v522_native_h_content_message_v15` | U1，一次更新 | T | 88,109,056 |
| Q_seed128_message | `configs/native_q_seed128_message_wisa3857.yaml` | `v522_native_q_seed128_message_v16` | R2，首次读取加两次更新 | 128 | 126,257,152 |

参数量包括 4096→3072 投影、更新模块、Q_seed（若有）和对齐参数；不包含冻结 Qwen、Wan 基础参数或 Wan LoRA。计数使用真实配置，在 meta device 上核对，无需载入大模型。

**代码路径与计算边界。**

- `physgen_native/multiround_knowledge.py::_encode_message`：冻结 Qwen 编码，返回完整 memory、有效位置 mask 和内容位置。完整内容不池化，不追加独立 R。
- `physgen_native/message_conditioning.py::MessageCondition`：投影、初始化工作状态、共享模块更新和最终 Wan 对齐。所有新的可训练参数归入 `conditioner`，由现有优化器与 checkpoint 路径完整管理。
- `physgen_native/message_conditioning.py::MessageBlock`：交叉注意力读取完整证据；Q_seed128 额外做双向工作区自注意力；随后 FFN 更新。
- `physgen_native/system.py::encode_condition`：正常训推入口；`encode_condition_rounds` 一次编码返回所有出口。
- `physgen_native/native_attention.py`：继续使用已有的全维 Wan Q/K/V/O 和 C mask；首帧直接 C 注入保持关闭。

H_content_message 的 U0 是投影后的内容，U1/U2 是一/两次模块更新。Q_seed128_message 的 R0 已经过一次完整模块读取，R1/R2 再追加一/两次更新。因此相同出口编号不代表两版执行同样次数的更新。

默认 H_content 训练出口为 1、2，默认推理出口为 1；Q_seed128 训练出口为 0、1、2，默认推理出口为 2。出口选择复用同一套参数，不构造额外副本。`--reasoning-round 0` 可作明确的零追加更新干预：H_content 得到 U0，Q_seed128 得到首次读入证据的 R0。

训练与推理主入口仍为单样本视频计算、梯度累积。条件模块支持变长 memory/content mask，原生读出也能屏蔽 C padding；本改动没有将现有 Wan 视频封装扩展成多样本 batch 采样器。

**冻结 Qwen，不安装 Qwen LoRA，也不训练新的评分头。**

新模式在 Qwen 加载时跳过 LoRA 安装。条件编码使用 no_grad，条件模块再次明确 detach Qwen 输出；更新 P/G/Q_seed 不会回传条件编码器。生成阶段保留原生 Wan 的学习安排：interface 冻结全部 Wan，joint 开启 Wan LoRA。

为了兑现设计中的固定现成 verifier，两版使用 `verifier.kind: qwen_native`。它读取冻结 Qwen 原有 LM head 的三个答案 logits：A 表示满足要求且可见过程连贯，B 表示可见违例，C 表示证据不足。这里的答案 C 与条件张量 C 无关。损失为三答案归一化后的 `-log P(A)`，temperature 默认 1。

该 scorer 没有可训练参数，没有 `QwenProcessCritic` 校准。因此两版阶段为 interface=64、joint=160；旧版的 calibration 阶段不出现在新配置中。旧版配置及其已训练评分头路径仍可加载。

judge 参数被冻结，但评分前向保留对视频的导数，反馈经过 VAE、16 步训练生成、C，再到新模块；该项反馈不更新 Wan 参数。它没有消除 Wan/verifier 反传的计算成本。评分仍按现有视图采样 6 帧，原生 VLM 判断也不是物理真值。

**在项目目录下分别启动。**

资源预检，不启动训练：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B train/train.py \
  --config configs/native_h_content_message_wisa3857.yaml --check-only

/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B train/train.py \
  --config configs/native_q_seed128_message_wisa3857.yaml --check-only
```

在已分配的 GPU 环境中训练，可用同一份旧 generator checkpoint 初始化两版：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B train/train.py \
  --config configs/native_h_content_message_wisa3857.yaml \
  --initialize-generator /path/to/existing_generator.pt

/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B train/train.py \
  --config configs/native_q_seed128_message_wisa3857.yaml \
  --initialize-generator /path/to/existing_generator.pt
```

`/path/to/existing_generator.pt` 替换为已有 native checkpoint。迁移复用 Wan LoRA 和兼容的 C 投影/对齐权重，旧 Qwen LoRA、旧工作区和旧 critic 不导入新版。新 G/Q_seed 需要学习。初始化报告明确列出保留、退休和新增的张量；这不是继续旧 optimizer 的 resume。

若从原始基础模型开始，省略 `--initialize-generator`。已准备两份独立 BLK-96G 单卡 YAML，复用既有环境与旧 v14 launcher 的 warm-start 路径：

- `archive/20261027_cleanup/train/native_h_content_message_1x96g.yaml`
- `archive/20261027_cleanup/train/native_q_seed128_message_1x96g.yaml`

两者 `slots_per_trial: 1`，本次没有提交。首次短程工程运行可在 train 命令增加 `--smoke`；此时两个阶段各执行一次更新。

恢复本版训练必须使用本次运行保存的配置：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B train/train.py \
  --config /path/to/run/config.yaml --resume /path/to/run
```

两版推理均由 checkpoint 内的配置恢复模型结构，无需另传版本开关：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B inference/infer_heldout.py \
  --checkpoint /path/to/h_content_run/latest.pt \
  --output /path/to/output_h_content --samples 4 --seed 42 --reasoning-round 1

/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B inference/infer_heldout.py \
  --checkpoint /path/to/q_seed128_run/latest.pt \
  --output /path/to/output_q_seed128 --samples 4 --seed 42 --reasoning-round 2
```

`inference/infer_shared_videophy2.py` 同样支持两版 checkpoint 和 `--reasoning-round`。现有 `--refine-c` 可在同一条采样轨迹中直接更新 C，使用固定的原生 verifier；默认关闭。

**本次已完成的软件验证。**

2026-10-26 使用现有 moviestory 环境执行：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B \
  -m unittest discover -s tests -p 'test_*.py' -q
```

121 项 CPU 测试通过，66.170 秒。其中 8 项新增集成测试覆盖两个版本：真实 Qwen 类的小尺寸模型编码；完整内容及固定 128 长度；checkpoint 重算梯度；padding 与批次不变性；现有 Wan 注入读出；生成与 verifier 梯度归属；保存/恢复/旧版本迁移；普通采样与在线 C 更新。

两版 `--check-only` 均返回 ready=true：现有 3,857 条记录（训练 3,666，验证 191），无缺失文件，无需预处理。未运行真实 8B/5B GPU 训推，也未宣称生成质量或物理分数提升。
