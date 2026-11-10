> 整理前的历史 README，于 2026-10-27 保存。文中的“当前”指当时版本；现在的入口和状态请看[根目录 README](../../README.md)。历史绝对路径按 cleanup_manifest.json 中的 moves 查找。

# V5.2.2 Native：由生成反馈训练的物理依赖条件

## 当前四套训练入口：v25–v28

按最新要求：**accumulation=4；冻结 verifier 224 步，LoRA verifier 256 步；Q 固定 128 槽位；memory/G 保持 4096 维，最终投影到 Wan 的 3072 维。** verifier 每隔 3 帧观察一次并补末帧，共 28/81 帧，保持原生 832×480。新训练删除 RGB process loss，使用 FM=1、temporal FM=0.1、联合阶段 support-NLL verifier=0.05；LoRA 校准 soft-CE=1。

| 方案 | 训练配置 | 提交到 blk-96g 的 YAML |
|---|---|---|
| H＋冻结 verifier | [v25](../../configs/native_h_content_message_verifier_frozen_simple_wisa3857.yaml) | [H frozen](../../train/native_h_content_message_verifier_frozen_simple_1x96g.yaml) |
| H＋verifier LoRA | [v26](../../configs/native_h_content_message_verifier_lora_simple_wisa3857.yaml) | [H LoRA](../../train/native_h_content_message_verifier_lora_simple_1x96g.yaml) |
| Q128＋冻结 verifier | [v27](../../configs/native_q_seed128_message_verifier_frozen_simple_wisa3857.yaml) | [Q frozen](../../train/native_q_seed128_message_verifier_frozen_simple_1x96g.yaml) |
| Q128＋verifier LoRA | [v28](../../configs/native_q_seed128_message_verifier_lora_simple_wisa3857.yaml) | [Q LoRA](../../train/native_q_seed128_message_verifier_lora_simple_1x96g.yaml) |

每份 YAML 使用一个 blk-96g slot。完整的学习率、系数理由、梯度路径、旧权重初始化和提交命令见 [设计与运行说明](../../docs/message_reasoning.md)。四个 `train/train_*_verifier_*_blk96g.sh` 默认预检，传入 `--submit` 才提交。不添加 hash/SHA 校验。

故障恢复修正：Qwen 分析改为四行、最多 100 词；达到 token 上限时保留已有非空分析继续训练，记录截断状态。teacher 观察同样简洁且允许部分有效输出。verifier 在 eval 模式下显式逐层重算，控制 28 帧原生视频反向传播的显存。

四套真实 blk-96g 阶段测试均完成，覆盖 interface／calibration（LoRA）／joint、accumulation=4、81 帧采样、28 帧原尺寸观察和完整 16 步梯度，峰值 63.6–64.6 GiB。[验证记录](../../analysis/recovery_simple_blk96g_20261026/verification.json)。阶段测试每阶段只运行一个更新，正式 224／256 步任务另行运行。

2026-10-27 再次恢复：Search 66384 / Run 67764（Q）与 Search 66385 / Run 67765（H）都耗尽重试，六次分配均为同一 H100 UUID。最后记录的显存约 45 GiB，容器 OOM 计数为 0；已有旧版模型对照也在该设备上复现 SIGKILL，信号发送者仍未确认。[本次诊断](../../analysis/recovery_lora_20261027/diagnosis.json)。两份断点均停在已保存的第 64 步，H 曾完成但未保存第 65 步。

已提交 **blk-96g、每任务一张 RTX PRO 6000** 的续训：Q = Search 66418 / Run 67798（slot 2，[YAML](../../train/native_q_seed128_message_verifier_lora_simple_resume64_20261027_1x96g.yaml)），H = Search 66419 / Run 67799（slot 3，[YAML](../../train/native_h_content_message_verifier_lora_simple_resume64_20261027_1x96g.yaml)）。恢复原模型、optimizer、scheduler、RNG 和对齐参数，保留原 256 步预算与 accumulation=4；每步原子保存并记录微批执行。恢复入口增加指定失败 GPU UUID 的启动拒绝检查；该检查只核对已分配单卡，不改变集群调度或访问其他 GPU。第 64 步断点及原日志已备份，4 项恢复测试通过。[提交记录](../../analysis/recovery_lora_20261027/submitted_jobs.json)。

2026-10-26 历史恢复：当时 H frozen=66356、Q frozen=66353 继续运行；H LoRA 的 66355→66365 和 Q LoRA 的 66357→66366 连续退出 137，均分配到 blk-96g 的 H100 slot 0。H LoRA 的 66385 从原 66355 目录第 60 步继续；Q LoRA 的 66384 从原初始化开始。临时 amp-80g 任务 66374/66375 已停止。

恢复入口每 4 步原子保存，并允许最多 2 次 Determined 重启；重启加载最新模型、optimizer、scheduler 和 RNG，保留 256 步预算和 accumulation=4。149 项完整 CPU 回归通过，随后包含新增首次启动检查的 3 项恢复专项测试通过。此改动提高中断恢复能力，不代表消除了外部 SIGKILL。完整的实验用途、pipe 版本、代码/YAML 路径、损失/学习率和 ckpt 位置见 [实验记录](../../analysis/recovery_simple_blk96g_20261026/experiments_v25_v28_20261026.json)，[失败与替代任务映射](../../analysis/recovery_simple_blk96g_20261026/jobs.json)。

## 历史原生几何修正：v21–v24

reasoning 首帧、verifier/teacher 视频观察、RGB 过程监督统一使用生成画面的 **832×480**。取消 `832×480 → 320×192 → 512×288` 的非等比缩放，也取消 RGB 监督的 208×120 输入和内部空间降采样（`sequence.scales: [1]`）。尺寸不符会报错；不通过拉伸适配。旧 v17–v20 只保留复现原预处理，后续使用下面四份修正版：

| 方案 | 配置 | 单卡训练入口 |
|---|---|---|
| H + 冻结 verifier，v21 | [配置](../../configs/native_h_content_message_verifier_frozen_geometry_wisa3857.yaml) | [YAML](train/native_h_content_message_verifier_frozen_geometry_1x96g.yaml) |
| H + verifier LoRA，v22 | [配置](../../configs/native_h_content_message_verifier_lora_geometry_wisa3857.yaml) | [YAML](train/native_h_content_message_verifier_lora_geometry_1x96g.yaml) |
| Q128 + 冻结 verifier，v23 | [配置](../../configs/native_q_seed128_message_verifier_frozen_geometry_wisa3857.yaml) | [YAML](train/native_q_seed128_message_verifier_frozen_geometry_1x96g.yaml) |
| Q128 + verifier LoRA，v24 | [配置](../../configs/native_q_seed128_message_verifier_lora_geometry_wisa3857.yaml) | [YAML](train/native_q_seed128_message_verifier_lora_geometry_1x96g.yaml) |

[几何修正与全面维度审计](../../docs/message_reasoning.md)。用真实 Qwen/Wan 类在 meta device 实例化四套生产配置后，没有实例化旧 rank-64 算子、紧凑 768 维注意力或额外评分头。`4096→3072` 条件适配与 Q_seed128 的固定 token 容量仍存在，不能称为无损；具体原生/新增映射及剩余限制见审计。未提交新 GPU 任务。

[验证记录](analysis/native_geometry_dimension_audit_20261026/verification.json)：135 项回归通过；最终原尺寸 RGB 监督改动后的 5 项专项检查通过（22.306 秒）；四版资源预检通过。未运行真实大模型 GPU 训练。

## 2026-10-26：四套 message + verifier 全流程（v17–v20）

已增加 H_content_message / Q_seed128_message × 冻结 verifier / 独立 verifier LoRA 四套配置，全部保留 verifier 和现有满宽 Wan 注入。评价实际视频的要求满足、前提响应、持续终止和属性连续性，显式区分支持、违例与未知，并检查答案标签偏置。LoRA 版增加独立校准阶段，只训练 judge 适配器；生成训练时冻结 judge，反馈经过视频回传到 C。生成侧 Qwen 在四版中都冻结，Wan LoRA 训练保留。

| 版本 | 配置 | 单卡入口 |
|---|---|---|
| v17：H + 冻结 verifier | [配置](../../configs/native_h_content_message_verifier_frozen_wisa3857.yaml) | [YAML](train/native_h_content_message_verifier_frozen_1x96g.yaml) |
| v18：H + verifier LoRA | [配置](../../configs/native_h_content_message_verifier_lora_wisa3857.yaml) | [YAML](train/native_h_content_message_verifier_lora_1x96g.yaml) |
| v19：Q128 + 冻结 verifier | [配置](../../configs/native_q_seed128_message_verifier_frozen_wisa3857.yaml) | [YAML](train/native_q_seed128_message_verifier_frozen_1x96g.yaml) |
| v20：Q128 + verifier LoRA | [配置](../../configs/native_q_seed128_message_verifier_lora_wisa3857.yaml) | [YAML](train/native_q_seed128_message_verifier_lora_1x96g.yaml) |

[完整设计、梯度归属、训练/恢复/推理/评估命令与限制](../../docs/message_reasoning.md)。共享训练入口自动执行全部阶段；新增 [heldout verifier 评估](../../inference/evaluate_message_verifier.py) 同帧比较基础 judge 与 adapter，并导出实际生成视频。冻结版共 224 更新，LoRA 版共 256 更新。LoRA 校准采用固定 observation teacher 的弱标签，不自动把参考视频判对，也不等同于真实物理标注。代码和配置已编写，未提交训练任务，实际生成质量尚待 GPU 运行验证。

[验证记录](analysis/implementation_message_verifier_v17_v20_20261026/verification.json)：129 项全量 CPU 回归通过；随后更新后的 9 项专项集成测试通过（与全量测试有重叠）；四份配置资源预检通过。

## 2026-10-26：H_content_message / Q_seed128_message 两套独立训推实现

新增 [H_content_message 配置](../../configs/native_h_content_message_wisa3857.yaml)（v15）和 [Q_seed128_message 配置](../../configs/native_q_seed128_message_wisa3857.yaml)（v16），由 [message_conditioning.py](../../physgen_native/message_conditioning.py) 构建统一 C，并接入现有全维 Wan 注入。前者保持内容长度 T，后者固定 128 个可学习工作 token；均取消 H_content/R 末端拼接，冻结 Qwen 且不安装其 LoRA。新版使用冻结 Qwen 原生答案 logits 反馈，没有新评分头或 calibration 阶段。

[训推命令、出口含义、checkpoint 迁移与验证记录](../../docs/message_reasoning.md)。两份单卡 YAML 已准备，未提交训练。121 项 CPU 测试通过；两版资源预检均通过。旧版默认入口保留，新版请显式选择对应配置。

## 2026-10-26：完整内容与全层多轮 Qwen reasoning（v14，显式配置启用）

新增 [v14 配置](../../configs/native_multiround_reasoning_wisa3857.yaml) 和 [单卡入口](train/native_multiround_reasoning_1x96g.yaml)，适配已经修正的 v13 全维注入。未提交训练任务；v13 配置和训练脚本的默认入口保持不变，旧 checkpoint 继续选择其原有实现。新参数使用独立的 `v522_native_multiround_reasoning_v14` revision。

- **完整内容保留**：要求、有效 clauses、analysis 的全部选中 token 构成 `H_content [B,T,4096]`；取消按段分配预算和平均池化。来源标记在文字前，只有完全相同的 clause 记录去重。文本预算 4096，超限明确报错，不静默截断。
- **适配后的文字分析**：生成分析时启用同一个物理 LoRA，离散生成保持 `no_grad`。初始上限 512 token；未到 EOS 时以 1024 上限重新生成，仍未完成则报错。缓存键包含输入、模板、预算、图像尺寸和 LoRA 参数版本；固定 teacher/verifier 的分析仍关闭 LoRA。
- **逐层证据记忆**：一次原生多模态编码保留全部 36 层的 K/V，每个为 `[B,8,L,128]`。保留原生 DeepStack 和 mRoPE；每轮只计算新的关系工作区，不重算完整前缀，不累积历轮工作区 KV，不切断训练梯度。
- **初始读出和追加两轮**：`R0=F(Q_seed;M_mem)`，`R(k+1)=F(Q_seed+A(Rk);M_mem)`。F 复用完整 Qwen 1–36 层，Qwen/LoRA/A 跨轮共享；S 初始为 32，每个 R 均为 `[B,S,4096]`。A 是 RMSNorm 和满宽 `4096→4096` 线性映射，没有额外 64 维瓶颈。
- **各轮可单独使用**：保留 `Zk=Concat(H_content,Rk) [B,T+S,4096]`。训练随机选择 0/1/2 个追加更新的出口，普通推理默认选 2；`encode_condition_rounds` 一次返回各轮 Z/C。工作区采用原生因果注意力，每个位置能读完整证据及允许的工作区前缀。
- **C 与全维注入对接**：共享 RMSNorm/Linear `4096→3072`，保留 v13 的 3072 维 LayerNorm/gain/bias 对齐，移除 v14 末端 roles 乘法。C 直接进入原生 `3072→3072` Q/K/V/O、24×128 heads；不恢复紧凑注意力、rank-64 算子、强度头或 VJEPA2。

实现分为 [不可变 KV 与共享工作区](../../physgen_native/recurrent_reasoning.py)、[输入、缓存、多出口和监督接口](../../physgen_native/multiround_knowledge.py)。当前视频训练仍为既有的单样本微批，因此每次 C 按实际长度使用，不填充。额外提供 `pack_conditions` 和 `generator(..., condition_mask=mask)`；需要填充时必须传有效 mask（True 为有效），不能靠零值 padding 代替屏蔽。这不把原有单视频采样器扩展为多视频批处理。

```python
outputs = model.encode_condition_rounds(conditions, first_rgb)
r0, r1, r2 = outputs['relation_states']  # [B,S,4096]
z0, z1, z2 = outputs['readouts']         # [B,T+S,4096]
c0, c1, c2 = outputs['conditions']       # [B,T+S,3072]
c1_only = model.encode_condition(conditions, first_rgb, reasoning_round=1)
```

`analysis_sft_loss(first_rgb, caption, clauses, target, duration=...)` 提供显式分析目标的 teacher-forcing；`relation_sft_loss(rk, question, answer)` 只从 Rk 和问题读出答案，不额外访问原始上下文。它们是可单独训练的监督接口，需要审核过的场景相关目标；当前生成训练没有把模型自写的分析当成正确标签，也没有悄悄添加物理标注数据或额外损失。现有 FM / 时间 FM / RGB 过程 / 冻结 verifier 反馈继续训练 C、物理 LoRA 和新工作区。

新工作区归属独立的 `reasoning` 参数组，在 interface/joint 更新，在 calibration 冻结；Wan 在 interface 全部冻结，joint 只开放既有 LoRA。checkpoint 保存工作区并支持精确恢复。旧权重初始化显式丢弃末端来源乘法和旧接口，新增工作区；从 v13/v14 继承已学对齐，从更早版本初始化对齐统计。初始化不恢复旧 optimizer。

```bash
bash tools/runtime_python.sh train/train.py --config configs/native_multiround_reasoning_wisa3857.yaml --check-only
# 对 v14 checkpoint 分别生成 C0 / C1 / C2；每次指定一个出口。
bash tools/runtime_python.sh inference/infer_heldout.py --checkpoint /path/to/v14.pt --reasoning-round 1
```

`--without-latent-reasoning` 在 v14 中选 R0（零次追加更新）；`--without-reasoning` 只关闭离散分析。运行记录分别保存实际出口和配置的最大更新轮数。

**113 项 CPU 检查通过（44.083 秒），其中新增 14 项 reasoning 检查；资源预检通过，3,857 条既有缓存可用。** 验证包括真实 Qwen3-VL 小模型的多模态缓存/完整重算数值等价、checkpoint 梯度等价、跨轮梯度、全维读取的 padding 屏蔽、BF16 前后向、冻结 verifier 到工作区的完整短轨迹反馈，以及版本迁移和恢复。见 [针对性测试](../../tests/test_multiround_reasoning.py)、[完整日志](analysis/implementation_v14_reasoning_20261026/cpu_tests.log)、[资源预检](analysis/implementation_v14_reasoning_20261026/resource_audit.json)。这些检查不代表完整 8B+5B GPU 资源验证或物理效果改善。

## 2026-10-26：删除 a、全维 C 对齐（v13，当前默认入口）

已按用户要求移除当前路径中的 a 预测头、`(a−0.5)r` Query 调制及强度损失。新版不读取 `v52/change`、`train_scale.json` 或 VJEPA2 缓存。Q 直接来自原生 block 的 `norm3(H_s)`；使用原生 3072→3072 Q/K/V/O、24×128 heads，保留原生文本分支、残差和 FFN。[读取实现](../../physgen_native/native_attention.py)

C 路径为 Qwen 上下文特征 → 既有 4096→3072 投影与来源编码 → **3072 维 LayerNorm → 可学习逐通道缩放和偏置 → C**。LayerNorm 无内部仿射，独立的 gain/bias 共 6,144 个参数；没有新的降维、状态算子、门控或 T5 特征模仿损失。[对齐实现](../../physgen_native/context_alignment.py)

新训练初始化时，从原训练 split 均匀取 32 条已有文本缓存，经过 **实际 Wan `text_embedding`** 后，包含原生 512-token padding，统计每通道均值 μ 和标准差 s，令 `gain=s, bias=μ`。统计只设置初始化尺度，语义对应由生成训练学习；不保证 C 与文本的注意力消息已经等幅。实际 BF16 文本模块统计覆盖 16,384 个 context tokens，其中未补齐 tokens 为 5,578：`RMS(μ)=0.00813892`、`RMS(s)=0.08905730`、context RMS `0.08942843`。[真实缓存记录](analysis/implementation_v13_20261026/native_context_statistics.json)

| 阶段 | 更新参数 | 损失 |
|---|---|---|
| interface，64 更新 | C 投影/既有来源编码、对齐 gain/bias、Qwen 知识 LoRA；整个 Wan（含已有 LoRA）冻结 | FM + 0.1 时间 FM + 0.1 RGB 过程 |
| calibration，32 更新 | verifier 评分头；生成分支冻结 | 软偏好排序 |
| joint，160 更新 | C、对齐参数、Qwen 知识 LoRA、Wan LoRA；Wan 基础权重冻结 | FM + 0.1 时间 FM + 0.1 RGB 过程 + 0.02 verifier |

Verifier 项仍沿完整短去噪轨迹，经生成视频、Wan 和对齐层回传到 C/知识 LoRA，该项不更新 verifier 或 Wan 参数。联合阶段 Wan LoRA 由生成类目标更新。

当前默认是 [v13 配置](../../configs/native_aligned_attention_wisa3857.yaml)，[单卡训练入口](train/native_aligned_1x96g.yaml) 执行完整三个阶段；需要限定更新预算时可给训练脚本传 `--stop-after`。旧生成权重迁移保留 C/Qwen/Wan LoRA，弃用旧接口，再初始化新的 affine；v13 精确恢复或 v13 权重迁移保留已学 affine，不重新校准。普通推理只读取 checkpoint，不依赖初始化缓存。

**99 项 CPU 检查通过（28.903 秒），资源预检通过，现有 3,857 条缓存可直接使用。** 新增检查包括：原生 Q 只投影一次、关闭 C 的输出精确一致、首阶段真实优化不改变非零已有 Wan LoRA、joint 解冻及学习率、verifier→C/对齐/Qwen 梯度归属、梯度重算、移除强度缓存依赖，以及迁移/保存恢复。[检查记录](analysis/implementation_v13_20261026/cpu_tests.log)。这些证明实现行为；尚未执行 v13 适配训练，不将其表述为生成质量或物理效果已经改善。

## 2026-10-26：全维 C 注入修正（v12，历史实现）

针对用户对图中 `3072→768`、`3072→64→3072` 和额外门控的质疑，已实现 [v12 全维注入](../../docs/fullwidth-injection-v12.md)。C 的读取复用选定 Wan block 的原生 **3072→3072 Q/K/V/O、24×128 heads**，取消额外的紧凑注意力、rank-64 状态算子和输出 sigmoid 标量门控。文本和 C 分别归一化注意力，在原生 O 之前合并消息；C 在 block 内 cross-attention 位置参与当前 FFN。首帧没有直接 C 注入，原生残差、时空 self-attention 和去噪调制保持。

连续可见变化强度 `a` 保留为轻量监督读出，并通过一个 3072 维向量软调制 C query；它不再乘在输出消息上。这个辅助标量读出仍含 `3072→192→1`，**C 消息不经过它**。Qwen 的 `4096→3072` 语义对接保留；Wan LoRA 是完整原生矩阵上的增量，不能与“整个注入消息只能通过 rank-64”混为一谈。Qwen reasoning、集中 C、无 VJEPA2、verifier→C 的完整反馈路径保持，阶段与损失仍为 3 阶段、4/1/4 项。

当时的 [v12 配置](../../configs/native_fullwidth_attention_wisa3857.yaml) 和 [单卡训练 YAML](train/native_fullwidth_1x96g.yaml) 保留用于该版本；当前默认已更新为上方 v13。v12 的旧 checkpoint 迁移继承 C/Qwen/Wan-LoRA/强度读出，不导入旧 optimizer；该版本 native reader 与 C 的语义对齐仍需训练。

**91 项 CPU 检查通过（27.425 秒）；资源预检通过，复用 3,857 条缓存。** 从真实 v11 step16 权重测得首个旧算子的两个矩阵秩均为 64；构造 RMS=1 的读出消息变化，其经过旧 `W_c` 后 RMS 约 `4.80e-16`。这是直接接口的代数限制，不是物理信息丢失比例或视频效果结论。[检查记录](analysis/queue_20261026_fullwidth_attention/cpu_tests.log)、[旧算子证据](analysis/queue_20261026_fullwidth_attention/old_rank_evidence.json)。

**Run66281 已于 07:21:39 UTC 在 BLK-96G 单卡完成，脚本耗时 245.10 秒。** 关闭 C 时与真实原生 Wan 的最大输出差为 0；生成目标、完整 16 步 verifier→C 反馈的 C 梯度 RMS 分别为 `4.33e-5`、`8.28e-7`。不做优化更新，随机评分头的导数不作为物理收益。[执行记录](analysis/queue_20261026_fullwidth_attention/job.json)。

**这次执行同时暴露了迁移对齐问题。** 在这一个未适配样本上，原生 O 投影前 C/text 注意力消息 RMS 比为 12.27–383.80；开启 C 对最终 FM 输出的改变量 RMS 为 1.4537，关闭 C 的输出自身 RMS 为 1.3936。幅度变化不是质量提升，旧 C 不能仅凭同为 3072 维就视为已对齐原生 K/V。强度 a 的单样本 SmoothL1 为 0.01038，全零变化基线为 0.00404；此时尚不能建立 a 的预测可靠性。当前 `ΔQ=(a−0.5)r` 只是一个标量控制的秩一 query 偏移，且训练只监督十层 a 的平均值；不将它称为已有效的物理信息增强。[复算证据](analysis/query_intensity_review_20261026/evidence.json)、[相关原论文/代码检索](analysis/query_intensity_review_20261026/literature.json)。

**66272 保留 v11 快照和旧注入结构**，它的 40 例评测不能代表 v12。当前不声明全维注入已带来涨分或解决长程一致性；本次分析没有把尚未验证的 a 替代结构加入运行模型。

## 2026-10-26：当天结果优先的 40 案例基线（v11 / Run66272）

按用户最新要求，先用已有 60 个 VideoPhy2 共享首图案例中的 **VP001–VP040** 做固定开发比较，复用历史 Wan-LoRA 视频和评分，不重新生成基准。它是项目的 I2V 子集比较，不是官方完整 T2V 榜单。目标是得到实际物理/语义涨分及可接受的完整视觉 demo；**目前尚未产生新版分数，不能声明 3–5 个百分点提升。** [固定协议与历史逐例结果](../../analysis/videophy40_20261026/protocol.json)。

| 已完成的 40 例参照 | SA 均分（1–5） | PC 均分（1–5） | SA、PC 均≥4 |
|---|---:|---:|---:|
| 主参照：joint400 Wan-LoRA，关闭旧 Planner/Adapter | 3.050 | 3.350 | 22.5% |
| 独立训练的 LoRA-only400 | 2.975 | 3.400 | 17.5% |

选择在新版评分前固定，不按胜负筛选；40 条 caption 与 3,666 条训练 caption 无完全相同项，这不等于已经排除语义或视觉近重复。评价仍用既有独立 VideoPhy2 AutoEval、32 帧与相同解码参数，未将该评分器用于训练。输出将包含配对变化、逐例胜负和按案例 bootstrap 区间；40 例联合通过率每例对应 2.5 个百分点，小幅点估计不能单独证明稳定收益。

**本次实际修正：**

- 复用主参照已有 **joint400 Wan attention LoRA（480 个张量，rank/alpha=32）**，只导入 Wan 的 LoRA，不导入旧 Planner、S 或 optimizer。复用 66180 step144 的 Qwen/C/接口权重并固定初始化快照，然后覆盖为上述已训练 Wan-LoRA。新增训练不是从随机 C 和零 Wan-LoRA 重头开始，已有训练预算单独记录。
- **Wan-LoRA 从首个新增生成更新就参与训练**，知识 LoRA、集中 C、状态依赖接口继续训练；不再等待数百步才开放生成器的适配能力。推理仍只采样一条 Wan 轨迹。
- 在全部噪声区间，对 FM 预测误差 `e = v − (noise − clean)` 施加固定的跨视频时间二次度量 `mean_lag ||e[t+lag]−e[t]||²/2`，latent 时间跨度为 1/2/4/8/16。该度量与 FM 联用保留相同的条件均值最优解，直接约束向量场误差的时间协调；**它不是物理速度目标，也不是额外未来预测器。** 低噪声（sigma≤0.35）仍保留覆盖全部 81 帧和长跨度的 RGB 有符号变化监督；高噪声不再通过非线性 RGB 单步恢复去拟合唯一未来。这是基于已有晚段幅度收缩证据的训练修订，效果仍由真实视频决定。
- **VJEPA2 保持停用，Qwen reasoning/C/verifier 均保留。** 后续校准改为观察当前带 C 的生成结果，joint 中仍由冻结 verifier 将视频梯度定向回传 C/知识分支。Qwen 基础判分的事实错误尚未解决，首轮不把未经校准的分数作为生成奖励，也不声称 verifier 已有物理收益。

完整 [v11 配方](../../configs/native_generation_baseline_wisa3857.yaml) 为 interface 64 / calibration 32 / joint 160，损失分别 **4 / 1 / 4**；本轮只执行前 **32 次新增更新 × 8 微批**，不是执行完整 256 步。四项生成目标明确为 FM、temporal FM、低噪声 RGB process、intensity；joint 将 intensity 替换为 verifier，不隐藏第五项。

**Run66272 已于 06:08 UTC 启动，blk-96g 单卡。** 自动顺序为：32 次训练→两段既有缓存留出 demo→固定 40 段新版 VideoPhy2 生成→独立评分→与两组历史参照配对汇总。训练和评分采用现有各自的解释器；阶段切换替换进程，不增加 GPU worker。环境、Determined 0.32.1、旧训练任务和数据缓存均未修改。[运行配置与初始化记录](analysis/queue_20261026_videophy_baseline/job.json)。

83 项行为检查通过（27.67 秒），包括低噪声与高噪声目标分工、晚段导数、FM 最优解保留、Wan-LoRA 精确导入及阶段梯度归属；3,857 条缓存完整可用。新 165 文件代码快照已固定。上述检查只证明实现行为，不是涨分证明。首轮 checkpoint 未经过 verifier 阶段，不能把它的任何收益归功于 verifier。

真实执行已完成首个正常 8 微批更新：109.16 秒、峰值 65.08 GiB；C、知识 LoRA、接口和继承的 Wan-LoRA 都接受生成训练，critic 梯度为零，符合首轮职责。见 [实际执行记录](analysis/queue_20261026_videophy_baseline/execution_evidence.json)。自动流程已开始，无须重跑历史基准；新 40 例分数仍待生成完成。

此前的 **v9 Qwen process** 已去掉 VJEPA2；进一步修正视频证据读取的 **v10 grounded Qwen** 已完成六对原始输入与 RGB 导数检查，判分质量仍不足以宣布有效，见下节。新的 v11 基线继承去除 VJEPA2 和保留 Qwen/C 的路径。既有 v8 运行保留。原 [简明介绍与流程图](../../docs/native-pipeline-guide.md) 和 [离线交互讲解页](../../docs/native-pipeline-explainer.html) 描述 v8，不能据此认为新版仍使用 VJEPA2。2026-10-26 的“完全冻结生成侧 Qwen”另文仍是设计建议，未替代这里保留连续 reasoning / 知识 LoRA 的实现。

## 2026-10-26：66215 结果与去除 VJEPA2 的 v9

**66215 于 03:29 UTC 完成，固定评估的是 v8 step16，不是 66180 的最新权重。结果证明 C 和连续 reasoning 能改变视频，尚未证明稳定改善物理效果或解决后段一致性。** 两个留出场景 × 两个噪声种子 × 两个 sigma 构成 8 个配对设置，不是 8 个独立场景。完整生成含正确 C、关闭 C、关闭连续 reasoning 三种，共 6 段。

| 对照 | 正确 C 的 FM 相对收益 | 正确 C 的过程误差相对收益 | 过程误差更低的配对数 |
|---|---:|---:|---:|
| 关闭 C | 1.484% | 1.033% | 7/8 |
| 关闭连续 reasoning | 0.516% | 0.216% | 4/8 |
| 替换文本条件 | 0.0367% | 0.0189% | 5/8 |
| 同为 16 更新的 naive T5→C（66201） | −0.461% | 2.821% | 4/8 |

这些小差异及很少的独立场景不能建立泛化优势。海浪条款全部来自首帧，旧文本替换还保留了这些条款，因此该替换也不是干净的物理条件干预。来源：[配对汇总](analysis/results_66215_20261026/paired_summary.json)、[逐视频变化量](analysis/results_66215_20261026/decoded_video_comparison.json)、[火柴及真实参考](analysis/results_66215_20261026/source0000042_with_reference.png)、[海浪及真实参考](analysis/results_66215_20261026/source0000086_with_reference.png)。

**66211 的训练时消融已完成，结论更明确。** 从第一步起关闭连续 latent reasoning，保留 Qwen 条件编码，同样训练 16 次更新；与完整模型配对后，FM 相对收益为 −0.0041%，过程误差相对收益为 −0.0064%，均各赢 4/8。这里是“分别训练”的对照，不是仅在推理时拔掉模块；这些差异不足以支持 reasoning 收益。它进一步支持修正训练职责：普通 FM/变化拟合尚未要求连续推理提供不可替代的物理信息。[完整汇总](analysis/results_66215_20261026/trained_no_reasoning_comparison.json)。

火柴正确 C 的结构较关闭 C 稳定，火焰发展也不同；但手部在火源附近停留过久，仍不是可靠的正确过程。真实缓存片段约 2.67 秒，并未完成大规模燃烧，不能把“火更大”当成更符合物理。海浪几种生成没有明确的视觉胜者。在 sigma=0.95 的配对恢复中，最后位置变化 skill 的两种子均值：火柴 −0.858，海浪 −0.016；这是相对零变化的恢复指标，不等于自由生成的物理评分。

v8 step16 尚未训练 verifier；该阶段 VJEPA2 未参与损失，66215 评估也未加载其观察器。因此这些不足不能归因于“VJEPA2 已把生成训坏”，移除它本身也不是效果改善证明。主要训练缺口是：早期 C 只接受普通生成/变化拟合，物理语义反馈要等 400 interface + 460 calibration 更新后才开始；旧反馈又从含真实未来的 FM 状态起步。

进一步读取 **66180 已有固定验证**，同四场景、同噪声种子、sigma=0.95，从 step16 到 step112：过程误差 0.13335→0.12613；末位置的合并变化 RMS 比 0.3289→0.2381，合并方向余弦 0.2456→0.2554，合并变化 skill 5.34%→6.49%。这里按变化能量合并，避免几乎静止样本的比值主导结论。`skill = 2 r cos − r²`，所以在方向基本不变时，缩小不准确的变化也能降低误差。该结果支持“恢复损失下降不等于后期变化方向学好”，而非证明完整自由生成必然冻结；见 [固定验证趋势](analysis/results_66215_20261026/v8_validation_trend.json)。新方案把额外过程判断施加于纯噪声生成，而不强迫输出任意放大变化幅度。

按用户要求，新 [配置](../../configs/native_qwen_process_wisa3857.yaml) 和 [单卡入口](train/native_qwen_process_1x96g.yaml) 实现：

- **不加载 VJEPA2 编码器或 checkpoint，不读 JEPA 缓存。** 保留全部既有 RGB change、VAE、T5、条款缓存，不新增数据集。
- **保留 Qwen 连续 reasoning 和集中 C。** 两轮共享 Qwen 后四层、工作 tokens 联合注意力、原生 Wan 状态依赖注入；内容读出新增文本要求的直接 span，避免仅靠条款/分析间接保留要求。文字分析本身不反传，连续状态与知识 LoRA 可训练。
- **Qwen 直接读取时间有序的实际生成视频和过程要求。** 共享冻结 Qwen 骨干产生前提—响应、过程发展、属性/来源连续性三个上下文读出，训练轻量评分头；这不是已完成微调的整个 MLLM。校准使用同一条件下实际 Wan 生成和参考的后段保持视图，Qwen 先描述视频证据再提供软偏好；不把静止或参考视频预设为绝对正确/错误，位置轮换避免固定参考位置线索。
- **先校准，再从生成训练开始反馈到 C。** 物理项固定评分器，沿 Qwen 视频编码→VAE→Wan→注入接口→C→Qwen 知识/推理 LoRA 求导；该项不更新 Wan/接口参数，生成目标仍更新相应模块。
- **物理反馈从纯噪声出发，对全部 16 个条件去噪步骤求导。** 每步保留完整 81 帧时间域，Qwen 均匀观察包括首尾的 21 帧；没有真实未来混入起点，也没有被忽略的模型前缀。它是 16 步条件采样的导数，尚不能等同 50 步 CFG 推理导数。推理仍是一条采样轨迹，可选同轨迹 C 修正已适配 Qwen 评分，不做多候选筛选或重启。

三阶段共 800 更新；阶段数与损失数遵守用户限制：

| 阶段 | 更新对象 | 损失数及内容 |
|---|---|---|
| calibration 128 | Qwen 视频评分读出，生成路径固定 | 1：软偏好排序 |
| interface 128 | reasoning/知识 LoRA、C、注入接口和强度头 | 4：FM、全时段有符号变化、连续强度、verifier→C |
| joint 544 | 上述生成参数及 Wan LoRA；评分固定 | 3：FM、全时段变化、verifier→C |

监督仍覆盖 81 帧与 1/2/4/8/16/32/64 帧跨度，作用于实际生成估计。全程共享 C 的一致性依赖、后段可见变化与 verifier 的过程反馈一起承担长程任务；不能用“共享 C”或“梯度非零”宣布任务已解决。

**执行状态：75 项 CPU 行为检查通过（29.43 秒）；3,857 条缓存可用（3,666 train / 191 validation）。Run66259 于 04:37:33 UTC 在 blk-96g 单卡完成全部三个阶段，峰值 65.09 GiB。** 三阶段分别 35.81/190.79/209.15 秒；接口和联合阶段完整 16 步纯噪声反馈到 C 的加权梯度 RMS 为 `7.43e-8` / `1.08e-7`，固定 verifier 不接收生成更新，校准阶段生成参数梯度为零。使用独立 155 文件快照；见 [实际结果](analysis/queue_20261026_qwen_process/result.json)。这证明实际执行及梯度归属，评分器只接受过一个校准更新，不是训练后物理收益证据。未修改环境、Determined 版本、v5.2.1 或旧实验快照；66176/66180 保持运行，66211 已自然完成。

**Run66260 于 04:46:36 UTC 以 137 退出，完整优化更新数为 0。** 它原计划执行 16 次 calibration 更新、每次 8 微批，实际只保存了首批两个判断视图；日志没有 CUDA OOM traceback，信号发送来源未确定。不能将 66259 的三阶段各一微批成功外推为正常累积训练已通过。没有据此认定硬件故障。见 [训练记录与失败日志](analysis/queue_20261026_qwen_process_training/job.json)。

Qwen 自己生成的偏好只提供可训练的语义监督，不是独立物理真值。校准留出指标必须与完整视频的中后段过程、合理条件改变、多个现象及种子一起判断；新路径不能用同一个 Qwen 自评循环来证明“强物理因果”。训练首批及固定验证会保存已计算的原始浮点帧和判断理由，避免只留下无法审核的分数；不为此额外采样。视频编码已移出校准训练，原始帧可逐值复现评分输入；后续查看时再生成 MP4。

**首批证据已经否定“换成基础 Qwen 判分就足够”这一假设。** 球拍/粉末案例中，原始教师给生成候选 98.99% 的软偏好，却主要依据“更活跃，暗示交互”，没有充分解释后期球、球拍与粉末的连续性。完整 21 帧显示前段球的上升和返回连续，不能将六张关键帧之间的差异直接叫作位置跳变；后段过程仍需仔细判断。[全部生成帧](analysis/results_66215_20261026/tennis_generated_allframes.png)。冰淇淋短片与后段保持版几乎相同，不应强制要求教师把保持版判为物理错误。

Run66262 复用这两对已保存视频，比较原判断与“先不看要求、逐视频描述早中晚，再比较”。**提示改动未解决问题**：仍出现把可见上升描述成下降、把近静止片段描述成逐渐融化等事实错误。重新读取的 MP4 上，原判断的球拍参考偏好从原始浮点输入时的 0.0101 变成 0.4864；编码、运行模式及硬件同时不同，不能把变化单独归因于压缩，也不能把它称为优化收益。观察优先的双顺序结果约为 0.5，没有建立判错能力。[完整判断记录](analysis/queue_20261026_qwen_observation/results/judgments.jsonl)。该实验选项未开启为默认训练教师。

**66263 已于 05:20:30 UTC 完成正常 8 微批更新及留出验证。** 该更新耗时 227.56 秒、峰值 40.05 GiB，只有评分头接受更新；保存 8 对训练和 8 对验证的精确浮点输入。[实际记录](analysis/queue_20261026_qwen_raw_evidence/first_update/result.json)。这补上了正常累积执行证据，但不能据此确定之前 137 的信号来源，也不是物理判分正确性证明。

**66264 已完成同两对视频的带时间独立图像检查。** 六帧图像版本正确描述了球的上升，并区分冰淇淋的缓慢变化与后段保持；但球拍参考偏好在交换呈现顺序后为 0.501/0.796，仍有明显顺序偏差，比较文字也在 128 token 处截断。输入类型、分辨率、采样帧数同时变化，不能把改善全部归因于其中一项。[判断原文](analysis/queue_20261026_qwen_raw_evidence/frame_observation/results/judgments.jsonl)。本机官方 Qwen3-VL video processor 与现有视频打包的 patch 和 grid 逐值相同，不支持把原先错误归因于 patch 排列。[输入检查](analysis/results_66215_20261026/native_qwen_input_equivalence.json)。

据此实现独立修订 [v10 配置](../../configs/native_grounded_qwen_process_wisa3857.yaml)：先对每段视频完成不看要求的时间事实描述，再结合两段完整描述和原始图像直接判定；正反呈现顺序的概率映射回同一语义后平均。去掉会截断的额外比较文字生成，保留生成侧连续 latent reasoning。对称判分消除最终目标的呈现顺序不一致，**不能修复错误的物理判断本身**。六个时间锚点覆盖首尾，逐帧过程监督仍覆盖全部 81 帧；稀疏语义观察不能保证识别所有短暂事件。阶段仍为 128/128/544，损失仍为 1/4/3，无 VJEPA2，无新数据集或新增推理视频。

**Run66265 于 05:31:49 UTC 完成**，复用 66263 的六对原始浮点帧（球拍、冰淇淋、火、凝胶、咖啡、湖面），没有加载 Wan 或重新生成视频。[实际结果](analysis/queue_20261026_grounded_qwen/result.json)。六帧输入导数全部非零，RMS 为 0.00105，峰值 17.37 GiB；该检查使用随机评分头，只检验 RGB 可微路径，不检验物理优化方向，也不是新版完整 Wan→C 反传的替代证据。

教师仍有实质不足：球拍描述遗漏了生成球先上升再返回的发展，初始悬空状态被描述为静置；正反判断在对称平均前的参考偏好差可达 0.761。冰淇淋慢变/保持的区别被识别，但六对开发样本不足以确定判错准确率。咖啡参考本身可见从两杯变为三杯，因此不能仅凭教师偏好生成候选，就判定教师错误。[原始帧](analysis/results_66215_20261026/raw_teacher_wisa0013758.png)。这些结果要求继续改进可见事实与过程的判断，不能把顺序对称或非零导数当作已经解决。v10 尚未取代 v9 默认入口，也未据此启动长校准训练；生成侧 reasoning、C 和 verifier 的主设计继续保留。

已准备与主配方相同训练目标的 [关闭连续 reasoning](../../configs/ablation_qwen_process_no_latent.yaml) 和 [naive T5→C](../../configs/ablation_qwen_process_naive_text.yaml) 配置，尚未提交新对照矩阵。它们共用 Qwen 视频 verifier，分别隔离生成侧连续推理、生成侧 Qwen 条件的作用；不能把第二项描述为整个系统完全没有 Qwen。

研究依据上，[VideoReward](https://arxiv.org/abs/2501.13918) 已使用视觉语言模型学习视频偏好，[Coconut](https://arxiv.org/abs/2412.06769) 已研究连续 latent 推理。这两项本身不应包装成首次提出。这里要建立的贡献是：**集中物理依赖 C 在 Qwen 连续计算中形成，经生成视频的条件化过程反馈获得训练，再由 Wan 当前状态选择和执行适用的依赖。** [Qwen3-VL](https://arxiv.org/abs/2511.21631) 的视频时间理解提供可用基础，但其通用评测不能替代本项目的物理判错率和生成改善证据。我们未引入这些论文的训练数据。

## 工作区修订：2026-10-25 联合 C 推理与条件响应反馈（v8）

**已实现核心模块和训练修订；Run66180 已开始 v8 正常训练，尚未获得训练后质量证据。Run66176 仍使用不可变 v7 快照运行，按用户要求保持执行。** 新配置为 [native_causal_response_wisa3857.yaml](../../configs/native_causal_response_wisa3857.yaml)，检查点标识 `v522_native_causal_response_v8`。既有 3,857 条缓存、集中 C、连续 reasoning、状态依赖注入、verifier→视频→C 的反馈全部保留。

这次改变由可定位的计算与优化问题驱动：

| 已确认的问题 | 已实现的核心修订 | 能证明到哪里 |
|---|---|---|
| C 工作区沿用自回归三角掩码，较早的工作 token 无法读取较晚工作 token 在推理中形成的新状态 | 原生 Qwen 后部层仍复用两轮；C 工作 token 之间改成联合注意力。初始多模态记忆仍保持原因果掩码，不能反向读取 C | 原生 Qwen 小型模块检查确认早期 C 对后期 C 的导数从零变为非零，记忆输出不变；尚不证明物理推理正确 |
| `softplus((E_correct-E_alternate)/T)` 可以通过使错误条件的视频更差而下降，正确视频不必改善；同时每微批计算两条生成分支 | v8 生成训练只保留正确条件的 FM、完整过程与相应阶段目标。25% 的非 CFG 丢弃微批仅去掉原生 T5 文本，保留 C 中的完整合法要求，使 C 直接承担条件生成职责 | 不再奖励错误分支退化；正确条件生成每微批由最多两条分支变为一条。是否形成有效依赖仍要看同种子完整生成 |
| 旧 verifier 的候选视频证据包含静态特征偏移；条件交叉差分只严格抵消加性偏置，不能保证交互头使用过程 | 候选证据改为 `tanh(f_t-f_0)`，过程读出减去同时间对的零响应读出；条件输入仍包含首帧、材料与依赖分析 | 对时间恒定的加性特征偏移不变；零特征变化给出零响应证据。全视频观察器有上下文和位置编码，因此不能把该性质夸大为“任意冻结 RGB 的特征都相同”或“静止就是错误” |

C 仍通过条件状态算子选择 Wan 当前状态的更新方式，而非独立预测另一段未来。视频时间在各注入位置重新进入读取；不同时间和空间的原生状态读取同一组经推理的依赖信息。长程监督继续覆盖全部 81 帧和 1/2/4/8/16/32/64 帧跨度，在实际生成估计上监督有符号变化，并按时间对分别归一化。推理内存的联合修订与长期视频一致性是不同层次，前者通过检查不等于后者已经解决。

三阶段按职责递进，每阶段分别 **3 / 2 / 3 项损失**：

| 阶段 | 更新内容 | 损失 |
|---|---|---|
| interface，400 更新 | 知识／推理 LoRA、C 映射、注入算子和连续强度头；Wan 冻结 | FM、完整过程变化、连续强度 |
| calibration，460 更新 | 生成路径固定，训练条件过程 verifier | 同场景过程排序、条件—过程交叉差分匹配 |
| joint，400 更新 | 生成目标更新 C、知识 LoRA、注入算子及 Wan LoRA；固定 verifier | FM、完整过程变化、verifier 反馈 |

joint 的 verifier 项专门更新知识／推理 LoRA 与 C 映射，Wan 和注入接口仍参与对 C 的求导，但不接收该项的参数更新。这样反馈的优化对象明确落在 C 上。反馈仍使用四步连通条件补全，解码完整 81 帧，V-JEPA 观察器读取覆盖首尾的 21 个双帧位置；RGB 过程损失则覆盖全部 81 帧。条件补全与 50 步自由生成之间的迁移尚待验证。普通推理保持一条 Wan 去噪轨迹，可选在线反馈也只更新该轨迹后续步骤使用的 C。

新增 [直接训练入口](train/native_causal_response_1x80g.yaml) 使用既有环境和单卡配置，已于 16:13:53 UTC 提交为 **Run66180**，分配到 `amp-80g` 的单张 A100 80GB；16:14:30 UTC 进入正常训练。149 个代码文件已独立快照并保存哈希，见 [任务记录](analysis/queue_20261025_causal_response/job.json)。入口直接进入正常训练，每 16 更新保存断点及输出固定留出视频，不再串联 VAE 检查、三阶段检查、未训练视频和 128 步门槛。400/460/400 是完整训练预算，不是必须等待完成才能分析结果的测试门槛。当前首个固定视频仅用于早期查看；可信效果比较仍需配对条件消融和多个既有留出场景，不能以单个视频宣布成功。

**推理入口另有一个实际修复：** `--refine-c` 原先用新字典覆盖 checkpoint 中的反馈配置，丢失 `metric: relative_rms`。现已改为只覆盖显式 CLI 选项，保留更新度量和原调度。66176 已完成的工程视频实际使用旧 Euclidean 更新，两个相对 C 更新分别为 `9.63e-8`、`8.08e-8`；该视频不作为 RMS 修正有效的证据，运行快照不被追改。

当前 **63 项短行为检查通过（24.14 秒）**，3,857 条缓存及划分检查通过；v5.2.1-mechanism 的 20 个文件、v5.2 的 103 个文件、66176 快照的 146 个文件哈希不变。检查覆盖注意力信息流、生成目标归属、单分支计算、响应读出性质、长跨度导数和推理策略保留。**16:27 UTC：v8 已完成 2 次正常训练更新。首个更新累积 8 微批，用时 323.97 秒、峰值 68.60 GiB；每微批 Wan 前向分支数为 1，C／知识／注入模块的梯度归属符合阶段一设计。** 这证明真实模型能够执行该修订，物理／视觉效果尚未证明。实现与验证记录见 [v8 记录](analysis/causal_response_20261025/implementation.json)。

第 16 步首个断点已接入 [自动对照任务](analysis/queue_20261025_causal_response/first_checkpoint_verification/job.json)：CPU 进程等待原子保存的断点，随后固定该文件并使用原 `det 0.32.1` 提交单卡比较，等待期间不预占 GPU。预先固定火焰、海浪两个留出场景，配对重建使用两个噪声种子；完整视频用每场景一个固定种子比较完整 C、关闭 C、关闭连续 latent reasoning，共 6 段。关闭 reasoning 同时作用于 CFG 的正、负条件，避免把部分消融误写成完整模块消融。训练继续运行，比较不修改其断点或状态。这是早期条件作用验证，不是多场景泛化或已校准 verifier 的效果证明。

### 16:51 UTC：收拢到 blk-96g 的独立任务

| 任务 | 内容与边界 |
|---|---|
| **Run66200**，blk-96g 单卡 RTX PRO 6000，已完成 | v8 三阶段各一个真实微批，新响应 verifier 的校准、联合阶段到 C 的反传和梯度归属均已实际执行；不是完整训练或质量证据 |
| **Run66201**，blk-96g 单卡 RTX PRO 6000，正常训练中 | [朴素文本条件对照](../../configs/ablation_native_text_causal_response.yaml)：缓存 T5 特征直接映射到 C，生成条件不调用 Qwen；同样数据顺序、种子、FM／过程／强度损失、映射和注入结构，先完成 16 次可恢复的正常更新，再导出相同留出场景的结果。已完成两次 8 微批更新 |
| **Run66211**，amp-80g 单卡排队 | 从训练开始关闭连续 latent reasoning，保留 Qwen 编码和显式依赖分析；同样的前 16 次正常更新及固定留出输出，区分 Qwen 条件编码与连续推理的贡献。66207／66209 在 blk 的同一 H100 分配上均以 137 退出，未完成更新；恢复任务未改模型、配置或软件环境，仅换到已有正常训练的资源池 |
| 首个 v8 断点对照，待权重 | CPU 等待器仍运行，权重就绪后自动提交单卡任务，无额外 GPU 等待占用。因上述重复终止，将待提交任务改为 amp-80g；尚未执行，不能记为已通过 |

三项新任务使用同一份 152 文件独立快照，现有 66176／66180 及其快照保持运行且不变。未重新启动已经结束的旧长梯度矩阵或旧版硬件对照。最新 67 项行为检查通过（26.12 秒），其中新增检查确认朴素条件分支确实不执行 Qwen 编码、其参数不收生成梯度，文本替换不改首帧或 Wan 原生文本。见 [任务总表](analysis/queue_20261025_blk_validation/inventory.json)。

17:02 UTC，66200 完整结束，三个阶段分别用时 43.09／19.61／76.25 秒，峰值 68.04／43.69／68.41 GiB。生成与校准阶段的实际参数梯度归属符合设计；冻结 verifier 经四步连通补全回到 C 的加权梯度 RMS 为 `4.811e-9`。这是实际反传及归属证据，不是物理优化收益。随后补交的 66207 复用相同代码快照，仅改变独立配置中的推理轮数和 run_name，避免只做推理时 knockout 的混淆。

66201 的前两次正常更新分别用时 492.05／405.23 秒，峰值 67.98／68.19 GiB；生成侧 Qwen 参数梯度为零，C 映射与注入接口有梯度，每微批 Wan 前向分支数为 1。66207／66209 均在首个优化更新之前退出，没有模型质量结果；退出码本身不足以判定硬件故障，具体终止来源尚未确定。两次失败的日志与原始配置均保留，未修改集群、Determined 或 Python 环境，也未额外占用 GPU 规避分配。

这个基线用于判断 Qwen 生成条件相对“文本直接映射 C”的价值。固定 verifier 的 Qwen 条件编码在后续阶段仍保留，不能把该基线称为整个系统完全不含 Qwen。它也不等于严格参数量或算力相同的对照：省去了知识 LoRA 的生成训练及 Qwen 的条件计算；外部训练样本和生成器接口相同。16 步用于早期可比较的训练结果，不能证明最终收敛优势。主模型和基线均保留断点，以便后续按相同训练进度继续比较。

## 运行中 v7 配方及设计依据

**C、reasoning、verifier→视频→C 的反馈和状态依赖注入全部保留。新的结构及训练代码已实现；长程物理一致性和生成质量尚未证明已经解决。** 当前入口为 [native_reasoning_operator_wisa3857.yaml](../../configs/native_reasoning_operator_wisa3857.yaml)，检查点标识 `v522_native_reasoning_operator_v7`。v6 配方保留用于复现；不能把改变推理计算及梯度归属后的 v7 当作旧断点无变化地恢复。所有训练材料仍来自现有 3,857 条缓存，不修改 v5.2.1-mechanism、缓存、旧检查点或实验快照。

### 哪些证据已经足够改变设计

| 已知问题 | 本次改变 |
|---|---|
| 其他场景 C 对 FM/过程指标影响极小；单纯增多 C tokens、T5 dropout 未建立依赖 | 把附加消息残差改为 C 选择的状态更新算子，并直接训练同首帧、同噪声下的条件对应关系 |
| v5 常规更新的 `sequence=0`，绝大部分步骤只有 FM 与强度监督 | 每个生成训练微批都监督完整 RGB 序列的有符号变化，覆盖 1/2/4/8/16/32/64 帧跨度；单个时间对分别归一化 |
| 来源分类 verifier 偏爱冻结；成对校准能识别冻结，但尚未证明真正使用条件 | 保留同场景连续过程排序，增加独立的条件匹配训练；评分头改为条件与过程的交互，去掉可独立输出分数的视频/条件偏置分支 |
| 完整 50 步 C 导数显著放大，K12 导数与全程改动 C 不一致 | 用完整短去噪补全过程训练反馈；输入是模型参数之外采样的 FM 状态，对全部 K 步求导，不存在被忽略却随参数改变的模型前缀 |
| 只有显式文字分析＋一次编码，缺少可训练的连续 latent 推理 | 复用 Qwen 最后 4 个原生语言层，进行 2 轮有初始证据支撑的连续 latent 更新；生成与 verifier 损失穿过全部推理轮次 |
| verifier 分数可以通过直接修改生成器改善，不能据此证明 C 变强 | v7 将 verifier 项的参数梯度定向到 C 映射与 Qwen；Wan/注入接口仍接受 FM、过程及条件使用三项监督 |

不继续扩大 K12/K50 × 更新幅度 × 种子的长测试矩阵。66129 已于 05:29:58 UTC 取消；66128 的**四步真实模型训练全部完成**后，其剩余重采样于 05:30:04 UTC 停止。旧训练两次反馈更新分别约 343/353 秒，裁剪前主梯度范数约 5811/20066；这些数据支持调整反馈任务，不能证明物理信息更强。记录见 [metrics.jsonl](checkpoints/v522-paired-verifier-real4steps-rtx-20261025_66128/metrics.jsonl)。

### C 的集中推理与对原生状态的作用

Qwen 先从首帧、文本和既有条款形成初始条件、交互、持续作用、条件改变后响应的分析；输入显式包含生成时长，避免把长时间尺度现象套入几秒片段。显式文字生成本身不反传。随后 [latent_reasoning.py](../../physgen_native/latent_reasoning.py) 读取 Qwen 后部层入口的合法多模态记忆和末层归一化前的内容状态，以共享的原生后 4 层作 2 轮连续更新，最后一次归一化、映射成 C。每轮都可读取同一份初始证据，后轮读取前轮 latent；没有独立推理 Transformer、未来状态网络或额外推理损失。使用原生多模态位置编码和注意力，推理层位于 DeepStack 视觉注入之后。没有可变 KV cache 或中途 detach。

这属于可训练的连续隐藏状态推理计算；其物理推理质量仍需训练后消融。连续 latent 推理已有 [Coconut](https://arxiv.org/abs/2412.06769) 等研究，不能把“循环隐藏状态”本身宣称为首次提出。这里的研究重点是：推理产物是否形成可被生成状态实际使用、能接受物理过程反馈的集中 C。

默认一次生成使用推理得到的 C；Wan 的原生三维时空状态 H 随时间位置、网络深度和去噪阶段不同，持续选择 C 中适用的信息。可选的在线反馈只在同一条去噪轨迹内修正 C，见下文。

旧接口虽有状态相关的注意力和门控，其残差内容主要是对读出消息的映射。新接口在 [conditioning.py](../../physgen_native/conditioning.py) 中明确计算：

\[
m=\operatorname{Attention}(H,C),\qquad
\Delta H=g(H,\sigma,a)W_o\big[\operatorname{SiLU}(W_h\operatorname{Norm}(H))\odot W_c m\big].
\]

C 因而选择**当前状态的低秩更新方式**。知识映射去掉常量 bias，来源编码改为乘性调制，零内容不能靠来源 embedding 单独产生残差。没有删除公共物理知识或做未经依据的全局均值消除。a 仍是连续局部可见变化强度，低 a 不会复制首帧或关闭原生生成。没有添加独立 M/S、未来状态预测器、轨迹、光流或物体类别路由。

这是一项待检验的物理条件作用设计；双线性交互、注意力或低秩结构本身不被宣称为首次提出。

### 三阶段，分别 4 / 2 / 4 项损失

| 阶段 | 训练参数 | 明确的目标 |
|---|---|---|
| interface：建立条件作用 | Qwen 知识 LoRA、C 映射、状态算子与连续强度头；Wan 冻结 | FM；有符号过程变化；条件使用；RGB 弱监督强度（4 项） |
| calibration：建立条件化反馈 | 固定整个生成路径，仅训练 verifier | 同场景过程排序；条件—过程匹配（2 项） |
| joint：联合改善 | 知识 LoRA、C 映射、状态算子及 Wan LoRA；verifier 固定 | FM；有符号过程变化；条件使用；verifier 反馈（4 项） |

第三阶段的**梯度归属**不是四项损失全部更新同一批参数：前三项更新知识、映射、接口及 Wan LoRA；第四项 verifier 只更新知识／连续推理 LoRA 与 C 映射。Wan 及接口仍参与完整可微计算，将视频反馈传至 C，但不接收这一项的参数梯度。离散伴随显式控制参数输入，没有 `detach(C)`，也不会清掉前三项已累积的生成器梯度。这直接约束了“物理反馈要打磨 C”的训练职责。

第三阶段固定已训练的强度预测头，保留其输入导数及门控/算子对它的使用；不另加第五项强度损失。强度头的数值标定可能随 Wan 表示变化，需要在实际训练中观察，不能仅凭固定权重声称标定不变。配置验证拒绝多余损失项。没有把关系分类、独立状态拟合和额外对齐藏进四个名称。

过程目标只比较多尺度 RGB 的有符号时间变化；FM 负责基本生成拟合。逐时间对归一化避免强烈的前期变化压低后期贡献。对这里的差分只作可见过程解释，不叫物理速度。长程训练从阶段一开始，阶段三继续。

条件使用目标在相同首帧、T5、噪声及 sigma 下比较正确 C 和替换文本条件产生的 C：`softplus((E_correct-E_alternate)/temperature)`，E 是实际生成估计的过程误差。保留当前首帧的观察条款、条款数量及来源标识，只替换文本要求，避免靠“少了观察条款”区分条件。两个分支都有导数。这是已有数据的条件对应监督，**不是有真实反事实未来的因果干预标签**；替换分支误差变大也不单独算成功，必须同时看到正确条件的绝对生成质量改善及有意义的选择性响应。

verifier 使用同一个冻结全视频 V-JEPA 观察器处理参考、生成和过程缺失视图；不把缓存中 isolated-pair 特征混成 full-context 特征。过程排序偏好由实际证据差异产生：本来静止的片段不会仅因“冻结操作”被硬标成物理错误。评分函数只输入候选视频及合法初始条件，参考未来只用于训练目标。其物理含义来自 Qwen 的依赖知识与实际过程约束，仍须检验材料/属性连续性、条件响应和多现象泛化；普通过程差异没有被包装成完整物理定律。

**条件匹配进一步使用过程与条件的交叉差分。** 仅比较 `s(V,c)-s(V,c')`，仍可能让 verifier 靠判断首帧与文字是否相符完成任务。因此，校准第二项实际比较：

\[
D=\big[s(V_{ref},c)-s(V_{candidate},c)\big]
 -\big[s(V_{ref},c')-s(V_{candidate},c')\big].
\]

目标由候选与参考的实际过程证据差异决定。任意纯视频分数 `a(V)` 与纯条件偏置 `b(c)` 都从 D 中代数抵消；必须学到条件相关的过程差异，才能改善这一目标。它仍是两项校准损失中的条件匹配项，不增加第三项或额外分支。行为测试覆盖这一抵消性质，以及在共享输出处，条件使用梯度指向条件内容而非共同的附加适配参数。这个性质约束了学习任务，不等价于已经识别真实物理因果变量。

### 反馈计算和正常生成的区别

[operator_training.py](../../physgen_native/operator_training.py) 从训练状态 `z_sigma=(1-sigma)z+sigma*noise` 出发，在 `[0.55,0.90]` 采样 sigma，做 4 个连通去噪步骤至零，再解码全部 81 帧并计算固定 verifier 的反馈。评分梯度走 `verifier → 全视频证据 → RGB → VAE → 全部4步Wan → 状态算子 → C → Qwen LoRA`。评分条件使用未适配 Qwen 并停止梯度，生成器不能仅修改 verifier 的条件输入来得分。

V-JEPA 在这里是读取候选视频的冻结观察器，**不等于物理判断器本身**。判断来自有具体条件输入、经过过程排序和条件交叉差分校准的 verifier。普通推理可只使用训练后 C，此时不加载观察器和评分头；需要在线修正时才加载它们。这不是删除 verifier 的训练作用。

这对**定义的条件补全训练目标**给出完整离散导数，保留所有视频时间位置。它不等于从纯噪声开始的 50 步目标，也没有把预训练 Wan 宣称为 4 步蒸馏模型。训练使用和 FM 相同的无 CFG 条件流，正常推理仍是 50 步、CFG=5。由条件补全到自由生成的迁移是要用实际视频检验的剩余问题。相较旧方案，主要变化是结构与监督职责，短链仅解决训练反馈的代价和求导范围问题。

默认预算是 400 个条件训练更新、460 个 verifier 更新、400 个联合更新；每次累积 8 微批。保留原来 800 个生成更新，增加明确列出的 verifier 阶段，**不能声称与旧实验等算力或已足够收敛**。先完成三个阶段的短真实模型检查，再按第一阶段的实际条件作用与中后段效果决定后续训练；不靠固定步数自动证明能力成立。

### 单次生成中的可选 C 反馈

[feedback_sampling.py](../../physgen_native/feedback_sampling.py) 在预先指定的噪声位置，利用该步已经计算的速度形成干净视频估计，固定 verifier 并仅求 C 的导数。修正后的 C 作用于**之后的步骤**；当前步骤使用已有速度继续，不重算当前 Wan 步、不重置噪声、不生成多个候选或重启整段视频。模型及评分器权重保持不变。C 更新是以初始 C RMS 为单位的梯度下降，不再把每条梯度强行放大到 C RMS 的 10%/50%。

10 月 25 日修正在线更新的尺度定义：若 `d=C.numel()`、`s²=mean(C_initial²)`，相对 RMS 位移的度量为 `||delta||²/(d*s²)`，对应梯度步是 `delta=-eta*d*s²*gradient`。此前只乘 `s²`，使均值型评分在增加等价 latent 元素时产生缩小的修正。新配置显式使用 `metric: relative_rms`；没有此字段的旧配置保留原更新，已保存 checkpoint 不被静默重解释。复制等价 latent 的行为检查验证更新不随复制倍数缩小，零梯度仍对应零更新。没有把任意梯度强制归一成固定百分比；这修正在线优化坐标，不是物理收益证明，也不改变训练中的 AdamW 或损失。

在线导数属于当前干净估计，不冒充最终 50 步视频的精确梯度；额外解码、观察和反向计算有成本。该选项默认关闭，待训练后的成对完整视频验证收益。开启方式：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B inference/infer_heldout.py \
  --checkpoint /absolute/path/latest.pt --samples 1 --refine-c
```

训练后可分别关闭显式文字分析与连续 latent 推理作消融；这些生成侧消融不改变冻结 verifier 的条件编码。关闭在线反馈时，采样与原路径逐元素一致。每个视频的 JSON 记录推理轮次、在线反馈位置与实际 C 更新幅度。

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B train/train.py --check-only
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B train/train.py --config configs/native_reasoning_operator_wisa3857.yaml --smoke --smoke-accumulation 8
# 阶段一训练入口；正式提交必须用独立代码快照和输出目录。
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B train/train.py --config configs/native_reasoning_operator_wisa3857.yaml --stop-after 400
```

已有 3,857 条缓存与模型文件检查通过。最新测试结果及队列记录见下方运行状态；新增检查使用真实 Qwen 模块的小型配置，验证每轮 latent 的生成反馈、原生 checkpoint 导数一致、评分条件不受知识 LoRA 改动、verifier 专门更新 C，以及单轨迹采样行为。它们不证明物理质量。SIGGRAPH 立论必须用实际结果证明：集中 C 对特定条件响应的可控作用、中后段过程改善、材料/场景组合泛化，且视觉质量同步保持。已有 [PhysMaster](https://arxiv.org/abs/2510.13809) 学习物理表示指导生成，[Phantom](https://arxiv.org/abs/2604.08503) 联合建模视频与物理 latent；因此“有一个物理 latent”不能单独作为新颖性结论。

新的执行链 [run_reasoning_training.py](../../tools/run_reasoning_training.py) 先做三阶段各一次、每次 8 微批的真实更新，再做一个完整单轨迹在线 C 反馈检查；通过后从头启动正常训练，先执行最终配方的前 128 个更新，保存可继续恢复的断点。随后在预先固定的火焰出现／传播、海浪、气泡形变、黏性液体四个既有留出场景上比较正确 C、关闭 C、同首帧替换文本的 C、关闭连续推理。配对去噪检查使用两个种子；完整视频先导出每场景一个固定种子的正确 C 与关闭 C，共 8 段。不挑选最佳种子，不把小幅平均差或错误条件退化单独当作收益。

这是最终三阶段配方的训练前缀，不是已完成完整训练。第一阶段 Wan 冻结，因此此时关闭 C 的生成仍可作为原生 Wan 对照；联合阶段则需要另列原生权重对照。正常三阶段配置另见 [native_reasoning_operator_1x96g.yaml](train/native_reasoning_operator_1x96g.yaml)。旧 Run66131/66133 的 v6 快照保留，任务状态见各自 job.json，不将旧检查报告冒充 v7 结果。

相关研究的进一步核对：[WMReward](https://arxiv.org/abs/2601.10553) 已用 V-JEPA 的上下文—未来预测误差引导生成，其附录也报告了突发溢流、点燃爆炸及复杂材料现象的失败，并指出预测 surprise 混合了物理与感知因素。[原文分析](https://arxiv.org/html/2601.10553v1) 说明把“容易预测”直接等同于“符合当前物理条件”并不充分。本方案继续集中解决**具体物理依赖 C 如何影响当前状态、如何持续影响中后段、verifier 如何对条件相关的过程变化提供反馈**。现有 V-JEPA2.1 权重包含 predictor，但本次没有因此堆入另一条未来预测分支；在当前证据下不把新增模块数量当作创新程度。

### 当前运行状态（2026-10-25 单卡恢复）

**旧版对照已结束：Run66162 成功，Run66163 失败。** 完全相同的 v5.2 原入口、checkpoint、种子和两个样本，在 RTX PRO 6000 上完成 2×50 步采样及两个 832×480×81 帧视频；H100 上完成首例全部 50 步后，在 VAE 解码／输出转换期间被 SIGKILL，未完成视频。RTX 的采样、解码峰值分别为 25.50、11.42 GiB；这不能代替失败 H100 的峰值。采样单步中位时间分别为约 6.10 秒和 16.99 秒。103 个旧版源码／配置哈希不变。新 C、reasoning、verifier 和新版运行包装不是复现此终止的必要条件；现有日志指向该 GPU 分配路径，但未确定信号发送者，不能宣称已证实硬件损坏。见 [原始对照与结论](analysis/legacy_v52_controls_20261025/comparison.json)。既有旧版短训练没有再提交，因为旧版推理已经提供了这一关键对照。

**16:12 UTC 核对：Run66176 在 node6 的单张 A800 80GB 上继续运行。** 使用原 `det 0.32.1`、原镜像和运行入口；没有修改软件安装或集群设置。完整 81 帧双分支 VAE 反传通过（35.33 GiB，67.8 秒）；三个阶段各 8 微批、共 3 次真实工程更新全部完成，峰值分别为 69.45／44.07／70.04 GiB。冻结 verifier 反传到 C 的加权梯度 RMS 为 `1.307e-8`，确认路径与归属，不证明物理收益。一段完整 50 步工程视频已输出，其在线 C 更新实际仍为旧 Euclidean 度量，配置覆盖问题已在工作区修复。正常训练完成到第 2 更新，每次约 9.6／10.2 分钟；训练数据不同，不能用两步损失作效果趋势。用户要求保持现有任务，v7 快照及其 128 步训练前缀保持不变；工作区 v8 的改动没有进入该任务。**尚无正式训练后质量结论。** 见 [任务状态](analysis/queue_20261025_amp80_recovery/job.json)、[工程更新实测](analysis/queue_20261025_amp80_recovery/results/integration.json) 与 [执行链](analysis/queue_20261025_amp80_recovery/results/pipeline.json)。

**54 项行为测试通过；3,857 条既有缓存检查通过；v5.2.1-mechanism 的 20 项受保护文件哈希不变。** 此前任务 [Run66143](analysis/queue_20261025_reasoning_single/job.json)，正常提交到 blk-96g，**`slots_per_trial: 1`，仅占一张卡**。执行顺序及本次修正见 [implementation.json](analysis/queue_20261025_reasoning_single/implementation.json)。独立代码快照的 143 个文件已保存哈希。该任务于 12:14:36 UTC 失败，尚无完成的微批或优化器更新。训练子进程返回 -9（SIGKILL），外层脚本错误地转成了 247；cgroup 没有 OOM 记录，主控日志没有记录抢占或主动 kill，终止来源尚未查明。

Run66135 在用户代码运行前失败：任务全局的 `PYTHONNOUSERSITE=1` 干扰 Determined 容器启动。该多余配置已撤回；继续使用既有 `det 0.32.1`、原软件环境和之前成功的提交环境，未升级、重装或重新配置 Determined、conda、镜像或集群。Run66140 随后正常通过容器启动并完成 8 次初始配对验证，但第一次训练微批的 VAE 反传出现 OOM：进程显存约 94.76 GiB，卡容量约 94.97 GiB，尚无优化器更新或检查点。

本次在完整条件分支外增加重算边界，避免同时保留正确 C、替换 C 两条 Wan/VAE 激活历史。两个分支仍有导数，所有帧、训练目标、模型参数保持相同；损失及各参数梯度的等价性已测试，原生 VAE 的双分支导数与前向保存张量量也已检查。没有用多卡、缩短视频或截断长期梯度处理这个 OOM。短工程检查不再重复初始验证；完整日志保存在文件中，紧凑进度同步显示到 Determined。Run66143 在记录首个微批结果前被终止，因此尚不能确认重算后的单卡训练峰值。

Run66148 已于 12:39:21 UTC 使用用户 v5.2 单卡模板重新提交，容器启动及全部首微批前向计算成功，但于 12:48:40 UTC 在首次反传中被 SIGKILL 终止；没有完成的微批或训练更新。最后一次 GPU 驱动记录为 82,803 MiB，cgroup OOM 为 0，信号发送来源未确定。只占一张 GPU，软件安装和集群配置不变。启动脚本与 v5.2 的进程级运行设置已比对一致；首个工程训练微批按计算步骤记录显存，SIGKILL 退出码保留为 137。见 [失败原因对照](analysis/submission_failure_review_20261025/comparison.json) 和 [新任务状态](analysis/queue_20261025_runtime_aligned/job.json)。这些调整用于定位和恢复执行，不构成生成效果改进的证据。

Run66161 已于 13:37:04 UTC 以相同单卡环境提交。修正了时间块重算仍在反传中一次重建多层高分辨率激活的问题：VAE 内部按原生残差块进行功能化重算，完整保留跨时间缓存导数。54 项检查通过，新增完整 81 帧输出及末帧向早期 latent 的梯度对齐覆盖 float32 和 bfloat16。真实预训练 VAE 的 81 帧双分支反传已通过，峰值 34.97 GiB，计算约 101 秒；此数值尚不包含 Wan/Qwen 常驻及其训练图，不能作为完整模型峰值。随后完整模型的首个微批通过反传，峰值 68.96 GiB，反传约 92.4 秒；但进程在 13:50:43 UTC 再次被 SIGKILL，优化器更新数仍为 0。终止前约一秒驱动占用 73,253 MiB，容器 OOM 为 0；具体信号来源待节点日志确认。当时没有正在运行的 v7 替代训练任务；当前恢复任务及原 v5.2 对照见本节开头。见 [实时状态](analysis/queue_20261025_decoder_blocks/job.json) 与 [解码反传结果](analysis/queue_20261025_decoder_blocks/results/decoder_backward.json)。

**目前没有 v7 训练后完整视频或效果统计，不能宣布长程一致性、物理质量或因果表示有效性已经解决。** 54 项检查是实现证据；真实效果证据将来自上述训练及固定对照，不复用旧版本的小百分比波动宣称成功。

## v1–v5 历史记录：完整对照、反馈作用区间与恢复任务

以下保留实验依据及旧配方；其中 50 步在线反馈、两阶段训练均指 v5，不是当前 v7 配方。

**C、可微 verifier、状态依赖注入均保留。长程一致性、有效因果信息及实际生成改善仍未获得充分证据，不能宣布已经解决。** 本轮替换失效的监督目标，修复反传的时间覆盖；不以删除 verifier 代替解决问题。VideoPhy2 仅作外部参考。

### 10 月 25 日已完成的统计与实质修正

66111、66112 完成全部 5 场景 × 2 噪声种子 × 7 对照；66119 完成其中海浪、气泡两场景的完整 50 步梯度对照。66118 与 66120 再次被 SIGKILL 终止，**并未完成实际模型四步训练或 learned verifier→C 效果测试**。

| 已完成实验 | 固定更新幅度 | 过程目标相对原始 C 的平均收益 | 相对等幅随机方向 |
|---|---|---|---|
| 末 12 步梯度，5 场景 × 2 种子 | C RMS 的 10% | +0.52%；按场景 bootstrap 区间约 −4.69% 至 +6.76% | 4/10 更好；随机方向自身平均 +0.60%，未建立梯度优势 |
| 同上 | C RMS 的 50% | −12.72% | 仅 2/10 更好，不能把增大更新当作修正 |
| 完整 50 步梯度，仅 2 场景 × 2 种子 | C RMS 的 10% | −30.02% | 仅 1/4 更好；不能外推为全部场景结论 |
| 将 C 换为其他场景，5 场景 × 2 种子 | 原有对照 | +0.37%，区间包含零 | 具体条件内容仍未体现可靠作用 |

汇总由 [summarize_feedback.py](../../tools/summarize_feedback.py) 从完整对照组重建，保留全部种子，不选取逐样本最佳扰动；见 [12 步汇总](analysis/feedback_review_20261025/oracle_k12_summary.json) 与 [50 步部分汇总](analysis/feedback_review_20261025/oracle_k50_partial_summary.json)。这些是使用参考未来的能力 oracle，不能当作物理评分或普通推理收益。

海浪 seed43 的 C 梯度 RMS 从 12 步的 `2.0446e-5` 放大到完整 50 步的 `15.9713`；这说明完整反向链不能只凭非零梯度判为有效。[视频取帧对照](analysis/feedback_review_20261025/wave_feedback.png) 中，大幅完整梯度更新也改变了后期构图和外观，不能把这种变化自动视作改善海浪物理过程。

检查代码发现，旧 C 专项把末 K 步的截断导数用于更新 C，却在重新完整采样时将新 C 应用于所有 50 步；被忽略的前缀也发生了变化。新的 `_sample_tokens` 支持固定原始 C 的前缀和修正 C 的后缀：**原有条件全程存在，反馈改动的作用范围与求导范围对应**。这不是把视频截短：所有 81 帧都生成并参与评价。它也不自动解决通用参数训练的截断偏差；当前先严格验证 C-only 更新，尚未据此改写正式训练目标。

[probe_condition_credit.py](../../tools/probe_condition_credit.py) 从同一次完整反向计算记录 1/4/12/25/50 步 C 梯度，检查放大出现的位置；对 0.01%、0.1%、1% C RMS 的下降、上升与随机方向分别完整重采样，比较全程改动和仅对应后缀改动。新的采样路径、逐步梯度记录与直接自动微分的一致性已有行为测试，当前 **35 项测试通过**。实际改善需要该实验的结果。

另一个发现是，step32 的 5 个场景中，平均 C 的两两余弦约 0.9966–0.9989，平均 attention value 的余弦约 0.9983–0.9995；见 [表示几何记录](analysis/feedback_review_20261025/condition_geometry.json)。这与通用适配成分占主导的解释相容，但平均值不能证明所有 token 坍缩，也不能把它套用于其他 checkpoint。当前不会仅凭此结果删除公共成分或再加一个未经验证的分支。

失败重现记录将 SIGKILL 关联到同一 GPU UUID；进程树 RSS 峰值约 9.4/16.4 GiB、容器 OOM 计数始终为零。节点内核日志不可访问，因此底层终止原因仍未确诊。66128 曾申请 blk-96g 的四个 slot，在明确分配的三张 RTX 卡上执行三个独立 worker，排除该 UUID；没有修改节点全局设置，也没有访问未分配设备。四步真实训练已完成，剩余重采样及排队的 66129 已主动停止，以转入架构和训练修正。配置、源码哈希和逐 worker 状态分别保存在 [恢复任务](analysis/queue_20261025_device_recovery) 和 [梯度任务](analysis/queue_20261025_condition_credit)。

66128 的真实模型检查最终完成全部四步，包括联合阶段。两次完整反馈更新约 343/353 秒，峰值 75.57/75.83 GiB，主梯度裁剪前范数约 5811/20066。检查点已保存。这说明替换 allocation 后原训练链能执行，并不证明评分方向或生成效果有效；大梯度不能作为物理信息更强的证据。

论文设计上，完整/截断梯度和 latent 修改本身不能作为新颖性依据。[DRaFT](https://arxiv.org/abs/2309.17400) 已比较完整与截断奖励反传；[Causal Writability](https://arxiv.org/abs/2609.15980) 已研究视频模型中物理行为写入位置与强度的作用；[Seeking Physics in Diffusion Noise](https://arxiv.org/abs/2603.14294) 使用内部表示训练 verifier 并施加梯度引导。我们的待证贡献仍应是**集中物理依赖 C 的具体内容，通过可信反馈产生可复现的过程改变，并泛化到多种现象**。本轮修复为这条路径建立可检验的优化基础，不把基础修复包装成已经成立的物理因果创新。

当前完整候选训练配置是 [native_paired_verifier_wisa3857.yaml](../../configs/native_paired_verifier_wisa3857.yaml)，检查点标识 `v522_native_paired_verifier_v5`。v4 回归校准已被下方新对照否定为默认候选，改用同场景成对校准。内部版本标识区分参数和目标，**不表示这些版本已经依次解决了问题**。v3 的参考过程损失用于梯度能力对照；它不是最终物理 verifier 的替代品。旧配置与旧结果保留用于复现。

重启后核实，Run65811 已是 `STATE_ERROR`，不再沿用“正在训练”的历史描述。新分支未修改该版本的 20 个受保护源码文件。训练仍只使用原有 3,857 条缓存；视频视图、在线采样及特征重算均来自这些已有样本。

### 已完成的反证及其实际含义

可重建汇总见 [evidence_20261024.json](analysis/evidence_20261024.json)，原始文件路径包含在其中。

| 已完成实验 | 实际结果 | 本轮处理 |
|---|---|---|
| 128 步内容 readout 训练后的留出集 C 替换 | 不同训练种子及硬件复核中，其他样本 C 仅使 FM 误差增加约 0.0004%–0.0141% | 增加 C tokens 或单独丢弃 T5 文本未建立有意义的条件使用；继续检验实际生成反馈 |
| 全视频 V-JEPA 特征上的真实/生成评分器 | 三个训练种子均可区分 8/8 留出真实/生成对，但三个种子都更偏好 8/8 全冻结真实视频 | 保留评分模块，改为同场景、逐时间对的过程证据校准 |
| 独立 VideoPhy2 对真实视频与全冻结版 | 两者 PC 均值均为 4.5；后段冻结为 4.375 | 不把该单项分数当作长程或物理成功标准 |
| 8 训练样本、32 更新的三种主路径比较 | 4 留出样本 × 2 生成种子上，三者最终 SA=4.25、PC=4.125、joint=0.625 | 这是小规模离散评分的相同结果，未建立生成收益，也不证明总体等价 |

以上负结果没有把“物理 C 不可能有效”作为结论。它们排除了仅凭损失下降、模块命名、非零梯度和极小百分比变化宣称成功。

### 17 时后的实际结果与失败处理

[三种校准方法的完整计数](analysis/queue_20261024_verifier_recovery/results/calibration/summary_counts.json) 已保存。三个 head 训练种子依次为 42、314159、271828，每个种子评价相同的 8 个开发场景：

| 校准目标 | 原视频优于全冻结的数量 | 原视频优于未参与拟合的后段冻结的数量 |
|---|---|---|
| 真实/生成来源分类 | 0/8、0/8、0/8 | 4/8、5/8、4/8 |
| 过程误差回归 | 2/8、5/8、1/8 | 5/8、4/8、2/8 |
| 同场景成对校准 | 8/8、8/8、8/8 | 8/8、8/8、8/8 |

这支持替换校准目标，并不能证明生成改善或多类型物理泛化。三个种子不是 24 个独立场景。在线训练现在比较同场景视图的连续过程证据，固定 verifier 的生成反馈形式保持一致；没有删除 verifier。已经停止原 66115 的回归 head 后续测试，保留完成的校准数据，新的训练与 C 反馈复用成对 head，不重复拟合。

66113、66114 均以退出码 137 失败，没有 Python/CUDA 异常栈；前者已经完成一次完整反传，峰值显存 58.79 GiB。只读节点检查 66116、66117 显示当前容器内存上限为 `max`，节点内存充足，但无权限读取内核 OOM 日志。因此 **不能宣布两次终止的根因已确认或已修复**。新 C 专项在合法条件预编码后释放不再使用的 Qwen、映射及旧 head，而非把 Qwen 长期放在 CPU；采样相等性测试覆盖释放前后。重跑以外层进程保存 RSS、cgroup OOM 计数和退出信号，保持完整 50 步导数及原有对照，用于复现并定位剩余运行问题。

新任务配置、独立代码快照和提交编号见 [paired_repair/jobs.json](analysis/queue_20261024_paired_repair/jobs.json)。本次 **33 项 CPU 行为测试通过**，3,857 条既有缓存与模型资源检查通过；真实模型四步更新及完整采样仍需等待任务结果。测试覆盖成对目标的梯度归属、交换比较的对称性、完整采样链导数及模块释放后采样值不变。它们不能替代生成质量证据。

### 当前已编码的完整闭环

1. **集中物理依赖 C。** 首帧、文本、已有条款及 Qwen 依赖分析生成 C；知识 LoRA 和映射接受真实生成反馈。保留初始条件、交互、持续作用、条件改变后响应的知识组织。普通条件编码不读取参考未来。
2. **持续注入。** Wan 完整时空状态在十个深度位置按视频时间、当前状态和去噪噪声读取 C。C 的时间恒定本身不算长程一致性；必须证明其内容对中后段过程有作用。
3. **保留并校准 verifier。** [evidence_verifier.py](../../physgen_native/evidence_verifier.py) 是有训练参数的条件评分模块。它读取实际视频的全时域证据及合法条件，逐时间对预测过程不匹配程度。参考未来只产生训练目标，不输入评分函数。生成更新固定 verifier 参数，并保留视频输入的导数。
4. **监督修正。** 对同一场景的原始、全冻结、中间过程缺失及实际生成视图，比较记录过程的有序联合证据；不再用“真实/生成来源”充当物理优劣标签。校准对照同时测试连续过程误差回归和有程度的成对偏好。后段冻结视图不参与离线 head 拟合，用于检查未训练编辑上的泛化。
5. **完整反传。** [adjoint.py](../../physgen_native/adjoint.py) 保存每个 Euler 步的 latent，反向时重算并释放单步计算图，累积 C、CFG 两路条件及生成参数的精确一阶导数。完整 50 步路径不在中间 detach；12 步对照显式截断前缀。这是离散采样器的导数，不是连续伴随近似，也不把逐步局部梯度简单相加。
6. **训练职责。** 每四次更新安排一次完整采样的 verifier 反馈更新，其余更新执行 FM 与连续强度监督。反馈更新有自己的优化步，不混在八个 FM 微批中稀释。前 400 步训练知识/接口，后 400 步同时训练 Wan LoRA。离线校准 verifier 后才开始这份候选训练配置。

完整反馈链为 `verifier → 视频证据编码器 → RGB → 原生 VAE → 50 步 Wan → 注入接口 → C → Qwen LoRA`。校准分支停止参考与候选特征梯度，只更新 verifier。生成分支不更新评分器权重。两条归属均有行为测试。

校准的当前数学目标是**记录过程的证据一致性**，尚不等于识别全部物理定律。它可能仍偏向单条参考过程，必须用生成结果、条件干预及未见场景检验。这个限制是需要继续解决的研究问题，不能靠把普通特征误差命名为“物理 verifier”消除。

800 次更新现在包含更多完整采样和反向计算，不能与旧 800 次更新声称等算力。训练记录实际时间、显存、更新类型与采样步数；正式方法比较还需匹配实际预算。

### 已提交的 blk-96g 实验

| 任务 | 内容 | 保存结果 |
|---|---|---|
| 66111、66112 | 末 12 步反传到 C；5 个开发样本、2 个噪声种子，分两片 | [12 步输出](analysis/queue_20261024_recovery/c_feedback_k12) |
| 66113、66114（已失败） | 完整 50 步反传；退出码 137，新快照重跑 | [50 步部分输出](analysis/queue_20261024_recovery/c_feedback_k50) |
| 66115（校准完成，后续主动终止） | 三种 verifier 监督 × 三个训练种子；回归 head 后续被成对 head 替换 | [运行阶段与结果](analysis/queue_20261024_verifier_recovery) |

C 对照包括原始 C、关闭注入、其他样本 C、两档梯度更新及等幅随机方向，均完整重新运行 50 步采样并导出视频。66111–66114 的更新使用参考未来，是训练能力 oracle；66115 的学习后 verifier 反馈不把参考未来用于 C 更新。两种证据不可混同。

任务均使用独立代码快照与独立输出目录，哈希分别见 [C 实验快照](analysis/queue_20261024_recovery/code_hashes.json) 和 [verifier 实验快照](analysis/queue_20261024_verifier_recovery/code_hashes.json)。运行中的快照不随主目录后续编辑改变。任务提交不等于测试通过；实时状态以 Determined 为准。

恢复前 CPU 行为测试 **30 项通过**，全部既有缓存和模型资源检查通过。它们证明实现与梯度归属，不证明生成质量。24 GiB 分阶段尝试在 VAE 反向处显存不足，现统一按用户要求排入 blk-96g；不再为了这次排队测试缩短真实去噪链。

16:30 UTC 左右的**未完成实验片段**已经显示正反两种结果：12 步反传、相对 C RMS 为 0.1 的更新，在火柴案例 seed42 上将过程误差从 0.05160 降到 0.04207，在海浪案例 seed43 上却从 0.01758 升到 0.01917。两个案例的重算 latent 最大差异均为 0，峰值约 58.75 GiB，说明这条实际反传路径运行成功，但效果尚不稳定。[首个案例的取帧对照](analysis/queue_20261024_recovery/c_feedback_k12/oracle_first_case_contact.png) 中，手部接触与火焰过程发生了变化；这不能替代完整视频的物理判断，也不能把参考拟合收益当成因果泛化。

### 论文立论与最接近工作的区别

核心研究主张是：**学习能持续支配生成过程的物理依赖 C，并通过可校准 verifier 的生成反馈使其有效。** 需要同时证明内容特异性、过程持续性和组合泛化，才可称为有效的物理因果表示。是否达到 SIGGRAPH 水平取决于这些实证及相对已有方法的贡献，不能从架构图预先保证。

| 原始文献 | 已有相关贡献 | 对本项目的实际约束 |
|---|---|---|
| [PhysMaster](https://arxiv.org/abs/2510.13809) | 图像物理表示作为条件，并用生成偏好训练表示 | “额外物理 latent＋生成反馈”本身已有先例；需要证明我们的依赖结构与反馈方法增加了什么 |
| [PHANTOM](https://arxiv.org/abs/2604.08503) | 在 Wan 上联合生成视频与 V-JEPA 物理 latent，双分支互相读取 | “latent 表达物理”并非空白；本方案将演化保留在原生 Wan，集中学习依赖条件 |
| [CoECT](https://arxiv.org/abs/2603.09094) | 因果事件推理及视觉/语言事件衔接 | 物理 CoT、事件顺序和显式因果文字不能单独充当新贡献 |
| [MoAlign](https://arxiv.org/abs/2510.19022) | 从视频特征中学习动态子空间并对齐生成模型 | 外观捷径与冻结评分问题已有相关观察；本项目依约不引入光流监督 |
| [LaMo](https://arxiv.org/abs/2605.23878) | 从已有视频学习 latent 变化先验并用于训练/采样 | “不用新增数据而学习动态 latent”同样需要更具体的方法区别 |
| [DRaFT](https://arxiv.org/abs/2309.17400) | 完整/截断采样链上的可微奖励训练 | 完整反传是技术基础，不能单独宣称创新；需证明它为何改善本任务的条件作用和后期过程 |

有潜力的贡献应落在一个可执行、可检验的闭环：物理依赖表示 C 的语义组织 → 当前原生状态下的作用 → 时间展开的 verifier 证据 → 返回 C 的生成反馈。当前正在验证其可行性，未声称已经发现可识别的真实因果变量。

后续效果分析必须保留全部种子和固定步长对照，不逐样本挑最佳扰动；按样本聚类统计，不把同一场景的多个时间位置当成独立样本。除逐时间段指标，还必须查看过程前提与响应、材料/属性连续性、出现和消散、视觉质量。已反复使用的开发样本不作为最终未见测试集。

当前候选配置入口：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B train/train.py \
  --config configs/native_paired_verifier_wisa3857.yaml --check-only
# 完整训练需提供校准后的训练集 verifier；本轮没有提交 800 步正式训练。
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B train/train.py \
  --config configs/native_paired_verifier_wisa3857.yaml \
  --initialize-critic /path/to/verifier_paired_seed42.pt
```

## v1 历史实现与原始对照说明

以下记录 v1 的设计及复现入口。其默认 verifier 和效果假设已受到上方实测反证约束，不作为当前有效性的结论或正式训练建议。

这版的核心对象是集中构建的 **Physical Dependency Condition，C**。它将首帧、描述和已有条款中的初始事实、相互作用、持续条件及条件改变的后果组织为知识条件，由 Wan 的原生时空状态按当前状态读取。C 接受实际生成任务与全过程监督的梯度。独立 M 组织器、独立 S 未来预测器及其未来 VAE 重建头全部移除。

`native` 指保留 Wan 预训练的全视频时空注意力、三维 RoPE、逐 token 噪声调制和原生输出头。它是生成侧的架构选择，不是“已经学会物理”的效果声明。

本目录独立于 `../v5.2.1-mechanism`，不修改 Run65811、已有训练缓存或其他版本。正式训练仍只使用 `wisa_4000` 中既有的 **3,857 个缓存样本：3,666 train / 191 validation**。在线推理、视频视图与模型采样属于训练计算，不新增训练数据集。新检查点标识为 `v522_native_physical_dependency_v1`。

## 问题证据与设计取舍

旧版 step100 的受控结果记录在 [旧版诊断](../../../v5.2.1-mechanism/analysis/horizon_step100_20261022/summary.json)。它们支持以下具体判断：

| 实测问题 | 核心原因 | 本版的对应改变 |
|---|---|---|
| h20 变化幅度比约 4.04%，方向余弦约 0.0123 | 确定性 S 在拟合从首帧无法唯一确定的未来；预训练时空计算也被拆散 | 删除独立预测瓶颈；用原生 Wan 的随机视频生成承担过程演化 |
| 压平所有未来预测，过程损失仅增加约 0.51% | 状态拟合可以通过近似静态、平均未来获得较低损失 | 对实际生成输出监督完整、有符号、多跨度的 RGB 和冻结视觉特征变化 |
| 关闭 M，未来 MSE 仅增加约 0.049% | 过程监督未更新 M/Qwen，关系分类又可绕过具体演化 | 从 step0 用生成与过程损失更新知识 LoRA、C 映射、注入接口 |
| 压平 S 后关系概率几乎不变 | 几个场景级关系分类并未要求使用过程证据 | 删除该训练目标；物理反馈针对视频过程证据和实际生成样本 |
| 静止 RGB 经因果 VAE 编码仍产生早期 latent 差异 | 编码器时间边界效应混入了所谓“变化” | 在连续解码后的 RGB 与固定视觉编码器上监督变化，不把原始 VAE 差分称为物理运动 |

旧目标已经监督所有未来位置，所以不能把问题解释为“第一阶段还没有监督后期”。重新加权旧辅助头的受控实验也未显示实质收益。这里同时改变表示职责、生成路径和监督位置；时间归一化只是完整目标的一部分。

## 集中物理依赖模块

实现集中在 [knowledge.py](../../physgen_native/knowledge.py)、[conditioning.py](../../physgen_native/conditioning.py) 和 [system.py](../../physgen_native/system.py) 的 `encode_condition`。

1. 冻结的 Qwen 只读取首帧、描述和已有条款，显式生成最多 192 tokens 的物理依赖分析：`OBSERVED / INTERACTION / PERSISTENCE / INTERVENTION`。它区分观察与条件预测，说明作用的前提、持续来源及条件改变后的响应。这里使用了实际自回归文本推理。
2. 同一 Qwen 基础模型启用知识 LoRA，重新编码上述输入与分析。每个条款和四种依赖问题各有一个 contextual readout，经归一化、线性映射与来源编码形成 C。当前缓存通常得到 8 个 C tokens。
3. 固定分析提供物理知识先验；生成损失训练的是连续 C 表示、知识 LoRA 和 C 的使用方式。**梯度不穿过离散文本采样，也不更新冻结的推理基础模型。**不能将其描述成端到端训练了离散推理策略。

公式：

\[
r=\operatorname{stopgrad}\{\operatorname{Generate}_{Qwen_0}(I_0,T,R)\},\qquad
C_\theta=P_\theta\{Qwen_{0+LoRA_\theta}(I_0,T,R,r)\}.
\]

C 是**具有物理依赖先验、受生成反馈训练的条件表示**。自然视频与语言先验并不能证明其中每个维度是可识别的真实因果变量；本文设计的可检验主张是：这些条件依赖是否被生成器使用，并改善相应过程和泛化。

## 如何注入并维持过程

[generator.py](../../physgen_native/generator.py) 按原生 Wan 计算完整视频。每三层读取一次 C，总共十个位置。共享的紧凑注意力以当前 Wan 状态为 query，以 C 为 key/value；每个位置具有自己的小型残差适配与软门控。视频时间与去噪噪声分别参与读取。

\[
H^{\ell+1}=Wan_\ell(H^\ell,\sigma)+A_\ell(H^\ell,C,\sigma,t).
\]

C 在一条采样轨迹中保持一致；不同时间、位置和去噪阶段的状态读取它时，响应可以不同。持续作用由原生状态演化与重复读取同一依赖条件共同承担，不再逐时间块递归预测另一套 S。

连续局部强度由当前 Wan 状态轻量预测，使用已有 RGB 结构变化缓存监督，并参与软调制。FM 梯度可以更新强度头。强度不是物理速度，低值不关闭 Wan，也不把未来硬复制为首帧。

新增映射约 12.60M 参数、接口约 14.03M、训练评分器约 1.06M、知识 LoRA 约 5.11M、Wan LoRA 约 47.19M；没有原来的约 596M S 网络。降低参数量本身不构成效果证明。

## 训练目标与梯度

默认 800 次 AdamW 更新、每次累积 8 个微批；前 400 步冻结 Wan 基础权重及其 LoRA，后 400 步开放 Wan LoRA。知识 LoRA、C 与接口从第一步参与真实 FM。评分器前 50 步主要接受校准，物理生成权重随后渐增；这不是旧版脱离生成的 C 初始化阶段。

| 目标 | 默认权重 | 监督对象与作用 |
|---|---:|---|
| FM | 1.0 | 完整未来视频的流匹配速度；它是去噪路径导数，不是视频中的物理速度 |
| 全过程 | 0.2 | 实际生成干净估计的 RGB 与冻结视觉特征变化，各占一半 |
| 连续强度 | 0.1 | 原生状态上的局部可见变化程度与软调制 |
| 评分器校准 | 0.2 | 停止梯度的视频证据上的成对偏好拟合，只更新评分器 |
| 配对生成物理反馈 | 0.05 | 固定评分器参数，保留视频输入导数，回传到生成器和 C |
| 自由采样反馈 | 0.05 | 每 8 步一次正常 50 步采样，末 4 步可微；从第 50 步开始 |

**全过程目标。** 对 81 个 RGB 时间位置，使用空间尺度 1/2/4，时间跨度 1/2/4/8/16/32/64 的有符号差分；同时用冻结的 Qwen 视觉编码器读取每个 latent-time 位置对应的真实相邻 RGB 帧，监督全部 21 个位置、跨度 1/2/4/8/16 的空间特征变化。视觉编码器输入的是生成结果，不承担未来预测。目标特征停止梯度。

\[
L_{seq}=\mathbb E_{i,j}\rho\left(\frac{[\phi(\hat V_j)-\phi(\hat V_i)]-[\phi(V_j)-\phi(V_i)]}{s_{ij}}\right)+\lambda_{state}L_{state}.
\]

`s_ij` 按**每个参考时间对**独立计算，保留幅度下限；强烈的早期事件不能设置全部后期时间对的归一化尺度。静止参考不要求凭空产生运动。目标同时覆盖中间过程、短期持续和初始至后段的联系，并不只约束最后一帧。

配对过程目标用于含真实视频信息的去噪训练，在 sigma≤0.85 激活；FM 覆盖 sigma∈[0.02,1]。自由采样视频使用过程评分，不要求不同合理随机未来逐像素复刻参考。

**Verifier。** Qwen 先生成具体视频过程证据分析，再对同一要求下的两段视频进行偏好判断；交换顺序各评一次，平均后形成软标签。视图包括过程停滞/中间缺失/局部顺序变化，以及当前模型的实际生成结果。编辑不是自动的“物理错误”标签。已有 JEPA 缓存只参与训练视图的时间窗口选择，不成为物理速度或推理输入。

评分器是训练时的轻量过程价值函数。校准时特征停止梯度；更新生成时评分器参数停止梯度，两条路径分开。正式反馈路径为：

\[
L_{phys}\rightarrow V\rightarrow Wan\rightarrow A\rightarrow C\rightarrow Qwen\ LoRA.
\]

自由采样保留正式 Euler50、shift5、CFG5 的完整前向，仅截断反向传播到末 4 步。末步可微不保证完整重采样后仍改善，因此提供了完整重采样的 C 干预工具。

目前真实小样本实验仍表明：通用 Qwen 对某些中间过程缺失的判断较弱，显式分析也未全面消除这一问题。评分器是补充物理反馈，**不能以其拟合损失下降证明长程或物理问题已解决**。全过程目标提供独立于这类判断的密集训练信号。

## 与 naive 注入怎样公平比较

四份配置使用同一数据、相同 C 数量/接口、学习率、阶段与连续强度目标：

| 配置 | 集中处理 | 生成目标 |
|---|---|---|
| [direct_fm](../../configs/ablation_direct_fm.yaml) | 普通图文 readouts，无显式依赖分析 | FM |
| [reasoned_fm](../../configs/ablation_reasoned_fm.yaml) | 显式依赖分析与依赖 readouts | FM |
| [reasoned_process](../../configs/ablation_reasoned_process.yaml) | 同上 | FM + 全过程 |
| [完整配置](../../configs/native_physical_wisa3857.yaml) | 同上 | 再加入校准和实际采样物理反馈 |

这样可以分别检验知识处理、过程监督和 verifier 的贡献。仅比较完整方案与原生 Wan，无法归因到物理因果设计。推理时关闭注入、关闭推理、替换 C 属于必要性干预，也不能替代分别训练的公平对照。

论文贡献应围绕“物理依赖条件的生成式学习与状态依赖使用”组织。以下结论需要实验支持，而不是模块名称支持：

- 修改某个前提表示，对应过程响应是否改变，而无关内容尽量保持；
- 中后段过程、属性和持续作用是否改善，而非只增加变化幅度；
- 未见材料/机制组合上的收益是否成立；
- 相同训练与生成预算下，物理、语义及视觉质量是否共同受益。

## 验证标准与证据边界

1. 行为测试检查原生 Qwen 视频打包、真实时序 VAE 解码与梯度、角色切换、梯度归属、静止/瞬态/错误方向、采样截断和检查点恢复。
2. 真实模型检查两阶段更新、8 微批累积、全部训练目标和完整自由采样反传。它们证明实现能训练，不证明训练完毕的质量。
3. `check_real_gradients.py` 固定样本与噪声，检查全过程梯度到 C/Qwen，并比较沿梯度更新 C 与同幅度随机更新，再完整重采样。该工具使用参考视频作为诊断 oracle；它不是普通推理，也不是泛化证据。
4. 正式验证按 sigma=0.25/0.65/0.95 分开报告，保存逐样本、h1…h20 原始变化能量、变化误差和方向。另报告 pooled skill/RMS，避免对接近静止样本的巨大比值直接求均值产生误导。
5. 完整生成比较使用同样本、同首帧、同初始噪声，重复生成种子和训练种子。物理判断独立于训练 verifier。统计置信与实际缺陷是否消除分别判断；“0.04→0.05”的 25% 相对收益仍可能完全不够。

[compare_evaluations.py](../../tools/compare_evaluations.py) 按样本聚类 bootstrap，保留每个样本的全部配对种子，报告绝对差、95% 区间、种子差和改善/退化样本数。少量生成种子不能估计充分的训练种子不确定性。禁止从单样本、单种子或小百分比改进推断稳定效果。

已有的 60 案例 [VideoPhy2](https://videophy2.github.io/) 输入和实际噪声可直接复用，独立 AutoEval 报告 SA、PC 及二者同时达标的比例。该项目子集是 I2V 条件比较，不等于完整官方 T2V 排名。面向论文的外部评测应同时覆盖 [VBench 2.0 的物理与常识维度](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/README.md) 及视觉质量，遵循官方输入协议。评测数据、评分器不进入训练。

完整训练与正式外部评测尚未完成，**目前不声称已解决长程一致性，也不声称接近 SOTA**。这份实现提供了可训练的完整方案与归因工具；最终效果必须由这些对照得到。

## 运行

环境沿用 `/home/liuzhirui/miniconda3/envs/moviestory`，所有基础模型本地加载，不下载模型或重新制作数据。生产入口使用单卡 blk-96g、原生 BF16、FP32 小型可训练权重、resident AdamW、梯度检查点，无训练 CPU offload。

```bash
cd /home/liuzhirui/Project/physGen/code/v5.2.2-native
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B train/train.py --check-only
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B tests/test_native.py

det -m http://10.130.130.5:4096 experiment create train/train_native_physical_blk96g.yaml train
# 小上下文目录，避免上传模型、缓存或检查点。
```

中断续训使用原运行的完整配置与检查点，不导入旧 M/S 参数：

```bash
CONFIG=/path/to/run/config.yaml RESUME=/path/to/run bash train/train_native_physical_blk96g.sh
```

正常留出集生成与同检查点推理干预：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B inference/infer_heldout.py --checkpoint /path/to/latest.pt --samples 12
# --condition-strength 0：关闭 C 注入；--without-reasoning：保留 C 接口但关闭显式分析。
```

已有 VideoPhy2 60 案例完整生成、独立评分和统计：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B inference/infer_shared_videophy2.py --checkpoint /path/to/latest.pt --output /path/to/new_output
/home/liuzhirui/miniconda3/envs/physvid/bin/python -I -B tools/evaluate_videophy2.py /path/to/new_output
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B tools/compare_evaluations.py --baseline /path/to/baseline --candidate /path/to/new_output --output /path/to/comparison.json
# 多种子重复 --baseline 和 --candidate，按相同顺序配对。
# 生成入口默认使用历史噪声；新增评测种子时，两臂使用相同 --noise-seed。
```

`export_video` 输出 81 帧、原时间戳对应 FPS 的 H264 Baseline/yuv420p CRF18；共享评测入口读取既有噪声 tensor，使用原协议的 Euler50、shift5、CFG5。每个视频旁的 JSON 记录实际采样设置与集中分析。

具体代码：主训练 [train/train.py](../../train/train.py)，完整配置 [native_physical_wisa3857.yaml](../../configs/native_physical_wisa3857.yaml)，部署入口 [训练 YAML](train/train_native_physical_blk96g.yaml)，受控诊断 [tools](../../tools)。

## 相关方法依据

- [DRaFT](https://arxiv.org/abs/2309.17400)：支持通过扩散采样末尾若干步骤反传的技术可行性，不构成本项目视频物理效果的证据。
- [Causal representation learning](https://arxiv.org/abs/2102.11107)：物理知识先验、观测相关性与可识别因果表示需要区分；本版不把语言命名当成因果识别。
