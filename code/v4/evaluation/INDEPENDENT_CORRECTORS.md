# v4 按 block 独立 corrector：实现与验证

更新：2026-09-12。当前默认配置为 `parameter_sharing: per_block`、`a_blocks: [5,10,15,20,25,30]`、`a_every: 5`。

当前 P₀ 已支持训练 GT 条件及 25% 无 GT 训练，暖起默认关闭；GT 输入层只属于独立初始化组。新旧初始化迁移和验证见 [专项说明](GT_CONDITION_PRIOR.md)。

## 1. 参数到底怎样复用

| 调用关系 | 参数关系 |
|---|---|
| 同一 step 的 A@5 与 A@10 等不同 block | 完整 corrector 的参数各自独立 |
| 不同去噪 step 的同一个 A@5 | 复用 A@5 自己的参数 |
| 同一位置内部两轮更新（A2 等） | 复用该位置自己的参数 |
| CFG 正／负分支在同一 block | 复用该 block 的权重，P 状态各自保存 |
| P₀ 初始化器与各位置 corrector | 参数独立，包括文本投影 |
| 可选 B@18 与全部 A | 完整参数独立 |

六套 A 各自独占 reader、writer、所有归一化与门控、文本／sigma 投影、类型 embedding、两个 Transformer block、状态更新头、读写 cross-attention。固定几何编码和架构超参数可以相同；P 仍按调用顺序传递，这是激活传递，不是参数共享。

同一训练 step 不会在每个 block 后执行优化器。整个前向完成后统一反向、梯度累积和一次 AdamW 更新；各 block 的梯度与 Adam 状态分别保存。后续损失可以通过 P/H 链训练较早的 corrector，这也不等于共享权重。推理过程不更新任何模型权重。

## 2. 实现与阶段

`physgen_v4/corrector.py` 在构造时分别创建 `A.5`、`A.10`、`A.15`、`A.20`、`A.25`、`A.30`，每个 ModuleDict 项包含独立 `core` 和 `site`；另有 `prior`、`B_core`、`B`。forward 按真实 Wan block 选择模块，不创建临时网络，不用缺省 A 替代未知位置。多位置 A 调用必须传入已注册的整数 block。重复、乱序、非法位置、与 a_every 冲突的布局，以及超出实际 Wan 深度的布局均会报错。

`ProcessWan` 的正式前向、逐位置 writer 消融和离线后缀重放均走同一分发入口。默认 30 层 Wan 的调用序列为 A@5→A@10→A@15→A@20→A@25→A@30；启用 B 时在 A@15 与 A@20 中间插入 B@18。P/H 形状、原生教师监督、跨去噪步接续和注意力方向保持原有定义。

| 阶段 | 可训练组 |
|---|---|
| A1／A2／A3 | initialization 和六个 A@block |
| B1 | 完整 B：B_core 与 B 适配器 |
| AB | initialization、六个 A@block、B |

B1 冻结全部 A 和初始化器。由于 B 不再借用 A 的核心，B1 必须训练自己的 B_core；新训 A 阶段没有训练过 B，不能把随机 B_core 冻结后只训练其适配器。B.b 与 writer 仍为零起步，最初对 P/H 为恒等映射，部分上游梯度首次为零是预期行为。

独立初始化组包括它自己的文本投影，暖起且关闭 L_prior 时可以完整排除该组的同步／裁剪／更新。AB 的 B 关闭时也排除完整 B 组。每个 A 的权重、梯度指标单独记录，例如 `weight/A@10_rms`、`grad/A@10_norm`；无须把不同位置混成一组。

## 3. Checkpoint 与旧实验

| 使用方式 | 行为 |
|---|---|
| 当前配置新训 A1 | 创建六套独立 A、独立先验与 B |
| 独立 checkpoint 的 RESUME | 恢复各位置权重、各自优化器状态、调度与游标 |
| 独立 checkpoint 的阶段 INIT_FROM | 严格恢复各位置权重，创建新阶段优化器，不清零 writer |
| 旧 checkpoint 缺少 parameter_sharing | RESUME／推理保留历史 shared 架构与旧参数顺序 |
| 旧 shared → 当前 per_block | 在 A／AB 使用 INIT_FROM；复制权重数值到各位置独立参数，创建新优化器 |

旧共享模型直接 RESUME **仍是旧共享模型**。要使用此次结构，应新训当前配置，或显式设置 INIT_FROM 迁移；两者与 RESUME 互斥。模型没有在训练中途自动拆分旧优化器状态。

仅拆分时，各 A 初始可具有相同数值，但 Parameter、存储、梯度、Adam 状态不共享，随后分别训练；旧 shared.text 也复制到独立 prior.text。B_core 从旧 shared 核心复制兼容权重。若同时将 interpolate 迁移为 cross_attention，则各位置新建注意力／读取门，全部 writer 和 B.b 才会重新归零；若同时替换 latent 初始化，则新建 image_text_gt 先验。这些迁移只能进入 A／AB，防止冻结新增的随机 A 或先验。原数据缓存可以继续使用。

加载前校验全部键、形状和独立 block 布局；不完整的 checkpoint、未声明的缺项、独立 block 改名或合并成 shared 均拒绝。训练、推理与归因评价使用相同的权重加载方法。历史 SharedCore 只保留在旧架构兼容分支中，当前 per_block 模型没有 shared 子模块。

## 4. 参数和显存预算

以下为默认完整宽度的参数枚举结果，包含 bias 与 LayerNorm：

| 部分 | 参数数 |
|---|---:|
| 初始化组，含独立文本投影 | 20,610,880 |
| 单个 A 完整 corrector | 136,224,515 |
| 六个 A 合计 | 817,347,090 |
| A 阶段可训练合计 | 837,957,970 |
| B 完整 corrector | 136,224,516 |
| AB／模型总参数 | 974,182,486 |

corrector 自身仍在每个 rank 完整复制，Wan 的 FSDP 不会将这些参数一起分片。所有 corrector FP32 权重约 3.63 GiB；以“全部权重 × 4 字节 ＋ 活动参数 × 12 字节（梯度和两份 Adam 状态）”计算，A／B／AB 每卡静态预算约 **12.99／5.15／14.52 GiB**。尚未包含 Wan 权重、激活、通信、重计算临时张量、VAE／教师、分配器；CPU 内存和 checkpoint 存储也随之增加。

独立参数改造没有更改资源分配；随后按用户的单卡测试要求，将96G入口改为1卡。当前 `train_wisa_native_p_4x48g.yaml` 实际是 ada-24g／2 卡，4x80g 是 amp-80g／4 卡，4x96g 是 blk-96g／1 卡。文件名不能证明容量，不能声称目前两张 24G 卡已经验证可完成训练。

不同 block 有不同 hidden，独立网络提供分别适配的能力；这是一项结构选择，不保证生成效果更好。小数据下增加参数也增加优化难度，实际收敛、泛化与完整视频收益仍需实测。

## 5. 实际验证与复现

**独立参数拆分时的 116 项 CPU 回归最终全部通过，其中独立参数与六位置集成 27 项；无跳过项。** 全量检查的两个 spawn 用例先遇到标准输入入口问题，改用模块入口后其中一个被沙箱 IPC 禁止；获准后在沙箱外重跑两项均通过，生产代码与测试用例没有为此降级。验证记录保留各次结果。

拆分时的结果保存在 [independent_correctors_review.json](results/independent_correctors_review.json)。测试使用实际完整 1664／3072 通道 corrector 和 CPU 小 token 网格，部分测试将先验内部宽度缩小为64；默认先验512维的完整参数预算通过 meta 枚举单独检查。

专项覆盖：全部模块／Parameter／存储的跨位置隔离，单 A@10 的真实反向与 AdamW 更新只改变本位置，所有六组的梯度覆盖，冻结 A 的独立 B 训练，跨 sigma 的本位置参数复用，两轮重计算梯度，完整模型 checkpoint 文件及每个 block 的独立 Adam 状态往返恢复，旧共享权重逐项迁移、组合迁移和不完整 checkpoint 拒绝。六位置集成使用真实 corrector 与冻结 TinyWan 替身，检查顺序 P 接续、FM 梯度、暖起梯度组、writer 消融与精确后缀重放。

专项复现：

```bash
cd /home/liuzhirui/Project/physGen/code/v4
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -m unittest -v \
  evaluation.test_independent_correctors evaluation.test_condition_prior
```

全部回归（使用模块入口，支持 DataLoader 的 spawn）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -m unittest -v \
  evaluation.contract_checks evaluation.test_condition_prior evaluation.test_cross_attention \
  evaluation.test_environments evaluation.test_independent_correctors evaluation.test_launchers \
  evaluation.test_monitoring evaluation.test_preparation evaluation.test_training_runtime
```

历史共享架构契约和注意力数学专项保留显式 shared 配置，用于兼容性回归；独立架构由独立参数专项和当前六位置集成验证。启动器测试按 YAML 的实际 resource_pool 检查卡数传递，避免把用户改过的资源池误判成代码失败。

本轮未提交集群任务，未运行真实 Wan／V-JEPA 生成、GPU 训练或多 GPU NCCL 验证。CPU 结果不代表上述路径或实际峰值已经通过；目标节点的短程训练开关和诊断方法见 [NCCL 检查方案](NCCL_HANG_CHECKLIST.md)，新结构需使用当前 per_block 配置。

## 6. 实验 61001 后的 AMP 修复与正式入口

用户随后提交的单卡实验 61001 在首次 backward 触发 `CheckpointError`。根因是 PyTorch 2.8 的 AMP 权重缓存跨越 no_grad 暖起与带梯度前向，造成激活重计算的保存张量不一致。共用训练入口现在显式关闭该缓存；BF16、checkpoint、P₀ 监督及各 block 参数归属继续保留；GT 初始化改造后暖起默认关闭，仅保留为显式对照。

此前 116 项未覆盖该组合。本次使用相同运行环境复现旧训练入口失败，并验证默认 512 维先验、六套独立 A、全参数梯度及训练控制流，共 22 项相关 CPU 回归最终通过。三份硬件 YAML 均显式 `PHASE=A1`、`TRAIN_CHECK_STEPS=0`，执行完整配置的 500 步 A1 与验证／保存。详见 [故障及正式训练说明](CHECKPOINT_AUTOCAST_FIX.md)。
