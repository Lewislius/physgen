# 共同实施契约：Physics Expert 方案一、二、四

版本：2026-11-10 / design-v2-final-for-implementation  
状态：**可进入实现的设计规格，未进行本项目 GPU 训练或生成评分；不是达标报告。**

本文件与所选方案的 Markdown 必须一起提供给编码 AI。主方案文件优先决定结构；本文件决定几何、训练、推理、测试与交付约定。不得自行把三个候选叠加成一个模型。

## 1. 目标与明确边界

目标不是以 SA 换 PC，也不是仅增加过线样本。目标同时覆盖：

- 固定 VideoPhy2-40：PC 均值至少 3.85、进一步达到 3.90；SA 均值至少 3.25；Joint 至少 15/40。
- 原先强样本继续提高物理细节、语义完整性和成像质量；原先失败样本增加联合成功。
- WMB、Physics-IQ、VBench 的同协议结果共同改善，并向同生成参数量、同输入条件、同推理预算的强基线竞争。
- 固定评测集合中不接受大色块、雪花、横线等明显画面崩坏。不得删视频、重抽种子或只展示较好结果。

本轮不设计方案三，不引入 V-JEPA 或其他新视觉教师，不使用教师特征损失、VLM 热图监督、干预学习、反事实训练对或推理期奖励搜索。已有 Qwen 作为输入条件编码器、已有 T5 作为原生文本编码器继续使用，不增加新的教师角色。官方评分器仅用于评测，不进入训练反向传播。

不使用“动作约束”。所有结构适用于静态平衡、接触、材料状态、形变、流体、光学、热学以及含运动的现象。不得用“运动更多”定义“物理更好”。

历史结构依据限定为 R1—R15。**禁止读取、导入、继承或以 R16 及之后候选的实现为起点。** 通用底座、公共基础设施可读取，但不得通过其后来修改的版本间接混入被排除设计。

## 2. 唯一允许的历史起点

项目仓库：`Lewislius/physgen`。固定审阅 commit：

```text
1defd6f4b41e42302bc3e4b9b2bf938abbbc686c
```

主代码基准：

```text
code/v5.2.2-native/experiments/physical_content_v7/execution_value_r10/physical_content/model.py
code/v5.2.2-native/experiments/physical_content_v7/request_bound_fusion_r15b/physical_content/model.py
code/v5.2.2-native/experiments/physical_scaling_q256/physical_scaling/model.py
code/v5.2.2-native/physgen_native/lora.py
code/v5.2.2-native/experiments/physical_content_v7/physical_content/supervision.py
```

R15B 代码可用于正确保留原生文本、真实 T5 长度、首帧与 block 拆分约定；不能默认启用其“先把物理 value 平均到请求 token”的读取机制。

事实与失败反例见 `05_EVIDENCE_AND_DECISIONS.md`。本轮不修改上述历史目录、不恢复历史训练任务、不覆盖旧分数和权重。建议独立工作目录 `experiments/physics_expert_redesign_v2/`，内部使用 P1/P2/P4 名称，不能命名为历史 R 版本。

## 3. 固定结构常量和动态维度

| 名称 | 默认值/形状 | 含义 |
|---|---|---|
| `D` | 3072 | Wan 隐空间宽度 |
| `num_layers` | 30 | Wan 原生 block 数 |
| `num_heads` | 24 | Wan attention 头数 |
| `head_dim` | 128 | 每头宽度 |
| `ffn_dim` | 14336 | 原生 FFN 中间宽度 |
| `qwen_width` | 4096 | 现有增强输入宽度 |
| `physics_tokens` | 256 | 物理槽数 |
| `enhancement_rounds` | 2 | 共享增强块迭代次数 |
| `hidden` | `[B,N,D]` | 实际 residual stream，未归一化 |
| `context` | `[B,L_pad,D]` | 原生 `native_text_context` 的输出 |
| `context_length` | `[B]` | 合法 token 数，含 EOS，不含人为补齐 padding |
| `C_route`,`C_execute` | `[B,256,D]` | 原有寻址内容、执行内容 |
| `groups` | `[256]` | 每个物理槽的既有关系分组 |
| `grid` | `[B,3]` | patch 后的 `(T_lat,H_patch,W_patch)` |
| `known_tokens` | I2V 为 `H_patch*W_patch` | 已知首帧的 token 数 |
| `times` | `[B,T_lat]` | 与 latent 帧对应的视频时间 |
| `sigma` | `[B]` 或现有 scalar | 扩散噪声时间，不是视频时间 |
| `modulation` | 原生 `[B,N,6,D]` 等可广播形状 | 严格沿用原生生成方式 |

480×832、81 帧若按当前 VAE 与 patch 配置编码，通常对应 21×15×26 的 token 网格；**实现必须从实际 `patch_embedding` 输出推导，禁止硬编码该示例。** 不同采样、padding 或 VAE 配置可能改变形状。

原生增强使用 32 个 Qwen-space attention heads，不要与 Wan 的 24 heads 混用。

## 4. 注入层数：先固定每 3 层，再比较每 5 层

以 0-based index 表达：

```python
sites = [i for i in range(30) if (i + 1) % condition_every == 0]
# every=3: [2,5,8,11,14,17,20,23,26,29]
# every=5: [4,9,14,19,24,29]
```

默认 `condition_every=3`。每 5 层是一个明确的结构对照，不是训练途中自动降低物理强度的策略。

所有注入点各有自己的 K/V/O、video-time 映射与必要的执行参数；只有增强模块共享。不要把同一张静态 residual 在十个位置重复相加：每个位置必须用当前视频状态重新读取 C。

不能把每 3 层训练的 checkpoint 直接删成每 5 层并宣称完成了公平结构比较。两个频率分别初始化、训练；报告不同参数量和计算量。第一阶段保持所有候选同为每 3 层，避免把频率与架构混为一个因素。

## 5. 原生 block 是实现基准，不是可自由改写的示意图

保留原生顺序：

```python
u_sa = norm1(h).float() * (1 + sa_scale) + sa_shift
h_sa = h.float() + self_attn(u_sa, ...) .float() * sa_gate
text_update = cross_attn(norm3(h_sa), context, native_context_lens)
h_text = h_sa + text_update
u_ffn = norm2(h_text).float() * (1 + ffn_scale) + ffn_shift
h_out = h_text + ffn(u_ffn).float() * ffn_gate
```

这里 `sa_gate`、`ffn_gate` 是**原生扩散时间调制**，不是新增的物理门控。三个方案只在各自文件规定的位置插入操作。

特别禁止：

1. 用 `norm3(h) + delta` 代替 `h + delta` 作为 residual stream。
2. 对原生 FFN 输入重复施加时间 modulation。
3. 先调用完整原生 block，再人工重复加一次该 block 的文本更新或 FFN。
4. 用另一种 padding mask 改写原生文本 attention，却把差异归于 Physics Expert。
5. 将 Qwen 物理槽的任意编号当作时空坐标应用视频 RoPE。
6. 将 `sigma` 当作视频帧时间；或者把帧序号当成真实秒数而不声明。

## 6. 统一的物理内容前端

保留 R10A 的 `Enhancement`、`ContentBlock` 和 `map_content` 的职责：

```text
冻结的输入条件特征
  → 关系内读取 + 全局读取
  → 两轮共享增强与组间交互
  → C_route / C_execute
```

`C_route` 继续承受原生词对齐辅助，`C_execute` 不承受直接词 MSE。共享增强底层仍会受到两种梯度，因此不能宣传“完全梯度隔离”。

Qwen 条件只来自允许的首帧、原请求及其分析。真实未来视频是生成目标，不得进入推理侧条件构建，也不得缓存为当前样本的“输入物理知识”。

首轮保持已有关系分组、256 tokens、两轮增强，不同步增加轮数或手工拆分“语义/物理/因果 token 预算”。不优化 C 余弦相似度、attention 熵或残差幅度。

## 7. 统一物理读取器

### 7.1 Query

物理 query 使用当前 residual 经本层 `norm3` 后，调用原生文本 Q 的**冻结 base**；使用同一原生 `norm_q`。它不是另一个新学 Q，也不把新文本 Q-LoRA 的增量偷偷共享给物理侧。

```python
def frozen_base_forward(linear, x):
    # 项目已核实的 LoRALinear 使用 .base, .lora_a, .lora_b。
    if isinstance(linear, LoRALinear):
        return linear.base(x)
    return linear(x)
```

不把同一个 base module 重新注册进多个子模块；把本层 attention 对象作为显式参数传入 reader，避免重复参数统计与 state_dict alias。

### 7.2 Key / value / 时间

- `W_key` 与 `W_value`：独立 full-width 3072×3072、bias=False，从对应原生文本 K/V 的 base 权重复制初始化，之后训练。
- `key_norm`：按旧实现复制，独立训练。
- `W_time`：256→3072、bias=False、零初始化，作用于既有 `time_features(times)`。
- 输出矩阵 `W_output`：3072→3072、bias=False、零初始化。
- 零值 reference slot 与既有关系先验保留，以隔离主改动；不增加新 null 分支或“预算控制器”。

对未来视频位置：

```text
q = native_norm_q(Q_base(norm3(h)) + W_time(video_time))
k = key_norm(W_key(C_route_effective))
v = W_value(C_execute)
m = Attention(q, [k, reference_key], [v, 0], existing_group_prior)
```

`m` 是多头拼接后的物理消息 `[B,N,D]`。已知首帧位置直接填零，不是待生成值。`C_route_effective` 在 P1 为请求增强后的 key 内容，在 P2/P4 为原 R10A `C_route`。

### 7.3 掩码和归一化

沿用 `group_log_prior(groups)`，不重新发明关系权重。支持不等长 batch 时，padding 物理槽的 logit 为负无穷；null/reference 仍有合法值，防止全 mask 产生 NaN。

attention 的 `1/sqrt(head_dim)` 只能施加一次。使用 `F.scaled_dot_product_attention` 时不再预除一次；用手工 logits 时显式除一次。PyTorch SDPA 的 bool mask 中 True 表示允许参与，与部分高级 attention API 的 mask 约定不同，接口测试必须验证。

### 7.4 计算与缓存

推理可缓存每个 CFG 分支、每一层的物理 K/V 和原生文本 K/V，因为 C 与文本在 denoising loop 内固定。P1 的 key enhancer 也可每个条件分支只计算一次。视频 query、m、文本输出随当前视频状态变化，不能跨 denoising step 缓存。

训练时，凡存在可训练参数的中间结果，不能跨 optimizer step 复用 autograd 图，不能 `detach()` 再当成等价优化。缓存 key 必须包含模型版本、条件内容、dtype、设备、site 与 CFG 分支。

## 8. 原生适配参数

共同开放原生文本 cross-attention **Q/O 的 rank-16 LoRA**，K/V base 保持冻结；LoRA `alpha=rank`，A 使用原实现初始化，B 零初始化。此项约 5,898,240 参数。

P1/P4 保留原 self-attention Q/K/V/O 与 FFN 的 rank-32 LoRA。P2 将这一组统一提高到 rank-64，把旧 `state_value` 的容量转移到实际生成运算中。

新增文本 LoRA 不是已证实的 SA 修复器。R11A 已尝试过更完整的 Q/K/V/O rank32 适配却未达标；本轮所有候选和关键控制都必须包含匹配的文本适配，不能把历史反例忘掉。

不能对已经包装为 LoRA 的线性层再次包装。P2 从 rank32 改 rank64若使用历史权重，必须显式迁移；默认采用 fresh official 初始化，不会加载形状不匹配的旧 adapter。

## 9. 默认训练处方：从第一步共同学习，不加新教师损失

### 9.1 初始机制训练

正式候选统一 `joint400`：400 optimizer updates，accumulation=4，共 1600 次样本使用。所有选定训练参数从第 1 步进入优化，warmup 32 updates；400 内 warmup 后固定峰值学习率。不得沿用父类 `set_phase()` 在前 128 步悄悄冻结新参数。

这是本轮新处方，不冒充与历史 128+272 完全相同。需要两个低成本匹配基线：

- 原 R10A 结构 + joint400；
- 原 R10A 结构 + 文本 Q/O LoRA16 + joint400。

历史 R10A/R1/R15B 视频仍作为输出质量参照，但不是每个新因子的严格单变量对照。

### 9.2 生成目标

沿用当前 FM 定义：

\[
z_\sigma=(1-\sigma)z_0+\sigma\epsilon,\qquad
v^*=\epsilon-z_0.
\]

I2V 已知首帧不计入生成误差；未知区域使用现有 `valid` mask。先复用历史损失，不通过同时改变 loss 给结构结果增加混淆：

\[
L_{gen}=0.8 L_{FM}+0.2 L_{tail},\qquad
L=L_{gen}+\lambda_{route}(s)L_{route}.
\]

`tail_fraction=0.25`，多尺度池化窗口保持原代码。`lambda_route` 在前 32 个更新从 0 升到 0.03，之后保持。`L_route` 只施加原始 C_route；P1 的请求增强后 route 不额外拉回静态词目标。

不加入特征蒸馏、区域热图、排序 reward、物理类别预测、动作标签、因果标签或专家正交损失。不存在 PC/SA 自动分配权重的在线控制器。

### 9.3 推荐优化参数

| 参数组 | 峰值学习率 |
|---|---:|
| enhancement | 5e-5 |
| physics_read（K/V/norm/time/reference；P1 state_value） | 2e-5 |
| physics_write（物理 O） | 3e-5 |
| native_self_ffn_lora | 2e-5 |
| native_text_qo_lora | 1e-5 |
| P1 request_key_enhancer | 2e-5 |
| P4 expert_in（A、B） | 2e-5 |
| P4 expert_out（O） | 3e-5 |

AdamW 的推荐显式起点：betas=(0.9,0.999)，eps=1e-8，二维矩阵 weight_decay=0.01；bias、norm、reference 参数不做 weight decay。若历史训练器实际不同，记录差异并让匹配基线使用相同新设置，不能声称完全沿用。

分组 grad clipping=1.0 可保留既有优化器设置。它不是推理时压制物理强度。BF16 矩阵运算、手工attention的logits/softmax使用FP32，residual/time运算按现有实现使用；SDPA/Flash后端需单独测试其数值行为，不宣称它与手工FP32逐位一致。不要启用 FP16 后把溢出当作模型机制失效。

### 9.4 dropout / CFG 条件

`text_dropout=0.1`。实现时必须核对并固定正负条件联动方式：原生文本与物理分析分支必须与 CFG 正负语义一致。不能出现训练时“空 T5 + 正请求 C”、推理时却使用不同规则而未声明的失配。

若当前缓存缺少一致的 negative 条件，先修复数据入口并生成一致缓存，而不是硬填全零 hidden 冒充正常负条件。保持首帧是否仍作为 negative 分支条件的原协议。

### 9.5 扩展训练不是盲目延长

已有缓存约 3857 条，400×4 仅 1600 次曝光。初筛后优胜方案可独立做总 2000 updates、有效 batch4（约 8000 次曝光），检查完整数据覆盖的效果；所有比较记录累计数据曝光与计算预算。

不是任意 400 失败方案都续训。先依据真实视频确定是否存在部分联合收益和可定位瓶颈。已有 R1 1000 步 PC 回落，不允许写“训练更久一定修复”。

面向更大范围竞争，另建高质量自然视频扩展集，具体规模按实际可用、授权数据决定。优先补充缺失物理类别、语义精确的同域样本和高成像质量样本；不用测试集未来视频，不用随机改条件的伪干预对。

## 10. 数据质量改进（不增加模型教师）

训练目标视频、原请求、物理分析必须匹配到同一缓存片段。审查实际采样范围、是否过程只在被裁掉的后半段发生、光照/相机变化是否被误写成物体变化、材料与数量是否被描述错误。

只对看得到的内容作事实描述；不能给静止图片编造未来结局当作输入“知识”。物理推断可以是条件化推断，但不得与真实训练片段冲突而仍作为精确标签。

优先使用已有视频、元数据与人工复核。允许离线修正文案，但新文案不是新教师特征，也不意味着自动正确。必须有版本号、样本级变更日志、重新编码缓存的 manifest。

采样与 splits 按原始视频/来源成组去重，防止同一个长视频不同 crop 分到训练和评测。静态与准静态现象不得因为 motion_score 较低被全部过滤。

## 11. 首帧与 T2V 的边界

I2V 的 `known_tokens` 从 geometry 推导。所有新增物理直接更新在 known slice 为 0。**这不等于内部首帧 hidden 永远不变**：全局原生 self-attention 可以间接改变其 hidden。采样器必须按既有协议持续提供正确的已知首帧；不能用“mask 了注入”替代输入条件的实现。

本轮默认评分继续是既有 I2V。要与 ProPhy 的 T2V 正式对标，需要独立启用真正 T2V：没有 GT 首帧、没有图像侧未来信息，所有 latent 位置有噪声并纳入生成损失。不能沿用 `prediction[:,:,1:]` 硬排除第一帧的 I2V loss。

T2V 不是把首帧 latent 填零就完成。必须通过官方 TI2V 无首帧路径、允许条件的 Qwen 文本输入及联合/单独 T2V 训练验证。该分支应独立报告，不把 I2V 600 提示词重放称为同协议 T2V。

## 12. API 和代码组织

建议新增以下模块，不 monkey-patch 未记录的全局行为：

```text
physics_expert_redesign_v2/
  common/enhancement.py          # 复用/封装 R10A
  common/physics_read.py         # 显式 raw message 与输出映射
  common/native_ops.py           # 原生 self/text/ffn 的无重复拆分
  common/lora_install.py         # 防重复包装；base accessor
  common/ownership.py            # 参数唯一归属与统计
  variants/p1_request_key.py
  variants/p2_pre_self.py
  variants/p4_cooperative.py
  train.py
  infer.py
  evaluate.py
  tests/
```

统一模型接口保留现有 `velocity(...)` 返回 `(velocity, auxiliary)`；若使用 dataclass 的新 `ConditionBundle`，必须提供适配器，不改变旧采样器的噪声方向。

统一 reader 返回：

```python
@dataclass
class PhysicsReadOutput:
    message: Tensor      # [B,N,D]，无输出 O，known slice=0
    write: Tensor        # [B,N,D]，已通过物理 O，known slice=0
```

P1 的 write 包含旧 state_value 交互；P2/P4 不包含。P4 需要 raw message 供条件专家使用，不能误把零初始化 O 后的 write 当成专家唯一输入。

`read_physics` 不负责调用原生文本 attention。native text 更新由 block adapter 精确调用一次。这样不再继承旧 Fusion 把 text 与 physics 混成一个返回值的歧义。

## 13. 参数注册、梯度和恢复

先完成所有模块与 LoRA 安装，再建立完整 trainable inventory，再创建优化器。每个 `requires_grad=True` 参数必须恰好归属一个 optimizer group；冻结参数不能进入更新组。不得仅按 `name.startswith('wan.')` 把新增文本、执行参数都混成一个组。

保存：模型结构 ID、默认配置、site 列表、base commit、源文件 hash、参数清单、训练数据 manifest、条件缓存 manifest、optimizer、scheduler、RNG、数据 cursor、precision 和采样设置。

允许恢复同一次运行；加载历史权重必须使用显式迁移器并报告 missing/unexpected keys。`strict=False` 不能成为静默跳过模块的理由。

gradient checkpointing 的 loop closure 要捕获当前 layer/site，不可让反向重计算调用最后一个 block。不要在整个冻结 Wan 上包 `no_grad()`：被冻结的权重不更新，但梯度必须穿过其运算回到 Physics Expert。

## 14. 必须通过的工程测试

### 14.1 初始化与数值

1. 新输出矩阵/LoRA-B 为零时，新模型与同一官方底座输出在 dtype 容许误差内一致。
2. 对“非零 trained adapter”的测试，必须先启用非零参数或运行几个真实 optimizer steps；不能在所有输出为零时要求源特征梯度非零。
3. 已知首帧的新增直接物理 residual 为零；不错误要求所有后续首帧 hidden 都不变。
4. 所有 logits mask 至少有合法 key；FP32 softmax 无 NaN/Inf。
5. 无额外 text/self-attention 重复前向。

### 14.2 梯度

6. 正常 FM 可训练物理 O、K/V、增强 execute 投影、route 投影、原生 LoRA。
7. P1 key enhancer O 激活后，Q/K/V 与上下游都有生成梯度。
8. P2 未来位置的原生 self-attention Q/K/V 都随物理写入变化，不仅 Q 变化。
9. P4 两个专家 output 投影首步可学习；后续 source/state 投影可学习；source=0 时该专家输出=0。
10. optimizer owner 检查与 trainable state_dict 覆盖率为 100%。

### 14.3 一致性

11. 同输入重复前向一致；checkpoint 开关与保存/恢复一致。
12. 正负 CFG 两臂分别正确缓存，不串用 C。
13. 原生文本 padding 处理与历史 baseline 一致；P1 binding 排除新增 padding。
14. 帧数、分辨率变化不破坏 view/reshape、时间 repeat_interleave 与 known mask。
15. `condition_every` 真正控制 selected sites，非注入层走原始 block。
16. 关闭物理分支的诊断不关闭语义 LoRA；结果不能标为“未经微调底座”。

本包 `checks/reference_math_checks.py` 仅用小型合成 block 验证若干代数与梯度性质，不等于实际 Wan 集成测试，更不等于生成性能证明。

## 15. 编码 AI 的交付要求

必须输出：实现文件、明确配置、CPU 单元测试、单个真实 Wan batch 的集成测试入口、完整参数统计、与历史脚本的接口适配、可恢复训练入口、固定协议推理入口、逐样本评分与视频清单。

实现完成与效果达标是两个独立状态。未生成评分时，将 `pc_mean/sa_mean/joint_pass_count` 设为 null，`quality_goal_achieved=false`；禁止按设计目标填充实际分数。

详细评测执行见 `04_EVALUATION_AND_EXECUTION.md`；来源见 `05_EVIDENCE_AND_DECISIONS.md`。

## 16. 额外显存与batch入口的实现约定

默认保持项目已经支持的每GPU一个视频，依靠gradient accumulation得到有效batch4。文档中的B维用于明确张量契约，不意味着现有variable-length batch已经支持；实现若暂不支持B>1，应明确assert，而不是静默混用不同视频的groups、valid mask或首帧数。

对物理attention，可以沿视频query维分块（例如起点`query_chunk_size=2048`，取决于实际显存），但每个query仍读取完整物理memory和同一reference。该分块应在FP32参考实现中与不分块数值一致。不能在每个key chunk上独立softmax，然后把输出相加。

训练时缓存仅限同一次forward中的重复条件构建；推理时才允许跨denoising step缓存已冻结的C/K/V。P1每个CFG分支只计算一次请求key增强，但native文本attention的query与输出仍每step重新计算。

## 17. 阶段扩展与历史权重迁移

`joint400`默认fresh初始化，便于解释结构贡献。后续总2000 updates的扩展建议从该候选自己的step400恢复，并将共同基线同步扩展；记录完整optimizer/RNG/data cursor，不重新把400步当作零成本起点。

2000阶段以相同峰值率延续，前400已完成部分不重跑warmup；400之后至2000按事先固定的cosine衰减到峰值的0.2。比较双方使用相同日程，不能只为新模型延长、只给基线400预算。

历史R10A/R15B warm-start可以作为单独生产研究，但不混入fresh对照。P2从rank32到64的精确线性迁移可用原A/B填入前32通道，新增B置零、A正常初始化，alpha/rank保持1；这只能保持LoRA线性映射不变，**不能保持将物理写入位置改变后的完整模型输出不变**。P4替换state_value也不是函数等价迁移。必须记录initialization_source与迁移清单。

配置中的本地权重、缓存、协议路径为null占位，需由编码AI依据实际存在的文件填写并验证。不得猜测GPU设备、自动下载新模型或自动启动训练。本包是设计与检查交付，不是远程执行授权。

## 18. 部署成本与没有增加的外部依赖

本轮延用已有条件特征缓存进行初筛训练；不要求Qwen在每个training step重新前向，更不要求另一个视觉模型在线提供teacher features。实际训练器应按现有工作流管理T5/Qwen/VAE的驻留或缓存，记录显存与前处理时间，不凭设计参数量保证某张GPU一定能容纳。

物理attention logit本身可较大：以N≈8190、24heads、257keys为例，完整FP32 logits约202MB（十进制），每个query chunk为2048时约50.5MB；这只是一个张量，不是全部激活显存估计。分块沿query维进行，保留完整key归一化即可避免改变数学输出。真实native attention backend与checkpointing仍需要实际测量。

三版均没有增加在线教师、未来状态编码器或评分模型。最终同规模比较仍需公开原有Qwen/T5/VAE及新增Physics模块，而不能仅用主干名称中的5B概括系统成本。
