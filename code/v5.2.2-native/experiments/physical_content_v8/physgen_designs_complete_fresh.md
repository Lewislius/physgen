# Physics Expert：方案一、二、四——完整单文件设计稿

版本：2026-11-10。本文包含全部设计规范、证据与三份默认配置。对应ZIP包含每5层配置、对照配置和CPU参考检查。

**设计定稿不等于训练效果已验证。本包未进行项目GPU训练或生成新评分。**


---

<!-- 原文件：README.md -->

# Physics Expert：方案一、二、四完整实现设计包

版本：design-v2 / 2026-11-10。方案三不在本包中。

**建议先实现P2-S3-R64与P4-S3-H512作为两个独立竞争方案；P1-S3用于验证读取/寻址改进。** 不把三版第一轮全部叠加。

目标：VideoPhy2-40同时PC≥3.85、争取3.90，SA≥3.25，Joint≥15/40；强样本继续提升；实际画质不崩坏；进一步在大规模、同任务、同预算评测中竞争。包内没有任何新模型已达标的声明。

## 阅读顺序

| 文件 | 作用 |
|---|---|
| 00_IMPLEMENTATION_CONTRACT.md（文件：`00_IMPLEMENTATION_CONTRACT.md`） | 公共张量、原生计算、训练、数据、精度、接口、梯度及恢复契约 |
| 01_P1_REQUEST_KEY_ENHANCEMENT.md（文件：`01_P1_REQUEST_KEY_ENHANCEMENT.md`） | 方案一：请求条件化物理key，原始全宽value保留 |
| 02_P2_DISTRIBUTED_NATIVE_EXECUTION.md（文件：`02_P2_DISTRIBUTED_NATIVE_EXECUTION.md`） | 方案二：每3/5层预写入已有原生block，不复制、不双前向差分 |
| 03_P4_COOPERATIVE_EXPERTS.md（文件：`03_P4_COOPERATIVE_EXPERTS.md`） | 方案四：物理—语义上下文条件专家，无新教师、无干预学习 |
| 04_EVALUATION_AND_EXECUTION.md（文件：`04_EVALUATION_AND_EXECUTION.md`） | 对照顺序、40例及大规模目标、画质与公平比较 |
| 05_EVIDENCE_AND_DECISIONS.md（文件：`05_EVIDENCE_AND_DECISIONS.md`） | 原始来源、历史反例、推导与新假设区分 |
| 06_HANDOFF_TO_CODING_AI.md（文件：`06_HANDOFF_TO_CODING_AI.md`） | 可直接复制给后续AI的开发任务说明 |
| configs/README.md（文件：`configs/README.md`） | 独立候选与对照YAML的使用说明 |
| checks/reference_math_checks.py（文件：`checks/reference_math_checks.py`） | CPU小张量代数/梯度检查；不是Wan项目实现 |
| checks/reference_math_checks_results.json（文件：`checks/reference_math_checks_results.json`） | 本次实际CPU检查结果，不包含生成评分 |

## 三版的边界

| 方案 | 独立变化 | 每3层参数估算 |
|---|---|---:|
| P1 | key读取原请求，value不经过请求预平均 | 713,657,344 |
| P2 | 物理写入提前到原生self-attention前，state_value预算转给rank64原生适配 | 638,553,088 |
| P4 | 两个来源条件专家替换state_value，保留全宽直接write | 675,908,608 |

以上是训练参数估算，不是新增推理总参数，也不是实际部署统计。原始R10A/R15B审计为670,010,368，详见E06。

已有Qwen只继续作为输入条件模型，T5作为原生文本编码器；不引入V-JEPA或其他新教师，不做热图监督/干预训练/动作约束。设计来源限定R1—R15与公共底座，不读取或继承R16及之后候选。

## 当前交付状态

完成：Markdown设计、YAML配置契约、预算推导和可运行的小型CPU参考检查。
未完成：与真实Wan的集成、项目GPU训练、视频生成、VideoPhy2/WMB/Physics-IQ/VBench新评分。

CPU检查验证的是函数关系、shape和梯度性质，不能推出PC、SA或Joint的数值。正式测试结果使用`results_template.json`从真实输出填写。


---

<!-- 原文件：00_IMPLEMENTATION_CONTRACT.md -->

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


---

<!-- 原文件：01_P1_REQUEST_KEY_ENHANCEMENT.md -->

# 方案一终版：原请求条件化物理键，完整保留物理执行值

设计 ID：`P1-RKE-v2`  
默认配置：`P1-S3`，每3层注入；对照为独立训练的`P1-S5`。  
状态：设计规格，未训练、未生成项目评测分数。  
必读公共约定：`00_IMPLEMENTATION_CONTRACT.md`（文件：`00_IMPLEMENTATION_CONTRACT.md`）。

## 1. 这一版解决什么问题

R10A 的直接物理value有PC收益，但其寻址未必足够贴合原请求。R15B试图先让每个请求token聚合物理value，再由视频读取；这种结构恢复了一部分Joint，却可能在中间聚合时丢失精细信息。[E02,E05]

这一版的决定是：

> **原请求帮助每个物理槽形成更适合寻址的key；生成视频仍直接读取该槽原来的完整物理value。**

不把物理value先平均到请求token上。不让视频attention通过两次归一化的乘积决定value。不增加三方打分器、语义保护系数或热图教师。

对上一版方案一的修订：去掉多项人工score组合和区域监督；用一个标准cross-attention完成“原请求→物理键”的条件化。该模块共享于所有注入点，只改变C_route，不改变C_execute。

## 2. 继承与改变清单

### 2.1 保留

- R10A的256个物理槽、两轮共享增强、关系内/全局读取及组间交互。
- 全宽C_route/C_execute，execute不做直接词MSE。
- 每site独立物理K/V/O、视频时间query、原有关系先验与零值reference。
- 原始物理state_value交互保留在本版；不把“改读取”与“改执行”一起混成大量变化。
- 原生self-attention与FFN LoRA rank32。
- 原生文本路径及原生padding约定。

### 2.2 只增加两项

1. 一个共享的、3072宽度的`RequestKeyEnhancer`，将当前原请求内容关联到物理键。
2. 原生文本Q/O rank16 LoRA，作为三个方案共同的语义适配配置；严格匹配控制实验。

不是另外生成一套请求记忆并向视频写入。不是物理开关路由器。

## 3. 为什么只改变key而不改变value

需要区分两个问题：

- **寻址**：当前视频位置应该使用哪条物理内容？
- **执行**：那条物理内容具体提供什么信息？

原请求有助于前者。例如同一画面存在玻璃、液体、桌面和投影区域，物理分析可能同时描述多种关系；原请求有助于明确当前生成应体现哪些对象、材质与现象。但不应因此先把多条物理value压成少数请求词对应的平均值。

形式上，本版保留：

\[
M=\operatorname{softmax}(Q_h K_{C^*_r}^{\mathsf T})V_{C_e},
\]

而不是：

\[
M=A_{h\to T}\,B_{T\to C}\,V_{C_e}.
\]

两者都可能有效，不能仅凭公式判定优劣。本版是可检验的替代，关键是**原始value矩阵不发生请求预聚合**。

“只有key改变”也不保证每个细节都被读出：softmax本身仍会平均多个value。因此本方案不能宣称消除了全部内容平均或保证了准确对象绑定。

## 4. 最小计算图

```text
图像/原请求/物理分析的既有输入特征
               │
      R10A全宽Enhancement
               │
        C_route   C_execute────────────────────────┐
           │                                      │
原生T5 context ──→ RequestKeyEnhancer                │
           │                                      │
       C_route*                                    │
           │                                      │
各层当前视频状态 → 物理K寻址 ───→ 直接读取原始C_execute
                                            │
                                 原物理执行/输出映射
                                            │
                                      视频生成块
```

不使用未来视频、额外教师、动作标签、反事实数据或推理期评分。

## 5. RequestKeyEnhancer的精确定义

输入：

```text
C_route : [B,256,3072]
T_valid : [B,L_valid_or_padded,3072]
text_mask: [B,L], True为合法实际token，包含EOS
```

使用24heads、head_dim128，标准cross-attention：

\[
R=\operatorname{MHA}\big(W_q\operatorname{LN}(C_r),
W_k\operatorname{LN}(T),W_v\operatorname{LN}(T)\big),
\]
\[
C_r^*=C_r+W_o R.
\]

MHA中的Q/K/V投影与输出投影合计4D²参数。不要在外部再套一套重复Q/K/V。这里的`R`只是正常attention输出，不是新引入的物理状态、可信度或因果变量。

实现约定：

- 两个LayerNorm均`elementwise_affine=False`，eps=1e-6。
- Q/K/V矩阵bias=False，Xavier初始化；O bias=False，零初始化。
- 不额外加FFN，不加专用sigma/video-time映射，不加null slot。
- 不给物理槽施加视频RoPE，不为槽位索引赋予物理空间含义。
- 新cross-attention使用合法原请求token；原生视频文本attention仍保留历史padding处理，不随它改变。
- `T`来自已有Wan的`native_text_context`，不是额外运行一个文本模型。
- 这一模块只实例化一份，所有site共享其C_route*。各site物理K仍是独立可训练映射。

为什么不是每site再建一套RequestKeyEnhancer：原请求与物理内容的基本对应不需要首先复制十套；保留独立site K可适配各层坐标。若共享是瓶颈，必须用真实生成对照证明后再研究，不在首版提前加十倍参数。

## 6. 完整层内顺序

这一版刻意不改变R10A的主要执行位置，以便先检验读取改动：

1. 原生self-attention作用于当前h，得到h_sa。
2. 原生文本attention读取原请求，得到t；此时尚未把t写回用于物理query。
3. 物理分支用同一pre-text状态h_sa读取C_route* / C_execute，得到m。
4. 保留旧执行公式：

\[
\delta_p=W_o^p\big(m+W_{state}\operatorname{LN}(h_{sa})\odot m\big).
\]

5. `h_mix = h_sa + t + delta_p`。
6. 执行原生FFN与原生时间调制。

已知首帧位置的物理delta为0。原生文本只调用一次。

本版没有宣称state_value不可替代。R15B对照显示它存在PC均值成本和部分Joint贡献，适合在读取机制确定后再通过匹配训练决定去留。[E05] P2/P4则各自给出了明确替代用途。

## 7. 可直接编码的核心模块

```python
class RequestKeyEnhancer(nn.Module):
    def __init__(self, width=3072, heads=24, eps=1e-6):
        super().__init__()
        assert width % heads == 0
        self.heads = heads
        self.head_dim = width // heads
        self.route_norm = nn.LayerNorm(width, eps=eps,
                                       elementwise_affine=False)
        self.text_norm = nn.LayerNorm(width, eps=eps,
                                      elementwise_affine=False)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        for layer in (self.q, self.k, self.v):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(self.o.weight)

    def split(self, x):
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)

    def forward(self, route, text, text_mask):
        assert text_mask.dtype == torch.bool
        assert text_mask.any(dim=-1).all()  # EOS至少合法
        q = self.split(self.q(self.route_norm(route)))
        t = self.text_norm(text)
        k, v = self.split(self.k(t)), self.split(self.v(t))
        message = F.scaled_dot_product_attention(
            q, k, v, attn_mask=text_mask[:, None, None, :],
            dropout_p=0.0,
        )
        message = message.transpose(1, 2).flatten(2)
        return route + self.o(message)
```

真实项目需要沿用设备、混合精度与checkpoint约定。不得把该模块输出直接写到视频；它只进入物理K。

前端使用：

```python
raw_content = enhancement(evidence)
route_effective = request_key_enhancer(
    raw_content.route, native_context, actual_text_mask
)
content = ConditionBundle(
    route=route_effective,
    execute=raw_content.execute,  # 原张量，不求token平均、不重新生成
    groups=raw_content.groups,
)
route_loss = enhancement.content_loss(raw_content, evidence)
```

如果为了`content_loss`保存raw_route，不要把它注册成运行时可训练“状态变量”；只是保留前向张量给原辅助目标。

## 8. 三个重要不变性

### 8.1 初始内容不变

W_o=0时，`C_route* == C_route`。新增模块不会一开始随机改变物理key；之后正常学习，不施加幅度上限。

### 8.2 value不被新模块改变

输出的C_execute与原增强输出完全相同。应使用相等性/存储引用测试，而不是通过某个cosine“差不多”证明。

### 8.3 没有独立请求写入捷径

若执行value全为0，且value投影无bias，物理消息为0，state_value乘积仍为0，物理输出为0。RequestKeyEnhancer不能绕开C_execute独立产生一个语言residual。

这与R11B新增独立请求memory不同。但这一性质不证明实际训练一定使用场景特异物理内容；它仍可能学习通用修正，需要生成对照。

## 9. 参数和计算预算

| 配置 | 估算训练参数 |
|---|---:|
| R10A每3层原结构 | 670,010,368 |
| RequestKeyEnhancer | +37,748,736 |
| 文本Q/O LoRA16 | +5,898,240 |
| **P1-S3** | **713,657,344** |
| **P1-S5** | **559,492,096** |

相对R10A训练参数增加约6.5%，不是严格同训练参数；总生成系统规模差异仍需分项报告。若声称同预算优势，应增加等参数的普通物理增强扩容对照，而非用“只多一点”省略统计。

新增attention大小为256×L_text，且每个条件只计算一次。推理可在denoising loop之前缓存C_route*；训练不能跨optimizer step缓存旧图。

## 10. 训练处方与梯度说明

主实验使用公共`joint400`，原数据/原样本顺序/原噪声采样规则固定。不默认使用新数据、无教师loss，无干预对。

词对齐辅助只施加raw C_route。若对C_route*也强制静态词对齐，可能抵销原请求条件化，属于另一个实验，不能悄悄开启。

本版有多个零初始化投影：增强route/execute、物理O、新binding O及LoRA-B。它们不一定在第一个step同时得到非零生成梯度。正确测试需要检查：

- 第一个step物理O能从非零anchor/value学习；
- 随后binding O可学习；
- binding O非零后，其Q/K/V可学习；
- 上游Enhancement仍接收生成梯度与既有route辅助。

不得为追求“首step所有参数都有梯度”加入未知用途的常数残差或可学习强度。

## 11. SA和PC共同改善的预期机制

物理知识的类型可能正确，但对应对象不准确。只提高C的表达力未必解决这种错位。本方案让每个物理槽读取**原始请求的原生文本空间**，从而有机会把物理key组织成更符合当前生成目标的寻址表示。

同时，value不经请求预平均，已有较强物理样本的局部信息仍有路径进入生成。目标是提升有用的条件使用，而不是减少物理影响换语义。

潜在问题同样明确：新增cross-attention可能学到重复已有Qwen request memory的功能，或者只增加公共文本成分。没有监督直接保证“语义绑定正确”。因此，不可仅凭attention图好看宣布成功。

## 12. 默认数据优化与结构分开

初始400保持旧缓存，先解释结构作用。后续使用自然视频监督改善上限时，再审查caption是否真实描述缓存片段、物理分析是否添加不存在条件、静态/光学/材料类别是否覆盖。

不把原始请求替换成长篇物理扩写。生成模型的T5输入仍是原请求；Qwen物理分析是独立输入条件，而不是重新定义用户要求。

不使用外部视觉老师，不训练“物理热图预测器”。没有高质量区域标签时，不用伪标签假装获得精确空间监督。

## 13. 推理及内容对照

正负CFG条件各计算自己的C和C_route*。不能把positive请求增强后的key复用于negative分支。

同首帧、原文请求、seed42、固定初始噪声、50步/shift5/CFG5。所有注入site固定启用，不逐视频调参数。

诊断对照解释应准确：

- 关闭RequestKeyEnhancer：检验当前checkpoint是否依赖新键，不能当作独立训练R10A。
- 固定训练均值C后仍运行RequestKeyEnhancer：保留了原请求信息，不能称为“所有样本条件已消除”。
- 替换reasoning时保持首帧与原请求不变；正常C不等于每句reasoning一定正确。
- 不允许拿错误C视频变差作为唯一证据。必须还优于匹配baseline。

这些是评测诊断，不是干预学习或反事实训练。

## 14. 专项测试

| 编号 | 检查内容 | 失败说明 |
|---|---|---|
| P1-T01 | binding O=0时route完全不变 | 初始化/残差实现错误 |
| P1-T02 | C_execute不变 | 错把value也增强或聚合 |
| P1-T03 | text padding改变但合法token不变，binding输出不变 | padding泄漏 |
| P1-T04 | 全零execute时物理更新为0 | 新增了未声明的独立语义写入 |
| P1-T05 | native text attention次数仍为1/site | 重复语义更新 |
| P1-T06 | O激活后不同原请求能改变route*与预测 | 新模块被断开/只缓存旧值 |
| P1-T07 | route辅助仅连接raw_route | 辅助目标与设计不一致 |
| P1-T08 | 多step后Q/K/V/O都有有效梯度 | 串联零初始化或detach错误 |
| P1-T09 | inference缓存与uncached一致 | 缓存分支或版本错误 |
| P1-T10 | 不加载/读取R16、教师权重或未来特征 | 实现超出设计范围 |

## 15. 最小机制实验

第一轮：P1-S3对比R10A+相同文本Q/O LoRA、相同joint400。

只有主结果显示有益趋势后，优先完成：

1. 同训练但无RequestKeyEnhancer；这是必需的结构对照。
2. 若需要证明“只改key”的优势，用同模块让key/value都改变的单独版本；它不是默认配置。
3. 每5层独立训练的成本对照。

不要同时扫宽度、轮数、十种门控、多个score系数。新增模块若不能超过相同文本适配的直接读取，就没有理由仅因解释合理而保留。

## 16. 达标标准

按公共评测文件同时评估PC均值、SA均值、Joint、通过率、强样本子集和真实质量。固定40目标保持PC≥3.85/争取3.90、SA≥3.25、Joint≥15。

如果P1只恢复SA但显著降低已有强样本PC，不宣布解决问题。若只增加PC不增加SA/Joint，则读键改进可能不足，应该与P2或P4进行独立方案竞争。

大规模结论必须来自独立样本、WMB、Physics-IQ、VBench同协议评估，不用固定16的局部改善替代。

## 17. 编码交付与禁止自行加项

交付共享binding模块、独立per-site reader接口、raw/effective route区分、native Q/O LoRA安装、参数审计、上述专项测试和训练/推理配置。

不引入三方匹配score、语义置信度、额外专家路由、视觉教师、区域loss、反事实数据、动作约束或推理筛选。P1不默认合并P2预写入或P4专家。先证明这一个新的内容读取环节有实际生成价值。

来源索引：[E01,E02,E04,E05,E06,E11]见`05_EVIDENCE_AND_DECISIONS.md`（文件：`05_EVIDENCE_AND_DECISIONS.md`）。


---

<!-- 原文件：02_P2_DISTRIBUTED_NATIVE_EXECUTION.md -->

# 方案二终版：分布式物理预写入与原生生成执行

设计 ID：`P2-DNE-v2`  
默认配置：`P2-S3-R64`，每 3 层注入，原生 self-attention/FFN LoRA rank64。  
对照配置：`P2-S5-R64`，每 5 层注入。  
状态：**设计定稿，等待真实实现、训练与生成评测。**  
先读：`00_IMPLEMENTATION_CONTRACT.md`（文件：`00_IMPLEMENTATION_CONTRACT.md`）。

---

## 1. 先回答“是不是空中楼阁”

上一版的“两处复制生成块 + 两次前向相减”不再作为本方案。原来的差分有数学意义，但不能由此推出 PC、SA 或画质必然提升；其计算与归因成本也过高。

本版只有一个明确的主结构变化：

> **把由当前视频状态读出的全宽物理消息，写入每个选定 Wan block 的实际输入 residual；随后只调用一次这个原生 block，让它的 self-attention、文本 attention、FFN 共同处理物理条件。**

这不是另建一个无法解释的执行器。执行器就是现有 Wan block。不是让物理分支变小，而是把它的作用位置提前，让它影响更多原生生成运算。

默认每 3 层执行一次，共 10 次；每 5 层为独立训练的 6 次配置。不是只在第 10、20 层放两个特殊大块。

这版可以逐项实现、检查输入、数梯度、审计参数与计算量。但完成这些只能建立实现正确性的信心，不能把“可实现”写成“已经达到 PC3.9/SA3.25/Joint15”。

## 2. 为什么选它，而不是只移动一个残差就宣称成功

### 2.1 历史上已经知道的东西

R10A 的 full-width `C_route/C_execute` 加直接写入得到过固定40 PC3.750，但 SA2.825、Joint7；物理能力有收益，联合能力不足。[E01,E02]

R13B 在 self-attention 前读取 C，主要用它调制原生 Q，并保留 self-attention 后的直接写入；其完整40为 SA2.875/PC3.525/Joint7。不能把“影响了原生 attention”直接当作成功。[E03]

R11A 也已经尝试过原生文本 Q/K/V/O LoRA 和文本先更新、物理后执行，其结果没有解决联合目标。因此“解冻文本”不是本版创新或成功证据。[E04]

### 2.2 本版具体改变什么

| 项目 | R10A / R13B 的相关实现 | P2 |
|---|---|---|
| 物理值 | 全宽 C_execute | 保留，不先请求平均 |
| 物理写入位置 | 主要在 self-attention 后；R13B 另调 Q | 在选定 block 的 self-attention 前 |
| 对 self-attention 的作用 | R13B 主要改 Q，原生 K/V 保持原视频状态 | Q、K、V 由同一个被物理更新的视频状态自然计算 |
| 文本读取时看到什么 | R10A 同层文本与物理并行 | 当前层文本读取已处理物理信息的状态 |
| 状态执行 | 外加 `state_value(state)*message` | 原生 attention 与 FFN 负责执行 |
| 训练容量 | 原 self/FFN LoRA rank32 | rank64，预算来自移除 state_value |
| 新视频大块 | 无/上版建议复制 | **不复制，不增加** |
| 教师/额外监督 | 既有 FM+route | 相同，不新增 |

本版不是把物理作用塞进 Q 的一个子通道，而是让已有生成块在同一输入状态上组织时空读取、语义读取和非线性变换。

这仍是一项待测机制假设，不是必然优于后写入的定理。最关键的匹配对照就是“同一 reader、同一 LoRA rank、同一训练处方，仅改 pre-self 与原来的写入位置”。

### 2.3 不能夸大“让原生运算执行”的新颖性

旧R10A的物理残差并不是写入后就不再经过原生生成运算：它当层会进入FFN，并继续经过后续block的self-attention、文本attention和FFN。**因此本版不能被宣传为“首次让物理进入原生生成算子”。**

本版要检验的更具体：在同一个选定site，把物理写入提前到self-attention之前，使同层Q/K/V、文本读取和FFN处理同一个物理更新状态；同时将外部状态矩阵预算转移到原生适配参数。两者都具有明确的实现差异，但都不是性能定理。

特别地，新位置与旧位置只相隔一次self-attention和一次文本读取，后续网络也可能自行吸收这个差异。若同预算P2-POST不比P2差，就应承认“预写入位置”不是有效新增因素。不能靠给同一残差重命名为“执行器”回避这个结论。

这里仍然值得主攻的原因是：现有物理C已显示局部物理收益，而原生生成路径的适配容量相对小；本版用一次标准残差条件输入和可审计的容量重分配检验两者能否更好共同学习，且不依赖未定义物理状态、额外教师或第二条生成前向。

## 3. 默认设计冻结项

主版固定以下选择，编码 AI 不得自行追加：

- R10A 256 tokens、4096→3072 全宽增强、两轮交互、route/execute 分离。
- 原始 C 直接读取，不默认加入方案一的 key enhancer，也不混用 R15B 的 value 预聚合。
- 每 3 层一个独立 K/V/O 物理读取器，层号 `[2,5,8,11,14,17,20,23,26,29]`。
- 物理 Q 复用本层原生文本 Q 的冻结 base；不复制新 Q 矩阵。
- 移除原 `state_value` 乘性交互。
- 原生 self-attention Q/K/V/O 与 FFN LoRA rank64，alpha64；30 层均启用。
- 原生文本 Q/O LoRA rank16，alpha16；文本 K/V 冻结。
- 所有被选择的训练参数从第 1 个 optimizer update 共同学习。
- 没有独立 sigma 调制器、置信度变量、动态强度预算、手工位置 mask、物理幅度裁剪或推理回退分支。
- 保留已有 video-time query、关系先验和 reference slot；不在本轮同时研究这些部件。

## 4. 张量定义与唯一主公式

只需要已有张量：

- `h`：原生 residual `[B,N,3072]`；
- `T`：原生文本 context；
- `C_r,C_e`：route 与 execute；
- `m`：当前位置读取的全宽物理 value；
- `delta`：物理输出映射后的 residual。

在选定层 l：

\[
m_l=\operatorname{Read}_l(\operatorname{norm3}_l(h_l),C_r,C_e,\text{video-time}),
\]
\[
\delta_l=W^O_l m_l,
\]
\[
\widetilde h_l=h_l+\delta_l,
\]
\[
h_{l+1}=\operatorname{WanBlock}_l(\widetilde h_l,T,\sigma).
\]

`delta_l` 的已知首帧 slice 为0。这里先在原生文本归一化坐标读取物理，再通过独立 O 映射到 residual 空间；**不能把未投影的 C value 直接相加到 residual。**

物理分支的 K/V/O 每个 site 独立，`m_l` 随当前 h_l 变化。C 本身可以在整个 denoising loop 中固定，但执行不是固定的逐层偏置。

## 5. 展开后的完整计算顺序

下式中的六个时间调制量全部来自原生 Wan，不新增任何“物理 gate”：

\[
u_l=\operatorname{norm1}_l(\widetilde h_l)(1+s_l^{sa})+b_l^{sa},
\]
\[
a_l=\widetilde h_l+g_l^{sa}\operatorname{SA}_l(u_l),
\]
\[
t_l=\operatorname{TextAttn}_l(\operatorname{norm3}_l(a_l),T),
\]
\[
b_l=a_l+t_l,
\]
\[
f_l=\operatorname{norm2}_l(b_l)(1+s_l^{ffn})+b_l^{ffn},
\]
\[
h_{l+1}=b_l+g_l^{ffn}\operatorname{FFN}_l(f_l).
\]

这里 `b_l` 作为 residual 中间结果与 `b_l^{ffn}` 作为原生 shift 是不同对象；代码采用 `hidden_text`、`ffn_shift` 避免命名歧义。

关键性质：

1. Q/K/V 都从 `u_l` 产生，因此物理条件可以改变选择关系与被传递的值。
2. 原生 RoPE 仍只施加到真实视频位置，不需要为 C 编造坐标。
3. 当前层原请求 attention 看到的是已经经过物理预写入与 self-attention 的状态，不需要另生成一个语义记忆库。
4. FFN 对同一个联合状态做表达；不会把两个视频最后混成一条。
5. 每个 selected block 的原生 self-attention、text attention、FFN 都只运行一次。

这能增加学习“如何执行物理”的空间，但不保证每个物理消息都会自动被解释正确。有效性来自真实生成训练及后续配对输出，而不是公式的形式。

## 6. 可以直接交给编码 AI 的前向骨架

```python
class P2NativeExecution(nn.Module):
    def forward_block(
        self, block, hidden, content, *, site, context,
        modulation, seq_lens, grid, freqs, context_lens,
        video_times, known_tokens, physics_enabled=True,
    ):
        if site in self.readers and physics_enabled:
            # reader 内只做物理读取，不计算 native text attention。
            result = self.readers[site](
                hidden=hidden,
                route=content.route,
                execute=content.execute,
                groups=content.groups,
                native_norm3=block.norm3,
                native_attention=block.cross_attn,
                video_times=video_times,
                known_tokens=known_tokens,
            )
            # result.write 已经是 [B,N,D]，known slice 为 0。
            hidden = hidden.float() + result.write.float()

        # 只有这一次 native block；不能再手动加 text/ffn。
        return block(
            hidden, modulation, seq_lens, grid,
            freqs, context, context_lens,
        )
```

若原生 backend 需要 positional argument 的不同顺序，写一个局部 adapter 并加等价性测试；不要在多个地方复制微小差异的原生 block 实现。

`physics_enabled=False` 是诊断开关，训练与正常推理始终为 True。关闭后原生 LoRA 仍保留，因此不是官方未训练底座。

## 7. 为什么它有机会提高 PC，也有机会提高 SA

### 7.1 PC 的路径

完整物理 value 不先通过短文本 token 的加权平均；每个 site 的原生注意力既能改关系，也能改值的传播。物理内容参与后续所有原生运算，而不只在 FFN 前留下一个独立 residual。

多个 site 使早、中、晚层都可以重新读取物理信息；不人为规定只有早层管运动、晚层只能保外观。静态反射、局部材质和细节同样可以利用这些通路。

### 7.2 SA 的路径

同一层文本读取基于物理更新后的实际视频状态，并具备 Q/O 的可训练适配。模型有能力学习“物理信息应该落实到哪个请求内容”，而不是事后靠一个额外语言分支重新覆盖视频。

然而它不会自动纠正错误 reasoning，也不能保证图像条件与文本矛盾时选对。训练样本语义准确性是必要组成，不用“native attention 会自己协调”替代数据质量。

### 7.3 强样本的继续提升

已经完成主要现象的样本，还可以在持续性、接触细节、材料响应、光学一致性、细小对象保留等方面改善。训练不能只保留“从失败变成功”的样本；评测按基线强样本分组继续比较 PC、SA 与实际质量。

## 8. 参数重分配，而不是盲目堆参数

R10A/R15B 的审计参数为670,010,368，其中增强227,580,928，物理读291,041,280，物理写94,371,840，原生 LoRA57,016,320。[E06]

每个物理 site 原参数38,541,312，其中 `state_value` 为 D²=9,437,184。P2 删除该矩阵，把原 self/FFN LoRA 从32提高到64，并加文本 Q/O LoRA16。

| 配置 | 增强 | 物理接口 | self/FFN LoRA | 文本 Q/O LoRA | 总训练参数估算 |
|---|---:|---:|---:|---:|---:|
| P2-S3-R64 | 227,580,928 | 291,041,280 | 114,032,640 | 5,898,240 | **638,553,088** |
| P2-S5-R64 | 227,580,928 | 174,624,768 | 114,032,640 | 5,898,240 | **522,136,576** |

这是按已审计模块数推算，真实代码必须重新统计。原生 LoRA 可在部署时合并，不应将“训练参数量”与“新增推理网络参数量”混为一谈。

默认不追加复制 block，原生大 attention 次数不增加。额外开销主要为每 site 对256个物理 token的cross-attention，以及 rank64 相对 rank32 的矩阵增量。

可选的预算竞争配置是 `P2-S5-R128`：每5层、原生rank128，总量约636.17M，接近 S3-R64。它研究“更多注入点 vs 更多原生适配容量”的预算分配，**不是**纯粹频率单因素消融；第一轮不必启动。

## 9. 初始化、学习与常见死梯度错误

### 9.1 正确初始化

- 官方 Wan 与现有训练侧拟合接口初始化，主实验 fresh。
- K/V 从对应原生 base 复制，保持非零。
- C_execute 起点包含现有 native anchor，不是全零。
- 物理 O 零初始化。
- 原生 LoRA A 非零随机、B零。
- 原有增强 route/execute 新投影沿用历史初始化。

不能同时把 value、C_execute、O 都清零。那会让 O 也可能缺少有意义输入。

### 9.2 首步与后续梯度

O=0 时，首次 backward 中物理 reader 上游的生成梯度可为0；O 本身可学习。O 更新后，K/V、C 与增强模块进入生成梯度路径。这是残差零初始化的预期行为，不应靠新增“激活强度变量”修复。

测试必须覆盖首次与数个optimizer steps后的情况。测试中检查非零梯度时，需要有非零误差和有效的未知区域。

### 9.3 不要误用冻结

冻结 native base 仅禁止其权重更新，不能对 native block 使用 `torch.no_grad()`。否则物理预写入的梯度无法穿过 block，方案主路径被切断。

## 10. 训练规格

默认完全使用共同契约的 `joint400` 与现有真实视频目标，不添加教师或新的中间监督。400 是机制初筛，不是在设计阶段承诺达标的预算。

需要将旧父类的分阶段逻辑覆盖为明确的 active parameter policy。特别是文本 Q/O 与 rank64 self/FFN LoRA 都从第1步共同学习；原历史128+272仅作为独立复现实验。

短预算完成后，根据真实视频、40例与独立验证样本决定是否进入完整数据覆盖训练。不能用固定16已经很高的PC替代完整40的SA/Joint。

训练扩展时优先解决 caption—片段失配、物理类别覆盖与完整现象学习，不引入动作约束，也不把所有物理视频过滤成高运动片段。

## 11. 推理规格

沿用同首帧、同原请求、持久化初始噪声、seed42、50步 shifted Euler、shift5、CFG5。所有site按训练配置固定启用。

不在失败案例中自动降低 physical guidance，不对高频做低通，不做逐视频seed选择。新增模型能力必须体现在固定协议的一次生成中。

每5层模型必须用它自己的训练权重和sites。条件缓存只用于加速，不改变输出公式。

## 12. 实现测试清单

除共同测试外，P2特别要求：

### P2-T01：原生起点

物理O与所有LoRA-B为0时，与原生同输入输出一致。FP32建议atol/rtol=1e-5量级；BF16先核对backend差异，不能用过大的容差掩盖重复计算。

### P2-T02：同层Q/K/V响应

使物理O非零，固定h/T/噪声，改变C。对选定层的原生 self-attention q/k/v 输入或输出做hook。未来位置的三者都应有响应。这个测试证明计算位置正确，不证明变化方向物理正确。

### P2-T03：调用次数

统计forward计数：一个selected block内原生 self-attention、cross-attention、FFN各一次。reader不得调用第二次native text attention。

### P2-T04：实际residual空间

对照错误实现 `norm3(h)+delta`，测试必须能区分并拒绝它。内部断言hidden是未经归一化的主residual。

### P2-T05：首帧

直接delta首帧为0；采样输入首帧严格来自允许条件。不要把native attention引起的first hidden变化误报为注入mask失败。

### P2-T06：有效参数

确认原`state_value`不在模型或optimizer中；rank64真实覆盖self QKVO与两层FFN；native text只包Q/O一次。

### P2-T07：梯度穿过冻结运算

native base不更新但delta与C有梯度。使用真实FM目标、数个optimizer updates验证，不能只检查任意输出sum。

### P2-T08：缓存和恢复

缓存开/关、checkpoint开/关、保存恢复后输出一致；正负CFG不共用错误的K/V。

## 13. 最少而有解释力的实验

主候选先只跑P2-S3-R64。额外机制控制按结果优先级启动：

| 控制 | 变化 | 要回答的问题 |
|---|---|---|
| P2-POST | 同reader、无state_value、同rank64，回到原self-attn后的并行写入 | 预写入位置是否有净价值？ |
| P2-R32 | 仅把原生rank64换回32 | 收益来自位置还是更多原生适配？ |
| P2-S5 | 每5层独立训练，其他规则同主版 | 注入频率/成本取舍 |

默认不要展开全部组合。推理开关只检查当前checkpoint依赖，不替代上述匹配训练。

必须和R10A、R15B、B400联合读取、原生LoRA基线比较，但正确标记训练处方、参数与数据是否匹配。

## 14. 结果解读与下一步必须具体

- **PC、SA、Joint与真实质量一起升：** 进入完整数据覆盖与多seed；不因单次40达标立刻宣称SOTA。
- **PC升、SA不升：** 检查文本条件匹配和同例语义遗漏；对照文本Q/O是否有实际贡献。下一项结构竞争是P4的来源条件执行，不是新增“语义保护分数”。
- **SA升、PC降：** 对照P2-POST与P2-R32，判断是否过早改变原生视频值传播；不要用总体SA抵销物理退步。
- **出现色块/横线/雪花：** 首先检查dtype、latent尺度、时间broadcast、native block重复调用与CFG错配；若实现正确仍出现，作为生成质量失败报告，不调低guidance伪装为同方案达标。
- **内容替换没影响，但整体优于baseline：** 可能主要学到额外生成适配。记录这一事实，不能宣称Qwen物理知识已有效执行。

这些分支用于定位而非自动回退。模型发布只能使用经过完整验证的一个固定配置。

## 15. 为什么我把它列为主攻方案

它保留目前最有依据的全宽内容与直接物理value，同时解决一个明确限制：外来残差只有有限的同层生成处理。新方案让现有完整生成块在每3层接收并执行物理信息，增加可学习生成能力而不复制大网络。

我对“这是可实施、可归因、值得优先验证的结构”有较高信心；对“未经训练就一定达到PC3.9、SA3.25、Joint15”的断言没有证据。目标应进入真实验收，而不是以复杂公式代替。

## 16. 编码AI必须交付的结果

完成 `P2NativeExecution`、common reader、rank64与textQO安装、默认与S5配置、上述8项专项测试、真实Wan单batch集成入口、参数清单、可恢复训练/固定协议推理脚本。

禁止提交复制block、双前向差分、额外教师、动作约束、可学习强度预算或读取R16代码的实现。未实际测试的分数全部为null。

来源索引：[E01—E08、E11]见`05_EVIDENCE_AND_DECISIONS.md`（文件：`05_EVIDENCE_AND_DECISIONS.md`）。

## 17. 首个实现里要看到的具体差异

编码完成后的审查不应只展示模块名，应打印/记录以下真实事实：

```text
P2-S3-R64:
  physics_sites = [2,5,8,11,14,17,20,23,26,29]
  native_block_forward_calls = 30 / velocity forward
  extra_copied_video_blocks = 0
  physics_reader_calls = 10 / velocity forward
  state_value_parameters = 0
  self_ffn_lora_rank = 64
  text_qo_lora_rank = 16
  physics_query_uses = native_text_q_frozen_base
  physical_values = original_C_execute
```

同一reader的`message`是随位置/层/去噪状态变化的张量，不是手工物理变量。第一次attention的Q/K/V会因为`hidden += write`自然响应；其变化方向需要训练，不能手动对Q/K/V分别加三个不明系数。

内存上保留原生gradient checkpointing，物理attention按query分块即可精确计算；每个chunk仍对全部256个C槽与reference做softmax，不能把key维拆开后分别softmax再相加。不得通过量化/低秩化C或减少物理内容维度来掩盖实现开销。

## 18. 对有效性预期的最终判断

本版的信心来自可定位的容量错配、现有物理直接值收益以及清晰的同预算控制，而不是“预写入天然更物理”。我愿意把它作为主攻候选，但不会把PC3.90、SA3.25、Joint15写成在400更新下已被理论支持的预测。

最终采用条件仍是三个数值目标、强样本继续改善、真实画质和跨基准共同成立。如果只有PC上涨，或者只是原生rank增大带来通用收益，就报告对应结论并以P4/读取设计继续竞争；不把较高的单项PC包装成已完成使命。


---

<!-- 原文件：03_P4_COOPERATIVE_EXPERTS.md -->

# 方案四终版：物理—语义上下文协同专家，无教师、无干预学习

设计 ID：`P4-SCE-v2`  
默认配置：`P4-S3-H512`，每3层注入、两个条件专家各512内部宽度。  
备选频率：`P4-S5-H512`，每5层独立训练。  
状态：可执行设计，未获得新训练结果。  
共同实现约定：`00_IMPLEMENTATION_CONTRACT.md`（文件：`00_IMPLEMENTATION_CONTRACT.md`）。

## 1. 这一版的主张

不新增V-JEPA、视觉教师或干预数据，不额外训练“因果分数”。

本版采用两个有明确输入来源的条件专家：

- **Physics Expert**：输入当前原生生成状态与实际读取的全宽物理消息；
- **Semantic–Context Expert**：输入同一个生成状态与原生文本attention实际读出的请求消息。

它们在原生FFN内部共同执行，不分别生成两条视频，也不通过softmax争夺一个固定输出配额。

“Context”指语义、场景与外观组织。没有干预数据或因果验证时，不将其命名为已经具备因果识别能力的Causal Expert。

## 2. 为什么不能只把两条LoRA改名为两个专家

如果两个普通LoRA都读同一个u，然后直接相加：

\[
B_1A_1u+B_2A_2u
=[B_1\ B_2]\begin{bmatrix}A_1\\A_2\end{bmatrix}u,
\]

它们可以合并成更高rank的单个LoRA，名称并没有带来真实分工。

本版不采用这种结构。两个专家分别受**不同实际条件消息**控制，并且具有非线性的状态—条件交互。因此它不是“两套同输入LoRA”的包装。

这也不意味着训练后一定出现可解释的严格物理/语义分工。结构只提供来源区别与协作能力；是否形成实际收益，需要相同预算的单专家对照和生成结果来回答。

## 3. 历史反例如何影响这次设计

R11B新增独立请求memory，固定40达到PC3.950但SA2.600、Joint6。关闭请求memory在局部控制中反而改善SA和Joint。因此本版不另造一套语言记忆向视频叠加。[E04,E07]

R15B通过请求token绑定物理value，恢复部分Joint但仍有SA损失。其state_value交互具有有利和不利效应，不能只按单项均值盲删。[E05]

R13A/R13B的调制尝试说明，少数条件矩阵影响了原生运算，并不等于已经学会生成正确现象。[E03]

本版保留有价值的full-width物理直接写入；用条件执行专家替换原state_value矩阵；上下文消息直接来自现有native text attention，而非独立请求生成通路。所有分支学习真实视频，不依赖教师热图或额外专家标签。

## 4. 一项关键的预算恒等式

旧R10A每个site有一张：

\[
W_{state}\in\mathbb R^{D\times D},\qquad D=3072,
\]

即9,437,184参数。

本版每个专家有三张矩阵：状态D→r、条件D→r、输出r→D。两个专家共：

\[
2\times3Dr=6Dr.
\]

取r=D/6=512，恰好：

\[
6D\cdot512=D^2=9,437,184.
\]

**因此两个专家替代的是旧state_value的预算，不是额外叠一个大网络。** 这也是选择512的明确依据，不是一个凭空的“物理状态维度”。

原始C、物理attention value和直接写入全部保持3072全宽。512只是两个额外条件执行算子的内部宽度，不能误实现为把整个物理内容压缩到512。

## 5. 完整的层内计算顺序

默认site为 `[2,5,8,11,14,17,20,23,26,29]`。

### 5.1 原生self-attention

先执行该层正常self-attention和时间调制，得到`h_sa`。这一步与P2不同：P4的主要改动在该层text/FFN部分，不先把物理写入self-attention输入。

### 5.2 全宽物理直接读取和写入

\[
m=\operatorname{Read}(\operatorname{norm3}(h_{sa}),C_r,C_e),
\]
\[
h_p=h_{sa}+W_o^p m.
\]

保留原始C_execute作为value，**不使用旧state_value乘法**。物理O是full-width。已知首帧的新增物理更新为0。

### 5.3 原生文本专家读取

\[
t=\operatorname{TextAttn}(\operatorname{norm3}(h_p),T),
\]
\[
h_{pt}=h_p+t.
\]

这是一次原生文本attention，带Q/O LoRA16。T仍为原请求的原生context。它看到的是物理写入后的当前视频状态，不再独立维护另一份请求memory。

不能计算t后再额外调用完整native block，因为那会重复self/text/FFN。

### 5.4 准备共同原生FFN输入

\[
u=\operatorname{norm2}(h_{pt})(1+s^{ffn})+b^{ffn}.
\]

两个专家都作用于这个已经包含视觉、文本和物理作用的状态；它们的直接条件来源不同。

### 5.5 两个条件专家

对任一来源消息s，定义：

\[
E(u,s)=O\left[\operatorname{GELU}(Au)\odot B\operatorname{LN}_0(s)\right],
\]

其中：

- A：D→512，当前视频状态投影；
- B：D→512，该来源消息投影；
- O：512→D，执行更新投影；
- `LN_0`：无affine的标准LayerNorm，仅规范两个来源的输入坐标尺度；
- 所有线性层bias=False；O零初始化。

物理专家与上下文专家分别为：

\[
e_p=E_p(u,m),\qquad e_t=E_t(u,t).
\]

最终：

\[
h_{out}=h_{pt}+g^{ffn}\left(\operatorname{FFN}_{native}(u)+e_p+e_t\right).
\]

`g_ffn`是原生时间调制，不新增路由权重。物理专家在known slice的输出设为0；上下文专家按原生生成路径正常运行。

## 6. 这几个中间张量都有直接来源

| 张量 | 来源与含义 | 是否新定义的抽象状态 |
|---|---|---|
| h_sa | 原生self-attention后的video hidden | 否 |
| m | 视频从C读取的attention value | 否 |
| t | 原生文本attention实际输出 | 否 |
| u | 原生FFN本来就使用的输入 | 否 |
| e_p | 物理条件执行增量 | 普通神经层输出 |
| e_t | 请求条件执行增量 | 普通神经层输出 |

没有“物理置信度”“语义预算”“因果强度”“冲突量”或其他缺少监督含义的变量。

乘性形式不是让两个专家互相压制；每个专家内部用来源消息调制**自己的**状态特征。没有`PC_weight + SA_weight = 1`，也没有限制专家输出只能为正或只能缩小物理作用。

## 7. 为什么这种协同不只是两条residual相加

协同发生在三个具体位置：

1. 物理直接写入后，原生文本attention在实际物理更新状态上读取原请求。
2. 两个条件专家共用已经吸收物理与原请求的u，但分别读取m与t，提供不同来源的执行能力。
3. 原生FFN持续处理完整的联合状态；两个专家是来源条件化增量，而不是替代主干的两个小模型。

这样可以学习：某条物理信息如何影响当前视觉状态，以及原请求涉及的对象、数量、材质或场景关系如何在这种物理变化中继续被表达。

这些是可学习能力，不是现成的逻辑保证。尤其不能宣称上下文专家必然修复所有SA错误。其是否值得保留由对应训练控制决定。

## 8. 为什么仍保留直接物理写入

只有两个512宽度的条件专家可能让丰富C被迫通过较小执行通路。保留3072→3072的物理O，使完整物理消息始终有直接进入视频状态的路径。

两个专家负责状态相关、来源相关的表达增强，不成为物理信息唯一瓶颈。这既继承R10A的直接value收益，也避免把“协同”实现成必须先由语义通道批准才能使用物理内容。

## 9. 条件专家可直接编码的定义

```python
class SourceConditionedExpert(nn.Module):
    def __init__(self, width=3072, inner=512, eps=1e-6):
        super().__init__()
        self.width = width
        self.eps = eps
        self.state = nn.Linear(width, inner, bias=False)
        self.source = nn.Linear(width, inner, bias=False)
        self.output = nn.Linear(inner, width, bias=False)
        nn.init.kaiming_uniform_(self.state.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.source.weight, a=math.sqrt(5))
        nn.init.zeros_(self.output.weight)

    def forward(self, u, source):
        # 输入坐标规范化，无学习门控或手工强度因子。
        source_n = F.layer_norm(
            source.float(), (self.width,), eps=self.eps
        )
        state_features = self.state(u)
        source_features = self.source(source_n)
        joint = F.gelu(state_features.float(), approximate='tanh')
        joint = joint * source_features.float()
        # 真正项目在统一autocast/dtype helper下调用，不能破坏BF16/FP32约定。
        return self.output(joint)
```

注意：`source_n`的float32与投影权重的dtype应通过统一autocast管理；不在每层反复把整个模型转dtype。CPU合成测试使用float32，真实Wan集成必须测试BF16路径。

层内骨架：

```python
h_sa = native_self_step(block, hidden, modulation, geometry)
read = physics_reader(h_sa, content, block, video_times, known_tokens)
h_p = h_sa.float() + read.write.float()
text_update = native_text_step(block, h_p, context, context_lens)
h_pt = h_p + text_update
u, native_ffn_gate = native_ffn_input(block, h_pt, modulation)

physics_extra = physics_expert(u, read.message)
physics_extra = mask_known(physics_extra, known_tokens)
context_extra = context_expert(u, text_update)

ffn_update = block.ffn(u).float()
ffn_update = ffn_update + physics_extra.float() + context_extra.float()
h_out = h_pt + native_ffn_gate * ffn_update
```

非site层调用一次完整原生block。site层使用上述拆分，不能再额外调用完整block。

## 10. 为什么要用raw message而不是zero-init write作为专家条件

物理O初始化为0。如果把它的输出`write`作为物理专家的唯一条件，而专家O也初始化为0，会形成更深的串联零路径，增加不必要的启动延迟。

本版物理专家读的是原始attention value `read.message`。它在初始化时来自非零anchor/C_execute与非零V，所以专家output投影从第一步就有可学习输入。

这没有绕过物理来源：message仍然由视频状态读取本样本物理C。若C_execute为0、V无bias，则message为0。

## 11. 零来源与梯度性质

由于source投影无bias、LayerNorm无affine：

\[
E(u,0)=0.
\]

因此即使当前视频状态u非零，物理条件专家也不能在source=0时仅靠一个独立bias产生“物理”残差。

但通用非零source仍可能被模型当作共享条件；这个性质不是“已经证明场景物理内容有效”。正确C必须在生成中优于匹配通用/错误C并且优于没有该结构的基线。

O为0时，首步A/B梯度可以为0，O本身可以学习。O更新后A/B获得梯度。两个专家输出O与物理直接O都只零初始化，不同时把源投影清零。

## 12. 参数与计算预算

旧state_value被替换，不保留重复旧矩阵。所有其他R10A模块及原生rank32 LoRA保留，新增文本Q/O rank16。

| 配置 | 估算训练参数 |
|---|---:|
| P4-S3-H512 | **675,908,608** |
| P4-S5-H512 | **521,743,360** |
| 相对R10A同频率的增量 | 仅文本Q/O约5,898,240 |

每site两个专家三组矩阵的总乘加规模也与旧D×D状态投影处于同一量级；增加了两个非线性与归一化，但不增加大视频attention次数。

可训练参数接近旧版不是等同于精确相同总FLOPs或延迟。应实际测tokens/sec、峰值显存、单视频延迟和全部推理模型成本。

## 13. 无干预学习时，如何训练分工

本轮不靠干预对、教师或手工专家标签。训练信号只有：

1. 真实视频的既有生成损失；
2. 原有raw C_route词对齐；
3. 明确的输入来源结构。

Physics Expert只能通过m这个额外来源获得直接物理条件；Context Expert只能通过t这个额外来源获得直接请求消息。两者都看到共享生成状态，因此不是严格的信息隔离，不能夸大为因果解耦。

原生Q/O LoRA、专家和Physics模块共同从第1步学习。source-conditioned专家使用公共学习率分组：state/source投影2e-5，output投影3e-5。

只有同一份联合生成损失，没有分别对专家加“PC loss”“SA loss”的假标签，也不按自动评分器动态重分配训练权重。

## 14. 训练数据如何使协作有意义

模型需要见到语义正确且物理现象完整的自然视频。重点是对象、材质、位置关系、时间变化和实际片段一致，而不是把文字描述扩写得更长。

高质量静态平衡、光学、透明材质、缓慢热变化及物体持续性样本都应保留。没有显著动作不代表没有学习价值。

初筛与历史配对保持同缓存；入选后再做完整数据覆盖和训练集质量修订。物理强但语义不符的视频不应当作“高PC所以优质”目标；同样，只有画面精美却缺少要求现象的视频也不是理想训练目标。

这是常规数据质量选择，不是反事实/干预学习，也不需要新的视觉教师。

## 15. 为什么这版有机会提高强样本，而不只是补弱样本

对一个已经大体正确的场景，物理消息可能包含尚未体现在结果中的局部材料响应、光学细节或持续性信息。物理条件专家允许这些信息依据当前生成状态以不同方式执行；上下文专家允许相关对象、属性和场景关系继续得到表达。

因为没有两个专家输出权重相加为1的竞争约束，它们可以同时产生有效增量。目标不是“PC越强，语义越被压低”，而是学习同一结果中两种条件相互支持的表示。

这需要在高分基线样本上用逐例变化和完整视频验证；不能只看Joint阈值是否过线。

## 16. 初始化与数值要求

- `physics output O`与两个专家的output矩阵为0；native LoRA-B为0。
- K/V、专家state/source矩阵非零。
- 模块无新增bias捷径；source LN无affine。
- native FFN保留原GELU、原时间调制、原rank32 LoRA。
- 两个source条件的标准化不会改变全宽直接物理write；不要再对直接物理write强行做norm、clamp或低通。
- 专家交互乘积及residual求和使用FP32，原生大矩阵遵循BF16/autocast配置。
- 使用独立RNG上下文初始化新专家，以免无意改变旧增强/主干的随机初始化对照。

## 17. 专项测试清单

| 编号 | 测试 |
|---|---|
| P4-T01 | 两专家参数量各3Dr，合计精确等于原state_value的D² |
| P4-T02 | state_value已移除，未与新专家重复保留 |
| P4-T03 | 所有新output为0时与官方起点等价 |
| P4-T04 | source=0时该专家输出严格为0，即使u非零 |
| P4-T05 | output激活后改变m/t分别改变对应专家输出 |
| P4-T06 | 首step expert output有梯度；若source/state无梯度，后续step应恢复 |
| P4-T07 | text attention只执行一次，且query来自h_p而非错误的pre-physics状态 |
| P4-T08 | 两专家共用同一个native FFN输入u，未重复LayerNorm/time modulation |
| P4-T09 | direct physics与physics expert在known slice为0 |
| P4-T10 | no_c关闭直接物理和物理专家，保留上下文专家与native LoRA |
| P4-T11 | 原生FFN只执行一次，非site层原生路径未改 |
| P4-T12 | optimizer覆盖所有专家参数，没有共享引用或重复参数组 |

`no_c`诊断结果不是未经训练的底座，因为上下文专家与LoRA仍然训练过。推理禁用某个专家也不能替代结构价值的匹配训练比较。

## 18. 必须有一个等参数单专家对照

为了区分“协同结构”与“只加容量”，设计一个单来源混合专家：

\[
E_{mix}(u,m,t)=O\left[\operatorname{GELU}(Au)\odot
(B_m\operatorname{LN}_0(m)+B_t\operatorname{LN}_0(t))\right].
\]

该单专家有4Dr_mix参数。取r_mix=768，则：

\[
4D\cdot768=6D\cdot512=D^2.
\]

与两个专家严格匹配此部分参数预算。若双专家没有胜过这个控制，就不能声称“专家分工”带来收益；可以选更简单或更快的有效版本。

首轮不必启动所有消融。优先顺序：主P4、匹配R10A+textQO、等参数单混合专家；需要定位时再独立训练删除一个专家的控制。

## 19. 默认配置

```yaml
variant: p4_source_conditioned_experts
condition_every: 3
physics:
  content: r10a_full_width
  reader: direct_original_values
  native_query: frozen_text_q_base
  state_value: false
  direct_write: true
  write_position: post_self_pre_text
experts:
  count: 2
  inner_width: 512
  physics_source: raw_physics_attention_message
  context_source: native_text_attention_output
  operation: gelu_state_times_linear_source
  source_norm: layernorm_no_affine
  routing: none
  output_bias: false
  output_zero_init: true
native_lora:
  self_ffn_rank: 32
  text_qo_rank: 16
training:
  preset: joint400
  teacher: none
  intervention_learning: false
```

正式配置以`configs/p4_s3.yaml`为准；这段说明不能被编码AI扩展成额外的`confidence`、`balance_gate`或`causal_state`。

## 20. 结果解读

- PC、SA和画质共同提升：继续做强样本与跨基准评估。
- 与等参数单专家持平：不夸大“协同”，选择更简单的有效结构。
- 只有PC提升：核查上下文专家是否真正提高原请求覆盖，而不是增加通用纹理；数据语义质量与上下文条件有效性是下一步，不新增第三个专家。
- 只有SA提升而PC下降：检查直接物理通路、原始value保留和state_value替代影响；不以SA抵销物理退步。
- 画面崩坏：先核查实现，再按真实失败处理。不能通过临时禁用专家、降低CFG或挑seed报告同一主模型达标。

## 21. 最终交付

交付两个`SourceConditionedExpert`、拆分后的native text/ffn操作、full-width直接读取、精确参数预算控制、专项测试、同预算单专家配置、恢复与固定协议推理入口。

必须只使用本文件规定的两个条件专家。没有干预数据，没有视觉教师，没有动作约束，没有新增因果标签。目标仍为PC≥3.85/3.90、SA≥3.25、Joint≥15/40，并在广泛评测上成立；这些是验收目标，不是预填结果。

来源索引：[E02—E07、E11]见`05_EVIDENCE_AND_DECISIONS.md`（文件：`05_EVIDENCE_AND_DECISIONS.md`）。


---

<!-- 原文件：04_EVALUATION_AND_EXECUTION.md -->

# 评测、实验顺序与达标判据

适用：P1-RKE-v2、P2-DNE-v2、P4-SCE-v2。  
这些配置都是待测候选；本文件不包含任何新候选的已完成质量结果。

## 1. 必须同时追求的三类改善

本轮不把目标缩成“只增加Joint”。完整目标是：

1. 未成功样本增加物理与语义联合成功；
2. 已经较强样本继续提高PC、SA或实际细节质量；
3. 真实成像与时间一致性保持高质量，不出现色块、雪花、横线等生成崩坏。

若基线PC已为5，评分没有更高档位，仍可用盲评的成对视频比较判断细节是否更真实；不能自行把评分扩成6分或用更大运动量替代物理进步。

## 2. 已核实的历史参照

| 模型/变体 | 固定40 SA均值 | PC均值 | Joint |
|---|---:|---:|---:|
| Native LoRA400 | 3.075 | 3.450 | 10/40 |
| R1-400 | 2.950 | 3.625 | 10/40 |
| R10A-400 | 2.825 | 3.750 | 7/40 |
| R13B-400 | 2.875 | 3.525 | 7/40 |
| R15B-400 | 2.925 | 3.700 | 10/40 |
| R15B-400 + B400联合读取 | 2.900 | 3.775 | 10/40 |
| R11B-400（不同数据修订） | 2.600 | 3.950 | 6/40 |

来源E01—E08。最后一行用于说明“单项PC很高不代表共同能力很强”，不能当作严格结构匹配基线。

本次未定位到名为“R13B+I”的可核验完整结果，不将它擅自改名为R15B+I，也不填入推测分数。

B400联合读取相对你提出的40例目标，PC3.775→3.900对应总评分增加5点；SA2.900→3.250对应总评分增加14点；Joint10→15对应净增5例。这只是差距换算，不表示靠五个特定样本优化就足够。

## 3. 首轮实验不展开全组合

### 3.1 必须固定的共同条件

- 源代码与base权重版本；
- 同一训练数据、样本顺序、sigma、噪声与条件dropout规则；
- `joint400`：400更新、累积4，全部选定模块共同训练；
- loss、optimizer、精度；
- 所有候选先使用每3层注入；
- 固定首帧、原请求、初始噪声和采样协议。

### 3.2 第一轮运行表

| ID | 结构 | 用途 |
|---|---|---|
| C0 | R10A原结构 + joint400 | 区分训练处方变化 |
| C1 | C0 + native text Q/O LoRA16 | 三个候选共享的语义适配基线 |
| P1 | 请求条件化物理key + C1其余结构 | 读取/增强路线 |
| P2 | pre-self物理写入、无state_value、native rank64 | 原生执行主线 |
| P4 | 两个source-conditioned专家替代state_value | 语义—物理协同路线 |

C0/C1不是新研究模块，而是避免重复误归因的必要对照。历史视频继续保留对比，不重复产生逐例最优拼接结果。

### 3.3 第二轮只解决最重要的剩余问题

P2优先比较同rank、同reader的POST位置控制；P4优先比较等参数单混合专家；P1优先比较无新key enhancer的C1。每5层配置仅在入选候选上独立训练。

不同时扫10种层号、5种rank、4种loss、多个guidance。先确认机制真实贡献，再扩大容量或改变频率。

## 4. 固定40例的硬目标与完整记录

```yaml
videophy40:
  pc_mean_min: 3.85
  pc_mean_stretch: 3.90
  sa_mean_min: 3.25
  joint_pass_count_min: 15
  catastrophic_visual_failure_count_max: 0
  best_of_n: 1
```

SA、PC为1—5评分；各自pass为>=4；Joint为同一视频二者都>=4。

程序必须验证：

```python
assert joint_count <= min(sa_pass_count, pc_pass_count)
assert len(set(case_ids)) == 40
assert all(1 <= s <= 5 for s in valid_scores)
```

评分解析失败、视频缺失与生成崩坏单独记录，不得静默丢掉该条或把未知评分置为“平均值”。覆盖不完整时不标记完整40达标。

均值与通过率同时报告。不同分布可以有相同PC均值，但通过率完全不同，不能将均值除以5冒充pass rate。

## 5. 逐样本输出字段

```json
{
  "case_id": "VP001",
  "model_id": "P2-S3-R64",
  "checkpoint_hash": "...",
  "seed": 42,
  "input_hash": "...",
  "initial_noise_hash": "...",
  "sa": null,
  "pc": null,
  "sa_pass": null,
  "pc_pass": null,
  "joint_pass": null,
  "video_path": "...",
  "visual_review": {
    "catastrophic_artifact": null,
    "color_blocks": null,
    "snow_noise": null,
    "horizontal_lines": null,
    "object_persistence": null,
    "requested_phenomenon": null,
    "material_and_optics": null,
    "review_note": ""
  }
}
```

未评测用null，不得使用目标值代填。上述review字段是人工/评测记录，不进入模型，也不成为新训练变量。

## 6. 弱样本、强样本与总体一起看

用预先选定基线定义组：

- baseline PC>=4的强物理样本；
- baseline SA>=4的强语义样本；
- baseline Joint成功样本；
- 其他样本。

每组报告PC/SA变化、胜/平/负、Joint新增/丢失，以及真实视频盲评结果。不能只显示新增Joint而隐去原有强样本退化；也不能只抬高强样本PC而长期不解决语义遗漏。

固定40历史上已被反复用于设计，应标为开发/回归集合。新增未用于调参的case做独立验证。扩展40+20的60例须分别报告原40、extra20、full60，不能拿其中最好的子集代替主目标。

## 7. 视觉质量检查不是“动作完成率”

完整视频检查的内容为：请求中现象是否呈现、对象/材质是否持续、空间关系是否正确、变化是否合理、静态关系是否稳定、画面细节是否自然。

对静态平衡或稳定折射，保持稳定是合理的；对需要变化的场景，无故静止是失败。不得统一使用“motion低=失败”或“变化多=物理好”。

固定40应逐条观看完整视频；大规模集至少全部自动检查格式/黑帧/异常编码并进行覆盖所有领域的人工盲审，评分异常与疑似伪影样本追加复核。不要只看三张抽帧拼图判断全部后程质量。

发布门槛要求所报告评测集合的明显崩坏观测数为0。有限样本0次不等于对所有未来输入数学保证0风险，但不降低该验收要求。

## 8. 大规模VideoPhy2与ProPhy

ProPhy v2表1报告的两条增强模型结果为：[E12]

| 模型 | ALL PC pass | ALL SA pass | ALL Joint | HARD PC | HARD SA | HARD Joint |
|---|---:|---:|---:|---:|---:|---:|
| Wan2.1-1.3B + ProPhy | 65.0% | 32.0% | 26.5% | 48.9% | 12.2% | 7.2% |
| CogVideoX-5B + ProPhy | 72.5% | 32.2% | 26.7% | 52.8% | 11.7% | 6.1% |

论文使用600个提示词，HARD为180；VBench质量维度使用相同600提示词。CogVideoX+ProPhy报告该套提示词质量分81.0，不是任意VBench集合上的可直接比较总分。[E12]

建议研究目标（非预测）：ALL PC pass>=76%、SA pass>=38%、Joint>=33%；HARD PC>=58%、SA>=20%、Joint>=14%。达标前先要求在完全对齐条件下超过参考，不把目标本身写进结果。

重要：当前项目主线是I2V，ProPhy主报告是T2V。已有项目600重放明确使用额外首帧，只是提示词协议的I2V适配。[E08b] 应分成：

- **I2V能力线**：同一首帧、同一原文、同种子，和同规模I2V基线比较；
- **T2V论文对标线**：真正无首帧模式，单独训练/验证条件入口，再与ProPhy同任务比较。

论文对judge的可复现细节不能从表格数字推断；应记录本方AutoEval版本和采样方式，能获得作者配置时重跑。不得仅因提示词一致就声称完全复现。

额外复核备注：论文表1的HunyuanVideo行出现SA19.2而Joint24.7，与Joint不大于SA的集合定义不符。此处不使用该行作定量目标，也不替作者猜测修正值。上面两条ProPhy行本身满足集合约束。最终比较优先使用可复核逐样本结果。

## 9. WMB

官方WMB覆盖350样本、7领域，分别测instruction following、physical adherence、common sense。[E15]

先用已有WMB70作为历史回归，再对入选模型运行完整350。报告各领域和各维度，不只报告总分。工业、机器人、自动驾驶等领域不能被自然场景的改善掩盖。

已有B400联合读取在WMB70为8.6000，R1为8.5714，且指令维度有下降。这是小幅点估计收益，不是已经全面超过基线。[E08]

图生视频用官方首帧和instruction；文生视频按官方规定组合首帧描述与instruction。不可混用输入规则。

## 10. Physics-IQ

固定使用一个完整的官方版本；Original与Verified分别报告，不能交叉使用测试未来视频来优化。区分I2V、V2V、原prompt/best-practice prompt、单样本/BoN。[E14]

已有项目20场景×3视角可用于历史回归，但不能冒充完整排行榜。按场景成组报告与bootstrap，三视角不是三个完全独立的物理事件。

本次核验的官方Verified资料包含Wan2.2-5B 27.7±0.9、CogVideoX-5B 31.8±1.5的记录；它们是校准参照，不被称为全部同规模模型的最终SOTA。完整协议超过35%可作为阶段性研究目标，不是本包候选的预测值。[E14]

正式SOTA声明遵循官方四次运行与标准差要求。不得使用BoN评分提升冒充单次生成改善。

光学、热学、磁学、流体、固体力学分开查看。已有B400联合读取Physics-IQ总分略升但光学明显退步，因此不能只盯整体均值。[E08]

## 11. VBench和实际成像

同时保留：

- 标准T2V或I2V套件，按模型任务选择，不混合总分；
- 与ProPhy对齐的600物理提示词质量评估，明确这是另一套prompt分布。

重点报告subject/background consistency、temporal flickering、imaging/aesthetic quality、语义维度与motion smoothness等原始维度。[E16]

Dynamic Degree用于诊断，不作为所有物理现象的统一最大化目标。不得通过冻结画面取得更高一致性，或通过增加随机运动取得更高动态分数。

## 12. 参数、算力与SOTA比较的账本

分别报告：

1. 生成主干参数；
2. 新增且不能合并的推理模块；
3. LoRA等训练参数；
4. 在线/离线Qwen、T5、VAE等其他模型；
5. 每视频前处理、denoising次数、延迟、显存与候选数量。

本轮系统仍使用Qwen3-VL-8B作为既有输入条件编码器，不能因为冻结就把它从系统成本里抹去。可以另列“5B生成主干”类别，但必须披露额外条件模型和模块。

P2的估算训练参数比R10A略少，P4接近，P1略多；这些不是精确相同总算力。对同参数量竞争的结论应采用相同输入任务和明确预算，而不是只比较名称中带5B的模型。

## 13. 真实训练迭代顺序

### 阶段A：实现正确性

本包合成CPU检查 → 新模块的实际CPU测试 → 单个真实Wan batch梯度与保存恢复 → 小规模固定输入生成。此阶段不声明质量达标。

### 阶段B：共同400更新

C0/C1与三个候选固定每3层。先看完整40与实际视频；固定16仅用于定位，不用于替代整体成绩。样本/seed不按候选分别挑。

### 阶段C：机制与频率

围绕胜出的P2/P4/P1做最少必要控制，再比较每5层。不能在发现失败后不停加不明变量，保持每个新增机制有明确的匹配训练结构消融；这里不构造干预学习数据。

### 阶段D：能力训练

选择有真实联合进展的结构，扩到完整现有数据覆盖，之后使用经授权、去污染、质量合格的自然视频扩展。记录训练总曝光和域分布，继续验证强样本质量。

### 阶段E：正式跨基准

冻结结构、checkpoint选择规则与推理设置。运行未参与设计的样本、多seed、完整WMB/Physics-IQ/VBench与VideoPhy2协议；不在结果不佳时换一套prompt改写方式而仍称同模型。

## 14. 效果不佳时如何修订，而不是降低目标

- 读取错位：优先P1与直接读取的生成对照，不增加语义置信度。
- 物理可读但执行效果弱：比较P2预写入/后写入、rank32/64，不用调大CFG代替模型学习。
- 语义覆盖持续不足：比较P4的上下文条件专家与等参数单专家，检查训练文本是否精确。
- 大量物理类别缺失：补自然视频覆盖，不要求每类都转成动作标签。
- 泛化退步：核查来源去重、视频片段质量与训练曝光，而不是测试集上best-of-N。

目标不变，但不能预先承诺每次修订都有效。失败候选保留其完整负结果，不以“还有理论潜力”无限延长训练。

## 15. 报告输出格式

每次报告包含manifest、config、代码版本、参数/计算预算、训练进度、逐例评分、聚合评分、种子、完整视频位置、质量复核、关键对照、结论与未完成项。

```json
{
  "design_id": "P2-DNE-v2",
  "implemented": false,
  "real_wan_integration_passed": false,
  "training_completed": false,
  "videophy40": null,
  "cross_benchmark": null,
  "quality_goal_achieved": false,
  "note": "Design specification only; scores must come from actual evaluation."
}
```

本文件给出目标和验证路径，不制造未做实验的信心数字。

## 16. 比较统计与checkpoint选择

固定40的seed42保留作历史可复现参照；入选候选至少使用另外两个预先固定的seed复测，不能仅挑其中最佳seed作为最终分数。正式Physics-IQ SOTA声明遵循其独立的四次运行要求。

均值差与Joint差使用同case配对统计。多seed报告每个seed和总体结果；bootstrap以case为cluster，先重采样case再保留该case的各seed，Physics-IQ以场景为cluster并保留其视角，避免伪独立。可以预先固定10000次重采样与随机种子作为分析协议，但不能据小样本窄区间保证泛化。

checkpoint选择使用独立开发集合上的PC、SA、画质联合规则，最终大规模测试集冻结后只评一次。历史40已是开发集，其PC3.9/SA3.25/Joint15即使满足，也不是未见数据泛化的充分证明。

为了继续提高强样本，允许在“均值及通过率都相近”的模型间使用预先定义的盲评细节胜率做选择；不引入权重不明的PC×SA×画质乘积分数。成对盲评随机交换左右，保留平局和原始备注，审阅者不看候选名。

## 17. 可直接使用的未运行结果模板

`results_template.json`已随包提供。真实训练后由脚本填入，不应手工把设计目标变成结果。合成检查写到`checks/reference_math_checks_results.json`，与实际模型报告分开保存。


---

<!-- 原文件：05_EVIDENCE_AND_DECISIONS.md -->

# 来源、事实边界与设计取舍记录

审阅日期：2026-11-10。  
项目内容固定到commit `1defd6f4b41e42302bc3e4b9b2bf938abbbc686c`。  
本轮只使用R16之前的设计与结果；没有读取或借用R16候选。

## 1. 项目事实来源

### E01 — 已完成40例汇总

[completed_full40_summary.csv](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/reports/completed_full40_summary.csv)

支持Native400、R1、R10A等结果；不同训练步数不可混成同预算结果。R1续训到1000的PC回落，不支持“只要更久就更好”。

### E02 — R10A设计与代码

[DESIGN.zh.txt](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/execution_value_r10/DESIGN.zh.txt)

[physical_content/model.py](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/execution_value_r10/physical_content/model.py)

支持两轮全宽增强、route/execute分工、原生Q复用、独立物理K/V/O、state_value乘法、时间与reference实现。代码事实不等于每一个子模块都已有独立有效性证据。

### E03 — R13B

[设计](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/interaction_query_r13b/DESIGN.zh.txt)

[正式比较结果](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/interaction_query_r13b/inference_outputs/physical_content_67610/step0400_original_fixed40_controls16_seed42/comparison_report.json)

支持“物理query调制+直接写入”的设计及40/60结果。不能把它当作P2所有Q/K/V共同重算的现成成功证据。

### E04 — R11A原生语义适配及历史成绩

[request_adapt_r11a/DESIGN.zh.txt](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/request_adapt_r11a/DESIGN.zh.txt)

[R11A_to_R13B_scores_20261105.csv](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/r11_r12/reports/R11A_to_R13B_scores_20261105.csv)

已尝试文本Q/K/V/O rank32、文本先物理后的顺序。新方案不应把文本LoRA本身包装成从未验证的突破。

### E05 — R15B与机制复盘

[设计](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/request_bound_fusion_r15b/DESIGN.zh.txt)

[实现](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/request_bound_fusion_r15b/physical_content/model.py)

[完整机制复盘](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/local_state_operator_r15c/review/r15b_mechanism_completed.json)

支持请求value聚合、C公共成分变化、state_value的混合效应及完整40成绩。不能从token cosine推出生成效用。

### E06 — 训练参数与400步审计

[completed400_training_audit.json](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/request_bound_fusion_r15b/reports/completed400_training_audit.json)

支持670,010,368参数分组、400更新/1600次样本使用。新方案参数为从这些模块与矩阵形状推导的估计，尚非真实部署模型统计。

### E07 — R11B反例与R12A记录

[reciprocal_fusion_r12a/DESIGN.zh.txt](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/reciprocal_fusion_r12a/DESIGN.zh.txt)

支持独立请求memory副作用、移除后局部收益，以及“文本先更新+C读取视频状态”已经尝试过的事实。本包没有把这些旧组合改名当成新方案。

### E08 — B400联合读取与跨基准

[b400_joint_read_eval/reports/results.zh.md](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/b400_joint_read_eval/reports/results.zh.md)

支持40/60、WMB70、Physics-IQ60视角分数以及仓库记录的视频复盘。本次没有重新播放全部原始视频，因此本包关于这些旧视频的具体描述来自该复盘，不冒充新的独立视觉判断。

### E08b — ProPhy提示词重放的输入任务限制

[prophy_videophy2_600/README.md](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/prophy_videophy2_600/README.md)

明确I2V首帧适配与ProPhy T2V主报告不同，也不保证judge完全一致。

### E09 — 缓存训练视频元数据

[cached_temporal_metadata_audit.json](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/request_bound_fusion_r15b/reports/cached_temporal_metadata_audit.json)

支持3857条记录与时间戳审计；它不是完整语义—片段一致性审计。

### E10 — 生成损失

[physical_content/supervision.py](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/physical_content/supervision.py)

高误差聚合是通用生成误差归约，不等价于物理标签或物理规律损失。

### E11 — 原生结构和LoRA接口

[Wan官方model.py](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py)

[Wan官方TI2V-5B配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)

[项目physical_scaling/model.py](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_scaling_q256/physical_scaling/model.py)

[项目LoRALinear](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/physgen_native/lora.py)

原生block的self→text→FFN顺序、norm/time语义与LoRA `.base/.lora_a/.lora_b`命名已经核对。官方main会变化，正式实验必须再锁定实际安装commit；本次读到的官方model.py blob SHA为`6982fa15db1156dfcfc50d480ae9a42d8525f83d`，这是文件blob，不是repo commit。

## 2. 外部一手资料

### E12 — ProPhy

[arXiv v2](https://arxiv.org/html/2512.05564v2)

[PDF](https://arxiv.org/pdf/2512.05564v2)

本次已通过PDF截图核对第6页Table1和第7页Table2；数值也与v2 HTML表格一致。引用其协议与报告数字，不认为其机制已经验证本包三个新设计。ProPhy使用额外VLM标注；本包选择不引入此监督，因此不能声称拥有等价训练信号。

### E13 — VideoPhy2官方定义

[官方README](https://github.com/Hritikbansal/videophy/blob/main/VIDEOPHY2/README.md)

支持SA/PC 1—5与Joint双阈值定义。数据集自身名称含action-centric，不意味着本项目必须采用动作约束。

### E14 — Physics-IQ官方

[Physics-IQ / Verified](https://github.com/google-deepmind/physics-IQ-benchmark)

支持Original/Verified、输入模式、场景视角、同规模记录及SOTA四次运行要求。网页/榜单会更新，本包不声称锁定了所有未来或全部同规模SOTA。

### E15 — WMB官方

[项目页](https://worldmodelbench-team.github.io/)

[官方实现](https://github.com/WorldModelBench-Team/WorldModelBench)

支持350样本、7领域及输入任务对应规则。

### E16 — VBench官方

[官方实现](https://github.com/Vchitect/VBench)

[VBench论文](https://arxiv.org/abs/2311.17982)

支持多维质量评测、I2V/T2V区分。不同prompt集合和版本的总分不可直接等同。

### E17 — PyTorch SDPA

[官方API说明](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)

支持bool mask True为允许、attention默认缩放与dropout需要显式设置等实现注意事项。实际依赖版本应锁定，不默认使用main文档的所有新API。

## 3. 三类陈述必须分开

| 类别 | 本包内容 | 可据此得出的结论 |
|---|---|---|
| 仓库事实 | 已完成结果、真实源码、参数审计 | 描述历史能力与实现 |
| 推导/实现性质 | 参数恒等式、零初始化等价、source=0输出0 | 证明数学/工程性质 |
| 新研究假设 | 请求key改善SA、预写入改善执行、协同专家共同提高质量 | 必须靠训练和生成验证 |

不要把第三类变成“已证明”；也不要因为第三类待验证就把设计写成无法执行的空泛建议。

## 4. 本次结构收敛记录

1. 删除方案三及所有新增视觉教师、热图监督。
2. P1从多项score融合收敛为一个标准请求→物理key cross-attention；执行value不经二次请求聚合。
3. P2删除复制大块与双前向相减，改为每3/5层的原生block输入预写入；同时把state_value预算转为rank64原生生成适配。
4. P4删除干预学习与“因果专家”未经验证的命名，用两个来源明确的条件专家替换state_value；其参数量可精确预算匹配。
5. 三版分别训练，不第一轮全堆一起；所有中间量来自已有视频/物理/文本运算。
6. 共同保持高目标，不使用降低CFG、后处理平滑、挑seed或自动关闭物理作为达标方式。

## 5. 没有做过的事

本次未访问项目训练服务器、未提交GPU任务、未运行完整Wan推理、未重新评分40例、未获得任何新候选的PC/SA/Joint。附带CPU测试已执行23项并通过，是合成数学/梯度检查，不是项目GPU训练或视频效果实测。Markdown、配置、预算与检查结果均在本次对话环境中创建。

## 6. 附带检查记录

reference_math_checks_results.json（文件：`checks/reference_math_checks_results.json`）记录本次23项CPU检查及torch/Python版本；检查脚本（文件：`checks/reference_math_checks.py`）可独立重跑，不下载任何模型。它验证小尺寸算子的数学性质，没有加载Wan checkpoint，因此不能作为真实Wan集成或质量报告。

方案二额外补充了旧残差也会进入后续原生层的事实，避免把“注入位置变更”夸大成首次原生执行。三版的新生成收益仍属于待验证假设。


---

<!-- 原文件：06_HANDOFF_TO_CODING_AI.md -->

# 可直接交给编码AI的实施任务

## 任务正文

请基于本包选定的**一个**候选实现Physics Expert；不要把三个候选自动合并。先读取`00_IMPLEMENTATION_CONTRACT.md`、对应方案文件、`04_EVALUATION_AND_EXECUTION.md`与来源文件。默认优先`P2-S3-R64`，或按用户明确选择实现P1/P4。

历史起点固定为Lewislius/physgen的commit `1defd6f4b41e42302bc3e4b9b2bf938abbbc686c`中R10A相关代码及公共底座。禁止读取或使用R16及以后候选，禁止修改旧实验结果/权重。新工作区独立。

第一步交付代码映射：列出将复用的R10A增强、loss、native文本、LoRA与采样接口；列出实际Wan版本和旧接口形状。若路径/签名不一致，先定位真实源码，不要凭相似名字导入后续版本。

第二步实现模型：明确`ConditionBundle`、`PhysicsReadOutput`、原生base访问、site选择、native运算拆分。P1只增强route；P2直接在residual预写入后调用一次native block；P4将两个来源条件专家放到同一native FFN中。所有native模块每个block只执行设计规定次数。

第三步安装训练参数并审计：每个参数恰好一个owner，LoRA不得重复包装。新增模块所有output初始化零，上游source/value保持非零。冻结权重不能使用`no_grad()`切断输入梯度。落实`joint400`而非静默继承旧128+272冻结规则。

第四步做测试：公共测试、方案专项测试、参数预算、CFG正负缓存、首帧mask、真实Wan单batch、gradient checkpointing开关、保存恢复。附带CPU检查仅作参考，不替代实际Wan测试。

第五步交付训练/推理/评测入口，但**除非用户明确授权，不自动提交GPU任务、下载新模型或使用外部API**。YAML中null路径需先依据本地实际文件填写并验证，不编造服务器路径。记录未完成项。

禁止新增：V-JEPA等教师、热图监督、动作约束、干预/反事实训练数据、因果置信度、语义保护门、动态注入预算、未知用途中间变量、自动调小guidance、best-of-N结果挑选。已有Qwen/T5只是已有输入编码器，不承担新教师损失。

## 代码交付目录建议

```text
physics_expert_redesign_v2/
  common/
    enhancement.py
    physics_read.py
    native_ops.py
    lora_install.py
    ownership.py
    condition_cache.py
  variants/
    p1_request_key.py       # 仅选定方案必需
    p2_pre_self.py
    p4_cooperative.py
  configs/
  tests/
  train.py
  infer.py
  evaluate.py
  README.md
```

只实现被选择的方案也可以，但公共代码不得假装已经支持另外两版。没有实际评分时`quality_goal_achieved=false`、分数字段null。

## 验收输出

请报告：代码差异、模块调用顺序、所有训练参数owner与数量、CPU/真实Wan测试分别通过哪些、训练入口命令、恢复规则、推理协议、尚未执行的步骤。参数统计分别列base/新增常驻模块/LoRA/总训练/系统外部条件模型。

数学性质通过不等于模型效果通过。PC≥3.85/3.90、SA≥3.25、Joint≥15/40以及高画质与跨基准竞争，是实际训练后的验收目标。既追求新增Joint成功，也保留并提高原本强样本的PC/SA和细节质量。

## 三个最容易写错的重点

P1：`C_execute`必须保持原张量，辅助loss对raw route而非effective route；请求attention只计算一次/条件，不写成十个复制模块。

P2：`hidden = hidden + physical_write`发生在真实residual上，然后调用一次当前native block；不是`norm3(hidden)+write`，不是复制大block，也不是两条forward相减。state_value删除，self/FFN LoRA rank64真实安装。

P4：条件专家读raw物理message和native text output；不用已经零初始化后的physical_write作唯一source；两个专家512宽加起来等于原state_value的参数预算。原生text与FFN不得重复计算。


---

# 默认YAML配置附录

以下YAML是待实现训练器的配置契约，不是可直接提交到旧runner的现成作业；null路径必须依据实际环境解析。

## p1_s3.yaml

```yaml
# 设计配置：不是已运行结果。paths=null须由实际环境解析。
schema_version: physics-expert-design-v2
status: design_not_project_training_results
project:
  repository: Lewislius/physgen
  ref: 1defd6f4b41e42302bc3e4b9b2bf938abbbc686c
  base_variant: execution_value_r10
  new_workspace: experiments/physics_expert_redesign_v2
  reuse_future_candidates: false
paths:
  wan_code: null
  wan_checkpoint: null
  qwen_checkpoint: null
  training_video_cache: null
  input_condition_cache: null
  interface_checkpoint: null
  fixed40_protocol: null
  independent_validation_protocol: null
  output_root: null
execution:
  auto_download: false
  auto_launch_training: false
  batch_size_per_device: 1
geometry:
  mode: native_i2v
  infer_from_patch_embedding: true
  default_rgb_size:
  - 480
  - 832
  default_rgb_frames: 81
  known_frames: 1
  exclude_known_frame_from_generation_loss: true
model:
  width: 3072
  num_layers: 30
  num_heads: 24
  ffn_dim: 14336
  qwen_width: 4096
  q_tokens: 256
  q_heads: 32
  rounds: 2
  condition_every: 3
  sites:
  - 2
  - 5
  - 8
  - 11
  - 14
  - 17
  - 20
  - 23
  - 26
  - 29
physics:
  enhancement: r10a_route_execute
  reader: direct_original_values
  query: frozen_native_text_q_base
  key_value_init: copy_corresponding_native_base
  output_init: zero
  group_prior: existing_group_log_prior
  reference_slot: existing_zero_value_reference
  video_time_query: true
  new_sigma_mapper: false
  direct_write: true
  state_value: true
  query_chunk_size: 2048
  mask_known_direct_write: true
  write_position: post_self_parallel_text
native_lora:
  self_attn_targets:
  - q
  - k
  - v
  - o
  ffn_targets: both_linear_layers
  self_ffn_rank: 32
  self_ffn_alpha: 32
  text_targets:
  - q
  - o
  text_rank: 16
  text_alpha: 16
  text_kv_trainable: false
  start_update: 1
request_key:
  enabled: true
  shared_across_sites: true
  width: 3072
  heads: 24
  norm: layernorm_no_affine
  qkv_init: xavier_uniform
  output_init: zero
  bias: false
  actual_text_tokens_only: true
  modify_execute: false
  ffn: false
experts:
  enabled: false
train:
  preset: joint400
  initialization: fresh_official
  updates: 400
  accumulation: 4
  seed: 42
  warmup_updates: 32
  schedule_after_warmup: constant_for_initial_400
  train_all_selected_parameters_from_update: 1
  gradient_checkpointing: true
  text_dropout: 0.1
  condition_dropout: paired_native_text_and_matching_negative_physics_cache
  precision: bf16_with_fp32_sensitive_ops
  optimizer: AdamW
  betas:
  - 0.9
  - 0.999
  eps: 1.0e-08
  matrix_weight_decay: 0.01
  norm_bias_reference_weight_decay: 0.0
  grad_clip_per_group: 1.0
  peak_lrs:
    enhancement: 5.0e-05
    physics_read: 2.0e-05
    physics_write: 3.0e-05
    native_self_ffn_lora: 2.0e-05
    native_text_qo_lora: 1.0e-05
    request_key: 2.0e-05
  save_updates:
  - 128
  - 250
  - 400
loss:
  velocity_target: noise_minus_clean
  mean_fm_weight: 0.8
  tail_weight: 0.2
  tail_fraction: 0.25
  tail_windows: exact_existing_supervision_py
  route_target: raw_route_only
  route_peak_weight: 0.03
  route_warmup_updates: 32
  extra_teacher_loss: false
  intervention_learning: false
  reward_training: false
inference:
  steps: 50
  shift: 5.0
  guidance: 5.0
  sampler: shifted_euler
  seed: 42
  persist_initial_noise: true
  best_of_n: 1
  cache_conditions_per_cfg_arm: true
evaluation:
  pc_mean_min: 3.85
  pc_mean_stretch: 3.9
  sa_mean_min: 3.25
  joint_pass_count_min: 15
  fixed40_count: 40
  catastrophic_artifacts_max: 0
  actual_results: null
variant: p1_request_key
design_id: P1-RKE-v2
config_id: P1-S3
parameter_budget:
  estimated_trainable: 713657344
  estimate_not_measured: true
```

## p2_s3.yaml

```yaml
# 设计配置：不是已运行结果。paths=null须由实际环境解析。
schema_version: physics-expert-design-v2
status: design_not_project_training_results
project:
  repository: Lewislius/physgen
  ref: 1defd6f4b41e42302bc3e4b9b2bf938abbbc686c
  base_variant: execution_value_r10
  new_workspace: experiments/physics_expert_redesign_v2
  reuse_future_candidates: false
paths:
  wan_code: null
  wan_checkpoint: null
  qwen_checkpoint: null
  training_video_cache: null
  input_condition_cache: null
  interface_checkpoint: null
  fixed40_protocol: null
  independent_validation_protocol: null
  output_root: null
execution:
  auto_download: false
  auto_launch_training: false
  batch_size_per_device: 1
geometry:
  mode: native_i2v
  infer_from_patch_embedding: true
  default_rgb_size:
  - 480
  - 832
  default_rgb_frames: 81
  known_frames: 1
  exclude_known_frame_from_generation_loss: true
model:
  width: 3072
  num_layers: 30
  num_heads: 24
  ffn_dim: 14336
  qwen_width: 4096
  q_tokens: 256
  q_heads: 32
  rounds: 2
  condition_every: 3
  sites:
  - 2
  - 5
  - 8
  - 11
  - 14
  - 17
  - 20
  - 23
  - 26
  - 29
physics:
  enhancement: r10a_route_execute
  reader: direct_original_values
  query: frozen_native_text_q_base
  key_value_init: copy_corresponding_native_base
  output_init: zero
  group_prior: existing_group_log_prior
  reference_slot: existing_zero_value_reference
  video_time_query: true
  new_sigma_mapper: false
  direct_write: true
  state_value: false
  query_chunk_size: 2048
  mask_known_direct_write: true
  write_position: pre_self
native_lora:
  self_attn_targets:
  - q
  - k
  - v
  - o
  ffn_targets: both_linear_layers
  self_ffn_rank: 64
  self_ffn_alpha: 64
  text_targets:
  - q
  - o
  text_rank: 16
  text_alpha: 16
  text_kv_trainable: false
  start_update: 1
request_key:
  enabled: false
experts:
  enabled: false
train:
  preset: joint400
  initialization: fresh_official
  updates: 400
  accumulation: 4
  seed: 42
  warmup_updates: 32
  schedule_after_warmup: constant_for_initial_400
  train_all_selected_parameters_from_update: 1
  gradient_checkpointing: true
  text_dropout: 0.1
  condition_dropout: paired_native_text_and_matching_negative_physics_cache
  precision: bf16_with_fp32_sensitive_ops
  optimizer: AdamW
  betas:
  - 0.9
  - 0.999
  eps: 1.0e-08
  matrix_weight_decay: 0.01
  norm_bias_reference_weight_decay: 0.0
  grad_clip_per_group: 1.0
  peak_lrs:
    enhancement: 5.0e-05
    physics_read: 2.0e-05
    physics_write: 3.0e-05
    native_self_ffn_lora: 2.0e-05
    native_text_qo_lora: 1.0e-05
  save_updates:
  - 128
  - 250
  - 400
loss:
  velocity_target: noise_minus_clean
  mean_fm_weight: 0.8
  tail_weight: 0.2
  tail_fraction: 0.25
  tail_windows: exact_existing_supervision_py
  route_target: raw_route_only
  route_peak_weight: 0.03
  route_warmup_updates: 32
  extra_teacher_loss: false
  intervention_learning: false
  reward_training: false
inference:
  steps: 50
  shift: 5.0
  guidance: 5.0
  sampler: shifted_euler
  seed: 42
  persist_initial_noise: true
  best_of_n: 1
  cache_conditions_per_cfg_arm: true
evaluation:
  pc_mean_min: 3.85
  pc_mean_stretch: 3.9
  sa_mean_min: 3.25
  joint_pass_count_min: 15
  fixed40_count: 40
  catastrophic_artifacts_max: 0
  actual_results: null
variant: p2_pre_self
design_id: P2-DNE-v2
config_id: P2-S3-R64
parameter_budget:
  estimated_trainable: 638553088
  estimate_not_measured: true
```

## p4_s3.yaml

```yaml
# 设计配置：不是已运行结果。paths=null须由实际环境解析。
schema_version: physics-expert-design-v2
status: design_not_project_training_results
project:
  repository: Lewislius/physgen
  ref: 1defd6f4b41e42302bc3e4b9b2bf938abbbc686c
  base_variant: execution_value_r10
  new_workspace: experiments/physics_expert_redesign_v2
  reuse_future_candidates: false
paths:
  wan_code: null
  wan_checkpoint: null
  qwen_checkpoint: null
  training_video_cache: null
  input_condition_cache: null
  interface_checkpoint: null
  fixed40_protocol: null
  independent_validation_protocol: null
  output_root: null
execution:
  auto_download: false
  auto_launch_training: false
  batch_size_per_device: 1
geometry:
  mode: native_i2v
  infer_from_patch_embedding: true
  default_rgb_size:
  - 480
  - 832
  default_rgb_frames: 81
  known_frames: 1
  exclude_known_frame_from_generation_loss: true
model:
  width: 3072
  num_layers: 30
  num_heads: 24
  ffn_dim: 14336
  qwen_width: 4096
  q_tokens: 256
  q_heads: 32
  rounds: 2
  condition_every: 3
  sites:
  - 2
  - 5
  - 8
  - 11
  - 14
  - 17
  - 20
  - 23
  - 26
  - 29
physics:
  enhancement: r10a_route_execute
  reader: direct_original_values
  query: frozen_native_text_q_base
  key_value_init: copy_corresponding_native_base
  output_init: zero
  group_prior: existing_group_log_prior
  reference_slot: existing_zero_value_reference
  video_time_query: true
  new_sigma_mapper: false
  direct_write: true
  state_value: false
  query_chunk_size: 2048
  mask_known_direct_write: true
  write_position: post_self_pre_text
native_lora:
  self_attn_targets:
  - q
  - k
  - v
  - o
  ffn_targets: both_linear_layers
  self_ffn_rank: 32
  self_ffn_alpha: 32
  text_targets:
  - q
  - o
  text_rank: 16
  text_alpha: 16
  text_kv_trainable: false
  start_update: 1
request_key:
  enabled: false
experts:
  enabled: true
  type: two_source_conditioned
  inner_width: 512
  sources:
  - raw_physics_message
  - native_text_output
  operation: gelu_state_times_linear_source
  source_norm: layernorm_no_affine
  output_init: zero
  bias: false
  softmax_router: false
  position: inside_native_ffn
  mask_known_physics_expert: true
train:
  preset: joint400
  initialization: fresh_official
  updates: 400
  accumulation: 4
  seed: 42
  warmup_updates: 32
  schedule_after_warmup: constant_for_initial_400
  train_all_selected_parameters_from_update: 1
  gradient_checkpointing: true
  text_dropout: 0.1
  condition_dropout: paired_native_text_and_matching_negative_physics_cache
  precision: bf16_with_fp32_sensitive_ops
  optimizer: AdamW
  betas:
  - 0.9
  - 0.999
  eps: 1.0e-08
  matrix_weight_decay: 0.01
  norm_bias_reference_weight_decay: 0.0
  grad_clip_per_group: 1.0
  peak_lrs:
    enhancement: 5.0e-05
    physics_read: 2.0e-05
    physics_write: 3.0e-05
    native_self_ffn_lora: 2.0e-05
    native_text_qo_lora: 1.0e-05
    expert_in: 2.0e-05
    expert_out: 3.0e-05
  save_updates:
  - 128
  - 250
  - 400
loss:
  velocity_target: noise_minus_clean
  mean_fm_weight: 0.8
  tail_weight: 0.2
  tail_fraction: 0.25
  tail_windows: exact_existing_supervision_py
  route_target: raw_route_only
  route_peak_weight: 0.03
  route_warmup_updates: 32
  extra_teacher_loss: false
  intervention_learning: false
  reward_training: false
inference:
  steps: 50
  shift: 5.0
  guidance: 5.0
  sampler: shifted_euler
  seed: 42
  persist_initial_noise: true
  best_of_n: 1
  cache_conditions_per_cfg_arm: true
evaluation:
  pc_mean_min: 3.85
  pc_mean_stretch: 3.9
  sa_mean_min: 3.25
  joint_pass_count_min: 15
  fixed40_count: 40
  catastrophic_artifacts_max: 0
  actual_results: null
variant: p4_source_conditioned_experts
design_id: P4-SCE-v2
config_id: P4-S3-H512
parameter_budget:
  estimated_trainable: 675908608
  estimate_not_measured: true
```

# 本次CPU参考检查状态

已实际运行 **23/23** 项并通过。Python 3.13.5，torch 2.10.0+cpu，设备CPU。仅验证小型合成算子的代数、shape、初始化和autograd性质。真实Wan集成、项目训练、PC/SA/Joint和跨基准评测均未运行。详细可重跑脚本与JSON记录见ZIP。
