# 方案四终版：物理—语义上下文协同专家，无教师、无干预学习

设计 ID：`P4-SCE-v2`  
默认配置：`P4-S3-H512`，每3层注入、两个条件专家各512内部宽度。  
备选频率：`P4-S5-H512`，每5层独立训练。  
状态：可执行设计，未获得新训练结果。  
共同实现约定：[`00_IMPLEMENTATION_CONTRACT.md`](00_IMPLEMENTATION_CONTRACT.md)。

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

来源索引：[E02—E07、E11]见[`05_EVIDENCE_AND_DECISIONS.md`](05_EVIDENCE_AND_DECISIONS.md)。
