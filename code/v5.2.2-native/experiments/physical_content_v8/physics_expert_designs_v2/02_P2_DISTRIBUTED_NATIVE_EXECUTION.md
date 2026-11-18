# 方案二终版：分布式物理预写入与原生生成执行

设计 ID：`P2-DNE-v2`  
默认配置：`P2-S3-R64`，每 3 层注入，原生 self-attention/FFN LoRA rank64。  
对照配置：`P2-S5-R64`，每 5 层注入。  
状态：**设计定稿，等待真实实现、训练与生成评测。**  
先读：[`00_IMPLEMENTATION_CONTRACT.md`](00_IMPLEMENTATION_CONTRACT.md)。

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

来源索引：[E01—E08、E11]见[`05_EVIDENCE_AND_DECISIONS.md`](05_EVIDENCE_AND_DECISIONS.md)。

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
