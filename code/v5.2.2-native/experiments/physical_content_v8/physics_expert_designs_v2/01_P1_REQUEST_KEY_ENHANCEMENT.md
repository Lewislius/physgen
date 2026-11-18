# 方案一终版：原请求条件化物理键，完整保留物理执行值

设计 ID：`P1-RKE-v2`  
默认配置：`P1-S3`，每3层注入；对照为独立训练的`P1-S5`。  
状态：设计规格，未训练、未生成项目评测分数。  
必读公共约定：[`00_IMPLEMENTATION_CONTRACT.md`](00_IMPLEMENTATION_CONTRACT.md)。

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

来源索引：[E01,E02,E04,E05,E06,E11]见[`05_EVIDENCE_AND_DECISIONS.md`](05_EVIDENCE_AND_DECISIONS.md)。
