# 基于 Wan 的初态外观—语义—因果状态转移方案：1–2 天 MVP 与论文路线

> 日期：2026-08-31  
> 原始诊断基线：**ACE-Router — Appearance-anchored Causal Event Residual Routing**  
> 修订后 1–2 天方案：**TRACE-Lite — Transition- and Region-Aware Causal Evolution Routing**  
> 远期完整方向：**TRACE-Field + RWC — Spatiotemporal Causal State-Transition Fields with Reader–Writer–Corrector Feedback**  
> 基础模型：优先 `Wan2.2-TI2V-5B`；`Wan2.1-T2V-1.3B` 作为低成本、纯文本对照  
> 重要边界：本方案**不以 PhysVid 为基础模型，不使用 PhysVid 的 local conditioning、negative physics prompt 或 checkpoint**。PhysVid/VideoPhy2 报告只作为问题证据库。
> 修订说明：全文按三个独立部分重新编号，每部分均从 1 开始。第二部分结合 [CoECT（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Chain_of_Event-Centric_Causal_Thought_for_Physically_Plausible_Video_Generation_CVPR_2026_paper.html) 与 [ProPhy](https://arxiv.org/abs/2512.05564) 重新审计设计；其中 TRACE-Lite **取代**第一部分的 ACE-Router 作为推荐 MVP，ACE 仅保留为消融基线。第三部分进一步分析 Reader–Writer/Corrector 的创新边界，并将其重构为 TRACE 的闭环远期版本。

阅读导航：第一部分解释失败原因并定义 ACE 基线；第二部分给出当前应执行的 TRACE-Lite、TRACE-Field 和四损失训练计划；第三部分只讨论通过前两阶段验证后才进入的 RWC 闭环扩展。三个部分的编号彼此独立。

# 第一部分：问题诊断、ACE-Router 基线与初始 MVP

## 1. 最终建议

最值得先做的不是继续扩写 prompt，也不是第一天就训练一个复杂 physics adapter，而是验证下面这个更严格的假设：

> **首帧外观负责定义事件初态并提供可辨识连续性，语义负责实体、角色与任务，物理负责指定哪些状态变量必须在局部时空中变化。只有未参与事件的实体、区域和属性才是不变量；参与事件的对象不能被整体当成不变量。**

原先“外观负责不变量”的说法过强，也会直接误导方法设计。更准确地，应把条件拆成三类变量：

| 变量类型 | 含义 | 冰融化示例 |
|---|---|---|
| 初始条件 $S_0$ | 只约束视频开始时真实存在的状态 | 开场有一块完整冰块，位于浅盘中 |
| 选择性不变量 $z_{inv}$ | 事件期间本来就不应被该过程改变的实体、区域或属性 | 相机、浅盘、桌面、背景；水与冰的材料来源连续性 |
| 转移变量 $q(t)$ | 必须随事件演化、不能被首帧永久锁死的量 | 冰块尺寸和可见占比下降，液态水面积上升，固相最终可消失 |

因此，冰块的“初始外观”不是跨视频不变量。冰块可以逐渐变小乃至不可见；需要保持的是**同一材料系统的连续演化与无关场景稳定**，而不是“冰块始终看起来像第一帧”。方法上也不应把整张首帧或整个事件对象硬冻结，只应把 $I_0$ 当作 $t=0$ 的边界条件，并把新增物理 residual 限制在正确的状态变量和时空 support 上。

原 ACE-Router 只处理了两个轴：Transformer 层深 \(\ell\) 和去噪步骤 \(k\)。新增论文指出了它仍未覆盖的两个轴：输出视频时间 \(f\) 和空间位置 \((x,y)\)。特别要强调：

```text
denoising step k != output video time f
```

在较早去噪步骤注入一条“冰逐渐融化”的全局条件，只表示较早规划这个概念，并没有告诉模型第 1、10、20、40 帧分别应处于什么状态。因此原 ACE 仍可能把“融化后的水”生成在所有帧，或者只在某一帧完成瞬时替换。

修订后的 1–2 天版本 TRACE-Lite 仍然无训练，但增加两个最低必要结构：

1. 使用 `Wan2.2-TI2V-5B` 的原生首帧 I2V 接口，把一张**事件发生前**的图像作为初始状态、实体可辨识性、相机和空间布局锚点；不把对象的尺寸、姿态、相态等转移变量当成永久外观约束。
2. 原始短 prompt 编码为语义条件，在所有 30 个 DiT block 中保持不变。
3. 将物理过程压缩为 3 个短状态阶段，而不是一段长 prompt；每个阶段只描述相对上一阶段必须可见的变化。
4. 为每个阶段设置重叠的输出时间权重，让 precondition、transition 和 terminal 条件分别主要作用到对应视频帧。
5. 使用手工框/粗 tube 构造事件区域，只在 actor、patient、接触邻域或材料变化区域注入物理残差；背景和无关实体不接收该 residual，但仍由原 Wan 路径自然生成，而不是被像素级冻结。
6. 在约 1/2～3/4 深度的 block 中，计算“正确阶段 − 对应失败反事实”的 cross-attention 残差；仍在去噪前 50%～70% 较强、后期衰减。

核心形式为：

\[
h'_{\ell,k,i}=h_{\ell,k,i}+r_{sem,\ell,k,i}
+\lambda_{\ell,k}
\sum_{m=1}^{M}
G_m(i)\Delta r_{m,\ell,k,i},
\]

其中：

- \(i=(f,x,y)\) 是输出视频 latent token，不是去噪步骤；
- \(\Delta r_m=\operatorname{CA}(h,c_m^+)-\operatorname{CA}(h,c_m^-)\)；
- \(c_m^+\) 是第 \(m\) 个阶段的正确可见变化；
- \(c_m^-\) 是该阶段最容易发生的失败反事实，如“结果已预置”“瞬间替换”或“状态回生”；
- \(G_m(f,x,y)=\alpha_m(f)\beta_m(f,x,y)\)；\(\alpha\) 是视频时间相位，\(\beta\) 是局部事件区域；
- \(\lambda_{\ell,k}=g_{layer}(\ell)g_{step}(k)\)：层深和去噪阶段的分解式权重。

这个 MVP 用很低成本回答四个递进问题：

1. 长 prompt 失败是否主要来自条件冲突和全层广播，而非物理知识不足？
2. 初态图像能否减少无因果依据的实体丢失、类别替换和身份漂移，同时允许事件要求的合理消失、变形、位移或相变，并避免静态 no-op？
3. 明确的输出时间相位能否减少“直接生成终态”、中间态缺失和状态回生？
4. 局部事件区域能否减少背景、无关人物或另一种物理过程被错误改写？

推荐实验把原 ACE 作为 global causal baseline，再与 `TRACE-Time` 和 `TRACE-ST` 对比。这样即使 TRACE-ST 无效，也能判断问题来自时间分配、空间定位还是 causal residual 本身，而不是把所有改动混成一个结果。详细实现、远期方案与闭环 RWC 设计分别见第二、第三部分。

---

## 2. 为什么长物理 prompt 反而让视频更差

你的扩写内容在人类看来非常正确，但它并不等价于适合视频生成模型的条件表示。

### 2.1 同一条全局文本同时描述互斥状态

长 prompt 同时包含：

- 碰撞前：滑板车直立、持续接近、尚未接触；
- 碰撞时：发生接触；
- 碰撞后：倾斜、减速；
- 末态：停止并保持倾斜。

普通 T2V cross-attention 没有被明确告知每句话属于哪一段时间。所有帧和所有层会同时看到“直立、倾斜、移动、停止”等互斥状态，容易出现：

- 开场就倾斜或已经接触；
- 全程静止以同时满足“保持一致”和“停止”；
- 物体在不同状态间形变，而不是发生连续运动；
- 镜头切换绕过真正的碰撞过程。

### 2.2 抽象逻辑句不是可视条件

`must`, `before`, `rather than`, `must not`, `cause-effect constraints` 对 LLM 很自然，但视频文本编码器主要从大规模 caption 学习视觉共现，并没有被训练成约束求解器。它可能知道 `scooter`、`trash can`、`tilting` 的视觉语义，却不会严格执行 20 条逻辑谓词的合取。

### 2.3 正向 prompt 中出现了错误概念

下面这句虽然是否定形式，但正向文本编码器仍会看到这些视觉概念：

```text
must not disappear, duplicate, teleport, or pass through
```

扩散模型未必能稳定处理逻辑否定。它可能反而激活“消失、复制、穿透”的相关表示。长串 negative concepts 放在正向 prompt 中风险很高。

### 2.4 物理词占据了外观和主体的注意力预算

原 prompt 只有两个主体和一个动作。扩写后，`scooter`、`trash can`、`collision`、`tilting` 被大量重复，且混入抽象词。T5 token 虽未必超过 512 长度，但 cross-attention 的有效注意力会被稀释。结果可能是主体外观下降、背景变杂、动作不再聚焦。

### 2.5 CFG 会放大整个条件差异，而不是只放大物理部分

标准 CFG 为：

\[
\epsilon=\epsilon_u+s(\epsilon_c-\epsilon_u).
\]

当 \(c\) 是一整段长物理描述时，CFG 同时放大：

- 主体和场景变化；
- 摄影、风格和光照变化；
- 正确物理状态；
- 互相冲突的状态；
- 模型不理解的逻辑词。

因此“物理描述更准确”不保证生成更准确。

### 2.6 结论

长 prompt 实验的负结果不是坏消息。它提供了一个很有价值的研究线索：

> **问题可能不是缺少物理文字，而是缺少对不同条件职责、层深和时间作用域的分离。**

---

## 3. 初态图像 + 高层语义是否有效

答案是：**有潜力有效，但前提是二者只约束各自擅长的变量。直接融合并不自动有效。**

### 3.1 三类条件应该各管什么

| 条件 | 应主要负责 | 不应强行负责 |
|---|---|---|
| 初态图像 \(I_0\) | 第 1 帧的实体、初始相态/姿态、颜色与材质线索、粗几何、初始位置、相机和背景 | 把参与事件的对象外形永久锁死；完整轨迹、碰撞响应和最终状态 |
| 原始短文本 \(p_{sem}\) | Actor/Tool/Patient、动作目标、场景大类 | 逐帧精确动力学、十几条否定约束 |
| 物理事件条件 \(p_{phy}\) | 哪些状态量改变、由何触发、何时/何处改变、变化方向与关键末态 | 无因果依据地重定义风格、主体类别、相机和无关背景 |

这可以概括为：

```text
Initial image = what is true at t=0 and what remains identifiable
Semantics     = what event should happen to which entities
Physics       = what must change, where, when, and because of what
```

这里不能把 `appearance` 与 `invariant` 画等号。应先做属性级审计：

```text
protected properties  -> camera, background, unaffected entities, category continuity
transition properties -> position, pose, size, shape, phase, amount, velocity, integrity
```

同一属性是否为不变量取决于事件。例如“红球撞墙”中颜色通常应保持、速度应改变；“纸张燃烧”中颜色、轮廓和完整性都应改变。TRACE-Lite 的空间 mask 只能回答“哪里允许额外物理干预”，不能精确分离一个对象内部哪些 feature channel 应保持。因此首版应把“不变量保护”表述为**减少无关改写的软保护**，不声称严格属性解耦。

### 3.2 为什么 `Wan2.2-TI2V-5B` 很适合作为首版基座

本地已有：

- [/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B](/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B)
- 30 个 DiT block；
- 5B 参数；
- 原生统一 T2V/I2V；
- 官方 I2V 实现先用 VAE 编码输入图像，再以时序 mask 将该图像 latent 写入首个时间片；初始化采样和每次 scheduler 更新后都会重新锚定这一已知时间片，而不是临时增加一个尚未对齐的 CLIP/DINO adapter；
- 720P、24 FPS，官方配置为 121 帧、50 steps；本地也已有较短帧数测试路径。

官方仓库明确说明 TI2V-5B 同时支持 T2V 与 I2V，并能在 4090 级显卡运行：[Wan2.2 官方仓库](https://github.com/Wan-Video/Wan2.2)、[TI2V-5B 模型卡](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)。

这一点也可在本地官方实现中直接核对：图像编码位于 [textimage2video.py](/home/liuzhirui/model/Wan2.2/wan/textimage2video.py:512)，首帧 latent 的初始化与逐步重锚定位于同一文件的 [mask 混合逻辑](/home/liuzhirui/model/Wan2.2/wan/textimage2video.py:550)。因此这里利用的是基座本来就学过的 I2V 条件路径，不是在两天实验中额外训练一个视觉接口。

### 3.3 为什么不建议首版给 Wan1.3B 硬加图像编码器

Wan2.1-T2V-1.3B 是纯文本 T2V。给它增加 DINO/CLIP 图像 token 后，没有现成对齐权重，必须训练 adapter；一两天内即使代码跑通，也无法区分“设计无效”和“adapter 尚未学会”。

因此：

- **完整 MVP：**用 Wan2.2-TI2V-5B；
- **廉价对照：**Wan2.1-T2V-1.3B 只测试文本语义 + 因果残差路由，不测试初态图像；
- **论文阶段：**再把视觉 adapter 移植到 1.3B，验证模型无关性。

### 3.4 初态图像也可能产生负作用

首帧图像不是越强越好。它是初始边界条件，不是全视频模板。它可能：

- 已经处于碰撞/破裂后的状态，使动作无法从前态开始；
- 物体距离太近，没有接近轨迹空间；
- 主体太小，接触不可见；
- 过强锁定姿态，让 I2V 倾向静态；
- 缺少某个关键实体，把错误永久锚定。

所以首帧必须是**pre-event neutral state**，并经过四项快速检查：

1. 关键实体全部存在；
2. 尚未发生目标事件；
3. 主体大小足以看见接触；
4. 留有运动和结果发生的空间；
5. 没有把本应变化的属性写成必须长期保持的视觉约束。

对“冰块融化”这类对象合理消失的场景，实体存在率也不能简单定义为“冰块每一帧都存在”。正确指标应是：开场冰存在；中段冰量连续下降且水量增加；后段允许固态冰完全消失，但不能无原因瞬间删除、换成无关物体或再次回生。

---

## 4. 如何正确理解 DiT 的“早层/中层/晚层”

这里必须区分两个完全不同的轴。

### 4.1 轴一：同一个去噪 step 内的 Transformer 深度

相关工作给出的共同线索是：

- 较早 block 更偏密集感知、背景和局部结构；
- 中间到约 2/3 深度的 block 更适合对象关系、动作、追踪和推理表示；
- 后期 block 更偏整合、输出准备和细节生成。

[From Image to Video](https://arxiv.org/abs/2502.07001) 在 24 层 WALT 上发现，大多数任务的最佳表示约位于模型 2/3 深度；点追踪稍早。  
[DiTFlow](https://arxiv.org/abs/2412.07776) 在 CogVideoX 的运动迁移消融中，block 20 明显优于 0、10、30。  
[Demystifying Video Reasoning](https://arxiv.org/abs/2603.16870) 的 40 层视频 DiT 分析指出，早层偏感知，中层窗口 20–29 对推理最关键，后层负责整合。

这些结果支持“物理关系优先放在中层”的设计先验，但不能写成普适定律。它们使用的模型和任务不同，必须在 Wan 上做自己的层窗口消融。

### 4.2 轴二：从高噪声到低噪声的 denoising steps

[Video Models Reason Early](https://arxiv.org/abs/2603.30043) 在受控迷宫和视频推理任务中发现，高层轨迹往往在前几个去噪 step 已基本确定，后续主要细化外观。DiTFlow 也显示在前 20%～40% 去噪阶段施加运动指导具有较好的效果/计算折中。

因此合理的初始假设是：

- 因果/运动指导应在高噪声到中噪声阶段较强；
- 最后 20%～30% step 应减弱物理文本干预，让纹理、边缘和身份一致性收敛；
- 这不等于“最后阶段不需要物理”，只是避免抽象条件持续扰动细节。

### 4.3 轴三：输出视频自身的物理时间

这是本次修订最重要的补充。去噪 step \(k\) 表示模型从噪声到样本的求解进度；输出时间 \(f\) 表示视频中“先发生什么、后发生什么”。二者不是同一个轴。

例如在所有输出帧上、只于早期去噪注入 `melted water`，仍可能让模型从第 1 帧开始就画出水。要描述 \(S_0\rightarrow S_1\rightarrow S_2\)，条件必须能区分输出 latent 的不同时间 token：

```text
early video tokens  -> intact/precondition state
middle video tokens -> visible transition state
late video tokens   -> terminal and persistence state
```

### 4.4 轴四：事件发生的空间区域

同一视频中可能同时有火焰、液体、人和雪地。物理变化通常只发生在 actor、patient、接触界面或材料区域。若相同 residual 加到所有 \((x,y)\) token，模型仍是各向同性响应：燃烧条件可能改变咖啡和人物，流体条件可能让背景产生形变。

因此完整路由至少是四轴函数：

\[
G=G(\text{layer }\ell,\text{ denoise }k,\text{ video time }f,\text{ space }x,y).
\]

### 4.5 对 30 层 Wan 的初始映射

| Wan block | 首版解释 | 条件策略 |
|---:|---|---|
| 0–7 | 感知、初始几何、背景 | 图像锚 + 原始语义；因果残差很弱 |
| 8–13 | 对象定位向关系表示过渡 | 中等因果残差 |
| 14–23 | 动作、关系、运动/推理候选窗口 | 强因果残差 |
| 24–29 | 表示整合与输出细化 | 快速衰减到语义基线 |

这只是初始化，不是最终结论。首日必须同时测 `0–9`、`10–19`、`14–23`、`20–29` 和 `all`。

---

## 5. 方法：ACE-Router

### 5.1 输入

对每个事件构造四个条件。

#### A. 初态首帧 \(I_0\)

事件发生前的中性状态图像，负责定义 $t=0$ 时的实体、状态、布局和相机，并提供后续实体/材料连续性的视觉线索；它不表示参与事件的外形、姿态或相态必须跨帧不变。

#### B. 原始语义条件 \(c_{sem}\)

直接使用用户的原始短 prompt，不做长扩写：

```text
A scooter collides with a trash can, the scooter tilting to one side before stopping.
```

#### C. 正确因果条件 \(c_+\)

原始 prompt 加一条紧凑、可见、肯定式的事件句：

```text
The same scooter visibly contacts the trash can, then tilts while slowing, and remains stopped beside it.
```

#### D. 最小反事实条件 \(c_-\)

只交换一个关键因果关系，其余实体和场景保持一致：

```text
The same scooter tilts and stops while it is still separated from the trash can; contact happens only afterward.
```

`c_-` 不是一长串“不要怎样”，也不直接作为整段 negative prompt。它用肯定式语言描述一个最邻近的错误过程，只用于产生一个和正确事件相反的局部注意力方向。

### 5.2 为什么使用“正确 − 反事实”而不是“长物理 − 空文本”

\(c_+\) 和 \(c_-\) 共享：

- 相同实体；
- 相同场景；
- 相同外观词；
- 相同动作词的大部分内容。

二者主要区别是接触与倾斜的顺序、是否存在碰撞后响应。因此它们的 cross-attention 差分更可能突出因果顺序，而不是再次改变主体颜色、背景和摄影风格。

这不是严格数学上的语义消去，因为 attention 是非线性的；但它是一个可被实验验证的近似。

### 5.3 从原始视频文本自动生成 \(c_+\) 和 \(c_-\) 的提示词

下面的提示词可直接交给具备稳定 JSON 输出能力的 LLM。它生成的是 ACE 的**单对全局条件**；第二部分 TRACE-Lite 会进一步把它拆成三个 phase。建议将“系统提示词”和“用户提示词”分别放入对应消息，不要把 LLM 的解释性文字送进 Wan T5。

系统提示词：

```text
You are a causal-condition compiler for a text-to-video diffusion experiment.
Given one original video prompt, create one compact correct causal condition c_plus
and one minimally different failure counterfactual c_minus.

Rules:
1. Preserve every explicitly stated entity, count, material, scene, and action goal.
2. Do not invent hidden forces, new objects, colors, camera motion, style, or outcomes.
3. c_plus must describe only visible states/actions in chronological order:
   precondition or trigger -> visible response -> terminal persistence when applicable.
4. c_minus must keep the same entities, scene, and nearly the same action vocabulary,
   but change exactly ONE causal/evolution relation. Choose the nearest observable
   failure from: outcome_preset, trigger_missing, cause_effect_reversal,
   instant_replacement, middle_deletion, terminal_reversal,
   unjustified_entity_loss, or no_op.
5. Describe c_minus affirmatively as what the wrong video DOES. Do not use a list of
   negations and do not use the words "not", "no", "without", "avoid", "must", or
   "should" in either condition.
6. Each condition must be one plain English sentence of at most 32 words. Use concrete
   visible verbs and states; avoid formulas, explanations, abstract physical laws,
   camera/style language, and quality adjectives.
7. Do not copy the whole original prompt. Output only the extra causal clauses that
   will later be appended to the unchanged original prompt.
8. If the original prompt is ambiguous, make the smallest visually conventional
   interpretation and record it in warnings. If no meaningful state change exists,
   lower confidence instead of fabricating one.
9. The c_plus/c_minus pair must differ mainly in event order, transition continuity,
   trigger-response binding, or persistence—not in appearance.

Return strict JSON only, using exactly this schema:
{
  "original_prompt": "verbatim input",
  "event_type": "short label",
  "changed_state_variables": ["visible variable and direction"],
  "preserved_context": ["shared entity/scene fact"],
  "causal_positive": "c_plus",
  "causal_counterfactual": "c_minus",
  "counterfactual_type": "one allowed type",
  "single_changed_relation": "one concise description",
  "warnings": [],
  "confidence": 0.0
}
The confidence must be a number from 0 to 1.
```

用户提示词：

```text
Compile this original video prompt:
{{ORIGINAL_VIDEO_PROMPT}}
```

滑板车示例的合格输出核心应接近：

```json
{
  "causal_positive": "The same scooter visibly contacts the trash can, then tilts while slowing, and remains stopped beside it.",
  "causal_counterfactual": "The same scooter tilts and stops while it is still separated from the trash can; contact happens only afterward.",
  "counterfactual_type": "cause_effect_reversal",
  "single_changed_relation": "tilting and stopping occur before rather than after contact"
}
```

使用前做四项程序化检查：两个句子中的实体集合相同；长度差不超过约 30%；`c_-` 只命中一个失败类型；两个句子不包含禁用否定词。然后仅编码 `original_prompt + causal_positive` 与 `original_prompt + causal_counterfactual`。若 LLM 改了实体数量、材质、背景或相机，必须重试或人工修正。

### 5.4 三条条件路径

```text
pre-event image I0
      │
      └── VAE first-frame latent ───────────────► initial state / continuity / layout

original short prompt c_sem
      │
      └── T5 → cross-attention in all blocks ──► entities / roles / event goal

compact c+ and counterfactual c-
      │
      └── T5 → mid-block attention difference ─► causal change / event order
```

三者在输入、功能和注入位置上分开，而不是拼成一段文本。

### 5.5 block 内的残差形式

Wan 原始 block 中，文本 residual 是：

\[
r_{sem}=\operatorname{CA}(\operatorname{Norm}(h),c_{sem}).
\]

ACE-Router 在指定 block 增加：

\[
\Delta r_{causal}
=
\operatorname{CA}(\operatorname{Norm}(h),c_+)
-
\operatorname{CA}(\operatorname{Norm}(h),c_-).
\]

最终：

\[
h' = h + r_{sem} + \lambda_{\ell,k}\Delta r_{causal}.
\]

如果 `lambda=0`，模型严格退回原 Wan 条件路径。这一点让调参和故障回退很安全。

### 5.6 层与 step 的分解路由

\[
\lambda_{\ell,k}=\lambda_0g_{layer}(\ell)g_{step}(k).
\]

首版固定：

```text
g_layer:
  blocks  0–7  : 0.10
  blocks  8–13 : 0.40
  blocks 14–23 : 1.00
  blocks 24–29 : 0.10

g_step for 50 sampling steps:
  steps  0–14  : 1.00
  steps 15–29  : 0.70
  steps 30–39  : 0.30
  steps 40–49  : 0.00

lambda_0 candidates: 0.25, 0.5, 1.0
```

不要第一轮就学习 router；固定 schedule 更容易解释因果。

### 5.7 如何从原始文本获得高质量 \(I_0\)

对纯文本 benchmark，推荐把 $I_0$ 的获得明确拆成“文本编译—候选生成—固定规则筛选”三步。不要直接让 T2I 看到完整事件 prompt，否则它很容易把碰撞后、融化后或破裂后的结果预置在首帧。

#### A. 从原始视频文本生成首帧状态描述

将下面的系统提示词交给 LLM：

```text
You are an initial-state image-prompt compiler for a controlled image-to-video
physics experiment. Convert one original video prompt into the single best still-image
description for frame 0, before the target event or state transition begins.

Rules:
1. Preserve all explicit entities, counts, materials, scene facts, and attributes.
2. Show every causally necessary entity clearly and simultaneously.
3. Place interacting entities with enough separation, free space, and scale for the
   later action, contact, deformation, transfer, or phase change to be visible.
4. Describe only the pre-event state using positive, directly visible language.
   Prefer "intact", "upright", "separated", "full", "dry surrounding surface",
   or equivalent initial-state wording. Do not narrate future motion.
5. Do not include the trigger, transition, terminal result, aftermath, motion blur,
   multiple moments, split panels, before-and-after layouts, or causal explanation.
6. Use one continuous shot with a static camera, clear subject scale, ordinary neutral
   lighting, and an unobstructed view. Choose the simplest viewpoint that exposes the
   interaction axis; do not add artistic style unless present in the input.
7. Treat event-varying properties such as position, pose, size, shape, phase, amount,
   velocity, and integrity as initial values only; list them under
   transition_variables_not_to_freeze.
8. If the input is ambiguous, make only the minimum layout/camera choice required for
   observability and record it under ambiguities.
9. pre_event_image_prompt must be plain English, 35-70 words, and must not contain
   "before and after", "sequence", "then", "finally", or multiple time points.

Return strict JSON only:
{
  "original_prompt": "verbatim input",
  "pre_event_image_prompt": "one positive still-image prompt",
  "required_visible_entities": ["entity"],
  "initial_states": ["entity: visible initial state"],
  "layout_requirements": ["observable spatial relation"],
  "protected_scene_properties": ["camera/background/unaffected entity property"],
  "transition_variables_not_to_freeze": ["variable expected to change"],
  "reject_if": ["observable reason to reject a generated candidate"],
  "ambiguities": []
}
```

用户提示词：

```text
Create the frame-0 image-state description for:
{{ORIGINAL_VIDEO_PROMPT}}
```

滑板车案例的 `pre_event_image_prompt` 可以是：

```text
A clear side view of one upright black scooter on a level street, positioned about two meters from one metal trash can along the same travel line. Both objects are fully visible at medium scale with open space between them and beyond the trash can, static camera, neutral realistic daylight.
```

冰融化案例应写成“完整冰块 + 干燥或仅微湿的邻近盘面”的单帧状态，而不是要求“冰块始终保持完整”。`reject_if` 可以列出“已经是大水洼”等审计条件，但不要把这些错误结果拼回正向 T2I prompt。

#### B. 用什么模型生成最合适

研究实验优先使用**固定版本的开放权重 T2I**，这样能保存 checkpoint、seed 和采样参数。当前更推荐 [Qwen-Image-2512 官方仓库](https://github.com/QwenLM/Qwen-Image) 作为高质量、提示遵循较强的首选；若只追求单张图最高质量且能接受服务依赖，可先用 Qwen-Image-2.0，但它不应与开放权重结果混在同一主表。资源或生态不匹配时，[FLUX.1-dev 官方实现](https://github.com/black-forest-labs/flux) 是可复现备选，但须记录其非商业权重许可。模型选择本身不是论文贡献，正式实验应在 10 条开发 prompt 上先盲测实体完整率、初态正确率和布局可用率，再固定一个生成器。

截至本次修订，本地 `/home/liuzhirui/model` 下未发现 Qwen-Image 或 FLUX checkpoint；已有 `Sana_1600M_512px_diffusers` 可用于先跑通 I0 管线，但 512px 与较小容量使它不宜直接作为“高质量 I0”主实验生成器。若 1–2 天窗口内来不及准备首选模型，应明确把 Sana 结果标成 smoke test，不能与后续高质量 T2I 结果混报。

不建议把“Wan 短视频的第一帧”作为默认 $I_0$：它会把视频模型自身的结果预置偏差带回首帧，并使 I0 质量与待评测基座耦合。只有没有独立 T2I 时，才把它作为降级方案并单独标注。

#### C. 候选生成、筛选与输入 Wan

建议固定如下协议：

1. 按 Wan 最终宽高比生成 `K=8` 个候选，短边至少 720 或长边至少 1024；先等比裁剪，再缩放到 Wan 输入尺寸，禁止拉伸变形。
2. 每个候选按预注册的 6 项二值规则筛选：实体/数量正确、初态正确、事件尚未发生、关键对象足够大、布局留有演化空间、无明显结构伪影。
3. 先用 VLM 做候选排序，再由人工只看首帧复核；选择过程不能查看任何后续视频结果。若全部不合格，整批按预定新 seeds 重生成，而不是挑选有利于某个方法的图片。
4. 同一条 prompt 的 M2–M6 和所有视频 seeds 共用同一张选定 $I_0$，避免把首帧差异误算成路由收益。
5. 保存原始图、裁剪图、生成模型完整 revision、正向 prompt、seed、采样步数、guidance、候选编号、筛选分数与人工否决原因。

若 benchmark 自带真实首帧且满足 pre-event 条件，应优先使用真实图；若真实首帧已经包含终态或关键实体不可见，则该样本不适合 I2V 公平对照，应标记或剔除，而不是用生成图悄悄替换。

最终流程为：

```text
original video text
  -> LLM initial-state compiler
  -> fixed T2I generates 8 candidates
  -> pre-registered image-only audit selects I0
  -> resize/crop once
  -> the same I0 enters all Wan I2V comparison methods
```

### 5.8 不使用初态图像的文本版本

为了保证与纯 T2V benchmark 公平比较，还必须报告：

```text
Wan T2V + c_sem + layer-step causal residual, without I0
```

如果只有 I2V 版本提升，说明主要收益来自首帧构图；如果文本版本也提升，才支持 causal routing 本身有效。

---

## 6. 1–2 天内的实际代码改动

### 6.1 目标文件

直接在官方 Wan2.2 分支上另建实验文件，不修改 PhysVid：

- [/home/liuzhirui/model/Wan2.2/wan/modules/model.py](/home/liuzhirui/model/Wan2.2/wan/modules/model.py)
- [/home/liuzhirui/model/Wan2.2/wan/textimage2video.py](/home/liuzhirui/model/Wan2.2/wan/textimage2video.py)
- 新建 `wan/textimage2video_ace.py`
- 新建 `experiments/ace_router/pilot_prompts.json`
- 新建 `experiments/ace_router/run_pilot.sh`

最安全的做法是复制出实验类/实验脚本，不覆盖官方推理实现。

### 6.2 `WanModel.forward` 增加两个 context

伪代码：

```python
def forward(
    self,
    x,
    t,
    context,
    seq_len,
    causal_pos_context=None,
    causal_neg_context=None,
    layer_scales=None,
    step_scale=0.0,
    y=None,
):
    sem = self.text_embedding(pad(context))
    pos = self.text_embedding(pad(causal_pos_context))
    neg = self.text_embedding(pad(causal_neg_context))

    for layer_id, block in enumerate(self.blocks):
        scale = step_scale * layer_scales[layer_id]
        x = block(
            x,
            context=sem,
            causal_pos_context=pos,
            causal_neg_context=neg,
            causal_scale=scale,
            ...
        )
```

### 6.3 `WanAttentionBlock` 只在指定层多算两次 cross-attention

```python
q = self.norm3(x)
r_sem = self.cross_attn(q, context, context_lens)

if causal_scale > 0:
    r_pos = self.cross_attn(q, causal_pos_context, context_lens)
    r_neg = self.cross_attn(q, causal_neg_context, context_lens)
    r = r_sem + causal_scale * (r_pos - r_neg)
else:
    r = r_sem

x = x + r
```

只有中间约 16 层额外计算两次 cross-attention，不需要额外完整 DiT forward，开销显著低于三分支全模型 CFG。

### 6.4 推理脚本编码三条文本

```python
context_sem = t5([prompt_original])
context_pos = t5([prompt_original + " " + causal_positive])
context_neg = t5([prompt_original + " " + causal_counterfactual])
```

标准 negative prompt/CFG 暂时保持官方配置。不要把长反物理列表再塞入 CFG 的 unconditional 分支。

### 6.5 step schedule

```python
def causal_step_scale(step_id, total_steps):
    ratio = step_id / total_steps
    if ratio < 0.30:
        return 1.0
    if ratio < 0.60:
        return 0.7
    if ratio < 0.80:
        return 0.3
    return 0.0
```

### 6.6 日志必须保存

每个视频旁保存 JSON：

```json
{
  "seed": 42,
  "original_prompt": "...",
  "pre_event_image_prompt": "...",
  "causal_positive": "...",
  "causal_counterfactual": "...",
  "layer_scales": [0.1, 0.1, "..."],
  "step_schedule": [1.0, 0.7, 0.3, 0.0],
  "lambda0": 0.5,
  "reference_image": "...",
  "i0_generation": {
    "model": "Qwen-Image-2512",
    "revision": "exact commit or model revision",
    "seed": 1001,
    "candidate_count": 8,
    "selected_candidate": 3,
    "selection_scores": [1, 1, 1, 1, 1, 1],
    "original_size": [1536, 864],
    "crop_resize": "center crop to target aspect ratio, then resize"
  },
  "checkpoint": "Wan2.2-TI2V-5B"
}
```

---

## 7. 滑板车案例的完整 MVP 输入

### 7.1 初态图像 prompt

```text
Side view of an upright black scooter two meters from a metal trash can on a clear street. Both objects are fully visible and separated, static camera, realistic daylight.
```

### 7.2 全层语义 prompt

```text
A scooter collides with a trash can, the scooter tilting to one side before stopping.
```

### 7.3 中层正确因果句

```text
The same scooter visibly contacts the trash can, then tilts while slowing, and remains stopped beside it.
```

### 7.4 中层最小反事实句

```text
The same scooter tilts and stops while it is still separated from the trash can; contact happens only afterward.
```

### 7.5 为什么它比 30 多行扩写更适合

- 每个条件只有一个职责；
- 初态首帧固定事件开始时的实体和空间关系，但不冻结后续姿态；
- 原始 prompt 不被替换；
- 正/反句共享主体词，差分主要落在接触和先后关系；
- 物理信号不进入所有层；
- 后期去噪不再被抽象逻辑持续扰动；
- 不要求文本编码器一次性满足几十个谓词。

---

## 8. 两天执行计划

### 8.1 Day 1 上午：4–6 小时

1. 复制官方 `WanTI2V` 与 `WanModel` 为 ACE 实验版本。
2. 加入三 context、layer scale 和 step scale。
3. 用一个 prompt、一个 seed、49 帧、较低分辨率完成 smoke test。
4. 验证 `lambda=0` 与原始 Wan 输出在数值/视觉上等价。
5. 记录额外显存和生成时间。

### 8.2 Day 1 下午：4–6 小时

选择 6 个机制 prompt，每个一个 seed，先筛 5 个层窗口：

```text
early : 0–9
mid-A : 10–19
mid-B : 14–23
late  : 20–29
all   : 0–29
```

只看：实体是否存在、事件是否发生、接触/顺序、画质是否明显崩。

如果所有窗口都无可见差异，先将 `lambda0` 从 0.25 提到 1.0；如果仍无差异，停止层路由，检查 T5 正/反句 embedding 和 cross-attention residual norm。

### 8.3 Day 2 上午：批量对照

从第 9 节固定选择 8 个机制 prompt × 2 seeds × 4 方法，共 64 个视频；先确认设置有效，再将冻结后的最佳设置扩展到 20 prompts × 3 seeds：

| 方法 | 初态图 | 文本 | 因果路由 |
|---|---|---|---|
| M0 | 无 | 原始短 prompt | 无 |
| M1 | 无 | 现有长物理扩写 | 无 |
| M2 | 有 | 原始短 prompt | 无 |
| M3 | 有 | 原始短 prompt | ACE 中层/step residual |

可选第五组：`M4 = 无图像 + ACE`，用于验证纯 T2V 泛化。

本地已有多 GPU，49 帧/较低面积的 5B pilot 在一天内可完成。第一轮不要直接跑 121 帧全量 VideoPhy2。

### 8.4 Day 2 下午：盲配对审阅

每个视频只回答七个问题：

1. 所有关键实体是否在其应该存在的阶段出现，合理相变/破裂后是否保持材料与事件连续性？
2. 初始状态是否正确？
3. 事件是否真正发生？
4. 接触/作用是否可见？
5. 原因是否先于响应？
6. 指定末态是否出现并保持？
7. 初态、实体连续性与画质是否不低于原始短 prompt？

不要用当前 caption-free PC 作为首轮主要选择标准。

---

## 9. 20 条多范围高质量精选 prompt

下面 20 条是**原始短语义 prompt**，不是长物理扩写。选择标准是：单镜头内可观察、关键实体少、触发和结果明确、3～5 秒内至少有一个可辨中间态，并能构造一个最小失败反事实。它们覆盖刚体接触、重力/摩擦、相变、不可逆变化、守恒转移、颗粒/扩散、弹性/耗散和多物理局部共存。

| ID | 范围 | 原始视频 prompt | 主要诊断 |
|---:|---|---|---|
| P01 | 碰撞—姿态 | `A black scooter rolls into a metal trash can, then tilts to one side and stops beside it.` | 接触先于倾斜；终态保持 |
| P02 | 两体碰撞 | `A red billiard ball rolls into a stationary blue billiard ball; after contact, the blue ball moves away while the red ball slows.` | 双实体身份；接触前后速度交换 |
| P03 | 击打反向 | `A tennis racket strikes an approaching tennis ball, sending the same ball back in the opposite direction.` | 小物体接触；方向反转 |
| P04 | 链式触发 | `A row of upright wooden dominoes falls in sequence after the first domino is pushed.` | 局部触发传播；时序链 |
| P05 | 摩擦耗散 | `A wooden block slides down a ramp onto a rough floor, slows continuously, and comes to rest.` | 连续减速；防止匀速/no-op |
| P06 | 重力坠落 | `A closed book is pushed past the edge of a shelf, falls, and lands open on the floor.` | 支撑消失后下落；中间态不可删除 |
| P07 | 失衡倾倒 | `A tall cardboard box is pulled sideways at its lower corner, tips over, and remains lying on one side.` | 施力—失衡—稳定终态 |
| P08 | 固液相变 | `A clear ice cube on a shallow plate gradually shrinks while a puddle of water grows around it.` | 冰减少/水增加；允许固态消失 |
| P09 | 软化融化 | `A pat of butter in a warm pan softens, loses its sharp edges, and spreads into a liquid pool.` | 轮廓连续变化；防瞬间替换 |
| P10 | 充气破裂 | `A rubber balloon inflates steadily until it bursts, leaving deflated fragments in place.` | 膨胀中间态；不可逆终态 |
| P11 | 脆性破裂 | `A glass bottle falls onto a hard tile floor, breaks on impact, and its fragments remain scattered.` | 接触触发破裂；禁止复原 |
| P12 | 脱落耗尽 | `A gust blows across a dry dandelion, detaching seeds that drift away while the seed head becomes depleted.` | 源减少；多小物体运动 |
| P13 | 液体转移 | `A transparent pitcher pours water into an empty clear glass; the pitcher level falls while the glass level rises.` | 源—流—目标耦合 |
| P14 | 有色液体 | `A bottle pours orange juice through a visible stream into a glass until the bottle holds less and the glass holds more.` | 流连续性；体积方向 |
| P15 | 颗粒堆积 | `Dry sand pours from a cup onto a flat table, emptying the cup as a conical pile grows below.` | 颗粒流；源减少/结果区增加 |
| P16 | 扩散混合 | `A drop of blue dye enters still clear water and gradually spreads outward, coloring an expanding region of the water.` | 局部到全局的连续扩散 |
| P17 | 弹性恢复 | `A hand presses a dry sponge flat, then lifts away as the same sponge gradually regains its shape.` | 压缩与恢复的有序双阶段 |
| P18 | 非弹性耗散 | `A rubber ball drops onto a hard floor and bounces several times to progressively lower heights before resting.` | 多次接触；峰值单调降低 |
| P19 | 撕裂 | `A sheet of paper pulled from both sides tears outward from a small notch and remains separated into two pieces.` | 裂纹传播；数量/身份连续性 |
| P20 | 多物理局部 | `On snowy ground beside a small campfire, a kettle pours coffee into a cup while the fire continues flickering.` | 液体与燃烧共存；雪地不应被改写 |

建议先用 `P01/P04/P08/P10/P13/P15/P18/P20` 做 8 条机制筛选，再在设置冻结后跑完整 20 条。P05、P06、P17 尤其容易被“几乎不动”或只给终态蒙混过关；P20 专门检测不同物理 residual 是否泄漏到无关区域。每条都应先用第一部分第 5.3 节的编译器生成 ACE 条件，再用第二部分第 3.2 节的 JSON 编译器生成 TRACE phases，而不是人工为不同方法写不等价的描述。

---

## 10. 成功与失败判据

### 10.1 进入下一阶段的最低成功线

M3 相对 M2 和 M0 至少满足：

- 严格事件完成率提升 ≥10 个百分点；
- cause-before-effect 正确率提升 ≥15 个百分点；
- 阶段条件化实体存在/连续率不下降；
- 初态连续性/画质配对偏好中，M3 不劣于 M2 的比例 ≥70%；
- no-op 比例下降，而不是靠更大随机运动取得分。

### 10.2 可能出现的结果及解释

| 结果 | 解释 | 下一步 |
|---|---|---|
| M2 提升实体，但 no-op 变多 | 图像锚有效，但压制运动 | 减弱/加噪首帧锚；加强早期因果 residual |
| M3 提升事件且画质稳定 | 核心假设成立 | 做可学习 sparse router 与更大诊断集 |
| M3 只增加随机运动 | residual 未真正绑定因果 | 加对象空间 mask/grounding，不增大 lambda |
| M3 画质下降 | 注入太晚、太强或层窗口错误 | 缩窄到 14–23；后 40% step 关闭 |
| 正/反 residual 接近 0 | T5 对顺序不敏感 | 改成状态 token encoder或小型 MLP/LoRA 对齐 |
| 长 prompt 与短 prompt 都失败 | 主要瓶颈可能是数据/模型容量 | 转向训练和 GPE-FM，而非继续 prompt engineering |
| 5B 成功、1.3B 失败 | 存在容量门槛 | 以 5B 为主，1.3B 只做下界 |

### 10.3 必须设置的止损点

若完整 20 prompts × 3 seeds 中：

- 接触/顺序无任何稳定提升；
- 正反 causal residual 的 norm 很小；
- 或提升完全由首帧构图解释；

则不要立即投入大规模训练。先将该方法降级为分析工具，转向显式状态 token、轨迹/接触监督或证据加权训练。

---

## 11. 这个 MVP 的创新性有多高

必须诚实区分“有意义的原型”和“足够投稿的完整方法”。

### 11.1 MVP 单独投稿不够

以下单点已经很拥挤：

- 外观与运动解耦；
- I2V 首帧保持身份；
- 中层表示更适合运动/推理；
- prompt refinement；
- training-free 多分支 guidance；
- motion graph 或阶段分解。

[Training-free Motion Factorization](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Training-free_Motion_Factorization_for_Compositional_Video_Generation_CVPR_2026_paper.html) 已经使用 motion graph 和不同 motion guidance。  
[LiON-LoRA](https://arxiv.org/abs/2507.05678)、SMRABooth 等已经研究 appearance/motion 干扰和稀疏位置/时间注入。  
[PhyT2V](https://openaccess.thecvf.com/content/CVPR2025/html/Xue_PhyT2V_LLM-Guided_Iterative_Self-Refinement_for_Physics-Grounded_Text-to-Video_Generation_CVPR_2025_paper.html) 已表明 LLM 物理 prompt refinement 可以成为强基线。  
[VIPER](https://arxiv.org/abs/2607.23472) 更进一步用目标图像 + 参考视频转移物理过程。

因此，“我们把图像、文本和物理信息放到不同地方”本身不够新。

### 11.2 可能形成论文贡献的真正核心

更有区分度的叙事是：

> **The initial image specifies a boundary state and continuity cues; physics specifies which state variables should depart from that state. Treating the whole appearance as invariant conflicts with genuine deformation, disappearance, and phase change. We therefore protect only event-irrelevant properties while routing causal changes to the appropriate variables and supports.**

完整论文应有四部分：

1. **Condition-role conflict 现象。**系统测量初态/连续性条件、语义实体和物理变化在 30×50 layer-step 网格中的 residual norm、cosine conflict 和输出因果效应，并区分应保护属性与应变化属性。
2. **Counterfactual causal residual。**不是泛化的 motion prompt，而是正确事件与最小反事实之间的差分。
3. **Role-aligned sparse routing。**外观、语义、因果变化在层深、step 和空间对象区域上的不同作用域。
4. **Grounded causal evaluation。**必须击败 no-op、结果预置、时间反转和错误实体，而不是只提高 caption-free PC。

### 11.3 与 VIPER 的边界

VIPER 使用参考视频作为物理示范，研究 visual in-context physics transfer。ACE-Router 首版不需要参考物理视频：

- 输入是一张外观/初始状态图；
- 物理来自紧凑文本正反事实；
- 目标是解决条件角色冲突和因果顺序；
- 不做“把参考视频的物理行为迁移到目标图像”。

若后续加入参考视频，必须重新评估与 VIPER 的重合，不能把它当作自然扩展而不讨论。

---

## 12. 从 MVP 扩展到顶会完整工作的路线

### 12.1 Layer-Step Conflict Atlas

在 Wan1.3B 和 Wan5B 上记录每层、每 step：

\[
N_{sem}^{\ell,k}=\|r_{sem}\|,
\quad
N_{causal}^{\ell,k}=\|\Delta r_{causal}\|,
\]

\[
C^{\ell,k}
=
\frac{\langle r_{sem},\Delta r_{causal}\rangle}
{\|r_{sem}\|\|\Delta r_{causal}\|+\epsilon}.
\]

还要做 intervention：只在一个窗口注入 residual，观察实体、接触、顺序和画质变化。只有激活可视化没有因果干预，说服力不够。

### 12.2 从固定 schedule 到可学习稀疏 router

不要训练大网络。只学习少量标量：

```text
30 layer gates × 4 denoising bins = 120 parameters
```

不为稀疏和 total variation 另加损失；把 layer gate 参数化为少量连续 Gaussian/triangular basis，把 denoise gate 限制为 4 bins，并用 event top-k/阈值直接控制激活数量。这样既能形成连续窗口，又遵守总损失最多四项，并可验证最佳窗口是否真位于中层/早期而不是手工选择。

### 12.3 对象级空间路由

当前因果 residual 对所有视频 token 生效，仍可能改变背景和无关人物。后续用原始 prompt 的 cross-attention 或检测器得到 Actor/Tool/Patient soft mask：

\[
h' = h+r_{sem}+M_{event}\odot\lambda\Delta r_{causal}.
\]

只在相关对象和接触邻域施加物理变化，背景和静止外观由语义/图像路径保留。

### 12.4 紧凑 Causal State Compiler

让 LLM 不再写文章，只输出固定 schema：

```json
{
  "entities": ["scooter", "trash can"],
  "protected_properties": ["same scooter category", "same trash can", "street and camera"],
  "transition_variables": ["scooter position", "scooter speed", "scooter pose"],
  "trigger": "visible scooter-can contact",
  "response": "scooter tilts and slows",
  "terminal": "scooter remains stopped beside can",
  "counterfactual": "scooter tilts before contact"
}
```

随后只把 `trigger + response + terminal` 压成一句 \(c_+\)，把 `counterfactual` 压成一句 \(c_-\)。Schema 用于审计，不把 JSON 全部喂给视频模型。

### 12.5 训练版 Grounded Evidence Alignment

如果 training-free residual 有趋势但不稳定，再进入训练：

- 冻结大部分 Wan；
- 只训练 cross-attention LoRA、120 个 gates 或小型 state projector；
- 在同一个 `L_gen` 内对 Actor/Tool/Patient 与 trigger→response 时空 tube 做 evidence-balanced flow matching，同时保留 tube 外预算避免画质下降；
- 将 no-op、倒序和结果预置放入统一反事实池，仍只使用第二部分第 8.3 节定义的四项总损失。

这一步可以继承既有报告中 GPE-FM 的证据加权思想，但它是建立在 Wan base 上的训练目标，不依赖 PhysVid 结构。

### 12.6 多基座与泛化

至少报告：

- Wan2.1-T2V-1.3B：低容量文本版；
- Wan2.2-TI2V-5B：主模型；
- 一个不同架构开放模型：验证 layer-step 角色路由不是 Wan 特例；
- seen mechanism / unseen entity combination；
- T2V 与 I2V 分开报告。

---

## 13. 推荐的论文问题与标题方向

### 13.1 最稳妥的问题定义

> 如何让视频 DiT 同时利用初态与选择性不变量、高层语义和因果物理变化，既不让异质条件互相覆盖，也不把参与事件的对象错误冻结？

### 13.2 不建议的 claim

- “模型学会了通用物理定律”；
- “中间层就是物理层”；
- “图像信息天然提高物理一致性”；
- “LLM 扩写失败，所以文本物理无效”；
- “只靠一个 training-free trick 解决世界模型物理”。

### 13.3 推荐 claim

> 我们识别并缓解了 initial-state/continuity conditioning、semantic grounding 与 causal state change 之间的条件角色冲突；通过层深—去噪阶段—输出时间—对象区域的稀疏路由，保护事件无关属性，同时允许指定状态变量发生连续变化。

### 13.4 标题草案

```text
Different Conditions Belong in Different Places:
Role-Aligned Causal Routing for Physically Coherent Video Diffusion
```

```text
Preserve What It Is, Control What It Does:
Appearance-Anchored Causal Event Routing for Video Generation
```

```text
An Effect Should Follow Its Cause:
Counterfactual Residual Routing in Video Diffusion Transformers
```

### 13.5 Venue 适配

| Venue | 更看重什么 | 当前路线需要补什么 |
|---|---|---|
| CVPR / ICCV | 视觉生成质量、清楚架构、跨模型实验、人评和可视化 | 最自然；需强 qualitative、严格对照和公开代码/数据 |
| ICML | 更一般的学习问题、机制分析、算法普适性 | 需要更正式的 condition conflict 定义、可学习 router、跨架构规律与更强理论/统计证据 |

当前路线更自然地面向 CVPR/ICCV。若只做 Wan 上的启发式窗口，ICML 说服力不足。

---

## 14. 与现有失败报告的对应关系

| 已观察问题 | ACE-Router 首版针对方式 | 仍不能直接解决的部分 |
|---|---|---|
| 关键实体缺失/身份漂移 | pre-event image + 原始语义全层保留 | 首帧本身实体错误时无效 |
| 无动作/no-op | 早—中去噪的因果 residual | 基座完全不会该动作时无效 |
| 结果先于原因 | correct vs counterfactual causal direction | 文本编码器对先后不敏感时需训练 |
| 接触缺失 | 中层强调 visible contact → response | 小物体被 VAE 压缩时仍困难 |
| 长 prompt 画质下降 | 不再全局输入长物理文本 | 初态首帧质量仍决定上限 |
| 物体类别替换 | 图像和 original prompt 双锚 | 运动中仍可能变形，需要对象 mask/训练 |
| 时间反转/末态回生 | c+ 包含 terminal persistence，c- 包含反顺序 | 5 秒长时记忆仍可能不足 |
| 流体/破裂机制替换 | 只提供事件方向 | 没有材料级训练时提升可能有限 |
| caption-free PC 高估 | 使用 grounded seven-question rubric | 需另建正式 evaluator |

---

## 15. 最后的研究判断

这条路线最值得保留的不是“多加一种条件”，而是一个更一般的原则：

> **首帧是初始边界，不是整段视频都应复制的外观模板；只有事件无关的实体、区域和属性才应保持。物理负责在特定时间、区域和状态变量上产生变化，高层语义负责把变化绑定到正确实体。三者不应被编码成一个同质 prompt，也不应在全部 DiT 层、全部去噪阶段、全部输出帧和全部空间区域以同样强度作用。**

原 ACE-Router 只能测试“中层正反事实 residual 是否有用”，不能充分测试“物理演化是否发生”。修订后的 1–2 天 TRACE-Lite 增加输出视频时间相位与粗空间事件 tube，才是当前推荐的最低完整版本。它仍不需要训练、参考物理视频、模拟器、奖励模型或 PhysVid。

但必须保持清醒：TRACE-Lite 仍是立项实验，不是最终论文贡献。简单的事件分解已有 CoECT，token-level anisotropic expert 已有 ProPhy。真正能支撑 CVPR/ICCV/ICML 的，应是后续 TRACE-Field 对开放词汇“状态转移场”的建模、四轴路由机制证据、多物理局部组合、middle-deletion/时间倒序反事实训练，以及在不牺牲外观和语义的前提下稳定提高严格演化完成率。

---

# 第二部分：结合 Physics Evolution 与 Anisotropic Generation 的修订方案

## 1. 两篇新增工作的准确启示

### 1.1 CoECT：知道物理终态不等于生成物理演化

[Chain of Event-Centric Causal Thought for Physically Plausible Video Generation（CoECT，CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Chain_of_Event-Centric_Causal_Thought_for_Physically_Plausible_Video_Generation_CVPR_2026_paper.html) 明确指出，已有方法常把物理现象压缩为静态 prompt 所定义的单一时刻。它的核心不是简单“把 prompt 写得更长”，而是：

1. 用 Physics-driven Event Chain Reasoning 将现象拆成因果有序事件；
2. 用物理公式和动态 scene graph 约束相邻事件；
3. 用 Transition-aware Cross-modal Prompting 将事件链转成语义条件和逐事件视觉关键帧；
4. 在 VAE latent 中插值关键帧，并加噪作为视频生成的视觉先验。

它的消融尤其重要：去掉 Interactive Keyframe Synthesis 后，PhyGenBench 平均分从 `0.66` 降到 `0.55`，即下降 `0.11` 绝对值、约 `16.7%` 相对值。作者据此强调，专门为不同物理阶段生成视觉关键帧，对跨帧动态锚定非常重要。[CoECT 方法与消融](https://arxiv.org/html/2603.09094v2#S4.SS4)

因此下面这个判断是成立的：

\[
\text{Physics Knowledge}\neq\text{Physics Evolution}.
\]

更严格地说，至少应区分：

\[
\text{endpoint recognition}
\neq
\text{transition realization}
\neq
\text{continuous dynamics}.
\]

模型知道冰和水、碰撞和倾倒，并不能保证：

- 初态确实出现；
- 中间状态可见；
- 变化方向单调或至少合理；
- 原因发生在结果之前；
- 终态不会在后面回生；
- 质量、体积、动量等耦合量近似一致。

### 1.2 CoECT 给我们的不是“复制关键帧流水线”

CoECT 已经占据了“公式约束事件链 + 逐事件图像编辑 + keyframe interpolation”的明确方法空间。若我们的远期方案也只是：

```text
LLM 拆事件 -> 生成三张关键帧 -> 插值 -> Wan 生成
```

创新性会很弱，而且还会引入：

- 多次图像编辑造成身份、材质和相机漂移；
- 错误关键帧对后续生成形成强错误锚；
- 线性 latent 插值不等于真实非线性动力学；
- 外部 T2I/编辑器的成本和闭源依赖；
- 关键帧数量固定时，对碰撞瞬态与连续相变难以统一。

上面这些是基于其方法结构作出的研究推断，不是 CoECT 作者的原文结论。我们的借鉴点应是“显式建模状态转移”，实现方式则改为 **DiT 内部的阶段—区域状态转移 residual field**。

### 1.3 ProPhy：物理指导必须能在 token 级各向异性响应

[ProPhy: Progressive Physical Alignment for Dynamic World Simulation](https://arxiv.org/abs/2512.05564) 对已有方法的概括与用户给出的理解一致：

- VideoREPA、偏好优化等方法主要在训练阶段把物理先验内化进模型，推理时缺少显式物理条件；
- WISA 虽然用 Mixture-of-Physics-Experts 提供显式物理类别，但主要是视频级路由；面对局部现象或多种物理共存时容易过粗；
- ProPhy 先通过视频级 Semantic Expert Block 选物理先验，再通过 token-level Refinement Expert Block 做细粒度路由；
- 它用 VLM attention 构造 token-level 物理区域监督，让不同位置对物理先验产生各向异性响应。[ProPhy 方法](https://arxiv.org/html/2512.05564v2#S3)

需要准确说明：ProPhy 的 token 数量本身包含压缩后的视频时间和空间维度，因此不能简单声称它“完全没有时间 token 级指导”。更准确的区别是：

> ProPhy 主要学习“某 token 对应哪一种物理类别/专家”，但没有显式规定该现象应沿着怎样的状态序列，从哪个视频阶段过渡到哪个阶段，也没有专门针对 middle deletion、结果预置和状态回生定义状态转移目标。

它自己也承认，VLM 标注区域存在噪声，简单的区域级物理分类只能捕捉粗粒度表面模式；未来需要结合控制方程获得更可解释的物理知识。[ProPhy 局限](https://arxiv.org/html/2512.05564v2#S5.SS1)

### 1.4 两篇论文合起来给出的缺口

| 问题 | CoECT 重点解决 | ProPhy 重点解决 | 仍值得研究的交叉缺口 |
|---|---|---|---|
| 物理事件是否有顺序 | 是，事件链 | 非主要显式结构 | 状态链怎样直接约束 DiT token |
| 中间态是否出现 | 逐事件关键帧 | 间接学习 | 不依赖关键帧编辑器的 transition field |
| 物理作用在哪里 | scene graph/视觉提示 | token-level router | 由 actor–trigger–patient 状态边确定区域 |
| 多物理共存 | 可拆多个事件 | 多专家、token 路由 | 多状态转移场的局部组合和冲突消解 |
| 开放世界物理 | LLM/公式检索 | 固定专家集合 | 开放词汇 transition operator，而非固定 law ID |
| 初态与选择性不变量 | 关键帧连续编辑 | 主干能力保留 | 初态边界、受保护属性与 transition variables 显式分权 |
| 失败反事实 | 不是核心训练单位 | 不是状态链反事实 | 结果预置、瞬间跳变、时间倒序、中间态删除 |

我们的合理切入点不是重复任一列，而是交叉缺口：

> **把物理表示为开放词汇的局部状态转移，并在 layer、denoise、output-time、space 四个轴上路由到 Wan token；同时显式区分初态、应保护属性和必须变化的状态变量。**

---

## 2. 对原 ACE-Router 的缺口审计

### 2.1 原 ACE 已经解决了什么

- 原始短 prompt 始终保留，避免长 prompt 替换语义主线；
- 首帧提供初始状态、实体连续性和布局锚，但不把事件对象整体定义为不变量；
- 正确—反事实 residual 比长篇物理说明更紧凑；
- layer gate 避免物理条件覆盖所有 Transformer 深度；
- denoise gate 避免抽象条件在细节收敛阶段持续扰动。

这些部分仍然保留。

### 2.2 原 ACE 没有解决什么

原公式：

\[
r_{\ell,k}=r_{sem}+\lambda_{\ell,k}(r_+-r_-)
\]

对同一层中的所有输出视频 token 使用相同的条件差分。它存在两个结构性缺口。

#### 缺口 A：没有 output-time assignment

`g_step(k)` 只控制第几个去噪步骤介入，不控制第几个输出帧应该是什么状态。因此：

- `terminal state` 可能覆盖第一帧；
- `contact` 和 `tilt` 可能同时出现在全片；
- `melting` 可能退化成冰—水的一次语义替换；
- `remains stopped` 可能压制前面的接近运动，形成 no-op。

#### 缺口 B：没有 spatial support assignment

`r_+-r_-` 加到背景、无关对象和所有角色 token。即使物理文本只说滑板车，残差仍可能改变垃圾桶、道路、人物和相机；多物理场景中，不同 residual 还会互相覆盖。

### 2.3 覆盖矩阵

| 失败 | 初态首帧 | ACE layer/step | ACE 正反事实 | 仍需 TRACE |
|---|---:|---:|---:|---:|
| 无因果依据的实体缺失/身份漂移 | 强 | 弱 | 弱 | protected-property mask 可继续增强 |
| 结果预置在开场 | 弱 | 否 | 部分 | 是，early-phase counterfactual |
| 中间演化缺失 | 否 | 否 | 部分 | 是，phase-local state deltas |
| 瞬间跳变 | 否 | 否 | 部分 | 是，overlapping temporal gates |
| 时间倒序/状态回生 | 否 | 部分 | 部分 | 是，late-phase anti-reversal |
| 无关区域被改变 | 部分 | 否 | 否 | 是，event support mask |
| 多物理局部共存 | 否 | 否 | 否 | 是，多 transition fields |
| 精确守恒/连续方程 | 否 | 否 | 否 | MVP 不解决，远期 edge coupling |

结论：ACE 不应被删除，但应降级为 `global causal baseline`。如果 TRACE 不能击败 ACE，就没有证据证明新增时间/空间结构真正有价值。

---

## 3. 修订后的 1–2 天方案：TRACE-Lite

### 3.1 设计目标

TRACE-Lite 只增加两种廉价先验：

1. **什么时候变：**输出视频 latent 上的 3 个重叠 phase gates；
2. **哪里变：**手工定义的粗事件 ROI/tube。

它暂时不训练 router，不引入物理专家，不调用 VLM 做在线 token 标注，也不要求生成多个关键帧。

### 3.2 输入 schema 与自动编译提示词

每条 pilot prompt 保存一个审计用 JSON，但它必须分两步产生：

1. **文本编译：**只根据原始视频 prompt 生成实体、状态变量、三阶段正反事实和 support 类型；
2. **首帧定位：**$I_0$ 生成后，再由 VLM/人工填写归一化 ROI。

仅凭文本不知道对象在实际首帧中的像素位置，因此让 LLM 直接输出 `[[0.10, 0.35, ...]]` 属于伪精确，会污染实验。文本阶段的 `roi` 必须为 `null`。推荐 schema 为：

```json
{
  "schema_version": "trace-lite-v1",
  "original_prompt": "...",
  "pre_event_image_prompt": "...",
  "pre_event_image": null,
  "entities": [
    {"id": "actor_1", "source_phrase": "...", "role": "actor", "visible_at_i0": true},
    {"id": "patient_1", "source_phrase": "...", "role": "patient", "visible_at_i0": true}
  ],
  "state_partition": {
    "initial_conditions": ["..."],
    "protected_properties": ["camera", "background", "unaffected entity identity"],
    "transition_variables": [
      {
        "entity": "actor_1",
        "variable": "pose",
        "initial": "upright",
        "direction": "upright_to_tilted",
        "terminal": "tilted and stopped"
      }
    ]
  },
  "phases": [
    {
      "id": "precondition",
      "center": 0.15,
      "sigma": 0.22,
      "positive": "...",
      "counterfactual": "...",
      "counterfactual_type": "outcome_preset",
      "affected_entity_ids": ["actor_1", "patient_1"],
      "support_type": "motion_corridor",
      "support_hint": "actor start region and path toward patient",
      "roi": null
    },
    {
      "id": "transition",
      "center": 0.50,
      "sigma": 0.24,
      "positive": "...",
      "counterfactual": "...",
      "counterfactual_type": "trigger_missing",
      "affected_entity_ids": ["actor_1", "patient_1"],
      "support_type": "contact_roi",
      "support_hint": "interface between actor and patient plus actor response region",
      "roi": null
    },
    {
      "id": "terminal",
      "center": 0.85,
      "sigma": 0.22,
      "positive": "...",
      "counterfactual": "...",
      "counterfactual_type": "terminal_reversal",
      "affected_entity_ids": ["actor_1", "patient_1"],
      "support_type": "result_region",
      "support_hint": "actor and patient around the intended terminal interaction region",
      "roi": null
    }
  ],
  "audit": {
    "required_opening_observations": ["..."],
    "required_intermediate_observations": ["..."],
    "required_terminal_observations": ["..."],
    "reject_opening_if": ["terminal result is already visible"],
    "ambiguities": [],
    "compiler_warnings": []
  }
}
```

其中 `protected_properties` 不能笼统写成“整个 actor 外观不变”；例如冰融化时，冰的尺寸、轮廓、相态和固态存在性必须放入 `transition_variables`，相机、盘子、背景以及材料来源连续性才进入保护项。

#### A. 从每条原始 pilot prompt 生成审计 JSON 的完整提示词

系统提示词：

```text
You are the TRACE-Lite audit-JSON compiler for a controlled physical video
generation experiment. Convert exactly one original short video prompt into a
three-phase causal state-transition specification. The JSON is for experiment
construction and auditing; it will not be sent wholesale to the video text encoder.

Core principles:
- The initial image is a t=0 boundary condition, not a promise that the whole object
  keeps the same appearance. Protect only event-irrelevant entities/properties.
- Put every property that should move, deform, shrink, disappear, change phase,
  transfer, break, or change amount into transition_variables.
- Physics clauses describe visible state changes, not abstract laws.

Compilation rules:
1. Copy original_prompt verbatim. Preserve all explicit entities, counts, materials,
   attributes, scene facts, and requested outcomes. Do not invent new causal objects.
2. Create exactly three phases: precondition, transition, terminal, with fixed
   (center, sigma) values (0.15,0.22), (0.50,0.24), (0.85,0.22).
3. Each positive clause is one plain English sentence of at most 26 words and describes
   only what should be visibly true/change in that phase.
4. Each counterfactual keeps the same entities and nearly the same vocabulary, but
   describes exactly one nearest failure affirmatively. Allowed types are:
   outcome_preset for precondition; trigger_missing, instant_replacement, or
   middle_deletion for transition; terminal_reversal, unjustified_entity_loss, or
   no_op for terminal. A disappearing object is a failure only when the target state
   requires it to remain; melting, pouring, burning, or depletion may legitimately
   reduce or remove the source state. Do not use "not", "no", "without", "avoid",
   "must", or "should".
5. The precondition pair tests whether the result is wrongly present at the opening.
   The transition pair tests trigger-response binding or visible intermediate change.
   The terminal pair tests completion and persistence versus reversal or unjustified
   loss of an entity that the terminal state requires.
6. Use only observable nouns, spatial relations, actions, and state changes. Do not
   output formulas, physical-law explanations, style text, quality text, or camera
   motion in phase clauses.
7. Classify each entity role as actor, patient, tool, result_material, environment,
   or unaffected, and mark whether it should be visible at I0. A result_material may
   be a later phase/state of an explicit source material; do not treat it as a newly
   invented independent object. Reuse stable entity IDs in affected_entity_ids.
8. protected_properties may contain camera, background, unaffected entities, and
   identity/material provenance when appropriate. Never place a required changing
   property in protected_properties.
9. Choose support_type from static_roi, motion_corridor, contact_roi, interface_roi,
   or result_region, and give a textual support_hint. Since no image is provided,
   every phase roi MUST be null. Never invent coordinates.
10. Create a 35-70 word positive pre_event_image_prompt that shows all necessary
    entities in the initial state, clearly visible and laid out with room for the
    event. It contains one moment only and does not narrate the future transition.
11. audit observations must be concrete enough for a human or VLM to judge from frames.
12. If the prompt is under-specified, use the smallest conventional visual
    interpretation and record it in audit.ambiguities. Do not hide uncertainty.

Return strict JSON only with exactly the trace-lite-v1 schema shown below. Use valid
double-quoted JSON, no comments, no Markdown, and no additional keys:
{{TRACE_LITE_V1_SCHEMA}}
```

在实际调用时，把上文完整 schema 原样替换 `{{TRACE_LITE_V1_SCHEMA}}`，用户消息为：

```text
Compile this pilot prompt into TRACE-Lite audit JSON:
{{ORIGINAL_VIDEO_PROMPT}}
```

程序化验证必须检查：三个 phase 是否齐全且顺序固定；所有 `roi` 是否为 `null`；实体 ID 是否都能追溯到原 prompt；保护项与转移变量是否冲突；正反句是否共享实体；每个反事实是否只改变一个关系；JSON 是否能被标准 parser 直接读取。任何一项失败都重试，不能静默修补后不留记录。

#### B. 看到 \(I_0\) 后填写 ROI 的多模态提示词

把选定首帧和上一步 JSON 一起交给 VLM：

```text
You are localizing event supports in the attached pre-event image for TRACE-Lite.
Use the supplied draft JSON. Do not change original_prompt, entities, state_partition,
phase texts, centers, sigmas, or counterfactual types.

For each entity whose visible_at_i0 is true, return a normalized [x1,y1,x2,y2] box in [0,1],
where (0,0) is the image top-left. Then propose one coarse phase ROI:
- static_roi: the changing object plus 10-20% context;
- motion_corridor: actor start box, patient box, and the corridor connecting them;
- contact_roi/interface_roi: the intended boundary between interacting entities plus
  both adjacent object parts;
- result_region: the area where the changed material/state is expected to appear.

The ROI is an intended conditioning support, not a claim about unseen future pixels.
If an entity is missing, occluded, too small, or ambiguous, set image_usable=false and
explain why. Do not invent a box. Set manual_review_required=true for every motion
corridor or future result region.

Return strict JSON only:
{
  "image_usable": true,
  "entity_boxes": [{"entity_id":"...","box":[0,0,1,1],"confidence":0.0}],
  "phase_rois": [{"phase_id":"precondition","roi":[[0,0,1,1]],"confidence":0.0}],
  "manual_review_required": true,
  "warnings": []
}

DRAFT_TRACE_JSON:
{{DRAFT_TRACE_JSON}}
```

VLM 输出只能作为初稿。人工需叠加可视化框复核，尤其是 motion corridor 与未来 result region；最终 JSON 应额外保存 `roi_source`、VLM 名称/revision、人工修改前后坐标和审核者。JSON 只用于构造条件与实验留档，送进 Wan T5 的只有 `original_prompt` 与各 phase 的短 `positive/counterfactual` 字符串。

### 3.3 输出视频时间相位及直观作用

令视频 latent 时间长度为 \(F'\)。例如 49 帧、VAE 时间 stride 为 4 时，\(F'=13\)。对第 \(m\) 个 phase 设置中心 \(\mu_m\) 和宽度 \(\sigma_m\)：

\[
\tilde\alpha_m(f)
=
\exp\left(-\frac{(f/(F'-1)-\mu_m)^2}{2\sigma_m^2}\right),
\]

\[
\alpha_m(f)
=
\frac{\tilde\alpha_m(f)}{\sum_j\tilde\alpha_j(f)+\epsilon}.
\]

首版使用：

```text
phase centers: [0.15, 0.50, 0.85]
sigma:         [0.22, 0.24, 0.22]
```

它的直观含义是：**不是把一条 prompt 分三次生成，也不是在第 17 个去噪 step 换一句话；而是在同一次 Wan forward 中，让不同输出时间 token 以不同权重接收三条阶段 residual。** 对上述参数，归一化权重大致为：

| 输出视频归一化时间 | precondition | transition | terminal | 直观作用 |
|---:|---:|---:|---:|---|
| 0.00（开场） | 0.874 | 0.126 | 0.001 | 强化正确初态，几乎不注入终态 |
| 0.25（前段） | 0.598 | 0.386 | 0.016 | 初态逐渐让位给变化开始 |
| 0.50（中段） | 0.180 | 0.639 | 0.180 | 中间变化/接触响应占主导 |
| 0.75（后段） | 0.016 | 0.386 | 0.598 | 从变化过渡到完成状态 |
| 1.00（结尾） | 0.001 | 0.126 | 0.874 | 强化终态保持，抑制回生 |

以 49 帧冰融化为例，VAE 压缩后约有 13 个时间 latent token：前几个 token 主要接收“完整冰块、几乎无水”，中间 token 主要接收“冰持续变小、水域扩大”，最后几个 token 主要接收“小残块/水持续存在”。若没有 $\alpha_m(f)$，所有 13 个 token 都会同时看到“完整冰”“正在变小”“已经融完”，模型就可能从第一帧直接画水，或在某一帧突然替换。

重叠 gate 的作用是让相邻状态平滑交接。若用三个硬切片，边界两侧会突然从一条 residual 跳到另一条，容易造成闪烁或形变；重叠后，一个 token 可以同时含有“仍有较大冰块”和“水开始增加”的连续过渡信息。归一化 $\sum_m\alpha_m(f)=1$ 还避免三条 residual 在重叠区简单叠成三倍强度。

这套权重只表达**先后分配**，不是精确物理速度。不同事件应允许调整中心和宽度：碰撞的 transition 可更窄，缓慢融化可更宽；但第一轮固定参数，避免逐样本手调造成结果不可比。`output video time f` 与 `denoising step k` 仍是两个正交轴：前者决定哪段成片接收哪种状态，后者决定在从噪声求解到细节的哪个阶段施加强度。

### 3.4 局部事件 support

对每个 phase 提供一个粗空间框或 tube，缩放到 latent patch grid，膨胀 10%～20%，再做 Gaussian blur：

\[
\beta_m(f,x,y)\in[0,1].
\]

首版允许三种 mask：

| 类型 | 适用场景 | 构造方式 |
|---|---|---|
| static ROI | 融化、燃烧、局部破裂 | 固定物体框扩张 |
| motion corridor | 滑板车接近、球碰撞 | 起点框、终点框和连线的膨胀并集 |
| contact ROI | 敲击、倾倒、接触触发 | actor/patient 框交界附近的扩张区域 |

手工 ROI 不适合作为论文最终方法，但非常适合验证“local routing 是否值得训练”。第一天就上 GroundingDINO/SAM/trackers，会把定位错误和方法错误混在一起。

### 3.5 阶段正反事实 residual

每个 phase 只描述一个可观察变化，并配一个最接近的失败反事实：

\[
\Delta r_m
=
\operatorname{CA}(q,c_m^+)
-
\operatorname{CA}(q,c_m^-).
\]

三类反事实分别对应：

| phase | 正确目标 | 失败反事实 |
|---|---|---|
| precondition | 正确初态/尚未发生结果 | outcome already present |
| transition | 可见中间变化/触发响应 | instant replacement or missing trigger |
| terminal | 结果完成并保持 | reversal/regrowth/rebound，或终态本应存在的实体无依据丢失 |

与原长 prompt 不同，反事实不会全局进入 official negative prompt；它只用于局部 residual difference。

### 3.6 四轴路由公式

设 \(i=(f,x,y)\)，完整 TRACE-Lite residual 为：

\[
r^{TRACE}_{\ell,k,i}
=r_{sem,\ell,k,i}
+\lambda_0g_{layer}(\ell)g_{denoise}(k)
\sum_m
\underbrace{\alpha_m(f)\beta_m(f,x,y)}_{G_m(i)}
\Delta r_{m,\ell,k,i}.
\]

四个轴各自承担不同职责：

```text
layer l       -> 哪种表征深度接收物理变化
denoise k     -> 在生成规划的哪个阶段介入
video time f  -> 哪段输出视频应该处于哪个物理阶段
space x,y     -> 哪些对象/接触区域应该发生变化
```

### 3.7 初态、受保护属性与转移变量

首版不额外向背景注入“保持不变”文本，因为这可能增加 no-op。空间 gate 只做最简单的 complement rule：

\[
G_{event}=\min(1,\sum_mG_m),\qquad
G_{background}=1-G_{event}.
\]

- `event region`：允许接受 causal residual，其中对象的位置、姿态、尺寸、形状、相态、数量或完整性可按事件变化；
- `protected complement`：只走原 Wan 的 self-attention、原始短文本和 I2V 条件，不接收新增 residual。

这不是冻结背景 latent；背景仍可自然变化，只是不被额外的物理 prompt 主动改写。也不能把 `protected complement` 直接解释成完整“不变量”：

- 空间外保护只能减少 residual 泄漏，不能保证像素或 feature 完全不变；
- event region 内同时包含应保持的实体连续性和应变化的物理状态，单一空间 mask 无法做属性级硬分离；
- 冰融化时，冰块框内的尺寸、轮廓和相态属于 transition variables，不能因来自 $I_0$ 就被保护；盘子、相机和背景才是此事件的主要保护对象；
- 滑板车碰撞时，滑板车类别/颜色通常应连续，位置、速度和姿态应变化。

因此 TRACE-Lite 只主张**局部 residual 隔离 + 基座连续性先验**。严格的属性级 invariant–variant 分权留给 TRACE-Field，通过 state schema、角色化 projector 或 feature projection 实现，不能由一个 `1-G_event` mask 冒充。

### 3.8 多物理场景的最小组合

对物理过程 \(p=1,\dots,P\)：

\[
r^{TRACE}=r_{sem}+\lambda
\sum_{p,m}\hat G_{p,m}\Delta r_{p,m},
\]

其中：

\[
\hat G_{p,m}
=
\frac{G_{p,m}}
{\max(1,\sum_{p',m'}G_{p',m'})}.
\]

如果咖啡、火焰和雪地同时出现：

- liquid-transfer residual 只进咖啡流、壶口和杯子区域；
- combustion residual 只进火焰与燃料邻域；
- 雪地背景不接收这两个 residual；
- 人体只保留原语义条件，除非 prompt 明确包含受热/着火等交互。

这只是 overlap normalization，不是完整多物理耦合；火加热容器再影响液体等连锁过程留到 TRACE-Field。

---

## 4. 两个完整 TRACE-Lite 案例

### 4.1 冰块逐渐融化

原始语义：

```text
An ice cube gradually melts as the temperature rises.
```

首帧：完整冰块放在浅盘中央，没有大滩水，静态相机。

| phase | video-time | \(c_m^+\) | \(c_m^-\) | ROI |
|---|---:|---|---|---|
| precondition | 0–35% | The intact ice cube is clearly visible with almost no surrounding water. | A finished puddle is already present at the beginning. | 冰块框 + 小范围盘面 |
| transition | 20–75% | The same ice cube visibly becomes smaller while a thin puddle grows around it. | A finished puddle suddenly replaces the intact ice in a single step. | 冰块框逐渐向盘面扩张 |
| terminal | 65–100% | Only a small remnant remains as the surrounding water persists and does not turn back into ice. | The ice becomes larger again or reappears after melting. | 小冰块 + 水区域 |

这个设计仍不能严格保证质量/体积守恒，但它比单一 `melting` 标签多提供了三个可验证信号：冰变小、水增加、不能回生。

建议额外记录人工 progress：

```text
ice-size:    [1.0, ~0.6, ~0.2]
water-area:  [0.0, >0.0, larger]
```

这些数字首版只用于评测，不直接输入模型。

### 4.2 滑板车撞垃圾桶

原始语义：

```text
A scooter collides with a trash can, the scooter tilting to one side before stopping.
```

| phase | video-time | \(c_m^+\) | \(c_m^-\) | ROI |
|---|---:|---|---|---|
| approach | 0–40% | The upright scooter moves continuously toward the separated trash can. | The scooter is already tilted and stopped beside the trash can at the beginning. | 滑板车起点到垃圾桶的 corridor |
| contact-response | 25–75% | The scooter visibly contacts the trash can, then begins tilting while slowing. | The scooter suddenly becomes tilted while still separated; contact occurs only afterward. | 两者接触区 + 滑板车 |
| terminal | 65–100% | The same scooter remains tilted and stopped beside the same trash can. | The tilted scooter becomes upright again after stopping. | 两个实体终点框 |

这里把三件事分开：

1. 接近不是接触；
2. 倾斜由接触触发；
3. 停止后不能重新直立、运动或消失。

---

## 5. TRACE-Lite 的最小代码改动

### 5.1 不改 Wan 的图像条件路径

继续复用官方 I2V：

- 图像先经 VAE 编码；
- 已知首个时间片在采样初始化和 scheduler 更新后重锚定；
- TRACE 只修改 DiT block 中额外 causal residual 的构造和注入范围。

### 5.2 新增配置

```python
@dataclass
class TracePhase:
    pos_context: torch.Tensor
    neg_context: torch.Tensor
    temporal_gate: torch.Tensor   # [F_lat]
    spatial_mask: torch.Tensor    # [F_lat, H_lat, W_lat]

@dataclass
class TraceConfig:
    phases: list[TracePhase]
    layer_scales: list[float]
    lambda0: float
```

### 5.3 在 block 中构造 token gate

Wan token 在进入 block 前由 \([F',H',W']\) flatten 成序列。TRACE mask 使用同样顺序：

```python
gate_m = temporal_gate[:, None, None] * spatial_mask
gate_m = gate_m.reshape(1, seq_len, 1).to(x.dtype)
```

若序列包含 padding，尾部补 0，不让 residual 进入 pad token。

### 5.4 block 伪代码

```python
q = self.norm3(x)
r_sem = self.cross_attn(q, sem_context, context_lens)
r_trace = torch.zeros_like(r_sem)

if trace_scale > 0:
    for phase in trace_phases:
        r_pos = self.cross_attn(q, phase.pos_context, context_lens)
        r_neg = self.cross_attn(q, phase.neg_context, context_lens)
        gate = phase.token_gate
        r_trace.add_(gate * (r_pos - r_neg))

x = x + r_sem + trace_scale * r_trace
```

三 phase 顺序循环显存最低；将 phase 维打包成 batch 可以更快，但会提高峰值显存。第一天优先循环，确认有效后再优化。

### 5.5 必须保留的三种运行模式

```text
global-ACE : 所有 f,x,y 使用同一 causal residual
TRACE-Time : 有 alpha_m(f)，beta=1
TRACE-ST   : 同时有 alpha_m(f) 和 beta_m(f,x,y)
```

这样可以分别回答：

- 提升来自事件分阶段，还是只来自更多文本？
- 空间 mask 真能减少泄漏，还是只是限制了运动？
- 四轴结构是否比原 layer-step 路由更有效？

### 5.6 lambda 与 schedule

沿用原 schedule 作为初值，但整体强度要更保守，因为三个 phase 会重叠：

```text
lambda0 candidates: 0.15, 0.30, 0.50
blocks 0–7:   0.05
blocks 8–13:  0.35
blocks 14–23: 1.00
blocks 24–29: 0.05

denoise 0–30%:   1.00
denoise 30–60%:  0.70
denoise 60–80%:  0.25
denoise 80–100%: 0.00
```

因为 \(\sum_m\alpha_m(f)=1\)，不同 phase 不会简单把 residual 强度叠加三倍。

### 5.7 新增日志

除了原日志，再保存：

```json
{
  "method": "TRACE-ST",
  "latent_grid": [13, 30, 52],
  "phase_centers": [0.15, 0.50, 0.85],
  "phase_sigma": [0.22, 0.24, 0.22],
  "roi_source": "manual",
  "event_masks": ["mask_pre.pt", "mask_mid.pt", "mask_post.pt"],
  "trace_residual_norm_by_phase": [0.0, 0.0, 0.0],
  "trace_residual_norm_by_layer": ["..."]
}
```

至少导出一次 `gate × residual norm` 的视频时间曲线，验证 middle phase 确实主要影响中间视频 token。

---

## 6. 修订后的 1–2 天执行与评测

### 6.1 Day 1：实现与单例验证

#### 上午

1. 在 ACE 实验分支加入 `TracePhase` 和 token gate；
2. 先使用全 1 spatial mask，验证 `TRACE-Time`；
3. 49 帧，较低面积，一个 seed；
4. 验证 `lambda0=0` 与官方 Wan 输出一致；
5. 验证三个 temporal gate 在每个视频 token 上求和约等于 1；
6. 输出每个 phase 的 residual norm 和时间分布。

#### 下午

1. 加入手工 static ROI/motion corridor；
2. 在“冰融化”和“滑板车碰撞”上跑 `ACE / TRACE-Time / TRACE-ST`；
3. 检查 mask reshape 是否与实际时空 token 对齐；
4. 检查 residual 是否只在 ROI 非零；
5. 观察 TRACE-ST 是否把运动过度锁死。

### 6.2 Day 2：最小对照

先用 4 类 prompt，每类 1 条、2 seeds：

1. 连续相变：冰逐渐融化；
2. 接触触发：滑板车撞垃圾桶并倾斜停止；
3. 守恒/转移：液体倒入杯中，源减少、目标增加；
4. 多物理局部：雪地中篝火旁向杯中倒咖啡。

六组方法共 48 个视频：

| ID | 方法 | 首帧 | 时间 phase | 空间 mask |
|---|---|---:|---:|---:|
| M0 | Wan + short prompt | 否 | 否 | 否 |
| M1 | Wan + long physics prompt | 否 | 文本隐式 | 否 |
| M2 | Wan I2V + short prompt | 是 | 否 | 否 |
| M3 | I2V + global ACE | 是 | 否 | 否 |
| M4 | I2V + TRACE-Time | 是 | 是 | 否 |
| M5 | I2V + TRACE-ST | 是 | 是 | 是 |

若 48 个视频能在第 2 天上午完成，再扩到 8 prompts；不要一开始跑 144 个，先确保新增结构有可见效应。

### 6.3 评测必须从“终态正确”升级为“演化正确”

首轮盲评回答：

1. 初态是否正确且结果没有被预置？
2. 至少一个非平凡中间状态是否清楚可见？
3. 状态变化是否方向正确、没有明显跳变？
4. trigger 是否先于 response？
5. 终态是否完成并保持？
6. 是否出现回生、反弹、重新直立或时间倒序？
7. 变化是否主要位于目标对象/接触区域？
8. 无关对象和背景是否保持合理？
9. 实体身份和整体画质是否不劣于 M2？

### 6.4 新增五个诊断指标

#### A. Static/Single-Moment Collapse Rate（SCR，越低越好）

满足任一项即判为 collapse：

- 终态在前 20% 帧已经出现；
- 开始与结束不同，但中间没有可辨别过渡；
- 全片只展示完成后的静态结果；
- 通过镜头切换直接跳过物理过程。

#### B. Intermediate Coverage（IC，越高越好）

```text
0: 没有中间态
1: 有模糊过渡但不可验证
2: 至少一个清楚中间态
3: 两个及以上有序中间态/连续可见演化
```

#### C. Monotonic/Directed Progress（MDP，越高越好）

对可测状态 \(q_f\) 和期望方向 \(d\in\{-1,+1\}\)：

\[
MDP=\frac{1}{F'-1}\sum_f
\mathbb{1}[d(q_{f+1}-q_f)\ge-\epsilon].
\]

冰大小应下降、水区域应增加；不适合单调假设的碰撞则改用阶段顺序分数。

#### D. Reversal Rate（RR，越低越好）

终态出现后又回到早期状态的比例，例如：

- 水重新变成完整冰块；
- 已停止滑板车重新加速；
- 破碎物重新拼回；
- 倒入杯中的液体无原因返回容器。

#### E. Locality Leakage（LL，越低越好）

事件 ROI 外出现与物理条件相关的异常变化，包括：

- 背景融化、燃烧、流动或形变；
- 无关人物姿态突然受影响；
- 第二种物理过程被错误专家覆盖；
- 相机用大幅运动掩盖事件。

### 6.5 最低成功线

M5 相对 M3 至少满足：

- SCR 下降 `>=20` 个百分点；
- IC>=2 的比例提升 `>=15` 个百分点；
- trigger-before-response 提升 `>=10` 个百分点；
- RR 下降 `>=10` 个百分点；
- LL 下降 `>=15` 个百分点；
- 阶段条件化实体存在/材料连续率不下降；
- 与 M2 比较，画质“不更差”的比例 `>=70%`。

样本很小时这些数值不是统计显著结论，只是决定是否值得进入训练阶段的工程门槛。

### 6.6 止损与诊断

| 结果 | 说明 | 下一步 |
|---|---|---|
| M4>M3，M5≈M4 | 时间 phase 有用，手工 ROI 无新增收益 | 保留 temporal router，重做 mask/定位 |
| M5 提高 locality 但 IC 不变 | 空间路由正确，状态表示仍太弱 | 训练 transition token/projector |
| M5 运动更少 | ROI 太窄或首帧锚太强 | 扩 corridor，减小图像/残差约束 |
| M4/M5 都直接终态 | phase prompt 对 T5 区分不足 | 提取 embedding/residual norm；转 state tokens |
| 中间态出现但抖动 | 硬边界或 phase 竞争 | 增大 Gaussian gate 重叠，用插值 control points 平滑参数化 |
| 多物理区域互相污染 | overlap normalization 不够 | 引入 role graph 和 learned arbitration |
| 只有初态图提升 | causal route 无效 | 不继续包装成物理方法，转向训练监督 |

---

## 7. 远期完整方案：TRACE-Field

### 7.1 研究问题升级

最终问题不应只是：

```text
Which physics expert should this video use?
```

而应是：

```text
Which entity should change,
from which state to which state,
after which trigger,
during which output-time interval,
in which spatial support,
while which variables and regions must remain invariant?
```

这把物理类别识别升级为 **causal state-transition field prediction**。

### 7.2 Causal State Graph Compiler

LLM/VLM 只输出固定 schema，不直接写长 prompt：

```json
{
  "entities": {
    "ice": {"role": "patient", "continuity": ["material provenance", "plate association"]},
    "water": {"role": "result_material"}
  },
  "protected_properties": ["static camera", "plate identity", "background"],
  "transition_variables": ["ice visible volume", "ice boundary", "solid fraction", "water area"],
  "transitions": [
    {
      "source_state": "solid ice, large volume",
      "trigger": "temperature increases above melting condition",
      "target_delta": ["ice volume decreases", "water amount increases"],
      "direction": ["monotonic_down", "monotonic_up"],
      "affected_entities": ["ice", "water"],
      "protected_entities": ["plate", "background"],
      "support_hint": "ice boundary and adjacent plate",
      "must_precede": "mostly melted terminal state"
    }
  ]
}
```

编译器输出的是状态边，不是长叙述。公式检索可以作为可选验证器，但不应成为所有开放世界事件的硬依赖。

### 7.3 从 fixed law experts 改为 open-vocabulary transition operators

简单复刻 combustion/liquid/refraction 等固定专家会与 WISA/ProPhy 高度重合，也限制未见物理组合。TRACE-Field 更适合使用共享低秩 operator：

\[
\mathcal{O}_{e}(h)
=
U\,\operatorname{diag}(a(e))\,Vh,
\]

其中 \(e\) 是由 `source_state + trigger + target_delta` 编码得到的 event-edge embedding，\(a(e)\) 由小型 hypernetwork 预测。这样：

- operator 参数共享，不为每条物理定律复制一个完整专家；
- 新物理现象可由状态变化语义组合；
- 同一种物理类别内也能区分不同角色和方向；
- 训练成本低于复制多层 Physical Branch。

### 7.4 可学习四轴 field router

对每个 event edge \(e\) 和视频 token \(i=(f,x,y)\)：

\[
G^{e}_{\ell,k,i}
=g^{e}_{layer}(\ell)
g^{e}_{denoise}(k)
g^{e}_{phase}(f)
g^{e}_{region}(h_{\ell,k,i},e).
\]

最终：

\[
h'_{\ell,k,i}
=h_{\ell,k,i}+r_{sem}
+\sum_eG^{e}_{\ell,k,i}\mathcal{O}_e(h_{\ell,k,i}).
\]

建议先采用因子分解而不是直接预测巨大的 \(L\times K\times F\times H\times W\) tensor：

- layer gate：30 个标量或 6 个连续 basis；
- denoise gate：4 个 bins；
- phase gate：8～16 个 temporal control points，经插值到 \(F'\)；
- region gate：token-wise scalar；
- event top-k：每 token 只激活 1～2 个 transition edges。

### 7.5 角色化空间 support

空间 mask 不应只问“哪里有 combustion”，而应区分：

| support | 含义 | 示例 |
|---|---|---|
| actor tube | 施力/运动主体 | 滑板车、球、手、容器 |
| patient tube | 接受作用的对象 | 垃圾桶、被击球、杯中液体 |
| interface tube | 相互作用边界 | 接触点、壶口、火焰—燃料边界 |
| result region | 新状态出现位置 | 水坑、碎片区、烟羽 |
| protected complement | 不应接收该 transition residual | 背景、无关人物、雪地 |

这比“物理类别 attention map”更接近因果结构：同一物体在不同阶段可能从 actor 变为 patient，不同状态边也可共享/竞争同一空间。

### 7.6 Dynamic support propagation

手工框只用于 MVP。远期可用三种互补信号：

1. 首帧检测/分割提供实体初始 mask；
2. Wan 中层 self-attention 或轻量 tracker 将实体 support 传播到后续时间；
3. transition residual norm 提供事件变化 saliency，并与实体 tube 相交。

推荐：

\[
M_e(f)=operatorname{Track}(M_e(0),h_{mid})
\cap
\operatorname{TopQ}(\|\Delta r_e(f)\|).
\]

仅依赖 VLM attention 容易继承 ProPhy 指出的区域噪声；仅依赖 residual saliency 又容易自证循环。二者与对象轨迹交叉能降低错误区域。

### 7.7 多物理组合与耦合边

独立现象可使用稀疏并行 fields；发生因果耦合时，用 event graph edge 连接：

```text
fire heats pot
pot transfers heat to liquid
liquid temperature rises
liquid begins boiling
```

对转移/守恒过程定义有向 flux edge：

\[
J_{a\rightarrow b}(f),
\]

并要求 source 与 target 的可见进度耦合：

\[
\Delta q_a(f)+\eta\Delta q_b(f)\approx0.
\]

这里的 \(q\) 可以是可学习的状态 probe，不必在开放世界中都是真实 SI 单位。对于液体体积、物体数量、颜色混合比例等可测任务，再使用显式量化监督。

### 7.8 Initial-state continuity 与 transition variables 分权

初态图像特征不应被整体标成 invariant，也不应与物理 feature 直接相加后交给同一个 gate。建议把保护/变化条件提升为“实体—属性—阶段”级别：

\[
h'=h
+M_{prot}\odot r_{continuity}
+M_{trans}\odot r_{transition}
+r_{semantic},
\]

其中 \(M_{prot}\) 与 \(M_{trans}\) 可在空间上重叠，但作用于不同属性子空间且职责不同：

- initial/continuity path 保持与事件不冲突的身份、材料来源、相机和场景线索；
- transition path 改变 schema 中明确列出的位移、姿态、形态、相态、数量、速度或完整性；
- semantic 始终绑定实体角色和目标事件。

在 event region 内也不能完全关闭 continuity path，否则物体会在变化中无依据地换类别；但 continuity path 也不能把冰块轮廓、气球尺寸或玻璃完整性锁死。首版远期实现优先使用显式 feature projection 或分组 channel gate，只去掉明显冲突于受保护属性的 transition 分量，不额外增加一个 orthogonality loss；若无法定义可靠属性方向，就退回软门控并如实报告，而不是宣称严格解耦。

---

## 8. 训练目标与反事实数据设计（总损失最多 4 项）

### 8.1 为什么普通 flow-matching loss 不够

普通生成损失可以在像素/latent 层重建视频，却不保证模型学会：

- 哪个中间态对事件成立是必要的；
- 终态不应提前出现；
- 哪个区域应该变化；
- 哪些变量必须反向耦合；
- 时间打乱后为什么物理上错误。

但这不意味着要为每个失败现象单独增加一个 loss。过多损失会带来难以解释的权重竞争、不同量纲的梯度冲突和巨大的调参空间，最终无法判断提升来自哪一项。这里采用一个硬约束：**包括基础生成目标在内，总损失最多 4 项；新需求优先并入已有目标或由结构约束解决。**

### 8.2 统一的反事实负例池

从真实物理视频构造：

| 负例 | 操作 | 对应失败 | 有效性约束 |
|---|---|---|---|
| endpoint preset | 把末段复制/前移到开头 | 结果预置 | 保持实体和场景相同，只改时间位置 |
| middle deletion | 删除或强压缩关键中间片段 | single-moment collapse | 首尾状态与正例尽量相同 |
| temporal reversal | 倒放或交换相邻阶段 | 时间倒序、回生 | 排除本来就时间对称的事件 |
| locality swap | 把 event mask/edge 绑定到背景或另一对象 | 错误区域、错误角色 | 不改原视频像素，只改条件—区域配对 |
| no-op | 重复首帧或冻结事件 ROI | 静止取巧 | 保持长度、背景与相机统计一致 |

这五类都进入同一个 `counterfactual pool`，不各自对应一个损失。每个正例每次随机采 1～2 个困难负例；负例生成参数、有效帧范围和排除原因必须记录。对弹性往返、周期运动等非单调过程，不能机械地把倒放判为错误，应由 state schema 决定该负例是否适用。

### 8.3 推荐的四项损失

总目标严格只有四项：

\[
\mathcal L
=\mathcal L_{gen}
+\lambda_{state}\mathcal L_{state}
+\lambda_{route}\mathcal L_{route}
+\lambda_{cf}\mathcal L_{cf}.
\]

#### A. \(\mathcal L_{gen}\)：基础 flow-matching 生成损失

沿用 Wan 的原始 flow-matching/velocity prediction 目标，作用于完整视频 latent。它负责基本画质、语义和生成能力，也是防止小模块只优化物理分数却破坏视频的主要锚点。首阶段冻结 backbone 时，该项主要训练新 operator/adapter 与原分布兼容；不再额外设置一个 quality loss。

#### B. \(\mathcal L_{state}\)：统一状态轨迹损失

每条 event edge 提供归一化状态目标 $q_e^*(f)$，例如冰大小 `1 -> 0`、水面积 `0 -> 1`，或碰撞的 `approach/contact/tilt/stopped` 阶段值。Reader/transition head 预测 $\hat q_e(f)$：

\[
\mathcal L_{state}
=
\frac{\sum_{e,f}m_{e,f}\operatorname{Huber}(\hat q_e(f),q_e^*(f))}
{\sum_{e,f}m_{e,f}+\epsilon}.
\]

phase、顺序、进度、终态保持和可量化的源—目标耦合都编码进同一条 $q^*$ 轨迹及有效性 mask $m$，不再拆成 `L_phase`、`L_order`、`L_progress` 或 `L_conservation`。不同物理量无法可靠量化时，只监督 0～1 的阶段进度，不伪造 SI 数值。

#### C. \(\mathcal L_{route}\)：局部作用域损失

让预测 gate $G_e(f,x,y)$ 对齐软 event-support target $M_e^*(f,x,y)$。首版使用一个带前景/背景权重的 BCE：

\[
\mathcal L_{route}
=-\frac1N\sum_i
\left[w_1M_i^*\log G_i+w_0(1-M_i^*)\log(1-G_i)\right].
\]

ROI 外的 target 为 0，因此 locality 与“减少无关区域改写”已合并在这一项中，不再单设 `L_local` 和 `L_invariant`。要注意它保护的是 event residual 的作用域，而非强制背景像素完全不变。

#### D. \(\mathcal L_{cf}\)：统一反事实排序损失

用同一状态一致性 scorer $s(v,c,G)$ 比较真实演化 $v^+$ 和从负例池采样的 $v^-$：

\[
\mathcal L_{cf}
=\max\left(0,\gamma-s(v^+,c,G)+s(v^-,c,G)\right).
\]

endpoint preset、middle deletion、reversal、locality swap 和 no-op 全部共享这一项。它负责“同样有正确实体/终态时，真实过程仍应优于错误过程”，不再分别建立五个 contrastive loss。

#### 不再增加的损失

- 路由稀疏：用 event top-k、低秩 operator 和有限 layer/denoise bins 直接限制容量；
- 时间/空间平滑：用 Gaussian phase basis、插值 control points 和 blur 后 soft mask 参数化；
- 不变量保护：由 `L_route` 的 ROI 外零目标、冻结主干和 `L_gen` 共同承担；
- anti-oscillation：由 memory EMA、gain clipping、hysteresis 和后期关闭 corrector 实现；
- Reader confidence：在 `L_state` 中预测带尺度的状态分布或做误差校准，不单列第五项。

建议第一轮只搜索三组辅助权重，而不是大网格：

```text
conservative: lambda_state=0.25, lambda_route=0.10, lambda_cf=0.10
balanced:     lambda_state=0.50, lambda_route=0.20, lambda_cf=0.20
strong:       lambda_state=1.00, lambda_route=0.30, lambda_cf=0.30
```

先用约 500～1000 steps warm-up 训练前三项并令 `lambda_cf=0`，确认生成与 gate 正常后再打开反事实项。每项先按有效元素数归一化，并分别记录梯度范数；若辅助项长期比 `L_gen` 梯度大一个数量级，优先降权而不是再加新的平衡 loss。

### 8.4 训练规模建议

#### Phase A：机制验证，1–2 周

- Wan2.1-1.3B；
- 冻结 backbone；
- 只训练 state head、phase/region router 和低秩 operator；
- 2K～5K 高可见性视频；
- 只覆盖碰撞、转移、融化/溶解、破裂四类；
- 每条样本只需 `video + original prompt + q* + soft event mask + counterfactual id`；
- 固定四项损失，Phase A 期间不接受新增 objective。

#### Phase B：论文主实验，3–5 周

- 扩展到 20K～50K clips；
- Wan1.3B 完整消融；
- Wan5B 训练 router/LoRA，不全量微调；
- 加入多物理 composite 数据和 counterfactual pairs；
- 建立自动标注—人工抽检流程；
- 若某种失败仍明显，先改负例采样、状态标注或路由参数化，不为单一 benchmark 症状增加第五个 loss。

#### Phase C：跨模型与评测，2–3 周

- 至少一个非 Wan 架构；
- VideoPhy2、PhyGenBench 和自建 Evolution/Locality subset；
- 人评、VLM 评估、对象追踪和状态 probe 联合；
- 报告质量、语义、物理、演化、局部性，而不是只报 PC。

---

## 9. 论文创新边界与必要消融

### 9.1 不能再作为核心 novelty 的点

- 用 LLM 把物理过程拆成事件链：CoECT 已做；
- 为每个事件生成关键帧并插值：CoECT 已做；
- 用 physical MoE：WISA/ProPhy 已做；
- 用 VLM attention 监督 token-level 物理区域：ProPhy 已做；
- 单纯在中层加入物理模块：已有多篇层选择/运动控制工作；
- 首帧保持外观：标准 I2V 能力。

### 9.2 更有潜力的核心贡献组合

1. **State-transition field，而非 physical category expert。**路由单位是有方向的 `source state -> trigger -> target delta`。
2. **四轴作用域。**统一建模 layer、denoise time、output video time 和 spatial support。
3. **Initial/protected/transition role separation。**显式区分初态边界、事件无关保护属性与必须演化的状态变量。
4. **Evolution counterfactuals。**专门训练和评测 outcome preset、middle deletion、time reversal、no-op 与 locality swap。
5. **Composable multi-physics edges。**独立局部场稀疏组合，耦合现象用 event/flux graph 传递。

### 9.3 与 CoECT、ProPhy 的方法边界表

| 维度 | CoECT | ProPhy | TRACE-Field 目标 |
|---|---|---|---|
| 主要表示 | 公式约束事件链 | 物理类别专家 | 开放词汇有向状态边 |
| 动态约束 | 事件关键帧 + 插值 | 学习到的 token prior | 输出时间 phase field + transition operator |
| 空间定位 | scene graph/视觉 keyframe | VLM 监督 token router | actor/patient/interface/result/protected supports |
| 多物理 | 多事件叙事 | 多专家 token 激活 | 稀疏 fields + causal coupling edges |
| 生成器改动 | 可接 off-the-shelf generator | 训练 Physical Branch | 冻结主干优先，小 router + low-rank operator |
| 反事实 | 不是演化训练核心 | 不是状态链反事实 | middle deletion/reversal/preset/no-op/locality swap |
| 初态/连续性 | 连续编辑关键帧 | 保留 backbone | 初态、protected properties 与 transition variables 分权 |

### 9.4 必须做的消融

```text
short prompt
long physics prompt
initial-state image only
global ACE
TRACE-Time
TRACE-Space without phases
TRACE-ST
TRACE-ST without counterfactual
TRACE-ST without protected complement
learned router without state edge
full TRACE-Field
```

还要交换：

- 正确 phase 顺序与倒序；
- 正确 ROI 与背景 ROI；
- 正确 event edge 与不相关 edge；
- 单物理与双物理 composite；
- seen law 与 unseen entity/law composition。

只有在这些交换实验中输出按预期变坏，才能证明 router 真正在控制因果演化，而不是单纯增加动态程度。

### 9.5 推荐论文命题

> Existing physics-aware generators often recognize physical endpoints or activate physics categories, but fail to realize where and when the corresponding state transitions should unfold. We model physics as open-vocabulary causal state-transition fields and route them anisotropically across model depth, denoising time, video time, and event-local spatial supports.

### 9.6 标题草案

```text
Physics Must Evolve Somewhere:
Spatiotemporal Causal State-Transition Fields for Video Diffusion
```

```text
From Knowing the Outcome to Generating the Process:
Anisotropic State-Transition Routing for Physical Video Generation
```

```text
Not Every Pixel Obeys the Same Event:
Composable Local Causal Fields for Physically Coherent Video Generation
```

### 9.7 Venue 判断

- **CVPR/ICCV：**最自然。重点是可见演化、局部路由、强 qualitative、严格 counterfactual 和跨基座实验。
- **ICML：**需要把 transition field 形式化为更一般的条件作用域学习问题，给出可学习稀疏路由的机制规律、泛化界面或更强统计结论；只在 Wan 上做启发式 mask 不够。

---

## 10. 修订后的最终决策

### 10.1 现在立刻做什么

不再把“原始短 prompt + 一个全局 causal residual”作为最终 MVP。将其保留为 M3 基线，并实现：

```text
TRACE-Lite
= original short semantics
+ native Wan5B pre-event initial-state/continuity anchor
+ three overlapping output-time phases
+ phase-specific correct-vs-failure residuals
+ manual coarse event ROI/tube
+ existing layer-denoise schedule
```

这比原 ACE 多出的工作主要是：phase JSON、时空 gate reshape、ROI 构造和演化评测，仍然可以在 1–2 天内完成。

### 10.2 什么时候值得继续

如果 TRACE-Time 明显减少结果预置/中间态缺失，且 TRACE-ST 进一步减少无关区域变化，那么远期 TRACE-Field 的两个核心假设分别得到支持：

1. 物理条件需要 output-time assignment；
2. 物理条件需要 event-local anisotropic support。

若只有图像锚有效、TRACE 对 ACE 没有稳定增益，则不应继续堆叠 router；应转向更直接的状态监督、关键帧/轨迹控制或训练数据问题。

### 10.3 最有价值的长期主线

最终论文不应声称“加入了更多物理知识”，而应证明：

> **物理知识只有被翻译成有方向的状态转移，并被分配到正确的视频阶段、正确对象与正确作用区域时，才可能成为物理演化；同时必须从初态中识别事件无关的受保护属性，而不能把整个参与事件的对象误当作不变量。**

这同时吸收了 CoECT 的 evolution insight 与 ProPhy 的 anisotropic insight，但方法上不依赖多关键帧编辑，也不复制固定物理类别 MoE，而是形成一条独立、可逐步验证且具有足够论文增量的路线。

---

# 第三部分：Reader–Writer/Corrector 的创新性与整合设计

## 1. Reader–Writer/Corrector 是否值得加入

### 1.1 直接结论

答案分成三层：

| 版本 | 创新潜力 | 是否建议 |
|---|---:|---|
| 在固定 block 保存 feature，再在后面 block 直接加回 | 低，约 `3/10` | 不作为论文核心 |
| 根据层/去噪步选择 Reader 与 Writer，并用残差纠正 | 中低，约 `4–5/10` | 可做强 baseline/分析工具 |
| 读取显式事件状态、写入因果索引记忆、按状态误差做局部闭环纠正 | 中高，约 `7–8/10` 潜力 | 建议纳入 TRACE-Field 核心远期方案 |

这里的分数不是审稿概率，而是相对当前公开工作的拥挤程度判断。

最重要的区分是：

```text
generic feature Reader/Writer
!=
causal state Observer/Memory/Corrector
```

前者只是“从哪里复制 feature 到哪里”；后者必须回答：

- Reader 读到的究竟是哪一个物理状态变量？
- Writer 保存的是哪条 event edge 在哪个视频阶段的进度？
- Corrector 根据什么可验证误差进行纠正？
- 纠正写到哪个对象、哪个输出时间、哪个 block 和哪个去噪阶段？
- 如何避免错误 Reader 形成正反馈并破坏画质？

### 1.2 为什么简单 Reader/Writer 已经不够新

#### A. 层选择和 feature injection 已很拥挤

[Stable Flow](https://openaccess.thecvf.com/content/CVPR2025/html/Avrahami_Stable_Flow_Vital_Layers_for_Training-Free_Image_Editing_CVPR_2025_paper.html) 已提出自动寻找 DiT 中对图像形成关键的 vital layers，并在这些层选择性注入 attention feature。FateZero、Pix2Video、MotionEditor 等视频编辑方法也会保存中间 attention/KV/feature，再注入后续分支或帧。[FateZero](https://openaccess.thecvf.com/content/ICCV2023/html/QI_FateZero_Fusing_Attentions_for_Zero-shot_Text-based_Video_Editing_ICCV_2023_paper.html)、[Pix2Video](https://openaccess.thecvf.com/content/ICCV2023/html/Ceylan_Pix2Video_Video_Editing_using_Image_Diffusion_ICCV_2023_paper.html)

因此“某层适合读、某层适合写”本身更像设计原则，不足以成为主要贡献。

#### B. 保存早期物理/运动先验并写回后续去噪已有直接工作

[PhaseLock（ICML 2026）](https://arxiv.org/abs/2606.06361) 报告 2-step I2V 输出有时比 50-step 输出更符合物理，并将其归因于后续去噪中的 phase erosion。它从 2-step latent 提取帧间 motion delta，再在完整去噪中用 Latent Delta Guidance 写回，报告平均物理一致性提升约 6.2 分，额外开销约 `1.06×` 时间、`1.02×` 显存。

所以“Writer 保存早期运动信息，防止晚期细节覆盖物理”已经存在非常接近的先例。

#### C. 生成器内部自读取与自纠正也已有直接工作

[Proprio](https://arxiv.org/abs/2605.28230) 直接使用冻结视频生成器在受控 latent 扰动下的 flow residual 作为内部物理 self-score，结合动态时空 mask，并用于 Best-of-N、初始噪声优化或二者组合。它已经形成了“模型读取自身内部信号并进行 inference-time refinement”的范式。[Proprio 方法](https://arxiv.org/html/2605.28230v1#S3)

[WMReward](https://arxiv.org/abs/2601.10553) 则用外部 latent world model 的未来预测 surprise，在推理时搜索或梯度引导视频生成。它说明闭环/反馈确实可能提高物理结果，但也占据了“外部动态评估器纠正采样”的方法空间。

#### D. 跨 step guidance 和角色化物理辅助信号也已有

[VPT](https://arxiv.org/abs/2607.04653) 已区分 agent、controlled object、passive object 和 background，并用 optical flow、role map、modality-decoupled denoising 与 cross-step auto-guidance 提高 Wan 的物理一致性。它还特别指出递归辅助预测会积累误差。[VPT 方法](https://arxiv.org/html/2607.04653v2#S3)

因此，我们不能把“Reader + Writer + Corrector”三个名字本身当创新。真正的区别必须来自读取内容、记忆结构、误差定义和局部写回机制。

### 1.3 RWC 应在 TRACE 中承担什么角色

TRACE-Lite/TRACE-Field 原本是开环控制：

```text
state-transition plan
    -> spatiotemporal router
    -> causal residual injection
    -> generated hidden states
```

它假设注入正确 residual 后模型会执行，但不能知道当前样本是否真的：

- 生成了中间态；
- 在正确时间发生接触；
- 已经提前出现终态；
- 在末段发生回生；
- 把变化泄漏到背景。

RWC 将其改成闭环：

```text
target transition field q*(f, entity, edge)
        │
        ▼
TRACE open-loop guidance ──► Wan hidden states
        ▲                         │
        │                         ▼
Local Corrector ◄── State Memory ◄── Reader/Observer
```

也就是说：

- **Reader/Observer：**从当前 Wan hidden token 读取局部事件状态和置信度；
- **Writer/Memory：**将读取结果写入低维、事件索引、去噪归一化的状态记忆；
- **Corrector：**比较目标状态曲线与当前记忆，只在偏差显著且 Reader 有信心时，将对应 transition operator 写回正确 token。

### 1.4 Reader 不应读取泛化 raw feature，而应读取状态

对每条 event edge \(e\)，Reader 至少预测：

\[
\hat q_e(f)=
\left[
q_{exist},
q_{progress},
q_{contact},
q_{response},
q_{terminal},
q_{reversal},
q_{locality}
\right].
\]

这里的 `q_exist` 必须是**实体—阶段条件化**目标，而不是要求所有原始对象永远存在。对融化、溶解、燃烧、倾倒和耗尽，source state 的存在度本来就可下降；Reader 还应同步读取 result material 或目标容器中的对应增加，以判断连续转移而非无依据删除。

示例：

#### 冰融化

```text
ice existence/size
water appearance/area
melting progress
regrowth probability
affected-region confidence
```

#### 滑板车碰撞

```text
scooter existence
distance-to-can trend
contact probability
post-contact tilt progress
stopped-state probability
re-acceleration/recovery probability
```

Reader 的输入不是全局平均 feature，而是 TRACE event support 内的 role-aware pooling：

\[
z^{read}_{e,\ell,k,f}
=
\frac{\sum_{x,y}G_e(f,x,y)h_{\ell,k,f,x,y}}
{\sum_{x,y}G_e(f,x,y)+\epsilon}.
\]

再由小型 probe 输出：

\[
(\hat q_e,\hat c_e)=R_{\phi}(z^{read},e,t_k,\ell),
\]

其中 \(\hat c_e\) 是置信度。Reader 不确定时 Corrector 应自动减弱，而不是强行纠正。

### 1.5 Writer 应保存低维 causal memory，而不是跨噪声复制 raw hidden

直接把 block feature 从去噪 step \(k\) 保存到 \(k+1\) 有三个问题：

- 不同噪声水平下 hidden 分布不同；
- raw feature 同时混合外观、语义、噪声和状态，难以解释；
- memory 很大，且容易把错误纹理/伪影永久写回。

推荐 memory slot：

\[
M_{e,k}
\in
\mathbb R^{F'\times d_s},
\]

其中 \(d_s\) 只保存状态、置信度和少量 event embedding。更新：

\[
M_{e,k}
=
\rho_{e,k}M_{e,k-1}
+(1-\rho_{e,k})
\operatorname{SNRNorm}(\hat q_{e,k}),
\]

\[
\rho_{e,k}
=
1-\operatorname{clip}(\hat c_{e,k},\rho_{min},\rho_{max}).
\]

高置信度读取更快更新 memory；低置信度读取只轻微改变既有估计。每个样本开始时 memory 清零，不跨视频保留。

Writer memory 按下面的 key 索引：

```text
(event edge, entity role, output video phase)
```

而不是固定 `combustion expert` 或一个全视频 motion vector。这是与 ProPhy/VPT/PhaseLock 拉开边界的重要部分。

### 1.6 Corrector 应纠正状态误差，而不是无条件重放 feature

目标状态曲线由 Causal State Graph Compiler 给出：

\[
q_e^*(f).
\]

闭环误差：

\[
\varepsilon_e(f,k)
=q_e^*(f)-M_{e,k}(f).
\]

Corrector：

\[
\delta h_{\ell,k,i}^{e}
=
\underbrace{G^e_{\ell,k,i}}_{\text{where/when}}
\underbrace{\chi(\hat c_e)}_{\text{confidence}}
\underbrace{\operatorname{clip}(K_e\varepsilon_e,-\delta,\delta)}_{\text{how much}}
\underbrace{\mathcal O_e(h_i)}_{\text{which transition}},
\]

\[
h'_{\ell,k,i}=h_{\ell,k,i}+\sum_e\delta h_{\ell,k,i}^{e}.
\]

这里 Corrector 不重新发明变化方向；变化方向仍来自 TRACE transition operator \(\mathcal O_e\)。Corrector 只根据状态误差调节强度和局部作用域。

安全规则：

1. Reader 输出对 Corrector 默认 `stop-gradient`，避免二者共同学会作弊的内部码；
2. correction norm 不超过原 Wan residual norm 的固定比例；
3. 最后 15%～20% 去噪默认关闭大幅状态纠正；
4. ROI 外 correction 为 0；
5. 置信度低于阈值时退回 open-loop TRACE；
6. 连续两次 correction 方向翻转时减小 gain，防止振荡；
7. background/appearance probe 同步监控质量退化。

### 1.7 两个闭环：block 内与 denoising step 间

#### Loop A：同一次 DiT forward 的 block-level loop

```text
blocks 0–9    : base perception / appearance
blocks 10–14  : Reader candidates
blocks 15–18  : state memory update / relation processing
blocks 19–23  : Corrector/Writer-back candidates
blocks 24–29  : consolidation, correction decay
```

同一次 forward 中先读后写，不需要额外完整 DiT 调用。但上面只是 Wan5B 的初始候选窗口，不是固定物理规律。

#### Loop B：多个去噪 step 之间的 memory loop

```text
denoise step k:
    read normalized state -> update M_k -> correct later blocks

denoise step k+1:
    read again -> compare with M_k and q* -> update/correct
```

这个 loop 用来检测：早期已经形成的正确演化是否在后续细化中被侵蚀，以及先前纠正是否真正持续。

### 1.8 Reader/Writer 层不应凭直觉固定：建立 RW Causal Atlas

已有论文关于中层/早期去噪的发现可作为先验，但不同架构、任务和 state probe 不一定共享最佳位置。应在 Wan 上分别测量：

#### Reader score

- phase/state 线性 probe 准确率；
- 正确演化与倒序/middle deletion 的可分性；
- actor/patient/contact 的定位准确率；
- 不同 seed 下的稳定性；
- 对输入 prompt 改动的敏感性。

#### Writer/Corrector score

- 单层干预对 IC/SCR/RR/LL 的因果效应；
- 对外观、身份和 VBench quality 的副作用；
- correction 能否在后续 block/step 保留；
- 错误 ROI/错误 event edge 写入是否按预期导致失败。

可以定义：

\[
V_{write}(\ell,k)
=
\Delta\text{EvolutionSuccess}
-\eta\Delta\text{QualityDamage}.
\]

Reader window 选 state observability 高的位置，Writer window 选 intervention value 高的位置。二者可以不同，这比简单宣称“中层适合物理”更有论文说服力。

### 1.9 预期能解决哪些问题

| 现有失败 | TRACE 开环 | RWC 闭环新增作用 | 预期把握 |
|---|---|---|---:|
| 终态在开场预置 | early phase 抑制 | Reader 检测 terminal-too-early，Corrector 强化 precondition | 中高 |
| 中间态缺失 | middle phase 注入 | Reader 检测 progress gap，提升 transition operator | 高潜力 |
| no-op/静止 | 因果 residual 促动 | Reader 检测 transition 区间 progress 过低 | 中高 |
| 时间倒序/回生 | phase + counterfactual | memory 记录已完成状态，检测 reversal | 高潜力 |
| 接触后无响应 | contact-response edge | contact probe 与 response probe 比较后纠正 | 中高 |
| 局部物理泄漏 | spatial support | locality Reader 检测 ROI 外 event activation | 中高 |
| 后期细化擦除运动 | denoise gate | memory 保留状态进度，检测后期侵蚀 | 高潜力 |
| 无依据的身份/外观漂移 | 首帧 + protected complement | continuity monitor 限制 correction norm | 中等 |
| 精确动量/体积守恒 | 只提供方向 | 若没有量化 state probe 仍不够 | 低至中 |
| 基座完全不会该现象 | 无法创造能力 | Corrector 只能在模型支持范围内重加权 | 低 |

“高潜力”不是保证提升。闭环效果的上限由 Reader 准确率和 transition operator 的可控性共同决定。

### 1.10 最主要的风险

#### Reader error amplification

Reader 把正确中间态误判为错误，Corrector 会主动破坏结果。必须做置信度校准和 open-loop fallback。

#### State-feature shortcut

Reader 可能学到水的纹理而不是融化进度，仍然只识别终态。middle deletion、倒序和同终态不同过程的 hard negatives 是必要的。

#### Memory staleness

早期高噪 feature 的判断可能在后期已失效。memory 需要 SNR/step conditioning，而不是无限累积。

#### Controller oscillation

多个 Corrector 在相邻 block/step 反复正负纠正会造成抖动、闪烁和形变，需要 gain clipping、hysteresis、correction EMA 和方向翻转降益，不再额外引入 temporal-TV loss。

#### Multi-event conflict

同一 token 同时属于火焰—容器、容器—液体两个事件边时，多个 corrector 可能冲突，需要 graph priority 或 learned arbitration。

#### Compute overhead

若 Reader 需要低清解码、外部 VLM 或额外完整 forward，计算会迅速接近 WMReward/Proprio。优先使用主干中间 hidden 的轻量 state probe。

---

## 2. 如何把 RWC 加入当前模型计划

### 2.1 1–2 天 MVP：只加 Reader audit，不把完整闭环塞进主实验

当前最稳妥顺序是：

```text
Day 1–2 primary:
    ACE baseline -> TRACE-Time -> TRACE-ST

Day 1–2 diagnostic add-on:
    Reader-only hooks and observability atlas

After TRACE signal is validated:
    Writer memory -> Corrector feedback
```

原因是：如果 TRACE 的 phase/ROI residual 本身不能改变视频，再加闭环 Corrector 只会放大无效信号；如果 Reader 尚不能分辨正确/错误状态，闭环甚至会比开环更差。

### 2.2 Reader-only 快速实现

在不改生成结果的情况下，对以下 block/step 注册 hook：

```text
blocks:        [5, 10, 14, 18, 22, 26, 29]
denoise steps: [0, 5, 10, 20, 30, 40]
```

对每个 phase 和 ROI 记录：

- hidden mean/std/norm；
- correct-vs-counterfactual residual norm；
- ROI 内外 residual ratio；
- phase token 的时间曲线；
- 相邻 phase representation distance；
- 相邻去噪 step 的 normalized state drift。

无训练 contrast score 可先用：

\[
A_{e,\ell,k,f}
=
\cos(\operatorname{LN}(z^{read}),\bar c_e^+)
-
\cos(\operatorname{LN}(z^{read}),\bar c_e^-),
\]

其中 \(\bar c\) 使用 Wan 自己的 text embedding 投影到 hidden dim。它只是可观测性 proxy，不能当物理真值。

### 2.3 Reader audit 的进入条件

满足下列至少三项才进入闭环开发：

1. 某个连续 block 窗口的 phase score 与预期 early/middle/late 顺序在 `>=70%` pilot 中一致；
2. 正确视频与 outcome-preset/middle-deletion/reversal 负例在该窗口可分；
3. 最佳窗口在不同 seed 上基本稳定；
4. phase prompt 交换会局部改变对应 phase score，而不是全片同时改变；
5. ROI 内 signal-to-background ratio 明显高于 1；
6. Reader score 与人工 IC/SCR/RR 标签至少呈稳定方向相关。

### 2.4 可选的 RWC-Lite，不纳入第一轮核心 claim

若 Reader proxy 看起来可信，可加一个无训练 scalar memory：

\[
M_{e,k}=0.8M_{e,k-1}+0.2A_{e,k},
\]

并只自适应 TRACE 强度：

\[
\lambda^{eff}_{e,k,f}
=
\lambda^{base}_{e,k,f}
\operatorname{clip}
\left(1+\kappa(q_e^*(f)-M_{e,k}(f)),0.5,1.5\right).
\]

安全设置：

- `kappa=0.25` 起步；
- 单步 gain 变化不超过 20%；
- Reader 低置信度时 `lambda_eff=lambda_base`；
- 后 20% 去噪关闭自适应；
- 只在 TRACE ROI 内生效。

将其作为 `M6 = TRACE-ST + scalar feedback`。若 M6 不稳定，立即回退 M5，不影响 TRACE-Lite 主结论。

### 2.5 训练版 RWC 的分阶段计划

#### RWC Stage A：State Reader，1–2 周

- 冻结 Wan；
- 从候选 block/step 提取 hidden；
- 用真实、倒序、middle deletion、endpoint preset、no-op 视频训练小 probe；
- 学 progress/order/contact/terminal/locality 与置信度；
- 先证明 state 可读，不做生成纠正。

#### RWC Stage B：Causal Memory + Corrector，2–3 周

- 加 event-indexed memory slots；
- 只训练 low-rank transition operator 和 gain network；
- Corrector 从零初始化；
- 严格复用第二部分训练计划的四项损失，不增加 RWC 专属 loss；
- 用 top-k/低秩结构、memory EMA、gain clipping 与 hysteresis 分别处理稀疏性和防振荡；
- 先在 Wan1.3B 上做全消融。

#### RWC Stage C：Wan5B 与多物理，2–4 周

- 在 Wan5B 只训练 Reader/Writer/Corrector adapter；
- 多 event edge top-k 路由；
- 测试 independent events 与 causally coupled events；
- 与 Proprio、PhaseLock、WMReward、ProPhy、VPT 做直接对比。

### 2.6 训练目标：严格复用四项总损失

RWC **不在第二部分的四项损失上做加法**。同一总式保持不变：

\[
\mathcal L
=\mathcal L_{gen}
+\lambda_{state}\mathcal L_{state}
+\lambda_{route}\mathcal L_{route}
+\lambda_{cf}\mathcal L_{cf}.
\]

RWC 需求按下表并入已有目标或结构规则：

| RWC 需求 | 处理方式 | 是否新增 loss |
|---|---|---:|
| Reader 预测进度、接触、终态与置信度 | 并入 `L_state`；置信度作为状态分布尺度/误差校准输出 | 否 |
| Corrector 只写入正确阶段和 ROI | 继续使用 `L_route` | 否 |
| 纠正后优于未纠正、倒序、no-op 等 | 把未纠正/错误纠正样本加入统一负例池，继续使用 `L_cf` | 否 |
| 生成质量与身份连续性 | 继续使用 `L_gen`，并冻结大部分 backbone | 否 |
| memory 跨 step 稳定 | SNR normalization + EMA；作为诊断曲线报告 | 否 |
| correction 不振荡 | gain clipping、hysteresis、方向翻转降益、后期关闭 | 否 |
| 稀疏与平滑 | event top-k、Gaussian basis、soft ROI 和低秩 operator | 否 |

这样 RWC 是对四个既有目标的**观测—记忆—控制实现**，而不是再叠一组难以权衡的 objective。若 memory drift 或 oscillation 仍失控，应先修改控制器/参数化；只有在四项主目标已稳定且消融证明结构规则无效时，才重新讨论损失设计，但主方案仍以最多四项为验收约束。

Corrector 训练要采用 `teacher-forced error + on-policy generated error` 混合。只在真视频 latent 上训练，会在生成失败状态上失效；只在模型生成视频上训练，又会过拟合某个 checkpoint 的错误。

### 2.7 直接竞争边界

| 方法 | 读什么 | 保存/评分什么 | 怎么纠正 | 我们必须不同在哪里 |
|---|---|---|---|---|
| Stable Flow | vital-layer attention | source feature | 层间 feature injection | 我们读显式事件状态，不做编辑复制 |
| PhaseLock | 2-step latent motion delta | early motion prior | 后续 latent delta guidance | 我们保存 event/role/phase 状态，不只帧差 |
| Proprio | flow residual | generator-native self-score | 搜索/优化初始噪声 | 我们做 block-local 状态误差反馈，不是全样本 residual score |
| WMReward | world-model future surprise | 外部动态 reward | 搜索/梯度 guidance | 我们不依赖外部 world model，且有 transition target |
| VPT | flow + role maps | learned auxiliary dynamics | cross-step checkpoint guidance | 我们不递归预测 dense auxiliary modality，使用 event memory |
| ProPhy | token physical category | token expert routing | physical branch injection | 我们读/纠正有方向的状态进度与阶段偏差 |

如果最终 RWC 只输出一个全局 physics score，再调 CFG 或优化 seed，它会与 Proprio/WMReward 太接近；如果只保存早期帧差再写回，会与 PhaseLock 太接近；如果只做 token 物理类别 probe，会与 ProPhy 太接近。

### 2.8 是否预期提升最终效果

有合理预期，但必须是条件式判断：

\[
\text{RWC gain}
\approx
\text{Reader accuracy}
\times
\text{Transition controllability}
-
\text{Feedback instability}.
\]

最可能提升的指标：

- Intermediate Coverage；
- Static Collapse Rate；
- cause-before-effect；
- terminal persistence / Reversal Rate；
- local event completion；
- late-denoise motion preservation。

不应预期仅靠 RWC 明显解决：

- 精确流体方程；
- 数值级质量/动量守恒；
- 基座训练集中完全没有的材料变化；
- 小物体已被 VAE 压没的接触；
- 严重错误的首帧构图。

### 2.9 最终建议

Reader–Writer/Corrector 应加入模型计划，但定位如下：

1. **当前 1–2 天：**TRACE-Lite 是主生成实验，RWC 只做 Reader observability hooks；
2. **确认 phase/ROI 有效后：**加入轻量 scalar memory 和 adaptive corrector，验证闭环价值；
3. **论文阶段：**将 Reader 升级为 causal state observer，将 Writer 升级为 event-indexed memory，将 Corrector 绑定到 TRACE-Field 的局部 transition operator；
4. **创新主张：**不是“我们有 Reader/Writer”，而是“我们把开放词汇局部状态转移建模成可观测、可记忆、可闭环纠正的四轴生成场”；是否能使用 `first` 之类表述，必须在更完整同期工作检索后决定；
5. **必要证据：**读写窗口的 causal atlas、错误状态可读性、错误 ROI/倒序干预、闭环稳定性以及对 Proprio/PhaseLock/ProPhy/VPT 的直接消融。

如果这些部分都做到，RWC 不只是一个附属模块，而可能成为 TRACE-Field 相比 CoECT 和 ProPhy 更明确的核心差异；如果 Reader 不能稳定读出中间状态，它就应保留为分析工具，而不能强行包装成闭环物理系统。
