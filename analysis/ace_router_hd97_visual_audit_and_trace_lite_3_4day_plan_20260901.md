# ACE-Router HD97 新旧结果审计与 TRACE-Lite 3–4 天增量方案

> 日期：2026-09-01  
> 核心问题：新版 conditioning + residual scaling 是否真正改善了生成；当前视频还暴露了哪些首要失败；下一轮如何从这些失败出发，用最小增量验证输出时间分阶段、空间各向异性与因果绑定。  
> 结论口径：**物理任务完成度优先于单帧清晰度。**“画面更干净”“终态更像”不等于“原因—中间态—结果被正确生成”。

## 1. 审计范围、配置差异与证据边界

本次直接检查了视频、`config.resolved.json`、`conditioning.compiled.json` 与 `diagnostics.jsonl`，并统一查看第 0/24/48/72/96 帧；对 P01、P02、P04、P06、P11 进一步按每 8 帧加密检查。全部实验共同使用 Wan2.2-TI2V-5B、1280×704、97 帧、24 FPS、50 steps、seed 42，成片约 4.04 秒。

| 组别 | 实际目录 | 样本数 | 关键设置 |
|---|---|---:|---|
| M3 旧 ACE | `outputs/ace_router_initial_v1_hd97/m3-initial-v1-hd97-20260901-125349` | 12 | I2V；initial-v1；CFG 5；`lambda0=0.5`；无 cap；block 8–23；旧 negative prompt |
| M3 新 3.5 | `outputs/ace_router/m3-20260901-104637` | 20 | I2V；compiled conditioning；CFG 3.5；`lambda0=0.1`；2% cap；block 14–23；空 CFG negative |
| M3 新 5 | `outputs/ace_router/m3-20260901-125349` | 12 | 与上一组相同，仅 CFG 5 |
| M4 旧 ACE | `outputs/ace_router_initial_v1_hd97/m4-initial-v1-hd97-20260901-135352` | 7 | T2V；initial-v1；CFG 5；`lambda0=0.5`；无 cap；block 8–23；旧 negative prompt |
| M4 新 3.5 | `outputs/ace_router/m4-20260901-104639` | 20 | T2V；compiled conditioning；CFG 3.5；`lambda0=0.08`；1.5% cap；block 14–23；空 CFG negative |
| M4 新 5 | `outputs/ace_router/m4-20260901-110224` | 20 | 与上一组相同，仅 CFG 5 |

这里需要澄清“旧版没有 conditioning 和缩放”的实验口径：旧配置并不是 `ACE disabled`，它仍计算 initial-v1 的 positive−counterfactual residual；“没有”指的是**没有新版 compiled semantic conditioning、没有 relative residual cap/自适应缩放**。旧版还会把原 prompt 前缀重复拼入 c+/c-，使用 `lambda0=0.5` 和旧 negative prompt。若把旧版写成“完全没有 causal conditioning”，会与实际 artifacts 不符。

严格三方同样例比较只有 M3 的 P01–P12、M4 的 P01–P07；其余视频只能用于发现“新版当前仍有什么问题”，不能用于宣称相对旧版提升。所有结论还受三个限制约束：只有一个 seed；旧/新同时改变了 semantic conditioning、正反事实文本构造、lambda、cap、layer/step schedule 与 negative prompt，因而不能把总差异归因给单一模块；当前没有人工盲评和轨迹测量。因此本文是**定性机制审计和下一轮实验决策依据**，不是统计结论。

## 2. 总体判断

最重要的判断不是“新版已经把物理生成做好了”，而是：

1. **新版首先取得了残差安全性和可控性的提升。**旧版 ACE 单个记录点的注入相对 semantic residual 可高达 15%–18%；新版 M3 被硬限制在 2%，M4 被限制在 1.5%。树枝状拓扑、巨大外物、背景/主体被条件过度改写等风险明显下降。
2. **M3（I2V）出现了一批真正更完整的过程，但提升不均匀。**P06 的“闭书离架—下落—落地打开”、P13 的壶倾斜并连续倒水、P15 的沙堆增长、P16 的染料区域扩张、P20 的倒水与火焰局部共存，证明当前基座与安全 ACE 并非完全没有状态演化能力。
3. **当前首要瓶颈已经从“残差太强”转向“残差作用域错误”。**全局 ACE 仍把一条原因—结果描述作用于全部输出时间 token 和全部空间 token，所以常得到正确终态，却跳过触发与中间态；或在正确区域之外生成手、液流、额外物体。
4. **M4（T2V）没有显示出稳定净提升。**没有初态图像后，实体数量、类别、位置与可运动空间无法固定；时间分阶段可以缓解结果预置，但不能凭空补回可靠的对象 grounding。因此下一轮应以 M3 为主线，M4 只做 TRACE-Time 的迁移性验证。
5. **CFG 5 不是通用更优点。**它有时增强状态变化（P02、P12、P19），也会增加复制、形变、溢出或过早结果（P07、P14 等）。当前更合理的默认值是 CFG 3.5，把 CFG 5 保留为压力测试；不应靠继续提高全局 CFG 来修复物理过程。

一句话总结：**新版已经把 ACE 从“可能破坏画面的强全局扰动”推进到“较安全、偶尔有效的全局因果方向”，但还没有成为可靠的因果状态转移控制器。**下一步的合理增量不是再堆 prompt、lambda 或 CFG，而是把已学到的“条件分权、关系差分、安全裁剪、中层稀疏注入”扩展到输出时间和事件区域。

## 3. 相比旧版已经出现的实际提升

**残差强度从不可控变为有明确预算。**对现有 diagnostics 的聚合如下；旧版为 summary 采样记录，新版为 full diagnostics，二者记录密度不同，但最大比例仍可直接说明 cap 是否生效。

| 组别 | `applied/semantic` 中位数 | P95 | 最大值 | 裁剪比例 |
|---|---:|---:|---:|---:|
| M3 旧 | 3.87% | 11.50% | 18.37% | 0%（无 cap） |
| M3 新 CFG 3.5 | 1.20% | 2.00% | 2.00% | 26.3% |
| M3 新 CFG 5 | 1.28% | 2.00% | 2.00% | 30.1% |
| M4 旧 | 4.16% | 10.17% | 15.51% | 0%（无 cap） |
| M4 新 CFG 3.5 | 0.55% | 1.50% | 1.50% | 11.9% |
| M4 新 CFG 5 | 0.56% | 1.50% | 1.50% | 11.6% |

这说明 `compiled conditioning + relation-only c+/c- + relative cap + mid_b + safe step schedule` 已经形成一个值得保留的设计范式：**先保证新增方向只占 semantic residual 的小比例，再讨论其是否物理正确。**所有记录均 finite，也没有数值爆炸。

**M3 的若干视觉/过程提升是明确的。**

- P05：旧版坡道/地面结构发生大幅重构，木块一度消失再出现；新版坡道保持稳定，木块确实沿坡运动并离开坡面。尚未正确表现粗糙面减速，但对象—环境关系更稳定。
- P06：这是最清楚的正例。旧版很早就把书变成打开状态，落地过程被压缩；新版先保持闭书、越过支撑边、连续下落，随后才在地面打开并保持。
- P07：新版箱体倾倒更完整，背景和主体数量更稳定；旧版箱体几何与外观跳变更明显。缺点是拉动原因仍不可见，CFG 5 又产生较强软化/折叠。
- P09：新版基本消除了旧版巨大勺子/手等外物，液态区域从黄油附近扩张，外观连续性更好；但固态黄油没有相应减少，仍只是“生成一滩液体”。
- P12：新版种子更接近细小冠毛，漂移区域更连续，显著好于旧版的深色颗粒/异物；CFG 5 的脱落更强。不过花头耗尽幅度仍不足。
- P02：新版 CFG 5 能更长时间保持红、蓝两球并让二者接近，优于旧版蓝球先消失。CFG 3.5 仍出现白球、手和主体出框，说明该收益不稳定。

**尚无旧版交集的新样例也证明“基座能做什么”。**P13/M3 能生成连续壶口—杯口水流，P15/M3 能生成持续沙流与增长的沙堆，P16/M3 呈现从紧凑染料点到逐步扩大的蓝色区域，P20/M3 能在火焰基本局部稳定时完成倾壶倒入。这些样例应作为 TRACE 迭代的 positive control：下一版不能为了修 P01/P11 而破坏这些已有能力。

但必须强调：上述提升主要是**布局、外观连续性、动作可见性和伪影控制**的提升；只有 P06、P16、P20 较接近完整物理过程，不能据此声称整体物理正确率已明显提高。

## 4. 当前视频仍然存在的核心问题

下面按“当前生成失败 → 样例证据 → 机制缺口”排列；这也是后续迭代的优先级，而不是先有方法再寻找适用问题。

| 当前失败 | 直接观察 | 为什么现有 ACE 解决不了 | 优先级 |
|---|---|---|---:|
| 结果预置/中间态删除 | P01 在接触垃圾桶前已经倾倒；P04 新版前半段全直立，随后几乎整排同时倒下；P10 未经历稳定充气便爆裂；P15/M4 开场已有沙堆；P19 缺口突然变成大裂缝 | denoise step 只控制“何时规划”，没有把 precondition/transition/terminal 分配给不同输出帧 | P0 |
| 触发—响应断裂 | P01 倾倒与接触分离；P04 没有首块推动与传播前沿；P17 按压幅度不足却直接进入手离开；P19 无明显双侧拉力便撕开 | 单个全局 c+/c- 只能表达关系方向，不能要求触发证据出现在响应之前的局部时间窗 | P0 |
| 身份/物质重复与幽灵残留 | P02 出现第三个白球/手；P11 碎片出现后完整瓶仍在空中；P17 抬走一个海绵后桌上又保留一个；P20/M4 多出容器 | global residual 没有 actor/result-material 的排他关系，也没有“同一实体由完整态转为碎片态”的 support 绑定 | P0 |
| 源—流—汇不耦合 | P08 水从画外/器具出现而冰不缩小；P13 的源端液位下降不清楚；P14 瓶子保持直立，橙汁从画外进入杯子；P15 杯内沙量没有清楚下降 | 文本知道“倒入/融化”，但没有把 source decrease、transfer tube、sink increase 作为同一事件边约束 | P0 |
| 状态进度不单调或不闭环 | P08 冰块尺寸几乎不减；P09 液池扩大但固态块保持；P12 花头耗尽不足；P18 反弹峰值不递减且结尾未静止 | 当前条件只要求终态概念，没有阶段进度、方向性和 terminal persistence 的专门 residual | P0 |
| 空间泄漏和错误作用对象 | P02 生成手/球杆；P14 出现额外杯子和溢流；P20/M4 出现额外容器；M4 多次用裁切、场景重构或错误类别替代动作 | 同一 causal residual 对所有空间 token 生效，没有 actor/trigger/patient/背景分区 | P1 |
| I2V 静态锚与事件变化冲突 | P08 冰块、P09 黄油、P17 海绵等都倾向保持首帧轮廓；模型宁愿在旁边新增液体/对象，也不修改被锚定主体 | 初态图像是必要边界，但当前没有显式告诉模型“哪些属性必须离开 I0”及其阶段性变化方向 | P1 |
| CFG 的全局增益无法专门强化物理 | CFG 5 有时增强脱落/裂纹，也会造成 P02 复制、P07 折叠、P14 溢出；CFG 3.5 更稳但部分任务 no-op | CFG 放大整个 conditional 分支，而非只放大正确物理边，不能充当 causal controller | P1 |

**逐样例的当前结论如下。**“较好”只表示相对当前批次，不代表严格合格。

| Pxx | M3 当前状态 | M4 当前状态 | 当前最值得解决的问题 |
|---|---|---|---|
| P01 滑板车碰撞 | 移动/倾倒更可见，但未接触便倾倒 | 类别/构图较旧版规整，但几乎开场即倒 | contact-before-tilt 与 terminal persistence |
| P02 双球碰撞 | CFG 5 较好保持两球；3.5 有白球、手、消失 | 3.5 接近静止，5 出现多球/错误位移 | 两球身份、接触后速度交换 |
| P03 网球击回 | 球拍与球相对稳定，但缺少清楚接触—反向 | 接触多在开场，后续轨迹含糊 | 接触瞬态与方向反转 |
| P04 多米诺 | 画面很干净，但由“全直立”跳为“几乎全倒”，旧版反而有手和传播前沿 | 基本静止/触发缺失 | 首块触发、局部传播、禁止 middle deletion |
| P05 下坡减速 | 坡道稳定、木块移动；离坡后缺少连续减速/停止 | 木块消失、重现或与坡道融合 | 分段速度进度与终态静止 |
| P06 书本掉落 | 本批最明确提升：闭书—下落—落地打开 | 对象常像灰板/薄片，书本语义退化 | 保持 M3 正例；验证 phase 不伤已有能力 |
| P07 箱体倾倒 | 倾倒完整但拉力缺失，CFG 5 形变较重 | 3.5 倾倒、5 撕开，结果对 CFG 敏感 | force/rope trigger 与刚体形状保持 |
| P08 冰融化 | 水区略增但冰几乎不缩，且出现外部水源 | 基本静止 | 冰量下降—水域增长的耦合单调性 |
| P09 黄油融化 | 无大外物、液池变大；固态块仍保留 | 有边缘软化但相变幅度不足 | 同材料守恒与 source decrease |
| P10 气球爆裂 | 跳过充气，爆裂后中心主体残留 | 开场已是充足终态尺寸，随后爆裂 | small→large→taut→rupture 三阶段 |
| P11 瓶子破碎 | 下落可见，但碎片先出现/完整瓶随后继续下落 | 同样存在完整态与碎片态并存 | impact-triggered exclusive state transition |
| P12 蒲公英 | 细粒子与漂移明显改善；花头耗尽不足 | 两个 CFG 都有较自然漂移，仍缺明确风触发 | progressive depletion 与来源一致性 |
| P13 倒水 | 壶倾斜、水流连接较好；源液位下降不够可验证 | 源容器被裁切，流/杯变化不清 | source–stream–sink coupling |
| P14 倒橙汁 | 左瓶不动、画外橙汁倒入杯中 | 开场即倒、额外杯/溢流，CFG 5 更重 | 正确源对象绑定与数量约束 |
| P15 倒沙 | 沙流和沙堆增长较好，杯内减少不清 | 开场已有堆，后段杯子离开 | 初态空目标区、源量下降、堆增长 |
| P16 染料扩散 | 当前最好的连续相变正例，蓝区渐进扩张 | 形成悬垂蓝团而非体积扩散 | 保护 M3 正例；验证空间/时间 gate |
| P17 海绵回弹 | 有按压与抬起，但出现被抬走/桌面残留的双海绵 | 压缩幅度极小，几乎只是手离开 | 同一主体的压缩—释放—恢复 |
| P18 衰减弹跳 | 多次上下运动，但峰值不单调、结尾不停止 | 3.5 出框，5 有多次弹跳但不衰减 | 接触计数、峰值递减、末态静止 |
| P19 纸张撕裂 | 大裂口突现，缺少缺口增长和足够分离 | CFG 5 有较好裂纹—分离，但缺拉力原因 | trigger-before-tear 与裂纹进度 |
| P20 雪地倒咖啡 | 壶、杯、火焰与雪地局部关系最好 | 有倒入但多出容器/裁切，CFG 5 流更弱 | 多物理局部隔离与实体数量 |

这些问题共同说明：**当前不是缺一段更详细的物理文字，而是缺“状态在何时、何处、由谁触发、作用到谁、完成后是否保持”的结构化分配。**

## 5. TRACE-Lite 对当前问题的可行性与必要修订

TRACE-Lite 值得做，但理由必须严格限定为：它直接针对本批视频最频繁的 `outcome preset / middle deletion / trigger missing / terminal reversal / locality leakage`，而不是因为“分阶段”本身新颖。97 帧经 Wan VAE 时间压缩后约对应 25 个 latent 时间位置，已有足够时间 token 承载多个可观察状态；当前 ACE 却让同一差分覆盖全部 25 个位置，这正好与观察到的失败一致。

**复杂度与适应性的折中结论是：采用受限的 `3/4` 二元自适应，而不是固定三段，也不是让 GPT 任意拆成 3–5 个事件。**所有视频都有 `start→evolve→settle` 三个功能窗口；只有当“触发”能独立于后续响应被肉眼观察时，才在中间插入一个 `trigger`，变成 `start→trigger→evolve→settle`。这些只是同一次生成里的重叠条件窗口，不是独立 clip，也不需要先构造事件图。

| 固定窗口 | 负责什么 | 主要阻止什么 |
|---|---|---|
| `start`：启动/边界 | 保留正确初态，让原因、接近或变化起点出现 | 开场直接终态、对象一开始就碎裂/融化/倒下 |
| `trigger`：可选触发 | 单独呈现接触、阈值、破裂点或释放瞬间 | 触发缺失、结果先于原因、触发与响应黏成一次跳变 |
| `evolve`：演化 | 显示触发后的主要中间过程或单调进度 | 中间态删除、瞬间替换、触发缺失、错误来源 |
| `settle`：收束/保持 | 完成结果并在尾部保持 | 回生、反转、幽灵源、持续振荡或结果消失 |

三段仍是当前 97 帧设置的默认：约 25 个 latent 时间位置分给三组重叠条件有足够覆盖，也覆盖本批最主要的开场、中段、结尾失败。四段只解决一种明确问题——把短触发与较长响应分开；它不是“复杂 prompt 就多切一段”。因此系统只有两个模板，仍容易比较、缓存和泛化。

具体选择只看一个问题：**能否分别指出“触发发生的可见证据”和“触发之后的响应证据”？**若不能，选三段；若两者都能被独立检查，选四段。优先复用现有 `conditioning.compiled.json` 的 `source_sections.event_type`、`prompts.causal_positive` 与 changed relation；当前文件没有单独的 `trigger` 字段，因此不新增一套 metadata。metadata 缺失时，由同一次短条件编译调用给出 `phase_count` 和一句 `split_reason`，不额外调用分类器。

| 现象拓扑 | 段数 | 阶段映射 | 当前样例 |
|---|---:|---|---|
| 连续、渐进变化，没有独立瞬时触发 | 3 | 启动/初态 → 连续演化 → 完成保持 | P08/P09 融化、P12 耗尽、P13–P16 转移/扩散 |
| 接触、阈值、破裂或释放先发生，响应随后出现 | 4 | 接近/累积 → 触发 → 响应演化 → 完成保持 | P01/P04/P06/P10/P11/P17/P19 |
| 重复衰减运动 | 通常 4 | 首次运动 → 首次接触 → 衰减 regime → 静止 | P18；不按每次弹跳继续加段 |
| 多物理并行 | 取最长主因果链的 3 或 4 段 | 各过程共享全局窗口，在每个窗口并列描述 | P20；不为每条并行事件分别调用 GPT |

四段必须同时满足两个硬条件：`trigger.check` 能在短帧窗中单独判定；`evolve.check` 描述的是不同的后续变化。若二者为空、同义或只是把一句话拆成两半，程序直接折叠回三段，不再请求 GPT 修补。首轮上限固定为四段；若五段才说得清，通常意味着 prompt 对约四秒视频过载，应先简化事件或增加帧数。

这条路线与 CoECT 式流水线的边界要写清：

- 每个样例最多一次离线 GPT 调用，在同一次调用中完成二元段数选择和短句生成；输出后缓存，Wan 推理时不调用 GPT。
- 不构造公式化事件链，不让 GPT 搜索事件节点、时间小数或空间坐标。
- 不为每段生成图片，不做关键帧编辑与插值；M3 仍只使用已有的一张 pre-event I0，M4 不增加图片条件。
- 不为每段再调用 VLM；空间实验只复用已有 I0 元数据或少量人工 ROI。
- 选中的 3/4 段文本一次批量做 T5 编码并缓存，在同一次噪声轨迹中联合生成，绝不分别生成再拼接。

**条件也保持最小化。**全视频继续使用原始 `global semantic prompt`；每个被选中的 phase 只新增一对 `positive/counterfactual` 短句和一条不进入模型的 `check`。不生成状态图、公式、数值时刻、逐段图片提示词或长篇物理解释。所有 phase 都保留正反事实，是因为 ACE 的有效经验本来就是“小而对齐的正确—最近失败差分”：`start` 对抗结果预置，可选 `trigger` 对抗触发缺失，`evolve` 对抗中间态删除，`settle` 对抗反转/幽灵源。

每对短句统一满足以下要求：

- 12–26 个英文词，只描述一项肉眼可见的主变化；最多带一项紧密耦合结果。
- 正反句使用完全相同的显式实体、数量和材质词，尽量复用句法，只改变一个关系。
- 反事实肯定地描述最接近的失败，不使用 `not/no/without/avoid/must/should`。
- 不重新描述画质、相机、背景，也不添加原 prompt 没有的施力者、容器、碎片来源或物质。
- `start` 不能提前写完整终态；可选 `trigger` 只写触发而不吞并响应；`evolve` 必须是可见中间过程；`settle` 必须含结果保持，而不重新触发新事件。

下面是可以直接交给 AI 的**单次简化编译提示词**。它只做一次 3/4 二元选择并同时生成短句，不输出公式、不生成图片，也不要求隐藏推理过程。

```text
You are a compact adaptive condition compiler for a 97-frame physical video.
Convert one original video prompt into either three or four short causal condition pairs.
Return strict JSON only. Do not explain your reasoning.

Every video has three base phases:
1. start: preserve the correct initial boundary and show the cause, approach, or onset.
2. evolve: show the main visible intermediate process after the cause.
3. settle: complete the requested result and keep it stable near the end.

Insert exactly one trigger phase between start and evolve only when both are true:
- a contact, threshold, rupture, release, or first impact is independently visible; and
- the later response has a different independently visible check.

Choose phase_count=3 for continuous progressive change with no separable trigger.
Choose phase_count=4 for start -> trigger -> evolve -> settle.
Never output fewer than 3 or more than 4 phases. More verbs, repeated cycles, or
parallel events alone are not reasons to add a phase.

Rules:
- Copy the original prompt verbatim into global_prompt.
- Never create event graphs, equations, numeric times, coordinates, extra objects,
  per-phase images, camera changes, or style descriptions.
- Write exactly one positive and one nearest-failure counterfactual for each selected phase.
- Each sentence must be 12-26 English words and describe one visible main change.
- A positive and its counterfactual must use the same explicit entity, count, and
  material nouns with nearly identical syntax, changing only one causal relation.
- Write counterfactual failures affirmatively. Never use not, no, without, avoid,
  must, or should.
- start must not reveal the completed result at the opening.
- trigger, when present, describes only the visible causal boundary, not the response.
- evolve must contain an observable intermediate process, not an instant replacement.
- settle must describe terminal persistence and, when relevant, exclude reversal,
  regrowth, continued oscillation, wrong source, or an intact ghost source through
  an affirmative counterfactual.
- For transfer, couple source decrease, a connected stream, and sink increase.
- For fracture or phase change, preserve material provenance from the original object.
- For repeated motion, describe the decreasing envelope, not every cycle.
- For parallel events, describe them within the same selected 3/4 phases; do not serialize them.
- check is one short visible criterion for offline audit and is not sent to Wan.
- split_reason is one short auditable sentence. It is not chain-of-thought.
- If trigger and evolve cannot have distinct checks, output phase_count=3.
- When phase_count=4, insert exactly this object between start and evolve:
  {"id":"trigger","positive":"","counterfactual":"","check":""}

Return exactly this schema:
{
  "schema_version": "trace-adaptive34-v1",
  "global_prompt": "",
  "phase_count": 3,
  "split_reason": "",
  "shared_entities": [""],
  "phases": [
    {"id": "start", "positive": "", "counterfactual": "", "check": ""},
    {"id": "evolve", "positive": "", "counterfactual": "", "check": ""},
    {"id": "settle", "positive": "", "counterfactual": "", "check": ""}
  ]
}
```

配套 user prompt：

```text
Compile this prompt into trace-adaptive34-v1 JSON.

ORIGINAL_VIDEO_PROMPT:
{{ORIGINAL_VIDEO_PROMPT}}

OPTIONAL_EXISTING_ACE_METADATA:
{{EXISTING_METADATA_OR_NULL}}
```

程序只做六项廉价校验：JSON 可解析；`phase_count` 只能为 3/4；ID 只能是三段序列或插入 `trigger` 的四段序列；四段时 `trigger.check` 与 `evolve.check` 非空且不同；实体词可追溯到原 prompt且每对正反句实体集合相同；禁用词未出现。使用 JSON schema/constrained decoding 时不自动多轮重试；四段条件不成立就确定性折叠成三段，其他失败记录后走人工/固定模板 fallback，避免编译器本身变成研究系统。

**各段动作的合理性用四个硬问题评判，不再建立复杂评分卡。**

| 阶段 | 必须回答“是”的问题 | 本批典型失败 |
|---|---|---|
| `start` | 正确初态是否出现？原因/接近/变化起点是否先于结果？ | 终态预置、无触发 |
| `trigger`（若有） | 接触/阈值/释放是否清晰出现，并且早于响应？ | 触发消失、因果倒序 |
| `evolve` | 是否看得到连续中间过程？变化是否落在正确实体/来源？ | 瞬间替换、错误水源、整排同时倒 |
| `settle` | 目标是否完成并在最后约 20% 保持？ | 回生、幽灵瓶、不衰减弹跳 |
| 全程 | 同一实体/材料是否连续，相机和无关区域是否稳定？ | 换物、复制、背景泄漏 |

任何一个必答项为“否”，对应阶段即失败，不用总分平均掩盖。对不同物理类型只额外看一个最有辨识力的量：碰撞看“接触先于响应”；转移看“源下降—连通流—汇上升”；相变/破碎看“原材料减少与结果材料增加同源”；弹跳看峰值包络递减；P20 看变化是否留在各自区域。这里不追求质量、动量或连续方程的伪精确数值。

修复也坚持一处失败只动一处：开场预置就改 `start` 对并略加宽启动窗；明确触发被中段吞掉且触发/响应各自可判时才从 3 切到 4；中间态消失就把 `evolve` 改写为更可见的部分状态并加宽中窗；尾部回生就改 `settle` 对；换物/闪烁先统一实体与材质词；错误区域只在 M3 加已有 I0 的粗 ROI。首轮不增加第五段、额外 GPT 调用或新模块，最后才考虑微调 cap/lambda。

**段间连续性依靠生成机制，而不是更多提示词。**所有被选 phase 共享同一原始 semantic、同一实体用词、同一噪声轨迹。程序只内置两套重叠软时间窗：三段沿用 `centers=[0.15,0.50,0.85]`、`widths=[0.22,0.24,0.22]`；四段使用 `centers=[0.12,0.38,0.64,0.88]`、`widths=[0.18,0.20,0.20,0.18]`。这些是全数据集固定的初值，不让 AI 生成时间参数。前一段的可见结果应自然成为后一段的起点，这通过 3/4 条 `check` 离线核对，不再生成额外 bridge prompt。M3 空间实验复用同一对象 ROI/corridor 并做软边界，M4 首轮仅做时间路由。所有 phase 合并后再使用现有安全 cap：M3 2%、M4 1.5%，四段也不会放大总 residual 预算。

因此首轮 TRACE-Lite 能验证的是一个很克制的命题：**连续过程用三段、显式触发过程用四段的二元路由，能否让正确初态、触发/中间过程和稳定终态更容易同时出现。**若它不能改善当前视频中的顺序、过渡、来源和保持问题，就停止增加复杂度，而不是继续扩展段数。

## 6. 与 PhysVid 的关系、创新空间与声明边界

PhysVid 不是“效果很差所以可以忽略”的弱基线。其论文在 Wan2.1-1.3B 上把约 5 秒视频划为 7 个约 0.7 秒的连续 chunk，为每个 chunk 生成 physics prompt，并训练额外的 chunk-aware cross-attention；推理时再用局部正例/反事实做 CFG。论文报告 VideoPhy PC 从 0.24 到 0.32、VideoPhy2 PC 从 0.6144 到 0.6411，但 VideoPhy semantic alignment 从 0.4570 降到 0.4302。作者也明确列出 VLM 标注幻觉、训练—推理时局部 prompt 来源不一致、不同 chunk 语义可能错位，以及每个 Transformer block 新增模块带来的成本问题。参见 [PhysVid 论文](https://arxiv.org/html/2603.26285) 的方法、消融与 limitations。

因此，**“把视频分段并给每段不同物理文本”本身既不够新，也不保证解决当前问题。**我们的增量只有在下面这些区别被真实验证时才形成方法价值：

| 维度 | PhysVid | 建议的 ACE→TRACE 路线 |
|---|---|---|
| 局部单位 | 固定连续时间 chunk 的物理描述 | 3/4 个功能窗口中的正确—最近失败状态差 |
| 条件来源 | 训练视频 chunk 的 VLM 标注；推理仅凭全局文本想象局部 prompt | 复用 ACE metadata；最多一次离线文本编译，推理时零 GPT 调用 |
| 时间结构 | chunk 对齐的 local cross-attention | 两套固定 soft-window 模板，仅按“连续/显式触发”二选一 |
| 空间结构 | 主要是时间 chunk 对齐，没有显式 actor–trigger–patient ROI | 首轮只在 M3 复用 I0 的少量粗 ROI/corridor，不额外调用定位模型 |
| 注入位置 | 每个 Transformer block 新增并训练 local pathway | 继承 ACE：中层窗口 + denoise schedule + output-time + space 四轴稀疏路由 |
| 反事实 | chunk 级不正确物理 prompt，通过 CFG 使用 | 与正确句实体/句法匹配的最近失败差分，并有 1.5%/2% 安全预算 |
| 初态/不变量 | 纯 T2V，未显式分权 | `I0 boundary / protected properties / transition variables` 三分 |
| 目标评测 | 以 SA/PC 聚合分数为主 | 额外测 SCR、intermediate coverage、trigger-before-response、reversal、identity、source–sink coupling、locality leakage |
| 训练成本 | 约 53k 视频，新增模块后两阶段训练 | 首先 training-free 验证机制；有效后才训练 state/router |

相比 PhysVid，我们最可能更好解决的不是笼统“物理常识”，而是本批结果中可具体命名的五类失败：

1. **结果预置与中间态删除**：以正确 phase 对抗对应的失败 phase，而不只是给 chunk 一段物理说明。
2. **触发—响应错位**：仅在触发确实独立可见时插入一个 trigger 窗口，使触发先于响应。
3. **初态约束与真实变化冲突**：I0 只定义边界；尺寸、形状、相态、速度、完整性等 transition variable 被明确允许并要求离开 I0。
4. **空间泄漏与多物理冲突**：例如 P20 只让液体 residual 进入壶口—流—杯区域，火焰和雪地不接收该 residual。
5. **当前失败驱动的因果评测**：明确判罚 no-op、ghost source、middle deletion、time reversal 和错误实体，而不是只依赖一个整体 PC 分数。

合理的创新表述应是：**从 ACE 中验证出的安全 counterfactual residual 出发，以低复杂度的 3/4 功能窗口把正确—最近失败差分路由到输出时间，并在有可靠 I0 时增加粗空间支持。**不能把“自动分段”单独当创新，也不能在当前阶段声称已解决严格守恒或连续动力学；真正的价值仍取决于是否修复本批可命名的生成失败。

如果 TRACE-Time 有效但 TRACE-ST 无增益，论文贡献会更接近已有 temporal local conditioning，创新性有限；如果 role-aware TRACE-ST 能在不伤 P06/P16/P20 的前提下同时减少 P11 的幽灵瓶、P14 的画外液流和 P20 的无关区域污染，则“因果状态转移场 + 条件角色分权”才开始形成明显区别。

## 7. 下一步 3–4 天的增量执行方案

目标不是跑更多视频，而是在 3–4 天内回答三个因果问题：`phase 是否有效`、`space 是否有效`、`收益是否来自路由而非更多文本`。

| 时间 | 实施内容 | 推荐样例 | 产出/验收 |
|---|---|---|---|
| Day 1 上午 | 实现可变长 3/4 phase context、两套固定 temporal gate、最终 residual cap；暂令 `beta=1` | P08 三段、P01 四段各做一次 smoke test | `lambda0=0` 严格回退；3/4 模板均归一；phase×residual 时间曲线正确 |
| Day 1 下午 | 接入一次性 `trace-adaptive34-v1` 编译与缓存；增加 `ACE-global`、`Phase-global-control`、`TRACE-Time` 和 shuffle 开关 | P01、P04、P06、P08 | 编译期最多一次 GPT；推理期零 GPT；能区分更多文本与时间分配 |
| Day 2 | 固定 CFG 3.5、seed 42、tau 2%，跑 8 条机制集；仅在四段样例补 fixed-3 对照 | P01/P04/P06/P08/P11/P13/P16/P18 | 每条输出统一 13 帧审计图；验证二元选择是否优于一律三段 |
| Day 3 上午 | 为 4 条手工构造 soft ROI/tube，先只做 M3 `TRACE-ST` | P01 corridor+contact；P08 ice+puddle；P13 source–stream–sink；P20 liquid ROI 排除火焰/雪地 | mask 与 latent flatten 对齐；ROI 外新增 residual 为 0；无明显运动冻结 |
| Day 3 下午 | 跑 `TRACE-Time` vs `TRACE-ST`，每条 seed 42；对有收益者补第二个 seed | P01/P08/P13/P20 | 判断空间路由是否减少提前倾倒、画外水源、背景泄漏 |
| Day 4（若有） | 冻结 3/4 选择规则和时间窗，只针对已定位的一处失败做小消融并补 2–3 seeds | 优先 P11 ghost bottle 或 P18 衰减弹跳 | 不增加第五段、不扫 event preset；输出 go/no-go 结论 |

**实现上应保持的约束：**

- CFG 固定 3.5；CFG 5 只在最终两个样例做鲁棒性检查，不参与调参。
- `lambda0`、cap、layer 14–23 与现有 safe step schedule 先保持不变；第一轮只增加 output-time，避免再次把变量混在一起。
- 所有 phase 使用完全相同的 semantic 主干；正反句实体集合与句法骨架匹配，每个 phase 只改变一个失败关系。
- 每个样例最多一次离线 GPT 编译且落盘缓存；三/四段 T5 context 一次批量编码，denoising 中不调用文本模型。
- 第一轮只人工标 4 个 ROI，不引入 GroundingDINO/SAM/VLM 在线定位；否则定位误差会与路由误差混在一起。
- 不在这 3–4 天内训练 router，不加入 Reader–Writer/Corrector，不生成多张关键帧，不实现模拟器或 reward search。
- P06、P16、P20 是保护性正例：若 TRACE 修复失败样例却明显破坏这三条，方案不应继续扩大。

**自适应只允许一个二元分支。**`continuous_3` 与 `triggered_4` 共用 layer、denoise、cap 和文本规则，只替换 phase 序列与一套固定时间窗。不能再按融化、碰撞、破碎、流体等类别分别调中心、宽度或强度；否则少量样例上的“自适应”会退化为人工 preset 搜索，既不高效也难以证明泛化。

## 8. 最小实验矩阵、指标与止损线

为证明方法价值，核心只跑 A/B/D 三组；C 仅在被选择为四段的样例上补跑，E 只跑 4 条空间样例，F 只跑 2–3 条顺序敏感样例。这样可以排除“多写文本”的混淆，又不把 3–4 天变成大矩阵。

| ID | 方法 | phase 文本 | output-time gate | spatial gate | 回答的问题 |
|---|---|---:|---:|---:|---|
| A | 当前 `ACE-global-safe` | 1 对 | 否 | 否 | 现有新版基线 |
| B | `Phase-global-control-adaptive34` | 与 D 相同的 3/4 对 | 否；每个时间均等权 | 否 | 仅增加文本是否就有效 |
| C | `TRACE-Time-fixed3` | 固定 3 对 | 是 | 否 | 在触发型样例上，插入 trigger 是否真有必要 |
| D | `TRACE-Time-adaptive34` | 自适应 3/4 对 | 是，两套固定模板 | 否 | 低复杂度二元自适应是否减少预置/跳变/触发缺失 |
| E | `TRACE-ST-adaptive34` | 与 D 相同 | 是 | 是，仅 M3 粗 ROI | 空间支持是否减少泄漏/错误来源 |
| F | `TRACE-Time-adaptive34-shuffled` | 与 D 相同 | 有但顺序打乱 | 否 | 改善是否真的依赖正确阶段顺序 |

统一人工审计七项：

1. 开场是否为正确初态，终态是否未预置；
2. 是否至少有一个可验证中间态；
3. trigger 是否先于 response；
4. 终态是否完成并在最后约 20% 保持；
5. 同一实体/材料是否连续，是否有复制、幽灵源或错误来源；
6. 可单调状态是否按目标方向发展；
7. ROI 外是否出现相关异常变化，画质和相机是否不劣于当前安全 ACE。

3–4 天工程 go 线建议设为：在 8 条机制集上，D 相对 A 至少让 3 条从“无中间态/错误顺序”变为“有清楚中间态且顺序正确”，并且优于 B，P06/P16 不退化；在被选为四段的触发型子集上，D 相对 C 至少让 2 条的“触发先于响应”更清楚；在 4 条空间集上，E 相对 D 至少修复 2 条错误对象/错误来源/区域泄漏，且不能以明显减少运动为代价。第二 seed 只补有收益样例并要求方向一致，不要求短期统计显著。

止损解释如下：

- 若 B 与 D 相对 A 的改善相同：收益来自更多文本，不来自 output-time routing；不要包装成 TRACE。
- 若 D≈A 且 phase residual norm 足够：T5 的 phase 差异没有映射成视频状态，下一步应考虑 state token/projector，而不是增大 lambda。
- 若触发型样例上 D≈C：可选 trigger 没有新增价值，应回到固定三段这个更简单的版本。
- 若 D>C 且只在触发先于响应上稳定改善：这才支持 3/4 二元自适应，而不支持更自由的分段。
- 若 D>A、E≈D：时间结构有效，手工 ROI 或空间 mask 设计无效；先保留 TRACE-Time，重做 grounding。
- 若 E 降低泄漏但使动作明显变少：ROI 太窄、motion corridor 不完整或 cap 按全局范数计算不当。
- 若 shuffled 与正确顺序同样好：方法只增加运动/语义强度，没有真正增强因果性。
- 若只有 M3 有效：这并不是坏结果，说明初态 boundary + grounded transition 是必要条件；应把 M4 保留为困难泛化，而非强行宣称 T2V 通用。

最终建议是：**先完成 `continuous_3 / triggered_4` 的 TRACE-Time 受控证伪，再只在 M3 上增加粗粒度 TRACE-ST。**本批结果已经证明 residual safety、I2V 初态和紧凑关系差分值得保留；下一步真正需要证明的，是一次编译、两套时间窗的轻量方案能否从“全局因果提示”升级为更可靠的输出时间干预。

参考材料：本地设计文档 [wan_appearance_semantics_causal_routing_mvp_20260831.md](/home/liuzhirui/Project/physGen/analysis/wan_appearance_semantics_causal_routing_mvp_20260831.md)，旧批严格审计 [QUALITY_AUDIT_20260901.md](/home/liuzhirui/Project/physGen/code/v1/outputs/ace_router/QUALITY_AUDIT_20260901.md)，以及 [PhysVid: Physics Aware Local Conditioning for Generative Video Models](https://arxiv.org/html/2603.26285)。
