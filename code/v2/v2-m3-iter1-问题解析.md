# v2 M3 iter1 问题解析：实际提示词、M3/M4 生成退化与改进路线

日期：2026-09-02  
M3 分析对象：`/home/liuzhirui/Project/physGen/code/v2/outputs/trace_writer/trace-m3-20260902-094055`  
M4 分析对象：`/home/liuzhirui/Project/physGen/code/v2/outputs/trace_writer/trace-m4-20260902-113035`  
v1 对照对象：`/home/liuzhirui/Project/physGen/code/v1/outputs/ace_router/m3-20260901-125349`、`/home/liuzhirui/Project/physGen/code/v1/outputs/ace_router/m4-20260901-110224`  
分析范围：按用户指定的取数快照，分析 M3/I2V P01–P10 与 M4/T2V P01–P02；分析期间随后生成的样本不追加入本报告。

## 1. 先给结论

用户观察是成立的：当前 v2 的部分结果明显不如 v1，尤其是 M3-P01/P02/P03/P04/P06/P08/P09/P10 和 M4-P02。问题包括背景重绘、类别/身份漂移、外部人物/手/球杆/液流侵入、对象消失、数量变化、状态副本共存、单帧突变、非刚体形变和因果次序错误。M3-P05 和 P07 相对可用，其中 P07 说明 fixed-five TRACE 并非必然破坏视频；它对“单一大物体、单一刚体旋转、空间支持连续且足够大”的任务更容易成功。

M4 给出了很重要的交叉证据。M4-P01 在约 0–10 帧先倒伏、11–16 帧又恢复直立、40–46 帧再次倒伏；M4-P02 的蓝球不是从第一帧静止等待碰撞，而是在约第 70 帧从画面上方生成并下落，球杆和手同时侵入。M4 没有首帧输入，五阶段 mask 又都是全画面，因此这些失败不能归咎于 M3 的首帧框失配；它们直接指向全局实体锚点、阶段文本和固定事件时钟本身。

同时也不能把所有失败都归咎于 v2：v1 的 P08–P10 也存在外部滴管、原物与结果共存、未充分充气就爆裂等问题，v1 M4-P01 还出现横向汽车条带伪影。这表明一部分是 Wan 底模的物理与实例守恒能力不足；v2 的主要问题是没有压住这些底模捷径，局部 TRACE residual 在若干样本上又进一步放大了它们。

但不能把问题只概括成“各阶段 prompt 写得不够好”。目前至少有七个相互叠加的原因，按证据强度排序如下：

1. **全局身份与场景锚点严重变弱。** M3-v1 的主 semantic context 为 158–192 tokens，M4-v1 的 P01/P02 分别为 262/292 tokens；v2 M3 只有 25–32 tokens，v2 M4 只有 21/25 tokens。更关键的是，v2 JSON 中很完整的 `protected_predicates` 并没有进入实际 T5 context。
2. **正例/反例不是足够严格的最小对。** 当前差分不仅包含“正确关系—错误关系”，也混入了对象属性、位置、动作、场景描述等差异。于是 `positive - violation` 可能变成“删除 upright domino / 删除球 / 改造 scooter”的捷径，而不是只修正因果顺序。
3. **静态首帧框与运动后的对象位置失配。** `role_union` 只覆盖首帧对象框；P02 terminal soft area 仅占画面 token 网格的 2.0%，P05 仅 0.9%。对象离开初始框后，终态保持提示基本写不到对象所在位置。
4. **固定阶段边界可能放大语义切换。** 当前仅用 1 个 latent token 做 `0.5/0.5` 交接，约每 4 个输出帧一个 latent token。P03、P06 的最大异常集中在 evolution→completion 的约第 64 帧附近。不过 P04 的删除发生在第 42 帧，P01 的最严重变化在第 78–85 帧，所以边界不是唯一根因。
5. **SafeCap 只约束单层、单步、CFG 前的 residual。** 同一 residual 会经过 10 个 Writer blocks 和 50 个去噪步；之后还进入 CFG 组合。局部稀疏 mask 下，逐 token 10% 的上限仍可能很强，当前又没有本次运行的 residual audit 数据来证明实际是否饱和。
6. **缺少封闭世界、实体生命周期和守恒约束。** P06/P09 会先生成结果、再保留原物；P08/P09 会引入滴管或外部液流；M4-P02 会先省略蓝球、后期再凭空生成。当前 schema 没有把“只允许这些实体存在”“源减少量与结果增加量耦合”“同一实例不能同时处于互斥状态”编译成持续运行约束。
7. **纯文本提示无法独自保证刚体、计数和守恒。** “同一个物体”“保持矩形”“运动传递”等文本，并不等价于几何刚性、实例一一对应、质量守恒或轨迹连续约束。P04、P06、P09 和 P10 已经说明模型会用删除、复制、外源注入或重绘来满足文字终态。

因此，iter1 不应该直接增加更多阶段或加大 `lambda0`。优先顺序应是：**恢复全局锚点 → 加入实体账本/封闭世界/守恒 → 改为严格最小对与 positive-only 对照 → 打开并扩展诊断 → 分别修正 M3/M4 的动态空间支持 → 做 CFG-aware 总预算和边界连续化**。

---

## 2. 当前真正参与生成的 prompt 到底有哪些

这里必须区分“生成 JSON 的 planner prompt”和“真正送入 Wan/T5 的生成 prompt”。

### 2.1 离线 planner prompt：不直接进入视频模型

文件：

```text
/home/liuzhirui/Project/physGen/prompt/generate_trace_writer_v1_plan.txt
```

它的作用是让离线 MLLM 根据 origin、首帧信息和可选物理元数据生成 `PXX-V2-planimg.json`。它规定：

- 输出五阶段 `setup/onset/evolution/completion/terminal`；
- 每阶段生成一个 positive 和 1–2 个 typed violations；
- 生成实体、状态、转移、因果约束、空间框和保护项；
- M3 使用 `visual_grounding`，可以从首帧推断身份、布局和框。

这个大 prompt **本身不会在视频推理时送给 T5 或 Wan**。它只影响 JSON 的质量。

### 2.2 真正被 T5 编码并参与生成的文本

当前 `CompiledTrace.context_texts()` 只返回以下文本：

1. `global_semantic`：全片主语义，作为 conditional 主 context；
2. 五个 `stage.<id>.positive`；
3. 每阶段的 1–2 个 `stage.<id>.negative.<k>`，即 violation/counterfactual；
4. 一个空字符串 `cfg_negative=""`，作为 unconditional CFG context。

对应代码：

```text
/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_compile.py:47
/home/liuzhirui/Project/physGen/code/v2/ace_router/wan_pipeline.py:236
```

实际 residual 形式为：

```text
stage_delta_j = CA(query, positive_j)
                - sum_k rho_jk * CA(query, violation_jk)

TRACE_delta = temporal_gate
              * spatial_gate
              * lambda0
              * layer_gate
              * denoise_step_gate
              * stage_delta_j
```

然后依次做逐 token cap 和全局 cap，并加到 blocks 14–23 的普通 semantic cross-attention 结果上。当前参数为：

| 参数 | 当前值 |
|---|---:|
| `lambda0` | 0.10 |
| Writer blocks | 14–23，共 10 层 |
| 去噪 step gate | 前 30% 为 0.5；30%–70% 为 1.0；70%–90% 为 0.3；最后 10% 为 0.1 |
| token cap | semantic token norm 的 10% |
| global cap | semantic sequence norm 的 2% |
| CFG | 3.5 |
| CFG negative | 空文本，token length 为 1 |

### 2.3 P01 的实际生成文本示例

P01 的主语义是：

> The same black scooter crosses the gray street into the same black slatted trash can, then tilts and stops.

五阶段实际文本如下。表中“反例”不是普通 negative prompt，而是在该阶段 residual 内被做 cross-attention 后从正例方向中减去。

| 阶段 | 正例 positive | 反例 violation |
|---|---|---|
| SETUP | The black scooter rolls rightward across the road gap while both objects retain their pictured upright appearances. | The pictured scooter is already sideways beside the can before crossing the open road gap. |
| ONSET | The scooter front reaches the slatted can before the pictured upright scooter begins rotating sideways. | The scooter rotates sideways over empty road while separation from the can remains visible. |
| EVOLUTION | Following contact, the same black scooter tilts sideways as its rightward travel visibly diminishes beside the upright can. | The scooter tilts beside the can while continuing its original rightward displacement unchanged. |
| COMPLETION | The scooter reaches a clear sideways pose and settles on the street immediately beside the unchanged slatted can. | The scooter keeps a partly upright pictured pose or continues translating past the can. |
| TERMINAL | The same black scooter remains sideways and motionless beside the same upright black slatted trash can. | 反例 1：The sideways scooter rises upright or resumes rolling across the street in final frames. 反例 2：Upright and sideways copies of the black scooter coexist beside the pictured can. |

它们的 T5 token lengths 为 18–28，主 `global_semantic` 为 26 tokens。

### 2.4 M4 实际生成文本示例：没有首帧并不等于没有锚点需求

M4 使用 `PXX-V2-plan.json`，不会把参考图作为模型条件；route 中 `spatial_mode=time`，五个 stage 的空间 mask 都是 100% 全画面。它仍然使用与 M3 相同的四类 T5 context：一条 global、五条 positive、对应 violations，以及空 CFG negative。

M4-P02 的实际 global 只有 25 tokens：

> One red billiard ball collides with one blue billiard ball and transfers visible motion on a billiard table.

实际五阶段文本如下：

| 阶段 | positive | violation |
|---|---|---|
| SETUP | The red ball rolls toward the stationary blue ball while a visible gap remains. | The blue ball is already moving away before the red ball approaches it. |
| ONSET | The red ball visibly contacts the blue ball before the blue ball starts moving. | The blue ball starts moving while visible separation from the red ball remains. |
| EVOLUTION | After contact, the blue ball moves away while the same red ball visibly slows. | The blue ball moves away after contact while the red ball keeps its original speed. |
| COMPLETION | The balls separate after collision with the blue ball moving and the red ball slower. | The collision ends without clear post-contact separation or changed motion in both balls. |
| TERMINAL | The same blue ball remains moving away while the same red ball remains slower. | 反例 1：The blue ball returns to its stationary pre-contact state during the final frames. 反例 2：Duplicated blue balls appear stationary and moving away at the same time. |

这些句子描述了正确事件，但没有形成足够强的**开场存在约束**。实际视频第一帧只有红球，蓝球约到第 70 帧才从上方出现；`one red + one blue` 虽然写在 global 和未消费的 `protected_predicates` 中，却没有实体存在张量、计数损失或首段 verifier 去保证它从 frame 0 就成立。与此同时，`billiard` 的数据先验召回了球杆和手，当前空 CFG negative 与短 global 都没有抑制它们。

M4-P01 的 global 甚至只有 21 tokens：

> One black scooter collides with one metal trash can, tilts, and stops beside it.

这里 `scooter` 没有明确限定 kick scooter，模型从第一帧就选择了 motor scooter。作为对照，v1 M4-P01/P02 的 semantic context 分别有 262/292 tokens，显式包含 initial scene、required entities、allowed changes 和 preserve throughout。v1 仍会失败，说明长 prompt 不是充分条件；但 v2 把这些锚点全部缩成 21–25 tokens，确实显著增加了布局、类别和实例数量的不确定性。

### 2.5 哪些 JSON 内容目前并没有真正作为生成 prompt 使用

以下字段虽然被保存、解析或校验，但没有进入 `context_texts()`：

- `protected_predicates`；
- `observable_check`；
- `entities[].mention` 的完整规范描述；
- `predicates/states/transitions/constraints` 的原始文本；
- `assumptions` 和 `warnings`。

其中 constraints 会间接决定 stage/violation 的选择，`affected_roles + support_kind + entity_boxes` 会间接决定空间 mask；但“exactly one ball”“camera remains static”“no hand/tool”“same rectangular proportions”等保护语义没有直接进入模型。

本次 M3 P01–P10 的所有 `protected_boxes` 都是空数组，而且所有 `grounding.approved` 都是 `false`。M4 P01/P02 按设计没有实体框，且仍然是 `approved=false`。因此当前 JSON 看起来包含保护约束，不代表生成时真正执行了这些保护约束。

`observable_check` 也只是离线可读说明，当前没有 Reader/VLM 在生成中或生成后检查它。

### 2.6 当前 JSON 到底“够不够”

答案需要分两层：

- **用于生成更完整的 prompt：大体够。** P01–P10 已有 entities、global、predicates、states、transitions、constraints、protected predicates、五阶段 positive/violation 和 observable checks，足以确定性编译一条比当前更强的 persistent global anchor，也足以生成初版视频后验 verifier。
- **用于直接保证设计目标：还不够。** 目前缺少显式实体生命周期/存在区间、封闭世界白名单、source→sink 守恒关系、刚体几何不变量、期望运动 corridor/target region 和真实事件时间。M3 的框还是静态且全部未批准；M4 没有空间布局约束。P08/P09 也没有明确写“不得出现滴管、手、外部液流或额外物质来源”。

所以最合理的处理不是重新让 MLLM 自由写更长散文，而是：先把现有 JSON 中已经足够的信息真正编译进模型；再用少量确定性字段补足 `entity_lifecycle`、`closed_world_entities`、`conservation_links`、`motion_corridor/target_region` 和 `event_progress`。这也是后文修复方案的基础。

---

## 3. 五阶段如何映射到 97 帧

当前 97 帧对应 25 个 latent temporal tokens，默认 `crossfade_tokens=1`：

```text
latent 00..03  SETUP core
latent 04      SETUP <-> ONSET
latent 05..07  ONSET core
latent 08      ONSET <-> EVOLUTION
latent 09..15  EVOLUTION core
latent 16      EVOLUTION <-> COMPLETION
latent 17..20  COMPLETION core
latent 21      COMPLETION <-> TERMINAL
latent 22..24  TERMINAL core
```

粗略映射到输出帧，边界位于约 16、32、64、84 帧。VAE 时域卷积会让影响扩散到相邻输出帧，因此不能把某个异常精确归因于单一 latent token；但可以检查异常是否在边界窗口聚集。

当前不是“五段视频分别生成后拼接”，仍然是一条 97 帧 latent 在一次去噪轨迹中联合生成。所谓衔接问题是相邻时间 token 接收的文本 residual 方向变化，不是传统剪辑接缝。

---

## 4. 当前输出与检测指标

### 4.1 评估方法和限制

本报告完成了：

- 对 M3 P01–P10 和 M4 P01–P02 全片每 4 帧抽样一次，每片 25 个观测点，时间间隔约 0.167 秒；
- 对 M3 P08–P10 再加密到每 2 帧一次，每片 49 个观测点，时间间隔约 0.083 秒；
- 对 P02 数量突变、P04 删除、P06 重绘、P08 外物侵入、P09 外部液流、P10 爆裂，以及 M4-P01/P02 的异常窗口做逐帧复核，时间间隔约 0.042 秒；
- 与最新 v1 M3/M4 同样本、同 seed=42 输出做并排观察；
- 在 240×132 下统一复算 RGB 相邻帧平均绝对像素差（MAD）、相邻 SSIM、二阶帧差、四个规范化边界帧差；
- 对容易分割的样本增加颜色/连通域启发式检测，例如 P04 的木色前景骤降、P10 的绿色轮廓面积与扩散范围；
- 检查 manifest、validation、route mask 面积、prompt token 数和 diagnostics。

表中的 MAD 定义为 `mean(abs(RGB[t]-RGB[t-1]))/255`；“最大跳变帧”指变化落到的目标帧编号。像素指标只检测“变化是否大/突兀”，不能判断变化是否正确。典型反例是 P04：对象被删除后画面长期静止，所以平均 MAD 很低，但语义质量极差。另一个反例是 v1 M4-P01：平均变化不大，但中后段出现一条横跨画面的汽车条带伪影。

v1 与 v2 也不是严格单变量实验：v1 CFG=5.0，v2 CFG=3.5；v1 使用长 semantic prompt，v2 使用短 global prompt；Writer、cap 和 step gate 结构不同。尤其 M4 中，v1 为 `lambda0=0.08/residual_cap=0.015`，v2 为 `lambda0=0.10/token_cap=0.10/global_cap=0.02`，两类 cap 不能按数值直接等价。因此表格用于定位退化，不用于声称某个单变量已经被因果证明。

### 4.2 技术完整性

M3 P01–P10 与 M4 P01–P02 均满足：

- 1280×704；
- 97 帧；
- 24 FPS，约 4.04 秒；
- manifest 标记 success；
- M3 的 manifest 明确记录并校验输入首帧及 SHA256；M4 明确记录 `image=null` 和 `image_is_model_condition=false`。

M3 首帧对输入图的 240×132 RGB SSIM 为 0.9881–0.9956；P08/P09/P10 分别为 0.9956/0.9952/0.9956。这说明 M3 的主要问题不是首帧读取错、尺寸错或编码首帧错，而是在后续时域生成中形成。

### 4.3 M3 P01–P10 时序变化指标

MAD 已归一化到 `[0,1]`。数值越大只表示相邻帧变化越多，并不自动表示越差。

| 样本 | v2 平均 MAD | v1 平均 MAD | v2/v1 | v2 最低相邻 SSIM | v2 最大跳变帧 | 解释 |
|---|---:|---:|---:|---:|---:|---|
| P01 | 0.019289 | 0.003627 | **5.32×** | 0.6776 | 79 | 后段背景、车型和主体大范围重绘 |
| P02 | 0.007376 | 0.002959 | **2.49×** | 0.9042 | 85 | 人体/手持续进入，59–63 帧红球复制 |
| P03 | 0.003776 | 0.004570 | 0.83× | 0.8907 | 68 | 均值低源于对象消失后静止；62–68 帧集中崩塌 |
| P04 | 0.001240 | 0.002815 | 0.44× | 0.8139 | 42 | 第 42 帧整排 domino 删除；低均值是假象 |
| P05 | 0.001783 | 0.001520 | 1.17× | 0.9831 | 59 | 时序平滑但滑动变翻滚，局部非刚体形变 |
| P06 | 0.008794 | 0.007139 | 1.23× | 0.7509 | 63 | 巨大闭书侵入并覆盖已生成的开书 |
| P07 | 0.002415 | 0.003176 | 0.76× | 0.9687 | 37 | 本轮最佳，连续但触地时仍挤压变形 |
| P08 | 0.004791 | 0.002838 | **1.69×** | 0.8790 | 89 | 透明杆/滴管样外物侵入，冰块未持续缩小并恢复锐利 |
| P09 | 0.002638 | 0.002218 | 1.19× | 0.9470 | 61 | 黄液增长但 butter pat 保留，约 71 帧起外部液流注入 |
| P10 | 0.004202 | 0.003330 | 1.26× | 0.9457 | 77 | 几乎未充气就爆裂，粉末/硬块式碎片扩散 |

进一步的边界证据：

- P03 第 64 帧 MAD=0.013664，异常峰集中在 61–68 帧；
- P06 第 64 帧 MAD=0.056466，四个边界的平均 MAD 是全片平均的 1.96 倍；其最大二阶帧差为 0.0950，是 v1 的 2.13 倍；
- P01 第 64、84 帧 MAD 分别为 0.032982、0.035581，但最大峰值在 78–85 帧形成一段持续重绘；
- P04 最大跳变在第 42 帧，而非精确边界；说明 stage 内容/对象保持失败同样重要。
- P08 后 80–96 帧 MAD 为 0.008616，而 v1 为 0.002653，约 3.25 倍；这对应冰块/透明外物的持续重构而非可信融化。
- P09 后 80–96 帧 MAD 为 v1 的约 2.13 倍，主要来自持续外部液流，而不是 butter 本体持续减少。

### 4.4 逐帧与任务级检测结果

| 样本 | 高频窗口与检测证据 | 物理含义 |
|---|---|---|
| M3-P02 | 逐帧检查 52–72：约 59–63 帧同时出现两个红球；蓝球仍在，人物/手也一直存在 | 实例数从 2 球变为至少 3 球，碰撞过程被复制捷径替代 |
| M3-P04 | 饱和木色 ROI 面积从 f41 的 68,767 降至 f42 的 8,634，单帧下降 **87.4%** | 不是逐块传播倒下，而是同步删除整排视觉证据 |
| M3-P08 | 逐帧检查 22–42：约 f22 起透明杆状物从右上进入并接触冰块；后段冰块重新恢复接近初始锐利立方体 | 外部工具替代自然融化，且发生 terminal reversal |
| M3-P09 | butter pat 视觉尺寸长期接近原状；约 f71 起黄色细流从画面上方持续注入。中心高饱和黄色面积 f0→f64 已约 26,672→110,433，而源块没有相应收缩 | 结果增长与源消耗未耦合，违反相变质量守恒并引入未声明外源 |
| M3-P10 | 完整轮廓保持到约 f38；此时绿色面积仅为首帧的 **1.022×**，f39–40 已开始破裂。f84 的绿色 bbox 达 `[212,0,1279,703]`，触及上/右/下边界 | 缺少“大幅充气”中间态；碎片不是 centered/in-place，而是无界粉末爆散 |
| M4-P01 | f0–10 第一次倒伏，f11–16 又恢复直立，f40–46 第二次倒伏 | 同一不可逆事件被执行两次，中间出现逆转；固定 stage 不等于单调事件进度 |
| M4-P02 | 首帧无蓝球；约 f70 蓝球从上方进入，f70–76 向下撞红球。f47 的最大跳变对应大球杆突然横穿画面 | 初态、方向、触发者和因果顺序全部错误；并非普通边界抖动 |

上述启发式在 1280×704 原帧上计算：P04 使用 OpenCV HSV `H=4..30,S>=90,V>=60` 并限制 `y=210..519`；P09 使用中心 ROI `x=250..1049,y=150..649` 内的 `H=15..45,S>=100,V>=80`；P10 使用 `H=35..95,S>=60,V>=35`。颜色阈值只作为支持证据，不替代实例分割或语义判断。P09 的不锈钢反光会污染黄色统计，M4-P02 的黑蓝球杆会污染蓝色阈值，所以最终事件结论均以逐帧视觉复核为主，数值只用于定位时间窗。

### 4.5 M4 P01–P02 指标与 v1 对照

| 样本 | v2 平均 MAD | v1 平均 MAD | v2/v1 | v2 最低 SSIM（帧） | v2 最大跳变帧 | 二阶突变证据 |
|---|---:|---:|---:|---:|---:|---|
| M4-P01 | 0.036890 | 0.006559 | **5.62×** | 0.4222（10） | 1 | 最大二阶差 0.1505，约为 v1 的 4.37× |
| M4-P02 | 0.020421 | 0.003217 | **6.35×** | 0.6230（47） | 47 | 最大二阶差 0.1605，约为 v1 的 8.46× |

M4-P01 的高变化主要来自开场快速运动、人物/背景漂移和“倒下→恢复→再倒下”。但 v1 M4-P01 也不是正确答案：它虽保持 kick scooter 类别，却在中后段生成一条横跨全宽、内部有汽车移动的水平条带。两者属于不同失败类型，不能只根据 MAD 宣称哪一个整体更好。

M4-P02 则可以较明确地判断 v2 更差：v1 至少从第一帧给出红蓝两球和稳定球桌，尽管中段也发生红球复制、且没有正确碰撞；v2 连蓝球初态都没有，随后引入手和球杆，再让蓝球从空中生成。M4 的交叉比较因此支持“长初态/数量锚点有价值，但仅靠长 prompt 仍不足以获得正确物理”。

### 4.6 validation 和运行诊断目前能说明什么

M3 P01–P07 均为 `0 errors + 2 warnings`，P08–P10 为 `0 errors + 1 warning`；M4 P01/P02 为 `0 errors + 0 warnings`。但 M4-P02 实际几乎所有关键事件都错了。这直接说明 validation 只证明 JSON 结构和因果引用可编译，**不证明视频满足 observable checks**。

更重要的是，本轮 `reader_mode=off`。12 个成片的 `trace.diagnostics.jsonl` 各只有一条普通 generation 记录，共 12 条；没有：

- 每层/每步 candidate 与 applied residual norm；
- token/global cap 激活比例；
- 相邻 stage residual cosine；
- ROI 内外 residual 泄漏；
- 各 latent time token 的 residual 能量；
- 可见事件与目标 stage 的对齐误差。

相比之下，当前 v1 对照每个样本有 500 条 residual diagnostics。v2 即使打开现有 AuditReader，也只记录总体 norm 和 cap，不包含设计文档要求的 per-stage、boundary cosine、ROI in/out 等关键项。因此现在只能从输出反推问题，不能直接证明是哪一个 stage residual 在何时失控。

### 4.7 空间 mask 覆盖与 M3/M4 的相反失配

下表为 `soft_area / 22×40 grid_area`，不是硬阈值后的像素比例：

| 样本 | SETUP | ONSET | EVOLUTION | COMPLETION | TERMINAL |
|---|---:|---:|---:|---:|---:|
| P01 | 21.7% | 18.4% | 23.7% | 23.7% | 14.3% |
| P02 | 17.5% | 10.6% | 21.4% | 17.5% | **2.0%** |
| P03 | 19.6% | 15.5% | 21.9% | 19.6% | 10.0% |
| P04 | 28.1% | 27.1% | 28.7% | 28.1% | 26.5% |
| P05 | 21.0% | 35.3% | 32.7% | 32.7% | **0.9%** |
| P06 | 24.7% | 23.7% | 26.1% | 29.7% | 11.4% |
| P07 | 40.4% | 40.2% | 41.1% | 41.1% | 40.2% |
| P08 | 58.8% | 58.8% | 58.8% | 58.8% | 58.8% |
| P09 | **88.4%** | **88.4%** | **88.4%** | **88.4%** | **88.4%** |
| P10 | **2.2%** | **2.2%** | **2.2%** | **2.2%** | **2.2%** |

P07 在所有阶段都覆盖约 40%，且是质量最好的一例；P02/P05 的 terminal 只保留首帧小对象框，明显无法覆盖运动后的终点。P10 也把膨胀和爆裂始终限制在首帧小气球的 2.2% 区域，无法约束扩张后的 envelope/fragments。相反，P09 因为把大平底锅作为 sink，五阶段均覆盖 88.4%，局部 residual 几乎退化为全局编辑，为外部液流和大面积重绘提供了空间。P04 虽然 mask 足够大仍失败，说明扩大 mask 不是万能修复。

M4 的五阶段空间 mask 全部为 100%，这避免了“物体走出首帧框”，却产生相反问题：任何 stage residual 都能改写整幅背景并召回手、球杆、行人或其他场景先验。由此可见，M3 需要动态 corridor/target，M4 需要实体感知的时变空间归属；二者不能共用“静态小框”或“全画面”这两个极端。

---

## 5. 逐样本问题解析

### P01：kick scooter 变成 motor scooter，背景被重绘

观察：

- 前半段还能保持首帧中的电动滑板车；
- 约 52–68 帧开始增加车身、座椅/外壳样结构，逐渐变成摩托式 scooter；
- 墙、树、绿篱、路边布局出现明显重绘和视角漂移；
- 最终没有可信地“接触后侧倒”，而是一个形态已改变的车辆停在垃圾桶附近。

原因判断：

- `scooter` 本身有 kick scooter / motor scooter 词义歧义；阶段内没有始终使用完整 canonical mention “the same upright black electric kick scooter”；
- “same identity、red cable detail、static side camera、background unchanged”都只在未使用的 `protected_predicates` 中；
- completion/terminal 强推“sideways”，模型可能把它解释成侧视外观，而非刚体倒伏；
- 终段持续 residual 加上缺乏背景 anchor，造成大范围重绘。

证据：统一复算后，v2 平均 MAD 为 v1 的 5.32 倍；后 80–96 帧 MAD 为 0.032290，而 v1 为 0.002412，约 13.4 倍。退化不是单个接缝，而是持续到结尾的大范围重绘。

### P02：球碰撞退化成人手操作

观察：

- 约第 32 帧手开始从上方进入；
- 后段人物躯干和双手占据画面，蓝球变成人为触碰/控制；
- 逐帧看 52–72 帧，约 59–63 帧出现两个红球并存，随后其中一个出框/消失；
- 没有形成干净、可读的“红球接触蓝球后动量转移”；
- 红球后段离开可见范围，exactly-one 的持续性也无法验证。

原因判断：

- billiards 的数据先验很容易召回球员/手，但“exactly one red/blue ball、static camera”未进入主 prompt；
- JSON 里没有直接执行的“画面中始终无人/无球杆”锚点；
- terminal mask 仅 2.0%，且位于首帧球的位置，蓝球移动后不再受终态提示稳定约束；
- v1 末段也出现了少量人物先验，说明这是底模弱点；v2 的短全局锚点和阶段差分把它显著放大。

### P03：接触后球拍、球和手发生突变，随后对象消失

观察：

- 约 56 帧球到达拍面；
- 60–66 帧球拍突然旋转/移动，约第 62 帧还出现手；
- 68–74 帧球拍和球相继离场或消失，只留下暗影/伪影与空球场；
- 目标要求是球反弹离开且球拍保持不变，实际是通过删除实体制造“分离”。

原因判断：

- “fixed racket、exactly one ball、same racket identity”未进入主语义；
- completion 正例要求 widening interval，反例要求“never gains separation”。这种非最小差分可能把“去掉其中一个对象”当成最容易增大间隔的路径；
- evolution→completion 的 latent 16 边界约对应第 64 帧，恰好覆盖崩塌窗口；
- terminal role_union 仍是首帧球和球拍框，无法跟随反弹后的球。

注意：P03 的平均 MAD 比 v1 低并不是更好，而是对象消失后帧间变化趋近于零。

### P04：整排 domino 瞬间删除，随后出现手和少量孤立木条

观察：

- 0–41 帧大部分 domino 基本直立；
- 第 42 帧附近整排突然只剩约 1–2 个直立木条，没有传播式倒下中间态；
- 后段出现手去操作孤立 domino；
- 数量、传播、相邻碰撞、终态保持全部失败。

这是本轮最清晰的“像素指标低但语义失败”样本。v1 对照能够生成从左向右的连续倒伏序列。

原因判断：

- planner warning 明确写着“exact domino count is intentionally not asserted”；这使数量保护先天不足；
- 即使 `protected_predicates` 写了 single straight row，它也未进入 T5；
- P04 的正反例词集合 Jaccard 仅 0.111，相邻 stage positives 仅 0.112，是 P01–P10 中最低组之一，远非严格最小对；
- completion 反例包含“far-right upright tail”，从该 context 做减法可能直接抑制 upright-domino 视觉特征，模型以删除而非旋转来满足目标；
- hand 是 domino 场景的常见数据先验，没有强全局 empty-scene anchor 去抑制。

### P05：总体可用，但“滑动”变成明显翻滚/形变

观察：

- 单块沿斜坡向右移动并到达地面，场景和数量总体稳定；
- 约 52–76 帧木块明显转动、呈菱形/不规则透视，动作更像翻滚而不是沿面滑动；
- 终点基本合理，时序指标与 v1 接近。

原因判断：

- prompt 强调 displacement 缩小，但没有把“块体姿态保持、接触面不换边”作为真正执行的条件；
- 纯文本 `same rectangular block` 不能强制刚体边长和角点距离不变；
- terminal mask 仅 0.9%，但此例主要靠底模先验完成，不能因此认为该 mask 设计正确。

### P06：终态提前出现并与源书本共存，随后巨大书本侵入

观察：

- 开始时闭合书在架上；约 20–24 帧，打开的书已经出现在地面；
- 一段时间内“架上闭书 + 地上开书”同时存在，违反实例守恒；
- 56–70 帧架上闭书放大并从前景跌落/覆盖地上开书，形成巨大伪影；
- 70 帧后额外书消失，只剩开书。

原因判断：

- global prompt 在全片直接包含 “lands open”，而 setup 局部 residual 不足以阻止终态提前生成；
- “同一本书、closed/open copies 不共存”只在 protected/terminal violation 中，且 terminal 出现得太晚；
- 把“Closed and open ... copies coexist”作为需要编码的反例，仍可能在 T5/cross-attention 中激活“两本书”的视觉概念；简单相减不保证像逻辑 NOT；
- 最大突变在 62–65 帧，正好覆盖 evolution→completion 交接窗口。

### P07：本轮正向样本，但仍有非刚体压缩

观察：

- 单一纸箱连续倾倒，数量、背景和方向保持良好；
- 最后能够停在侧面；
- 接触地面前后，箱体边缘略鼓、底角压缩、矩形比例改变。

为什么它相对成功：

- 单主体、单宏观旋转、没有实例间复杂交换；
- 空间 mask 在全部阶段都约为 40%，不会因对象移动立刻失配；
- stage 语义与默认五段时间分配较一致。

它也说明下一轮应保留 P07 作为 non-regression control，不能只在失败样本上调参。

### P08：冰没有融化，模型改用透明杆/滴管制造“水”

观察：

- 0–21 帧冰块几乎完全不变；约 f22 起一条透明细杆从右上进入，随后与冰块接触并形成叉状/玻璃碎片样结构；
- 中段冰块短暂圆化、膨胀或重构，但没有形成“固体 footprint 单调缩小、液体 footprint 单调增大”的联动；
- 后段冰块又恢复接近初始的锐利立方体，只出现少量不稳定透明区域；
- v1 也出现明显滴管/玻璃棒，所以这是强底模先验；v2 没有消除它，且后段变化 MAD 是 v1 的 3.25 倍。

原因判断：

- `transparent water spreads` 很容易让模型调用“滴管/倒水”视觉先验；现有 JSON 的 protected predicates 没有写 `no tool / no external liquid source`；
- 冰块和 plate 合并后的 mask 固定覆盖 58.8%，没有单独的 source-decrease 与 sink-increase 区域；
- violation 明写 “ice regains its large sharp pictured shape”，虽然理论上被相减，但 cross-attention 差分不是符号 NOT，仍可能激活恢复锐利冰块的表征；
- 没有任何在线/后验检查真正执行 `cube contracts as reflective area expands`。

### P09：黄液增加但 butter 不减少，后段又从画面外注入液流

观察：

- 约 22–28 帧黄色 puddle 已在 butter pat 周围出现，但矩形块长期保留近似原高度、体积和锐利边缘；
- puddle 继续扩大，形成“原块 + 熔化结果”共存；
- 约 f71 起，一条黄色细流从画面上方持续灌入中心，直到结尾；
- v1 也有 butter 与大 puddle 共存，但没有 v2 后段这样持续、清晰的外部液流。

原因判断：

- P09 的所有 stage mask 都因大 pan sink 扩张到 88.4% 画面，TRACE 几乎可在全图生成黄色内容；
- 正例反复强调 `pool spreads`，却没有一个可执行的等量/单调约束要求 `solid area or height decreases`；
- protected predicates 只保护材质、锅和相机，没有封闭外部物质来源；
- positive 与 violation 的词集合 Jaccard 仅 0.154，差分同时改变了液体、固体、形状和范围，容易学到“多生成黄色区域”而不是“同一 butter 相变”。

### P10：没有明显充气过程，直接爆成粉末和硬块

观察：

- 到 f38，绿色完整轮廓面积最多只比首帧增加约 2.2%，远未达到 prompt 所说的 `large taut balloon`；
- f39–40 已开始破裂，直接跳过主要 inflation/evolution 中间态；
- 结果更像绿色粉末爆炸加多个硬质几何块，不像一个橡胶膜撕裂后形成少量相连/卷曲薄片；
- f84 后绿色区域触及画面上、右、下边界，违反 `fragments remain centered/in place`；
- v1 同样未充分充气就爆裂，说明底模对这类不可逆相变的默认表示就很弱；v2 的扩散范围和后段持续变化更大。

原因判断：

- P10 的 mask 五阶段始终只有首帧小气球区域 2.2%，膨胀后与飞散碎片立即走出控制区；
- fixed-five 只规定文本时间，不保证可见面积达到阈值后才允许 rupture；
- global 从全片一开始就同时包含 `inflates, bursts, leaves fragments`，底模倾向压缩过程，直接选取最显著的爆裂终态；
- `no inflation tool or person` 虽已写入 protected predicates，但没有进入实际 context；更关键的橡胶薄膜拓扑/碎片形态约束也不存在。

### M4-P01：同一倒伏事件发生两次，中间反向恢复

观察：

- 第一帧即生成 motor scooter 而不是 kick scooter，并有未要求的行人；
- f0–10 scooter 快速倾倒，f10 附近模糊最严重；
- f11–16 又重构为直立 scooter；
- f40–46 在 trash can 旁第二次连续倒下，之后保持；
- v1 M4-P01 保住了 kick scooter 类别，但中后段出现带汽车的水平条带伪影，因此两个版本都不合格，只是失败机制不同。

原因判断：

- M4 global 只有 21 tokens，`scooter` 类别歧义没有消解；
- time-only mask 为 100%，阶段 residual 可重绘整幅场景；
- 没有单调事件进度或 hysteresis，模型可以 `tilted→upright→tilted`；
- setup、completion、terminal 分别强调不同姿态，但共享身份锚点太弱，相邻 operator 不是同一刚体轨迹上的小增量。

### M4-P02：蓝球后期从空中生成，球杆成为真正触发者

观察：

- frame 0 只有一个红球，计划要求的静止蓝球缺失；
- 早段黑色球杆和手反复进入，约 f47 球杆横穿画面并产生全片最大跳变；
- 约 f70 蓝球从画面上方出现，f70–76 向下撞向红球；
- 最终是“外来蓝球/球杆撞红球”，而非“红球沿桌面撞静止蓝球并传递动量”；
- v1 M4-P02 至少从第一帧生成红蓝两球和稳定球桌，但也出现红球复制且未正确碰撞。v2 在初态存在性和触发者上退化更明显。

原因判断：

- `one red/one blue` 没有被编译成 frame-0 existence 和全时域 count constraint；
- M4 无首帧时尤其依赖完整 scene/layout anchor，而实际 global 仅 25 tokens；
- `billiard` 强召回 cue/hand，当前 CFG negative 为空、protected 中也没有 `no cue/no person`；
- stage 文本只描述运动关系，没有约束运动平面和方向；蓝球从上方进入也能被语言模型粗略视为“blue ball moves and contacts red ball”。

---

## 6. Prompt 质量和“衔接”到底占多大责任

### 6.1 Prompt 质量确实有问题，但最严重的是信息没有进入真正的 prompt

M3 P01–P10 的 v2 `global_semantic` 仅 25–32 tokens；对应 v1 semantic 为 158–192 tokens。M4 的差距更大：v2 P01/P02 只有 21/25 tokens，而 v1 为 262/292 tokens。v1 主语义明确包含：

- required visible entities；
- at the beginning；
- required state changes；
- allowed-to-change variables；
- preserve throughout。

v2 把这些内容拆进结构化 JSON，却只把一句短 global 和局部 stage 文本送进模型。结构化程度提高了，但底模真正能看到的全局约束反而减少。这是 v2 退化最可信的首要原因。

### 6.2 当前 violation bank 不是可靠的“逻辑取反器”

对 M3 P01–P10 做一个粗糙的词集合 Jaccard 检查：

- stage positive 与对应 violation 平均重合仅 0.215；
- 相邻 stage positive 平均重合仅 0.184；
- P04 分别只有 0.111 和 0.112；P09 的 positive/violation 只有 0.154；P10 的相邻 positives 只有 0.112。

M4 P01/P02 的对应均值为 0.250/0.247；M4-P02 即使词面重合较高仍然失败，说明 Jaccard 只是最小对纯度的粗筛，不是充分指标。Cross-attention 输出也不是线性的符号逻辑，`CA(c+) - CA(c-)` 不保证只删除错误关系。反例中出现的 “copies、upright tail、closed book、duplicated blue balls” 等内容仍可能被模型表征并污染生成。

### 6.3 阶段衔接是放大器，不是唯一根因

当前 1-token crossfade 从某阶段 core 权重 1.0 变到边界 0.5，再到下一 token 的 0，离散导数较大。如果相邻 residual 方向本来就不一致，简单 `0.5*r_j + 0.5*r_{j+1}` 不能保证视觉连续。

P03/P06 的异常支持边界风险，但 P04 在 f42 删除、P10 在 f39–40 提前爆裂、M4-P01 在 f0–16 倒伏后恢复，都不发生在某个唯一理想边界。所以只把 `CROSSFADE_TOKENS` 从 1 改到 2 可能缓解局部跳变，却解决不了身份、数量、事件逆转、静态框和负向语义污染。

### 6.4 开环固定时间与真实事件没有同步

五阶段是规范化事件时间，不知道视频里真正何时接触、落地或打开：

- P06 的“落地打开”在约 24 帧就提前出现；
- P05 的主要滑动反而到约 48 帧后才明显；
- P04 在 evolution 初期直接删掉整排，而非形成传播前沿。
- P10 应在 completion 才 rupture，却约 f39–40 就爆裂；
- M4-P02 的蓝球到约 f70 才出现，早期 setup/onset 连被撞对象都不存在。

这属于 Stage Alignment Error。没有在线 Reader 时，固定边界只能是一种先验，不能保证对所有物理类型都对齐。

### 6.5 “提示词问题”和“底模问题”必须拆开

P04 的 v1 能生成连续 domino cascade，而 v2 删除整排；M4-P02 的 v1 能从开场保持两种颜色球，而 v2 后期才生成蓝球。这些是 v2 信息通路/残差设计造成退化的强证据。

但 P08–P10 的 v1 同样明显失败：P08 用滴管制造水、P09 保留 butter 同时生成大 puddle、P10 未充分膨胀就爆裂。这里即使恢复 v1 长 prompt，也未必能解决；需要显式 closed-world/conservation/geometry 控制、候选 verifier，甚至针对物理变化的数据或结构性条件。正确结论是：**prompt 与衔接确实有问题，但不是全部问题；它们目前既丢掉关键锚点，又没有为底模缺失的物理能力提供替代约束。**

### 6.6 M3 与 M4 暴露的是两种相反的空间问题

- M3 有首帧身份信息，但静态框会随运动失配；P02/P05/P10 的后期主体走出小 mask。
- M4 没有首帧，当前直接把 stage residual 施加到全画面；这会让局部因果词汇改写背景并召回伴随物体。
- 因此“统一扩大 mask”会让 M4 更糟，“统一缩小 mask”又让 M3 丢失运动目标。应采用实体轨迹/attention-derived support，而不是固定面积策略。

---

## 7. iter1 推荐的可落地修复方案

### 7.1 第一优先级：恢复“全局不可变锚点”

增加一个确定性 `global_anchor_compiler`。M3 可先以 60–120 tokens 为目标，M4 因为没有图像提供布局/身份，可允许 100–180 tokens；长度只是预算，真正要求是下面的信息全部出现：

- 每个实体的 canonical mention 和精确数量；
- 初始状态；
- 总体事件；
- 允许变化的属性白名单；
- 身份、颜色、材质、刚体比例、相机和背景保持项；
- 不允许新增的 agent/object，优先用肯定式场景描述而不是只写 `no/not`。

例如 P02 可改成：

> A locked side-view camera shows exactly one glossy red billiard ball and exactly one glossy blue billiard ball on unchanged green cloth. The empty space above the table remains empty. Both balls keep the same round shape, size, color, and identity. Only the two ball positions and velocities change: the red ball moves right, makes visible contact, slows, and the blue ball then moves right.

这里的核心不是一味增加长度，而是保证“不可变化量”持续存在于每一步、每个空间位置的 main context 中，而不是只存在于局部 stage residual 或 JSON 备注中。M4 还必须显式编译 `from the first frame`、左右/上下关系、运动平面、空区域和完整入镜要求，不能把布局完全留给一句事件摘要。

### 7.2 第二优先级：把 stage prompt 改成共享锚点的最小关系对

不要让正反例分别自由写整句。改用模板：

```text
shared anchor: same entities, same camera, same unrelated attributes
positive relation: CONTACT precedes TILT
counterfactual relation: TILT precedes CONTACT
```

两句应只替换一个 relation/order token span。例如 P01 ONSET：

```text
positive:
The same electric kick scooter reaches the same trash can; contact occurs before tilt.

counterfactual:
The same electric kick scooter reaches the same trash can; tilt occurs before contact.
```

共同的身份部分在差分中近似抵消，差分更集中于 before/after。全局身份由 global anchor 单独承担。

同时增加两个必要消融：

1. `positive-only`：只加 stage positive，不减 violation；
2. `minimal-pair contrast`：使用模板化最小对。

若 positive-only 明显优于当前 contrast，说明 violation subtraction 是主要污染源，不应继续堆更多反例。

### 7.3 第三优先级：加入封闭世界实体账本与守恒耦合

仅靠 protected sentence 还不够。建议在 schema/compiler 中增加确定性字段：

```text
closed_world_entities: [red_ball#1, blue_ball#1, table#1]
entity_lifecycle: visible(red_ball#1,t)=1, visible(blue_ball#1,t)=1
forbidden_new_roles: [person, hand, cue, tool, external_source]
state_exclusion: closed_book#1 + open_book#1 are the same slot, not two instances
conservation_link: decrease(source_solid) ~= increase(product_liquid)
monotonic_progress: melt_area nondecreasing; solid_area nonincreasing
```

实现上可先分三层，不必一次训练新模型：

1. 把实体白名单、全时域存在和禁止外源编译进 persistent anchor；
2. 用 tracker/segmenter 在生成后计算 count、source/sink 面积联动和额外 agent，作为 seed verifier；
3. verifier 稳定后，再将其做成可微 guidance 或 latent trust-region。

这直接针对 P02/P04/P06/P08/P09/M4-P02 的共同问题：模型目前把“得到正确终态”当作开放式图像编辑，而不是对同一组实体做守恒状态转移。

### 7.4 第四优先级：动态目标区域与终态 mask

当前 M3 只有首帧 box。对会移动的实体，至少要在 plan/编译产物中显式表示：

```text
initial_box
allowed_motion_corridor
expected_target_region
terminal_support = dilated(corridor union target_region)
```

具体修改方向：

- terminal 不再默认退回首帧 `role_union`；
- P02 的蓝球 terminal 应覆盖碰撞后的右侧通道；
- P03 的球 terminal 应覆盖拍面右侧的 outgoing corridor；
- P05 的木块 terminal 应覆盖坡底和右侧地面，而不是首帧 0.9% 小框；
- 对 P04 这种多实例行列，不能只用一个 row box；至少需要 count/integrity mask 或多个代表性实例锚点；
- P08/P09 应拆开 `source solid` 与 `sink liquid` mask，分别施加 decrease/increase，而不是把整只 plate/pan 当成统一生成区域；
- P10 需要随 envelope 扩张的 predicted support，burst 后切换为受限 fragment corridor，不能一直使用 2.2% 初始框；
- 首轮 pilot 的 grounding 必须人工查看 overlay 并置 `approved=true`，不要把全部 unapproved box 直接用于结论实验。

可以自动把 motion ROI 的补集作为 background-protect 区域，避免要求 MLLM 为墙、地板、绿篱逐一画框。

M4 不能简单继续使用 100% mask。两个不依赖外部首帧的方案：

- **同一次去噪内的 attention-derived support**：用 global entity mention 的 cross-attention/saliency 得到粗对象区域，经过时域平滑和面积上限后路由 stage residual；
- **低分辨率 draft→track→重生成**：先只用 persistent anchor 生成低成本草稿，检测实体轨迹和事件时刻，再把轨迹/区域作为第二次 T2V 的结构条件。它不使用用户首帧，但属于两阶段生成，应作为独立 M4 消融而不是偷偷改变定义。

### 7.5 第五优先级：CFG-aware、support-aware 的总预算

当前 cap 是单层单步且位于 CFG 前。建议：

- 记录 CFG 后相对 base conditional 的最终 noise-prediction delta；
- cap 目标除以 `guide_scale`，或直接在 CFG 后做 trust-region；
- global cap 的参考 norm 改成相同 active support 内的 semantic norm，而不是全序列 semantic norm；
- 对 10 层累积设置 layer-group 总预算，而不只是每层各自 2%；
- 初始保守 sweep：`lambda0 ∈ {0.03, 0.05, 0.10}`、token cap `{0.03, 0.05, 0.10}`、global cap `{0.005, 0.01, 0.02}`，先看 audit 后再选择，不按单样本手调。

M3-v1 的 `lambda0=0.1/residual_cap=0.02` 与 v2 数值相近却在多例更稳定，因此不能直接断言“0.1 一定过大”；M4-v1 则更保守（`lambda0=0.08/residual_cap=0.015`，早期 step gate 0.25，而 v2 为 0.5）。更可信的判断是：短 global anchor、非最小对、局部/全局 support 和预算共同改变了有效扰动强度，必须在统一 CFG 和相同 anchor 下做消融。

### 7.6 第六优先级：边界安全的 residual morph 与单调事件时钟

短期可以先跑已有 `crossfade_tokens=2`，但更合理的版本是：

1. 在边界计算相邻 stage operators 的 cosine；
2. 若方向一致，正常平滑插值；
3. 若 cosine 很低或为负，只保留共同分量，衰减冲突分量；
4. 对相邻 latent time token 增加 residual derivative cap：限制 `||D_f-D_{f-1}||`；
5. 使用 C1 连续的 smoothstep/Hann 权重，而非只有一个 `0.5/0.5` 点。

这可以称为 **Boundary-Cosine Trust Region**。它不需要训练，也不会根据文本数量放大总预算。

但 M4-P01 的“倒下→站起→再倒下”说明只平滑边界还不够。再增加一个低自由度、单调的 `event_progress p(t)` 与 hysteresis：一旦 `tilt/contact/rupture` 被 verifier 判定发生，后续 stage 不允许回到互斥前态；completion 只有在 evolution 的可见阈值达到后才解锁。P10 可用 `area_ratio >= target` 才允许 burst，P02 可用 `both_entities_visible && gap≈0` 才允许 transfer。

### 7.7 第七优先级：把诊断补到设计文档要求的水平

pilot 默认 `READER_MODE=audit`，每个 step/layer/stage 至少记录：

- positive、每个 violation、positive-minus-negative 的 norm；
- positive/negative cosine；
- 相邻 stage operator cosine；
- token cap fraction、global cap coefficient；
- 每个 latent time token 的 applied norm；
- ROI 内/外 applied norm；
- CFG 后相对 base prediction 的 delta；
- 边界前后 residual derivative。

生成后自动计算：

- adjacent SSIM/MAD、光流速度和二阶光流 jerk；
- mask 外背景 LPIPS/SSIM 和非预期光流；
- GroundingDINO/SAM2 + tracker 的实例数量与 identity continuity；
- 刚体对象角点距离/长宽比变化；
- JSON `observable_check` 派生的任务指标。

只有把 residual 指标和视频异常帧对应起来，才能判断该降 lambda、改 prompt，还是改 route。

---

## 8. 更有开创性的方向

### 8.1 Anchor–Dynamics 双流 Writer

将条件明确拆成两条：

```text
Anchor stream:
身份、数量、材质、背景、相机、刚体比例；全时域持续、低强度

Dynamics stream:
每阶段唯一允许变化的状态量；时空路由、可变强度
```

当前系统只有短 global semantic + 动态 stage delta，保护信息被丢在 JSON。双流能让“什么不能变”和“此刻什么要变”不再互相争夺同一个文本 residual。

对 M4，Anchor stream 还承担 scene construction：先稳定“从第一帧有哪些实体、在哪里、空白区域是什么”，Dynamics stream 才负责碰撞或倒伏。这样可以避免用全画面 stage residual 同时承担造景和改运动。

### 8.2 基于不变量的 residual 投影

对已知刚体/数量不变量，构造一个近似保护子空间：

- 用首帧对象特征或 DINO patch features 表示 identity anchor；
- 如果 stage residual 与 identity-preserving gradient 强冲突，则投影掉冲突分量；
- 背景区域的 TRACE delta 严格投影为零；
- 对刚体对象，限制 residual 导致的非仿射形变，只允许近似 SE(2)/SE(3) 运动分量。

这是比“再加一句 keep shape”更接近几何约束的方法。

### 8.3 JSON 驱动的因果 verifier 与候选选择

不立即做昂贵的在线闭环，先生成少量候选 seed，再用 JSON 自动编译 verifier：

- P02：红球运动 → 接触 → 蓝球运动，红球减速；始终两球；
- P04：倒下数量随时间单调增加，总实例数不变；
- P06：始终一本书；质心先离架下降，接触地面后再打开；
- P07：四边长度比例稳定，角点轨迹连续。
- P08：固体面积单调下降、液体面积单调上升，无工具/外源；
- P09：source 减少与 sink 增加负相关，无来自画面边界的液流；
- P10：先达到最小膨胀面积阈值再破裂，fragment bbox 不触边；
- M4-P02：frame 0 即有恰好两球，二者沿桌面同一轴运动，无 cue/hand/person。

用 verifier 排序 seed，既能得到短期收益，也能积累未来 Reader/Corrector 的监督信号。

### 8.4 事件时间扭曲，而不是继续增加阶段数

保留五种职责，但允许一个低自由度、单调的 time warp 将规范化阶段映射到实际事件进度。warp 可由：

- 离线任务类型先验；
- 低分辨率候选的接触/运动检测；
- 未来可靠 Reader 的状态概率；

来决定。它比让 MLLM自由输出五个时间点更稳，也比固定所有任务在第 32/64 帧转阶段更符合快速碰撞、缓慢滑动和传播过程的差异。

### 8.5 物理脚手架：让视频模型负责外观，不独自发明动力学

对底模反复失败的 P04/P08/P09/P10，可以把问题拆成“粗物理状态轨迹 + 视频外观渲染”：

- 刚体/碰撞：用简化 2D/3D simulator 产生质心、姿态、接触时刻和 point tracks；
- domino：产生单调传播的实例姿态序列；
- 融化/扩散：产生 source/sink 面积曲线、level-set 或粗 segmentation sequence；
- balloon：产生 envelope 面积曲线和破裂时刻，破裂后只给少量薄膜 fragment tracks。

再将 optical flow、depth、segmentation、point trajectory 或 sparse control tokens 作为结构条件交给 Wan。这样模型只需生成真实纹理、遮挡和光照，不再凭语言独自求解守恒和接触动力学。若不能修改底模输入，可先把这些轨迹用于候选筛选或 residual projection。

### 8.6 针对反复失败物理类型的小规模 LoRA/adapter

若恢复 anchor、最小对、动态 support 和 verifier 后，P08–P10 仍在多个 seed 中系统失败，就应承认它是能力缺口而不是继续堆 prompt。可用合成/实拍短序列训练轻量 adapter，监督内容不是完整视频语义，而是：实体持续、相变 source/sink 联动、刚体比例、事件前后顺序和无外源。训练前必须先建立 base/anchor-only/TRACE 三组对照，避免用微调掩盖 Writer 本身的问题。

---

## 9. 最小、可执行的实验矩阵

不要立刻重跑 20 个。分两层：

- 快速机制集：M3-P01/P02/P04/P06/P07，加 M4-P01/P02；覆盖身份漂移、数量、删除、状态共存、成功对照和无首帧模式。
- 物理能力集：M3-P08/P09/P10；只在 anchor/contrast 机制通过后运行，用于判断是否必须加入守恒脚手架或微调。

第一轮 seed=42；机制明确后再扩到 seeds `{42, 43, 44}`。M3 与 M4 分开报告，不能用一个均值掩盖相反的空间问题。

| 组 | 主 semantic | stage residual | 空间/边界 | 目的 |
|---|---|---|---|---|
| A | 当前短 global | `lambda0=0` | 无 TRACE | 判断短 global 自身能生成到什么程度 |
| B | v1 长 semantic | `lambda0=0` | 无 TRACE | 分离“prompt 长锚点”与 ACE/TRACE 收益 |
| C | 新 deterministic anchor | `lambda0=0` | 无 TRACE | 单独测结构化 anchor 编译收益 |
| D | 新 anchor | positive-only | 当前 mask，lambda 0.03/0.05 | 判断 violation subtraction 是否污染 |
| E | 新 anchor | 当前自由文本 contrast | 当前 mask，同预算 | 给现有 TRACE 一个受控基线 |
| F | 新 anchor | 模板化 minimal pair | 当前 mask，同预算 | 检查真正 relation contrast |
| G | 同 F | minimal pair | crossfade 1→2 | 只测边界平滑，不混入其他修改 |
| H-M3 | 同 F | minimal pair | dynamic corridor/source-sink + support-aware cap | 验证 M3 静态框修复 |
| H-M4 | 同 F | minimal pair | full-frame vs attention-derived support | 验证 M4 全画面泄漏修复 |
| I | 同 H | minimal pair | monotonic event clock/hysteresis | 修复重复事件、提前爆裂和逆转 |
| J | v1 条件与参数复现 | v1 ACE | v1 | 公平参考，不用历史视频替代受控对照 |

所有组必须统一：checkpoint、首帧、seed、分辨率、steps、solver、shift 和 CFG。当前历史 v1/v2 的 CFG 不同，下一轮应固定同一 CFG 后再比较。

推荐 Go/No-Go 判断：

- B/C 明显优于 A：全局 anchor 缺失是主因；比较 B 与 C 可判断新 compiler 是否达到 v1 长 prompt 的信息覆盖；
- D 明显优于 E：当前 violation subtraction 有害；
- F 优于 D/E：最小关系对提供有效因果增益；
- G 只改善边界但不改善数量/身份：符合预期，不夸大 crossfade 作用；
- H-M3 改善 P02/P05/P10 终态、H-M4 减少 cue/hand/background 泄漏：证明两种模式需要不同动态 support；
- I 消除 M4-P01 的重复倒伏和 P10 的提前爆裂：证明单调事件时钟有效；
- 正确顺序与 shuffled 顺序无差异：停止宣称模型利用了因果顺序，回到表示层重做。

---

## 10. 下一轮验收指标

### 通用质量

- M3 首帧 fidelity 不低于当前；M4 单独检查 frame-0 entity/layout completeness；
- 全片主体 identity embedding 连续；
- 实例数量与计划一致，不出现新增手、人、工具或复制体；
- route 外背景光流、LPIPS 和颜色漂移不高于 base/v1；
- 边界 ±4 帧的 MAD、SSIM、flow jerk 不形成异常离群峰。
- 除计划声明的新产物外，不允许从画面边缘或空区域生成新的物质来源。

### 物理过程

- P01：contact frame < first-tilt frame；垃圾桶保持直立；scooter 长宽/关键点比例稳定；
- P02：红球先动、接触后蓝球再动；始终恰好两球；没有人手；
- P03：接触后球速度方向翻转；球拍中心和姿态基本固定；球始终存在；
- P04：fallen-count 单调增加、总 domino count 保持、传播前沿连续；
- P05：块体质心沿坡面/地面连续，姿态变化受限，速度包络下降；
- P06：始终一本书，先离架下降、后接触、再打开；
- P07：刚体角点间距和长宽比变化受限，落地后角速度衰减。
- P08：冰 solid area 单调下降、water area 单调上升，两者显著负相关；无杆/滴管/外部液体；
- P09：butter 高度/固体面积下降与 puddle 增长同步；顶部边界无进入液流；
- P10：完整 envelope 面积达到预设增幅后才 rupture；碎片为有限橡胶薄片且 bbox 不触画面边界；
- M4-P01：只出现一次 `upright→tilted`，不发生恢复直立或第二次倒伏；
- M4-P02：frame 0 即有一红一蓝两球，二者始终沿桌面平面运动，红球接触早于蓝球起动，无 cue/hand/person。

阈值不应先拍脑袋固定；先以 v1、base Wan 和 P07 当前成功结果建立分布，再设置 non-inferiority 与改进阈值。

---

## 11. 最终建议

当前最值得立即做的并不是重写更多华丽的 stage 文本，而是修复“结构化信息生成了、却没有真正喂给模型”的断层，并补上实体守恒。建议 iter1 的实现顺序为：

1. 编译并注入 persistent global anchor；
2. 增加 closed-world entity ledger、全时域 existence 与 source→sink conservation 字段；
3. 增加 positive-only 与 minimal-pair 两种 stage residual，先证明 violation subtraction 是否有益；
4. 默认 pilot 开启并扩展 AuditReader，把 residual、边界和 ROI 指标与异常帧对齐；
5. M3 实现 dynamic corridor/source-sink support，M4 对照 full-frame 与 attention-derived support；
6. 用 M3-P01/P02/P04/P06/P07 与 M4-P01/P02 做统一 CFG 的小矩阵；
7. 再评估 2-token/smooth boundary、单调 event clock 与 Reader-driven time warp；
8. P08–P10 若仍跨 seed 失败，转向 verifier、物理脚手架或小规模 adapter，不继续堆自然语言反例。

一句话概括：**v2 当前不是“因果信息太少”，而是“因果 JSON 很丰富，但真正贯穿全片的身份/数量/场景锚点太少，局部阶段差分又不够纯净，同时没有把状态转移变成封闭、守恒、单调的同一实体过程”；先修信息通路、实体账本和残差可观测性，再谈增加控制强度。**

## 附录：证据文件

- v2 推理入口：`/home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer.py`
- v2 Wan pipeline：`/home/liuzhirui/Project/physGen/code/v2/ace_router/wan_pipeline.py`
- prompt 编译：`/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_compile.py`
- TRACE residual：`/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_writer.py`
- 固定时间路由：`/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_masks.py`
- AuditReader：`/home/liuzhirui/Project/physGen/code/v2/ace_router/controllers.py`
- planner prompt：`/home/liuzhirui/Project/physGen/prompt/generate_trace_writer_v1_plan.txt`
- 本轮 M3 输出：`/home/liuzhirui/Project/physGen/code/v2/outputs/trace_writer/trace-m3-20260902-094055`
- 本轮 M4 输出：`/home/liuzhirui/Project/physGen/code/v2/outputs/trace_writer/trace-m4-20260902-113035`
- v1 M3 对照：`/home/liuzhirui/Project/physGen/code/v1/outputs/ace_router/m3-20260901-125349`
- v1 M4 对照：`/home/liuzhirui/Project/physGen/code/v1/outputs/ace_router/m4-20260901-110224`
