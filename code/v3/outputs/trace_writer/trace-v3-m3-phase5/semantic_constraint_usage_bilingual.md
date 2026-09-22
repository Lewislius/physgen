# 语义与约束使用报告（可读版）

> 来源：`semantic_constraint_usage_bilingual.csv`。阶段正、负提示词读取自每个 P 的 `trace.contexts.compiled.json`，因此下文展示的是推理时实际编码的完整文本，而不是输入 plan 中仅用于规划的原始描述。
> 阶段中文译文读取自经过人工整理的 `prompt_table_bilingual.md`，只用于阅读；实际推理编码的仍是所列英文原文。

## 阅读说明

- `c1`、`c2` 等是当前 P 内部的约束 ID；不同 P 中相同编号的具体含义可能不同。
- `positive` 是该阶段实际编码的正提示词。
- `negative.N` 是实际编码的第 N 个负提示词；多个负提示词按照所列权重加权。
- 各 P 使用折叠区块，避免所有长文本同时展开。

## 总览

| P | 时间路由 | 阶段文本实际选中约束 | Global couples | 仅校验约束 |
|---|---|---|---|---|
| P01 | triggered | setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5 | c3 | c2 |
| P02 | triggered | setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5 | c3 | c2 |
| P03 | triggered | setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5 | c3 | c2 |
| P04 | triggered | setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5 | c3 | c2 |
| P05 | multi_transition | setup→c1; onset→c1; evolution→c2; completion→c5; terminal→c6,c7 | 无 | c3,c4 |
| P06 | multi_transition | setup→c1; onset→c1; evolution→c5; completion→c6; terminal→c7,c8 | 无 | c2,c3,c4 |
| P07 | triggered | setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5 | c3 | c2 |
| P08 | continuous | setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5 | c3 | 无 |
| P09 | continuous | setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5 | c3 | 无 |
| P10 | multi_transition | setup→c1; onset→c1; evolution→c2; completion→c5; terminal→c6,c7 | 无 | c3,c4 |
| P11 | triggered | setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5 | c3 | c2 |
| P12 | continuous | setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5 | c3 | 无 |
| P13 | continuous | setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5 | c3 | 无 |
| P14 | continuous | setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5 | c3 | 无 |
| P15 | continuous | setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5 | c3 | 无 |
| P16 | continuous | setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5 | c3 | 无 |
| P17 | multi_transition | setup→c1; onset→c1; evolution→c5; completion→c6; terminal→c7,c8 | 无 | c2,c3,c4 |
| P18 | triggered | setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5 | c3 | c2 |
| P19 | continuous | setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5 | c3 | 无 |
| P20 | continuous | setup→c1; onset→c1; evolution→c2; completion→c3; terminal→c4,c5 | c3 | 无 |

## 每个 P 的完整内容

<details>
<summary><strong>P01</strong> — triggered；setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5</summary>

### P01 概要

- 时间路由类型：`triggered`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`c2`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains scooter: exactly 1; trash can: exactly 1. The reference
> first frame fixes these identities and initial layout. [INITIAL STATE] gap=wide, pose=upright,
> motion=rolling_right. [PHYSICAL EVENT] The same black scooter crosses the gray street into the same
> black slatted trash can, then tilts and stops. [PERSISTENT PROTECTION] same black scooter identity
> and red cable detail; same black slatted metal trash can; trash can remains upright; gray street,
> curb, hedge, concrete wall, daylight, and static side camera remain unchanged. [CLOSED WORLD] No
> additional external source, extra copy, unlisted object enters or appears; unoccupied scene regions
> remain empty. [CONSERVATION] Source loss and result gain stay coupled: tracked logical entity slots
> remain consistent; sideways tilt grows while rightward displacement diminishes. No result appears
> from an undeclared external source. [TERMINAL STATE] The same logical entity slots persist without
> duplicates; gap=contact, pose=tilted_sideways, motion=stopped remains visible through the final
> frames.

#### 中文解释

> [实体清单] 封闭场景包含 scooter：恰好 1 个；trash_can：恰好 1 个。参考首帧固定这些身份和初始布局。[初始状态]
> gap=wide、motion=rolling_right、pose=upright。[物理事件] 同一辆黑色踏板车横穿灰色街道，撞上同一个黑色条板式垃圾桶，随后侧倾并停止。[持续保护]
> 保持同一辆黑色踏板车的身份和红色线缆细节；保持同一个黑色条板式金属垃圾桶；垃圾桶保持直立；灰色街道、路缘、树篱、混凝土墙、日光和静止侧视相机保持不变。[封闭世界]
> 不允许新增外部来源、额外副本、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；侧向倾斜增大，同时向右位移减小。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；gap=contact、motion=stopped、pose=tilted_sideways 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`136`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: scooter: exactly 1, state gap=wide, pose=upright,
> motion=rolling_right, required; trash can: exactly 1, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VALID RELATION] the scooter front visibly
> reaches the trash can happens BEFORE stem and deck rotate sideways.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 踏板车（scooter）: 恰好1个，状态 gap=wide, pose=upright, motion=rolling_right，必需; 垃圾桶（trash
> can）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 踏板车前端明显抵达垃圾桶先于车把立杆与踏板向侧面转动发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`139`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: scooter: exactly 1, state gap=wide, pose=upright,
> motion=rolling_right, required; trash can: exactly 1, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] stem and deck rotate
> sideways is already present BEFORE the scooter front visibly reaches the trash can.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 踏板车（scooter）: 恰好1个，状态 gap=wide, pose=upright, motion=rolling_right，必需; 垃圾桶（trash
> can）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 在踏板车前端明显抵达垃圾桶发生之前，车把立杆与踏板向侧面转动已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c2`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`138`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: scooter: exactly 1, state gap=wide, pose=upright,
> motion=rolling_right, transitioning; trash can: exactly 1, transitioning. Counts and identities stay
> fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] the scooter front
> visibly reaches the trash can happens BEFORE stem and deck rotate sideways.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 踏板车（scooter）: 恰好1个，状态 gap=wide, pose=upright, motion=rolling_right，正在过渡; 垃圾桶（trash
> can）: 恰好1个，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 踏板车前端明显抵达垃圾桶先于车把立杆与踏板向侧面转动发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`139`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c2`

> [STAGE ENTITY INVENTORY] Present now: scooter: exactly 1, state gap=wide, pose=upright,
> motion=rolling_right, transitioning; trash can: exactly 1, transitioning. Counts and identities stay
> fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] stem and deck
> rotate sideways happens BEFORE the scooter front visibly reaches the trash can.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 踏板车（scooter）: 恰好1个，状态 gap=wide, pose=upright, motion=rolling_right，正在过渡; 垃圾桶（trash
> can）: 恰好1个，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 车把立杆与踏板向侧面转动先于踏板车前端明显抵达垃圾桶发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`141`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: scooter: exactly 1, state gap=contact, pose=tilted_sideways,
> motion=stopped, required; trash can: exactly 1, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VALID RELATION] The coupled changes occur
> together: stem and deck rotate sideways WHILE rightward displacement decreases to zero.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 踏板车（scooter）: 恰好1个，状态 gap=contact, pose=tilted_sideways, motion=stopped，必需; 垃圾桶（trash
> can）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 耦合变化同时发生：车把立杆与踏板向侧面转动，同时向右位移减小至零。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`148`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3`

> [STAGE ENTITY INVENTORY] Present now: scooter: exactly 1, state gap=contact, pose=tilted_sideways,
> motion=stopped, required; trash can: exactly 1, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] Only one coupled change
> occurs while the other remains unchanged: stem and deck rotate sideways; rightward displacement
> decreases to zero.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 踏板车（scooter）: 恰好1个，状态 gap=contact, pose=tilted_sideways, motion=stopped，必需; 垃圾桶（trash
> can）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 只有一个耦合变化发生，另一个保持不变：车把立杆与踏板向侧面转动；向右位移减小至零。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`143`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: scooter: exactly 1, state gap=contact, pose=tilted_sideways,
> motion=stopped, transitioning; trash can: exactly 1, transitioning. Counts and identities stay
> fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The coupled changes
> occur together: stem and deck rotate sideways WHILE rightward displacement decreases to zero.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 踏板车（scooter）: 恰好1个，状态 gap=contact, pose=tilted_sideways, motion=stopped，正在过渡;
> 垃圾桶（trash can）: 恰好1个，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 耦合变化同时发生：车把立杆与踏板向侧面转动，同时向右位移减小至零。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`133`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3, c4`

> [STAGE ENTITY INVENTORY] Present now: scooter: exactly 1, state gap=contact, pose=tilted_sideways,
> motion=stopped, transitioning; trash can: exactly 1, transitioning. Counts and identities stay
> fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] The declared
> cause occurs BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 踏板车（scooter）: 恰好1个，状态 gap=contact, pose=tilted_sideways, motion=stopped，正在过渡;
> 垃圾桶（trash can）: 恰好1个，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`173`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: scooter: exactly 1, state gap=contact, pose=tilted_sideways,
> motion=stopped, required; trash can: exactly 1, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VALID RELATION] The terminal facts
> pose=tilted_sideways, motion=stopped REMAIN through the final frames. [AND] The same logical slot
> has only its valid state; mutually exclusive facts pose=upright, pose=tilted_sideways do not
> coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 踏板车（scooter）: 恰好1个，状态 gap=contact, pose=tilted_sideways, motion=stopped，必需; 垃圾桶（trash
> can）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实 pose=tilted_sideways（侧倒）、motion=stopped（停止） 在最后几帧持续保持。 [且]
> 同一逻辑槽位只具有其有效状态；互斥事实 pose=upright（直立）与 pose=tilted_sideways（侧倒） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`165`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: scooter: exactly 1, state gap=contact, pose=tilted_sideways,
> motion=stopped, required; trash can: exactly 1, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] The valid terminal state
> returns to its mutually exclusive earlier state. [AND] The same logical slot has only its valid
> state; mutually exclusive facts pose=upright, pose=tilted_sideways do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 踏板车（scooter）: 恰好1个，状态 gap=contact, pose=tilted_sideways, motion=stopped，必需; 垃圾桶（trash
> can）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 pose=upright（直立）与
> pose=tilted_sideways（侧倒） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`161`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: scooter: exactly 1, state gap=contact, pose=tilted_sideways,
> motion=stopped, required; trash can: exactly 1, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] The terminal facts
> pose=tilted_sideways, motion=stopped REMAIN through the final frames. [AND] Copies of the same
> logical slot appear in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 踏板车（scooter）: 恰好1个，状态 gap=contact, pose=tilted_sideways, motion=stopped，必需; 垃圾桶（trash
> can）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实 pose=tilted_sideways（侧倒）、motion=stopped（停止） 在最后几帧持续保持。 [且]
> 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：front_contact 必须先于 side_tilt。 |
| `c2` | `requires` | 因果依赖：side_tilt 只有在 front_contact 之后才能发生。 |
| `c3` | `couples` | 耦合约束：side_tilt, rightward_stop 必须共同变化；关系为“侧向倾斜增大，同时向右位移减小”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 pose=tilted_sideways, motion=stopped 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 pose=upright, pose=tilted_sideways 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "side_tilt",
    "before": "front_contact",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "cause": "front_contact",
    "effect": "side_tilt",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "type": "requires"
  },
  {
    "effects": [
      "side_tilt",
      "rightward_stop"
    ],
    "id": "c3",
    "relation": "sideways tilt grows while rightward displacement diminishes",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "pose=tilted_sideways",
      "motion=stopped"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "pose=upright",
      "pose=tilted_sideways"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=2个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"scooter":"derived_motion_envelope","trash_can":"unresolved_static_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P02</strong> — triggered；setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5</summary>

### P02 概要

- 时间路由类型：`triggered`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`c2`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains red ball: exactly 1; blue ball: exactly 1. The
> reference first frame fixes these identities and initial layout. [INITIAL STATE] gap=wide,
> red_motion=rolling_right, blue_motion=stationary. [PHYSICAL EVENT] The glossy red ball crosses the
> green billiard cloth into the glossy blue ball and transfers motion on contact. [PERSISTENT
> PROTECTION] exactly one glossy red ball and one glossy blue ball; ball colors, size, and roundness
> remain unchanged; green billiard cloth, dark top rail, gray wall, lighting, and static camera remain
> unchanged. [CLOSED WORLD] No additional cue, cue stick, external source, extra ball, extra copy,
> hand, person, tool, unlisted object enters or appears; unoccupied scene regions remain empty.
> [CONSERVATION] Source loss and result gain stay coupled: tracked logical entity slots remain
> consistent; red rightward displacement decreases while blue rightward displacement begins. No result
> appears from an undeclared external source. [TERMINAL STATE] The same logical entity slots persist
> without duplicates; gap=contact, red_motion=slower_right, blue_motion=moving_right remains visible
> through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 red_ball：恰好 1 个；blue_ball：恰好 1 个。参考首帧固定这些身份和初始布局。[初始状态]
> blue_motion=stationary、gap=wide、red_motion=rolling_right。[物理事件]
> 有光泽的红球穿过绿色台球桌布撞上有光泽的蓝球，并在接触时传递运动。[持续保护]
> 恰好一个有光泽的红球和一个有光泽的蓝球；球的颜色、大小和圆度保持不变；绿色台球布、深色顶边、灰墙、照明和静止相机保持不变。[封闭世界]
> 不允许新增台球杆、台球杆、外部来源、额外球体、额外副本、手、人物、工具、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；红球向右位移减小，同时蓝球开始向右位移。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；blue_motion=moving_right、gap=contact、red_motion=slower_right 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`158`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: red ball: exactly 1, state gap=wide, red_motion=rolling_right,
> required; blue ball: exactly 1, state blue_motion=stationary, required. Counts and identities stay
> fixed; no duplicate state copy is created. No additional cue, cue stick, external source, extra
> ball, extra copy, hand, person, tool, unlisted object exists. All globally protected identity,
> shape, material, scene, and camera attributes stay unchanged. Declared source loss and result gain
> remain coupled. [VALID RELATION] the red and blue glossy ball boundaries visibly meet happens BEFORE
> rightward blue displacement begins.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 红球（red ball）: 恰好1个，状态 gap=wide, red_motion=rolling_right，必需; 蓝球（blue ball）: 恰好1个，状态
> blue_motion=stationary，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：台球杆、台球杆、外部来源、额外球体、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 红、蓝两个光泽球的边界明显接触先于蓝球开始向右位移发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`161`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: red ball: exactly 1, state gap=wide, red_motion=rolling_right,
> required; blue ball: exactly 1, state blue_motion=stationary, required. Counts and identities stay
> fixed; no duplicate state copy is created. No additional cue, cue stick, external source, extra
> ball, extra copy, hand, person, tool, unlisted object exists. All globally protected identity,
> shape, material, scene, and camera attributes stay unchanged. Declared source loss and result gain
> remain coupled. [VIOLATED RELATION] rightward blue displacement begins is already present BEFORE the
> red and blue glossy ball boundaries visibly meet.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 红球（red ball）: 恰好1个，状态 gap=wide, red_motion=rolling_right，必需; 蓝球（blue ball）: 恰好1个，状态
> blue_motion=stationary，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：台球杆、台球杆、外部来源、额外球体、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 在红、蓝两个光泽球的边界明显接触发生之前，蓝球开始向右位移已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c2`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`160`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: red ball: exactly 1, state gap=wide, red_motion=rolling_right,
> transitioning; blue ball: exactly 1, state blue_motion=stationary, transitioning. Counts and
> identities stay fixed; no duplicate state copy is created. No additional cue, cue stick, external
> source, extra ball, extra copy, hand, person, tool, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VALID RELATION] the red and blue glossy ball boundaries visibly meet
> happens BEFORE rightward blue displacement begins.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 红球（red ball）: 恰好1个，状态 gap=wide, red_motion=rolling_right，正在过渡; 蓝球（blue ball）: 恰好1个，状态
> blue_motion=stationary，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：台球杆、台球杆、外部来源、额外球体、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 红、蓝两个光泽球的边界明显接触先于蓝球开始向右位移发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`161`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c2`

> [STAGE ENTITY INVENTORY] Present now: red ball: exactly 1, state gap=wide, red_motion=rolling_right,
> transitioning; blue ball: exactly 1, state blue_motion=stationary, transitioning. Counts and
> identities stay fixed; no duplicate state copy is created. No additional cue, cue stick, external
> source, extra ball, extra copy, hand, person, tool, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VIOLATED RELATION] rightward blue displacement begins happens BEFORE
> the red and blue glossy ball boundaries visibly meet.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 红球（red ball）: 恰好1个，状态 gap=wide, red_motion=rolling_right，正在过渡; 蓝球（blue ball）: 恰好1个，状态
> blue_motion=stationary，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：台球杆、台球杆、外部来源、额外球体、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 蓝球开始向右位移先于红、蓝两个光泽球的边界明显接触发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`162`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: red ball: exactly 1, state gap=contact,
> red_motion=slower_right, required; blue ball: exactly 1, state blue_motion=moving_right, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional cue, cue stick,
> external source, extra ball, extra copy, hand, person, tool, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VALID RELATION] The coupled changes occur together: rightward
> red displacement decreases WHILE rightward blue displacement begins.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 红球（red ball）: 恰好1个，状态 gap=contact, red_motion=slower_right，必需; 蓝球（blue ball）: 恰好1个，状态
> blue_motion=moving_right，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：台球杆、台球杆、外部来源、额外球体、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 耦合变化同时发生：红球向右位移减小，同时蓝球开始向右位移。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`169`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3`

> [STAGE ENTITY INVENTORY] Present now: red ball: exactly 1, state gap=contact,
> red_motion=slower_right, required; blue ball: exactly 1, state blue_motion=moving_right, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional cue, cue stick,
> external source, extra ball, extra copy, hand, person, tool, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VIOLATED RELATION] Only one coupled change occurs while the
> other remains unchanged: rightward red displacement decreases; rightward blue displacement begins.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 红球（red ball）: 恰好1个，状态 gap=contact, red_motion=slower_right，必需; 蓝球（blue ball）: 恰好1个，状态
> blue_motion=moving_right，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：台球杆、台球杆、外部来源、额外球体、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 只有一个耦合变化发生，另一个保持不变：红球向右位移减小；蓝球开始向右位移。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`164`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: red ball: exactly 1, state gap=contact,
> red_motion=slower_right, transitioning; blue ball: exactly 1, state blue_motion=moving_right,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> cue, cue stick, external source, extra ball, extra copy, hand, person, tool, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VALID RELATION] The coupled changes occur
> together: rightward red displacement decreases WHILE rightward blue displacement begins.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 红球（red ball）: 恰好1个，状态 gap=contact, red_motion=slower_right，正在过渡; 蓝球（blue ball）:
> 恰好1个，状态 blue_motion=moving_right，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：台球杆、台球杆、外部来源、额外球体、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [有效关系] 耦合变化同时发生：红球向右位移减小，同时蓝球开始向右位移。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`156`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3, c4`

> [STAGE ENTITY INVENTORY] Present now: red ball: exactly 1, state gap=contact,
> red_motion=slower_right, transitioning; blue ball: exactly 1, state blue_motion=moving_right,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> cue, cue stick, external source, extra ball, extra copy, hand, person, tool, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] The declared cause occurs
> BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 红球（red ball）: 恰好1个，状态 gap=contact, red_motion=slower_right，正在过渡; 蓝球（blue ball）:
> 恰好1个，状态 blue_motion=moving_right，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：台球杆、台球杆、外部来源、额外球体、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [违规关系] 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`204`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: red ball: exactly 1, state gap=contact,
> red_motion=slower_right, required; blue ball: exactly 1, state blue_motion=moving_right, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional cue, cue stick,
> external source, extra ball, extra copy, hand, person, tool, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VALID RELATION] The terminal facts red_motion=slower_right,
> blue_motion=moving_right REMAIN through the final frames. [AND] The same logical slot has only its
> valid state; mutually exclusive facts blue_motion=stationary, blue_motion=moving_right do not
> coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 红球（red ball）: 恰好1个，状态 gap=contact, red_motion=slower_right，必需; 蓝球（blue ball）: 恰好1个，状态
> blue_motion=moving_right，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：台球杆、台球杆、外部来源、额外球体、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实
> red_motion=slower_right（较慢向右）、blue_motion=moving_right（向右运动） 在最后几帧持续保持。 [且] 同一逻辑槽位只具有其有效状态；互斥事实
> blue_motion=stationary（静止）与 blue_motion=moving_right（向右运动） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`191`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: red ball: exactly 1, state gap=contact,
> red_motion=slower_right, required; blue ball: exactly 1, state blue_motion=moving_right, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional cue, cue stick,
> external source, extra ball, extra copy, hand, person, tool, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VIOLATED RELATION] The valid terminal state returns to its
> mutually exclusive earlier state. [AND] The same logical slot has only its valid state; mutually
> exclusive facts blue_motion=stationary, blue_motion=moving_right do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 红球（red ball）: 恰好1个，状态 gap=contact, red_motion=slower_right，必需; 蓝球（blue ball）: 恰好1个，状态
> blue_motion=moving_right，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：台球杆、台球杆、外部来源、额外球体、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实
> blue_motion=stationary（静止）与 blue_motion=moving_right（向右运动） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`189`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: red ball: exactly 1, state gap=contact,
> red_motion=slower_right, required; blue ball: exactly 1, state blue_motion=moving_right, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional cue, cue stick,
> external source, extra ball, extra copy, hand, person, tool, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VIOLATED RELATION] The terminal facts red_motion=slower_right,
> blue_motion=moving_right REMAIN through the final frames. [AND] Copies of the same logical slot
> appear in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 红球（red ball）: 恰好1个，状态 gap=contact, red_motion=slower_right，必需; 蓝球（blue ball）: 恰好1个，状态
> blue_motion=moving_right，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：台球杆、台球杆、外部来源、额外球体、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实
> red_motion=slower_right（较慢向右）、blue_motion=moving_right（向右运动） 在最后几帧持续保持。 [且] 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：boundary_contact 必须先于 blue_depart。 |
| `c2` | `requires` | 因果依赖：blue_depart 只有在 boundary_contact 之后才能发生。 |
| `c3` | `couples` | 耦合约束：red_decelerate, blue_depart 必须共同变化；关系为“红球向右位移减小，同时蓝球开始向右位移”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 red_motion=slower_right, blue_motion=moving_right 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 blue_motion=stationary, blue_motion=moving_right 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "blue_depart",
    "before": "boundary_contact",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "cause": "boundary_contact",
    "effect": "blue_depart",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "type": "requires"
  },
  {
    "effects": [
      "red_decelerate",
      "blue_depart"
    ],
    "id": "c3",
    "relation": "red rightward displacement decreases while blue rightward displacement begins",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "red_motion=slower_right",
      "blue_motion=moving_right"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "blue_motion=stationary",
      "blue_motion=moving_right"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=2个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"blue_ball":"derived_motion_envelope","red_ball":"derived_motion_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P03</strong> — triggered；setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5</summary>

### P03 概要

- 时间路由类型：`triggered`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`c2`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains ball: exactly 1; racket: exactly 1. The reference first
> frame fixes these identities and initial layout. [INITIAL STATE] gap_state=right_separated,
> direction=leftward_toward_racket. [PHYSICAL EVENT] The yellow tennis ball crosses the dark green
> court leftward into the black-and-white racket and departs rightward after contact. [PERSISTENT
> PROTECTION] exactly one yellow tennis ball; same black racket frame, white strings, and white grip;
> dark green backdrop, blue strip, green court, lighting, and static camera remain unchanged. [CLOSED
> WORLD] No additional external source, extra copy, unlisted object enters or appears; unoccupied
> scene regions remain empty. [CONSERVATION] Source loss and result gain stay coupled: tracked logical
> entity slots remain consistent; rightward reversal accompanies reopening separation from the string
> bed. No result appears from an undeclared external source. [TERMINAL STATE] The same logical entity
> slots persist without duplicates; gap_state=right_departing, direction=rightward_away remains
> visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 ball：恰好 1 个；racket：恰好 1 个。参考首帧固定这些身份和初始布局。[初始状态]
> direction=leftward_toward_racket、gap_state=right_separated。[物理事件]
> 黄色网球在深绿色球场上向左运动并撞上黑白球拍，接触后向右离开。[持续保护]
> 恰好一个黄色网球；保持同一黑色球拍框、白色拍弦和白色握把；深绿色背景、蓝色条带、绿色球场、照明和静止相机保持不变。[封闭世界]
> 不允许新增外部来源、额外副本、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；向右反向运动伴随与拍弦床重新拉开距离。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；direction=rightward_away、gap_state=right_departing 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`145`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state gap_state=right_separated,
> direction=leftward_toward_racket, required; racket: exactly 1, required. Counts and identities stay
> fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] the yellow ball
> visibly reaches the white racket strings happens BEFORE horizontal travel changes from leftward to
> rightward.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 网球（ball）: 恰好1个，状态 gap_state=right_separated, direction=leftward_toward_racket，必需;
> 球拍（racket）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 黄色网球明显抵达白色拍弦先于水平运动方向从向左变为向右发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`148`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state gap_state=right_separated,
> direction=leftward_toward_racket, required; racket: exactly 1, required. Counts and identities stay
> fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] horizontal
> travel changes from leftward to rightward is already present BEFORE the yellow ball visibly reaches
> the white racket strings.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 网球（ball）: 恰好1个，状态 gap_state=right_separated, direction=leftward_toward_racket，必需;
> 球拍（racket）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 在黄色网球明显抵达白色拍弦发生之前，水平运动方向从向左变为向右已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c2`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`147`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state gap_state=right_separated,
> direction=leftward_toward_racket, transitioning; racket: exactly 1, transitioning. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] the
> yellow ball visibly reaches the white racket strings happens BEFORE horizontal travel changes from
> leftward to rightward.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 网球（ball）: 恰好1个，状态 gap_state=right_separated, direction=leftward_toward_racket，正在过渡;
> 球拍（racket）: 恰好1个，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 黄色网球明显抵达白色拍弦先于水平运动方向从向左变为向右发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`148`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c2`

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state gap_state=right_separated,
> direction=leftward_toward_racket, transitioning; racket: exactly 1, transitioning. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> horizontal travel changes from leftward to rightward happens BEFORE the yellow ball visibly reaches
> the white racket strings.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 网球（ball）: 恰好1个，状态 gap_state=right_separated, direction=leftward_toward_racket，正在过渡;
> 球拍（racket）: 恰好1个，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 水平运动方向从向左变为向右先于黄色网球明显抵达白色拍弦发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`144`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state gap_state=right_departing,
> direction=rightward_away, required; racket: exactly 1, required. Counts and identities stay fixed;
> no duplicate state copy is created. No additional external source, extra copy, unlisted object
> exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The coupled changes
> occur together: green interval closes then reopens rightward WHILE horizontal travel changes from
> leftward to rightward.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 网球（ball）: 恰好1个，状态 gap_state=right_departing, direction=rightward_away，必需; 球拍（racket）:
> 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 耦合变化同时发生：绿色间隔闭合后重新向右打开，同时水平运动方向从向左变为向右。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`151`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3`

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state gap_state=right_departing,
> direction=rightward_away, required; racket: exactly 1, required. Counts and identities stay fixed;
> no duplicate state copy is created. No additional external source, extra copy, unlisted object
> exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] Only one coupled
> change occurs while the other remains unchanged: green interval closes then reopens rightward;
> horizontal travel changes from leftward to rightward.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 网球（ball）: 恰好1个，状态 gap_state=right_departing, direction=rightward_away，必需; 球拍（racket）:
> 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 只有一个耦合变化发生，另一个保持不变：绿色间隔闭合后重新向右打开；水平运动方向从向左变为向右。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`146`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state gap_state=right_departing,
> direction=rightward_away, transitioning; racket: exactly 1, transitioning. Counts and identities
> stay fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The coupled changes
> occur together: green interval closes then reopens rightward WHILE horizontal travel changes from
> leftward to rightward.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 网球（ball）: 恰好1个，状态 gap_state=right_departing, direction=rightward_away，正在过渡;
> 球拍（racket）: 恰好1个，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 耦合变化同时发生：绿色间隔闭合后重新向右打开，同时水平运动方向从向左变为向右。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`133`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3, c4`

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state gap_state=right_departing,
> direction=rightward_away, transitioning; racket: exactly 1, transitioning. Counts and identities
> stay fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] The declared
> cause occurs BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 网球（ball）: 恰好1个，状态 gap_state=right_departing, direction=rightward_away，正在过渡;
> 球拍（racket）: 恰好1个，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`181`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state gap_state=right_departing,
> direction=rightward_away, required; racket: exactly 1, required. Counts and identities stay fixed;
> no duplicate state copy is created. No additional external source, extra copy, unlisted object
> exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The terminal facts
> gap_state=right_departing, direction=rightward_away REMAIN through the final frames. [AND] The same
> logical slot has only its valid state; mutually exclusive facts direction=leftward_toward_racket,
> direction=rightward_away do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 网球（ball）: 恰好1个，状态 gap_state=right_departing, direction=rightward_away，必需; 球拍（racket）:
> 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实 gap_state=right_departing（向右离开）、direction=rightward_away（向右远离）
> 在最后几帧持续保持。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 direction=leftward_toward_racket（向左朝球拍）与
> direction=rightward_away（向右远离） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`169`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state gap_state=right_departing,
> direction=rightward_away, required; racket: exactly 1, required. Counts and identities stay fixed;
> no duplicate state copy is created. No additional external source, extra copy, unlisted object
> exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] The valid
> terminal state returns to its mutually exclusive earlier state. [AND] The same logical slot has only
> its valid state; mutually exclusive facts direction=leftward_toward_racket, direction=rightward_away
> do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 网球（ball）: 恰好1个，状态 gap_state=right_departing, direction=rightward_away，必需; 球拍（racket）:
> 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实
> direction=leftward_toward_racket（向左朝球拍）与 direction=rightward_away（向右远离） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`165`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state gap_state=right_departing,
> direction=rightward_away, required; racket: exactly 1, required. Counts and identities stay fixed;
> no duplicate state copy is created. No additional external source, extra copy, unlisted object
> exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] The terminal
> facts gap_state=right_departing, direction=rightward_away REMAIN through the final frames. [AND]
> Copies of the same logical slot appear in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 网球（ball）: 恰好1个，状态 gap_state=right_departing, direction=rightward_away，必需; 球拍（racket）:
> 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实 gap_state=right_departing（向右离开）、direction=rightward_away（向右远离）
> 在最后几帧持续保持。 [且] 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：string_contact 必须先于 horizontal_reverse。 |
| `c2` | `requires` | 因果依赖：horizontal_reverse 只有在 string_contact 之后才能发生。 |
| `c3` | `couples` | 耦合约束：court_gap_cycle, horizontal_reverse 必须共同变化；关系为“向右反向运动伴随与拍弦床重新拉开距离”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 gap_state=right_departing, direction=rightward_away 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 direction=leftward_toward_racket, direction=rightward_away 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "horizontal_reverse",
    "before": "string_contact",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "cause": "string_contact",
    "effect": "horizontal_reverse",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "type": "requires"
  },
  {
    "effects": [
      "court_gap_cycle",
      "horizontal_reverse"
    ],
    "id": "c3",
    "relation": "rightward reversal accompanies reopening separation from the string bed",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "gap_state=right_departing",
      "direction=rightward_away"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "direction=leftward_toward_racket",
      "direction=rightward_away"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=2个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"ball":"derived_motion_envelope","racket":"unresolved_static_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P04</strong> — triggered；setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5</summary>

### P04 概要

- 时间路由类型：`triggered`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`c2`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains first domino: exactly 1; remaining row: one declared
> collective group. The reference first frame fixes these identities and initial layout. [INITIAL
> STATE] first_pose=upright, row_progress=upright_regular_row. [PHYSICAL EVENT] The pale wooden domino
> row on the beige floor falls from its near-left first domino toward the far-right end. [PERSISTENT
> PROTECTION] single straight row with the pictured regular spacing; pale wooden material and
> rectangular piece identity remain stable; beige floor, plain wall, lighting, perspective, and static
> camera remain unchanged. [CLOSED WORLD] No additional external source, extra copy, hand, person,
> tool, unlisted object enters or appears; unoccupied scene regions remain empty. [CONSERVATION]
> Source loss and result gain stay coupled: tracked logical entity slots remain consistent; near-left
> fall launches a rightward adjacent-contact front. No result appears from an undeclared external
> source. [TERMINAL STATE] The same logical entity slots persist without duplicates;
> first_pose=fallen_rightward, row_progress=fallen_row remains visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 first_domino：恰好 1 个；remaining_row：一个已声明的集合组。参考首帧固定这些身份和初始布局。[初始状态]
> first_pose=upright、row_progress=upright_regular_row。[物理事件] 米色地面上的浅色木质多米诺骨牌列，从左近端第一块开始向最右端依次倒下。[持续保护]
> 保持图示规则间距的单一直线骨牌列；浅色木质材质和长方形骨牌身份保持稳定；米色地面、纯色墙、照明、透视和静止相机保持不变。[封闭世界]
> 不允许新增外部来源、额外副本、手、人物、工具、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；左近端骨牌倒下会启动向右传播的相邻接触前沿。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；first_pose=fallen_rightward、row_progress=fallen_row 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`150`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: first domino: exactly 1, state first_pose=upright, required;
> remaining row: one declared collective group, state row_progress=upright_regular_row, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, hand, person, tool, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] the near-left first domino reaches its right neighbor happens BEFORE
> falling boundary advances left to right.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 第一块多米诺骨牌（first domino）: 恰好1个，状态 first_pose=upright，必需; 其余骨牌列（remaining row）:
> 1个已声明集合组，状态 row_progress=upright_regular_row，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 左近端第一块多米诺骨牌触及其右侧相邻骨牌先于倒下边界从左向右推进发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`153`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: first domino: exactly 1, state first_pose=upright, required;
> remaining row: one declared collective group, state row_progress=upright_regular_row, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, hand, person, tool, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] falling boundary advances left to right is already present BEFORE the
> near-left first domino reaches its right neighbor.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 第一块多米诺骨牌（first domino）: 恰好1个，状态 first_pose=upright，必需; 其余骨牌列（remaining row）:
> 1个已声明集合组，状态 row_progress=upright_regular_row，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 在左近端第一块多米诺骨牌触及其右侧相邻骨牌发生之前，倒下边界从左向右推进已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c2`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`152`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: first domino: exactly 1, state first_pose=upright,
> transitioning; remaining row: one declared collective group, state row_progress=upright_regular_row,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, hand, person, tool, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VALID RELATION] the near-left first domino reaches its right neighbor
> happens BEFORE falling boundary advances left to right.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 第一块多米诺骨牌（first domino）: 恰好1个，状态 first_pose=upright，正在过渡; 其余骨牌列（remaining row）:
> 1个已声明集合组，状态 row_progress=upright_regular_row，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 左近端第一块多米诺骨牌触及其右侧相邻骨牌先于倒下边界从左向右推进发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`153`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c2`

> [STAGE ENTITY INVENTORY] Present now: first domino: exactly 1, state first_pose=upright,
> transitioning; remaining row: one declared collective group, state row_progress=upright_regular_row,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, hand, person, tool, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VIOLATED RELATION] falling boundary advances left to right happens
> BEFORE the near-left first domino reaches its right neighbor.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 第一块多米诺骨牌（first domino）: 恰好1个，状态 first_pose=upright，正在过渡; 其余骨牌列（remaining row）:
> 1个已声明集合组，状态 row_progress=upright_regular_row，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 倒下边界从左向右推进先于左近端第一块多米诺骨牌触及其右侧相邻骨牌发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`152`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: first domino: exactly 1, state first_pose=fallen_rightward,
> required; remaining row: one declared collective group, state row_progress=fallen_row, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, hand, person, tool, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] The coupled changes occur together: first piece rotates toward its right
> neighbor WHILE falling boundary advances left to right.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 第一块多米诺骨牌（first domino）: 恰好1个，状态 first_pose=fallen_rightward，必需; 其余骨牌列（remaining row）:
> 1个已声明集合组，状态 row_progress=fallen_row，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 耦合变化同时发生：第一块骨牌向右侧相邻骨牌旋转，同时倒下边界从左向右推进。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`159`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3`

> [STAGE ENTITY INVENTORY] Present now: first domino: exactly 1, state first_pose=fallen_rightward,
> required; remaining row: one declared collective group, state row_progress=fallen_row, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, hand, person, tool, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] Only one coupled change occurs while the other remains unchanged: first
> piece rotates toward its right neighbor; falling boundary advances left to right.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 第一块多米诺骨牌（first domino）: 恰好1个，状态 first_pose=fallen_rightward，必需; 其余骨牌列（remaining row）:
> 1个已声明集合组，状态 row_progress=fallen_row，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 只有一个耦合变化发生，另一个保持不变：第一块骨牌向右侧相邻骨牌旋转；倒下边界从左向右推进。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`154`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: first domino: exactly 1, state first_pose=fallen_rightward,
> transitioning; remaining row: one declared collective group, state row_progress=fallen_row,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, hand, person, tool, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VALID RELATION] The coupled changes occur together: first piece rotates
> toward its right neighbor WHILE falling boundary advances left to right.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 第一块多米诺骨牌（first domino）: 恰好1个，状态 first_pose=fallen_rightward，正在过渡; 其余骨牌列（remaining
> row）: 1个已声明集合组，状态 row_progress=fallen_row，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 耦合变化同时发生：第一块骨牌向右侧相邻骨牌旋转，同时倒下边界从左向右推进。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`143`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3, c4`

> [STAGE ENTITY INVENTORY] Present now: first domino: exactly 1, state first_pose=fallen_rightward,
> transitioning; remaining row: one declared collective group, state row_progress=fallen_row,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, hand, person, tool, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VIOLATED RELATION] The declared cause occurs BUT the required target
> change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 第一块多米诺骨牌（first domino）: 恰好1个，状态 first_pose=fallen_rightward，正在过渡; 其余骨牌列（remaining
> row）: 1个已声明集合组，状态 row_progress=fallen_row，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`196`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: first domino: exactly 1, state first_pose=fallen_rightward,
> required; remaining row: one declared collective group, state row_progress=fallen_row, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, hand, person, tool, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] The terminal facts first_pose=fallen_rightward, row_progress=fallen_row
> REMAIN through the final frames. [AND] The same logical slot has only its valid state; mutually
> exclusive facts row_progress=upright_regular_row, row_progress=fallen_row do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 第一块多米诺骨牌（first domino）: 恰好1个，状态 first_pose=fallen_rightward，必需; 其余骨牌列（remaining row）:
> 1个已声明集合组，状态 row_progress=fallen_row，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实
> first_pose=fallen_rightward（向右倒下）、row_progress=fallen_row（整列已倒） 在最后几帧持续保持。 [且] 同一逻辑槽位只具有其有效状态；互斥事实
> row_progress=upright_regular_row（直立整齐排列）与 row_progress=fallen_row（整列已倒） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`183`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: first domino: exactly 1, state first_pose=fallen_rightward,
> required; remaining row: one declared collective group, state row_progress=fallen_row, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, hand, person, tool, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] The valid terminal state returns to its mutually exclusive earlier
> state. [AND] The same logical slot has only its valid state; mutually exclusive facts
> row_progress=upright_regular_row, row_progress=fallen_row do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 第一块多米诺骨牌（first domino）: 恰好1个，状态 first_pose=fallen_rightward，必需; 其余骨牌列（remaining row）:
> 1个已声明集合组，状态 row_progress=fallen_row，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实
> row_progress=upright_regular_row（直立整齐排列）与 row_progress=fallen_row（整列已倒） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`176`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: first domino: exactly 1, state first_pose=fallen_rightward,
> required; remaining row: one declared collective group, state row_progress=fallen_row, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, hand, person, tool, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] The terminal facts first_pose=fallen_rightward, row_progress=fallen_row
> REMAIN through the final frames. [AND] Copies of the same logical slot appear in mutually exclusive
> states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 第一块多米诺骨牌（first domino）: 恰好1个，状态 first_pose=fallen_rightward，必需; 其余骨牌列（remaining row）:
> 1个已声明集合组，状态 row_progress=fallen_row，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实
> first_pose=fallen_rightward（向右倒下）、row_progress=fallen_row（整列已倒） 在最后几帧持续保持。 [且] 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：left_contact 必须先于 rightward_front。 |
| `c2` | `requires` | 因果依赖：rightward_front 只有在 left_contact 之后才能发生。 |
| `c3` | `couples` | 耦合约束：left_first_fall, rightward_front 必须共同变化；关系为“左近端骨牌倒下会启动向右传播的相邻接触前沿”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 first_pose=fallen_rightward, row_progress=fallen_row 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 row_progress=upright_regular_row, row_progress=fallen_row 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "rightward_front",
    "before": "left_contact",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "cause": "left_contact",
    "effect": "rightward_front",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "type": "requires"
  },
  {
    "effects": [
      "left_first_fall",
      "rightward_front"
    ],
    "id": "c3",
    "relation": "near-left fall launches a rightward adjacent-contact front",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "first_pose=fallen_rightward",
      "row_progress=fallen_row"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "row_progress=upright_regular_row",
      "row_progress=fallen_row"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=2个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"first_domino":"derived_motion_envelope","remaining_row":"derived_motion_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P05</strong> — multi_transition；setup→c1; onset→c1; evolution→c2; completion→c5; terminal→c6,c7</summary>

### P05 概要

- 时间路由类型：`multi_transition`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c2; completion→c5; terminal→c6,c7`
- 同时进入 Global 守恒文本的 couples 约束：`无`
- 仅校验、未进入生成张量的约束：`c3,c4`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains block: exactly 1; ramp: exactly 1; rough floor: one
> declared collective group. The reference first frame fixes these identities and initial layout.
> [INITIAL STATE] location=upper_left_ramp, motion=still. [PHYSICAL EVENT] The small light-wood block
> slides down the light-wood wedge ramp onto the rough gray floor and stops. [PERSISTENT PROTECTION]
> same small rectangular light-wood block; light-wood wedge geometry and narrow ramp tip remain fixed;
> rough gray floor, pale wall, lighting, and static side camera remain unchanged. [CLOSED WORLD] No
> additional external source, extra copy, unlisted object enters or appears; unoccupied scene regions
> remain empty. [CONSERVATION] Source loss and result gain stay coupled: tracked logical entity slots
> remain consistent. No result appears from an undeclared external source. [TERMINAL STATE] The same
> logical entity slots persist without duplicates; location=right_gray_floor, motion=stopped remains
> visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 block：恰好 1 个；ramp：恰好 1 个；rough_floor：一个已声明的集合组。参考首帧固定这些身份和初始布局。[初始状态]
> location=upper_left_ramp、motion=still。[物理事件] 小块浅色木块沿浅色木质楔形斜坡滑下，进入粗糙的灰色地面并停止。[持续保护]
> 保持同一小型长方形浅色木块；浅色木质楔形几何和狭窄斜坡尖端保持固定；粗糙灰色地面、浅色墙、照明和静止侧视相机保持不变。[封闭世界]
> 不允许新增外部来源、额外副本、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒] 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；location=right_gray_floor、motion=stopped 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`143`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: block: exactly 1, state location=upper_left_ramp,
> motion=still, required; ramp: exactly 1, required; rough floor: one declared collective group,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] first visible block displacement down the wooden slope happens BEFORE
> position descends rightward to gray floor.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 木块（block）: 恰好1个，状态 location=upper_left_ramp, motion=still，必需; 斜坡（ramp）: 恰好1个，必需;
> 粗糙地面（rough floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 木块首次明显沿木质斜面发生位移先于位置向右下方移动至灰色地面发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`146`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c3`

> [STAGE ENTITY INVENTORY] Present now: block: exactly 1, state location=upper_left_ramp,
> motion=still, required; ramp: exactly 1, required; rough floor: one declared collective group,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] position descends rightward to gray floor is already present BEFORE
> first visible block displacement down the wooden slope.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 木块（block）: 恰好1个，状态 location=upper_left_ramp, motion=still，必需; 斜坡（ramp）: 恰好1个，必需;
> 粗糙地面（rough floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 在木块首次明显沿木质斜面发生位移发生之前，位置向右下方移动至灰色地面已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`145`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: block: exactly 1, state location=upper_left_ramp,
> motion=still, transitioning; ramp: exactly 1, required; rough floor: one declared collective group,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] first visible block displacement down the wooden slope happens BEFORE
> position descends rightward to gray floor.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 木块（block）: 恰好1个，状态 location=upper_left_ramp, motion=still，正在过渡; 斜坡（ramp）: 恰好1个，必需;
> 粗糙地面（rough floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 木块首次明显沿木质斜面发生位移先于位置向右下方移动至灰色地面发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`146`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c3, c4`

> [STAGE ENTITY INVENTORY] Present now: block: exactly 1, state location=upper_left_ramp,
> motion=still, transitioning; ramp: exactly 1, required; rough floor: one declared collective group,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] position descends rightward to gray floor happens BEFORE first visible
> block displacement down the wooden slope.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 木块（block）: 恰好1个，状态 location=upper_left_ramp, motion=still，正在过渡; 斜坡（ramp）: 恰好1个，必需;
> 粗糙地面（rough floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 位置向右下方移动至灰色地面先于木块首次明显沿木质斜面发生位移发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`138`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: block: exactly 1, state location=right_gray_floor,
> motion=sliding, required; ramp: exactly 1, required; rough floor: one declared collective group,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] The declared change position descends rightward to gray floor occurs in
> its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 木块（block）: 恰好1个，状态 location=right_gray_floor, motion=sliding，必需; 斜坡（ramp）: 恰好1个，必需;
> 粗糙地面（rough floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 已声明的变化“位置向右下方移动至灰色地面”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`138`
- 权重：`1.0`；violation：`wrong_direction`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c5`

> [STAGE ENTITY INVENTORY] Present now: block: exactly 1, state location=right_gray_floor,
> motion=sliding, required; ramp: exactly 1, required; rough floor: one declared collective group,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] The same declared property changes in the opposite or out-of-range
> direction.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 木块（block）: 恰好1个，状态 location=right_gray_floor, motion=sliding，必需; 斜坡（ramp）: 恰好1个，必需;
> 粗糙地面（rough floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 同一项已声明属性朝相反方向变化，或变化超出规定范围。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c5`
- violation 中声明的候选约束：`c5, c6`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`139`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: block: exactly 1, state location=right_gray_floor,
> motion=stopped, transitioning; ramp: exactly 1, required; rough floor: one declared collective
> group, transitioning. Counts and identities stay fixed; no duplicate state copy is created. No
> additional external source, extra copy, unlisted object exists. All globally protected identity,
> shape, material, scene, and camera attributes stay unchanged. Declared source loss and result gain
> remain coupled. [VALID RELATION] The declared change rightward displacement contracts to zero occurs
> in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 木块（block）: 恰好1个，状态 location=right_gray_floor, motion=stopped，正在过渡; 斜坡（ramp）: 恰好1个，必需;
> 粗糙地面（rough floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 已声明的变化“向右位移收缩至零”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`138`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5, c6`

> [STAGE ENTITY INVENTORY] Present now: block: exactly 1, state location=right_gray_floor,
> motion=stopped, transitioning; ramp: exactly 1, required; rough floor: one declared collective
> group, transitioning. Counts and identities stay fixed; no duplicate state copy is created. No
> additional external source, extra copy, unlisted object exists. All globally protected identity,
> shape, material, scene, and camera attributes stay unchanged. Declared source loss and result gain
> remain coupled. [VIOLATED RELATION] The declared cause occurs BUT the required target change does
> not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 木块（block）: 恰好1个，状态 location=right_gray_floor, motion=stopped，正在过渡; 斜坡（ramp）: 恰好1个，必需;
> 粗糙地面（rough floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c6, c7`
- violation 中声明的候选约束：`c6, c7`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`175`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: block: exactly 1, state location=right_gray_floor,
> motion=stopped, required; ramp: exactly 1, required; rough floor: one declared collective group,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] The terminal facts location=right_gray_floor, motion=stopped REMAIN
> through the final frames. [AND] The same logical slot has only its valid state; mutually exclusive
> facts motion=sliding, motion=stopped do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 木块（block）: 恰好1个，状态 location=right_gray_floor, motion=stopped，必需; 斜坡（ramp）: 恰好1个，必需;
> 粗糙地面（rough floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实
> location=right_gray_floor（右侧灰色地面）、motion=stopped（停止） 在最后几帧持续保持。 [且] 同一逻辑槽位只具有其有效状态；互斥事实
> motion=sliding（滑动）与 motion=stopped（停止） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`167`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c6`；violation 声明的候选约束：`c6`

> [STAGE ENTITY INVENTORY] Present now: block: exactly 1, state location=right_gray_floor,
> motion=stopped, required; ramp: exactly 1, required; rough floor: one declared collective group,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] The valid terminal state returns to its mutually exclusive earlier
> state. [AND] The same logical slot has only its valid state; mutually exclusive facts
> motion=sliding, motion=stopped do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 木块（block）: 恰好1个，状态 location=right_gray_floor, motion=stopped，必需; 斜坡（ramp）: 恰好1个，必需;
> 粗糙地面（rough floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实
> motion=sliding（滑动）与 motion=stopped（停止） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`166`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c7`；violation 声明的候选约束：`c7`

> [STAGE ENTITY INVENTORY] Present now: block: exactly 1, state location=right_gray_floor,
> motion=stopped, required; ramp: exactly 1, required; rough floor: one declared collective group,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] The terminal facts location=right_gray_floor, motion=stopped REMAIN
> through the final frames. [AND] Copies of the same logical slot appear in mutually exclusive states
> at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 木块（block）: 恰好1个，状态 location=right_gray_floor, motion=stopped，必需; 斜坡（ramp）: 恰好1个，必需;
> 粗糙地面（rough floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实
> location=right_gray_floor（右侧灰色地面）、motion=stopped（停止） 在最后几帧持续保持。 [且] 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：first_slide 必须先于 pictured_descent。 |
| `c2` | `changes` | 方向变化：转移 e1 中的效果 pictured_descent 必须按声明方向发生。 |
| `c3` | `precedes` | 前置顺序：edge_crossing 必须先于 gray_floor_stop。 |
| `c4` | `requires` | 因果依赖：gray_floor_stop 只有在 edge_crossing 之后才能发生。 |
| `c5` | `changes` | 方向变化：转移 e2 中的效果 gray_floor_stop 必须按声明方向发生。 |
| `c6` | `persists` | 持续约束：状态 z2 的事实 location=right_gray_floor, motion=stopped 必须维持到最终帧。 |
| `c7` | `excludes` | 互斥约束：事实 motion=sliding, motion=stopped 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "pictured_descent",
    "before": "first_slide",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "effect": "pictured_descent",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "transition": "e1",
    "type": "changes"
  },
  {
    "after": "gray_floor_stop",
    "before": "edge_crossing",
    "id": "c3",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "cause": "edge_crossing",
    "effect": "gray_floor_stop",
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "type": "requires"
  },
  {
    "effect": "gray_floor_stop",
    "id": "c5",
    "source_refs": [
      "origin"
    ],
    "transition": "e2",
    "type": "changes"
  },
  {
    "facts": [
      "location=right_gray_floor",
      "motion=stopped"
    ],
    "id": "c6",
    "source_refs": [
      "origin"
    ],
    "state": "z2",
    "type": "persists"
  },
  {
    "facts": [
      "motion=sliding",
      "motion=stopped"
    ],
    "id": "c7",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=3个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"block":"derived_motion_envelope","ramp":"unresolved_static_envelope","rough_floor":"unresolved_static_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P06</strong> — multi_transition；setup→c1; onset→c1; evolution→c5; completion→c6; terminal→c7,c8</summary>

### P06 概要

- 时间路由类型：`multi_transition`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c5; completion→c6; terminal→c7,c8`
- 同时进入 Global 守恒文本的 couples 约束：`无`
- 仅校验、未进入生成张量的约束：`c2,c3,c4`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains book: exactly 1; shelf: exactly 1; floor: one declared
> collective group. The reference first frame fixes these identities and initial layout. [INITIAL
> STATE] position=left_shelf, support=wood_shelf_supported, book_state=closed. [PHYSICAL EVENT] The
> blue-edged closed book leaves the left wooden shelf, falls through the open wall space, and lands
> open on the wood floor. [PERSISTENT PROTECTION] same cream pages and dark blue cover edges; left
> wooden shelf and vertical support remain fixed; pale wall, white baseboard, wood-plank floor,
> lighting, and static side camera remain unchanged. [CLOSED WORLD] No additional external source,
> extra copy, hand, person, tool, unlisted object enters or appears; unoccupied scene regions remain
> empty. [CONSERVATION] Source loss and result gain stay coupled: tracked logical entity slots remain
> consistent. No result appears from an undeclared external source. [TERMINAL STATE] The same logical
> entity slots persist without duplicates; position=wood_floor, support=floor_supported,
> book_state=open remains visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 book：恰好 1 个；shelf：恰好 1 个；floor：一个已声明的集合组。参考首帧固定这些身份和初始布局。[初始状态]
> book_state=closed、position=left_shelf、support=wood_shelf_supported。[物理事件]
> 蓝边合拢的书离开左侧木架，穿过墙前的开放空间下落，并以打开状态落在木地板上。[持续保护]
> 保持同样的奶油色书页和深蓝色封面边缘；左侧木架和竖直支撑保持固定；浅色墙、白色踢脚线、木板地面、照明和静止侧视相机保持不变。[封闭世界]
> 不允许新增外部来源、额外副本、手、人物、工具、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；book_state=open、position=wood_floor、support=floor_supported 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`160`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: book: exactly 1, state position=left_shelf,
> support=wood_shelf_supported, book_state=closed, required; shelf: exactly 1, required; floor: one
> declared collective group, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, hand, person, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VALID RELATION] the book passes the visible right end
> of the wooden shelf happens BEFORE book moves right of shelf then downward.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 书（book）: 恰好1个，状态 position=left_shelf, support=wood_shelf_supported,
> book_state=closed，必需; 书架（shelf）: 恰好1个，必需; 地板（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 书越过木架可见的右端先于书先移至书架右侧再向下运动发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`163`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c3`

> [STAGE ENTITY INVENTORY] Present now: book: exactly 1, state position=left_shelf,
> support=wood_shelf_supported, book_state=closed, required; shelf: exactly 1, required; floor: one
> declared collective group, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, hand, person, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VIOLATED RELATION] book moves right of shelf then
> downward is already present BEFORE the book passes the visible right end of the wooden shelf.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 书（book）: 恰好1个，状态 position=left_shelf, support=wood_shelf_supported,
> book_state=closed，必需; 书架（shelf）: 恰好1个，必需; 地板（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 在书越过木架可见的右端发生之前，书先移至书架右侧再向下运动已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c2, c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`162`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: book: exactly 1, state position=left_shelf,
> support=wood_shelf_supported, book_state=closed, transitioning; shelf: exactly 1, transitioning;
> floor: one declared collective group, required. Counts and identities stay fixed; no duplicate state
> copy is created. No additional external source, extra copy, hand, person, tool, unlisted object
> exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] the book passes the
> visible right end of the wooden shelf happens BEFORE book moves right of shelf then downward.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 书（book）: 恰好1个，状态 position=left_shelf, support=wood_shelf_supported,
> book_state=closed，正在过渡; 书架（shelf）: 恰好1个，正在过渡; 地板（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 书越过木架可见的右端先于书先移至书架右侧再向下运动发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`163`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c2, c3, c4`

> [STAGE ENTITY INVENTORY] Present now: book: exactly 1, state position=left_shelf,
> support=wood_shelf_supported, book_state=closed, transitioning; shelf: exactly 1, transitioning;
> floor: one declared collective group, required. Counts and identities stay fixed; no duplicate state
> copy is created. No additional external source, extra copy, hand, person, tool, unlisted object
> exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] book moves right
> of shelf then downward happens BEFORE the book passes the visible right end of the wooden shelf.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 书（book）: 恰好1个，状态 position=left_shelf, support=wood_shelf_supported,
> book_state=closed，正在过渡; 书架（shelf）: 恰好1个，正在过渡; 地板（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 书先移至书架右侧再向下运动先于书越过木架可见的右端发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c5`
- violation 中声明的候选约束：`c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`149`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: book: exactly 1, state position=airborne, support=unsupported,
> book_state=closed, required; shelf: exactly 1, required; floor: one declared collective group,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, hand, person, tool, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VALID RELATION] The declared change book moves right of shelf then
> downward occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 书（book）: 恰好1个，状态 position=airborne, support=unsupported, book_state=closed，必需;
> 书架（shelf）: 恰好1个，必需; 地板（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 已声明的变化“书先移至书架右侧再向下运动”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`146`
- 权重：`1.0`；violation：`effect_missing`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: book: exactly 1, state position=airborne, support=unsupported,
> book_state=closed, required; shelf: exactly 1, required; floor: one declared collective group,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, hand, person, tool, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VIOLATED RELATION] The declared cause occurs BUT the required target
> change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 书（book）: 恰好1个，状态 position=airborne, support=unsupported, book_state=closed，必需;
> 书架（shelf）: 恰好1个，必需; 地板（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c6`
- violation 中声明的候选约束：`c6, c7`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`150`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: book: exactly 1, state position=wood_floor,
> support=floor_supported, book_state=open, transitioning; shelf: exactly 1, required; floor: one
> declared collective group, transitioning. Counts and identities stay fixed; no duplicate state copy
> is created. No additional external source, extra copy, hand, person, tool, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VALID RELATION] The declared change blue-edged
> covers separate occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 书（book）: 恰好1个，状态 position=wood_floor, support=floor_supported, book_state=open，正在过渡;
> 书架（shelf）: 恰好1个，必需; 地板（floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 已声明的变化“蓝边书封分开”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`150`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c6`；violation 声明的候选约束：`c6, c7`

> [STAGE ENTITY INVENTORY] Present now: book: exactly 1, state position=wood_floor,
> support=floor_supported, book_state=open, transitioning; shelf: exactly 1, required; floor: one
> declared collective group, transitioning. Counts and identities stay fixed; no duplicate state copy
> is created. No additional external source, extra copy, hand, person, tool, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] The declared cause occurs
> BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 书（book）: 恰好1个，状态 position=wood_floor, support=floor_supported, book_state=open，正在过渡;
> 书架（shelf）: 恰好1个，必需; 地板（floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c7, c8`
- violation 中声明的候选约束：`c7, c8`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`188`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: book: exactly 1, state position=wood_floor,
> support=floor_supported, book_state=open, required; shelf: exactly 1, required; floor: one declared
> collective group, required. Counts and identities stay fixed; no duplicate state copy is created. No
> additional external source, extra copy, hand, person, tool, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VALID RELATION] The terminal facts position=wood_floor,
> book_state=open REMAIN through the final frames. [AND] The same logical slot has only its valid
> state; mutually exclusive facts book_state=closed, book_state=open do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 书（book）: 恰好1个，状态 position=wood_floor, support=floor_supported, book_state=open，必需;
> 书架（shelf）: 恰好1个，必需; 地板（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实
> position=wood_floor（木地板）、book_state=open（打开） 在最后几帧持续保持。 [且] 同一逻辑槽位只具有其有效状态；互斥事实
> book_state=closed（合拢）与 book_state=open（打开） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`181`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c7`；violation 声明的候选约束：`c7`

> [STAGE ENTITY INVENTORY] Present now: book: exactly 1, state position=wood_floor,
> support=floor_supported, book_state=open, required; shelf: exactly 1, required; floor: one declared
> collective group, required. Counts and identities stay fixed; no duplicate state copy is created. No
> additional external source, extra copy, hand, person, tool, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VIOLATED RELATION] The valid terminal state returns to its
> mutually exclusive earlier state. [AND] The same logical slot has only its valid state; mutually
> exclusive facts book_state=closed, book_state=open do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 书（book）: 恰好1个，状态 position=wood_floor, support=floor_supported, book_state=open，必需;
> 书架（shelf）: 恰好1个，必需; 地板（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 book_state=closed（合拢）与 book_state=open（打开） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`177`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c8`；violation 声明的候选约束：`c8`

> [STAGE ENTITY INVENTORY] Present now: book: exactly 1, state position=wood_floor,
> support=floor_supported, book_state=open, required; shelf: exactly 1, required; floor: one declared
> collective group, required. Counts and identities stay fixed; no duplicate state copy is created. No
> additional external source, extra copy, hand, person, tool, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VIOLATED RELATION] The terminal facts position=wood_floor,
> book_state=open REMAIN through the final frames. [AND] Copies of the same logical slot appear in
> mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 书（book）: 恰好1个，状态 position=wood_floor, support=floor_supported, book_state=open，必需;
> 书架（shelf）: 恰好1个，必需; 地板（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实
> position=wood_floor（木地板）、book_state=open（打开） 在最后几帧持续保持。 [且] 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_edge_crossing 必须先于 wall_corridor_fall。 |
| `c2` | `requires` | 因果依赖：wall_corridor_fall 只有在 pictured_edge_crossing 之后才能发生。 |
| `c3` | `precedes` | 前置顺序：pictured_floor_contact 必须先于 blue_cover_open。 |
| `c4` | `requires` | 因果依赖：blue_cover_open 只有在 pictured_floor_contact 之后才能发生。 |
| `c5` | `changes` | 方向变化：转移 e1 中的效果 wall_corridor_fall 必须按声明方向发生。 |
| `c6` | `changes` | 方向变化：转移 e2 中的效果 blue_cover_open 必须按声明方向发生。 |
| `c7` | `persists` | 持续约束：状态 z2 的事实 position=wood_floor, book_state=open 必须维持到最终帧。 |
| `c8` | `excludes` | 互斥约束：事实 book_state=closed, book_state=open 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "wall_corridor_fall",
    "before": "pictured_edge_crossing",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "cause": "pictured_edge_crossing",
    "effect": "wall_corridor_fall",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "type": "requires"
  },
  {
    "after": "blue_cover_open",
    "before": "pictured_floor_contact",
    "id": "c3",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "cause": "pictured_floor_contact",
    "effect": "blue_cover_open",
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "type": "requires"
  },
  {
    "effect": "wall_corridor_fall",
    "id": "c5",
    "source_refs": [
      "origin"
    ],
    "transition": "e1",
    "type": "changes"
  },
  {
    "effect": "blue_cover_open",
    "id": "c6",
    "source_refs": [
      "origin"
    ],
    "transition": "e2",
    "type": "changes"
  },
  {
    "facts": [
      "position=wood_floor",
      "book_state=open"
    ],
    "id": "c7",
    "source_refs": [
      "origin"
    ],
    "state": "z2",
    "type": "persists"
  },
  {
    "facts": [
      "book_state=closed",
      "book_state=open"
    ],
    "id": "c8",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=3个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"book":"derived_motion_envelope","floor":"unresolved_static_envelope","shelf":"unresolved_static_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P07</strong> — triggered；setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5</summary>

### P07 概要

- 时间路由类型：`triggered`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`c2`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains box: exactly 1; floor: one declared collective group.
> The reference first frame fixes these identities and initial layout. [INITIAL STATE]
> corner_state=centered_support, pose=vertical. [PHYSICAL EVENT] The centered tall tan cardboard box
> shifts at one lower corner, tips across the gray floor, and remains on its side. [PERSISTENT
> PROTECTION] same tall tan cardboard surface and rectangular proportions; smooth gray floor, white
> wall, baseboard, lighting, and static frontal camera remain unchanged; no pulling hand or tool is
> added. [CLOSED WORLD] No additional external source, extra copy, unlisted object enters or appears;
> unoccupied scene regions remain empty. [CONSERVATION] Source loss and result gain stay coupled:
> tracked logical entity slots remain consistent; bottom shift is followed by increasing rotation
> toward floor. No result appears from an undeclared external source. [TERMINAL STATE] The same
> logical entity slots persist without duplicates; corner_state=shifted_sideways,
> pose=horizontal_side_supported remains visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 box：恰好 1 个；floor：一个已声明的集合组。参考首帧固定这些身份和初始布局。[初始状态]
> corner_state=centered_support、pose=vertical。[物理事件] 居中的高大棕褐色纸箱从一个底角开始偏移，在灰色地面上倾倒，并保持侧躺。[持续保护]
> 保持同一高大棕褐色纸板表面和长方形比例；光滑灰色地面、白墙、踢脚线、照明和静止正视相机保持不变；不添加拉拽的手或工具。[封闭世界]
> 不允许新增外部来源、额外副本、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；底部偏移之后，朝地面的旋转逐渐增大。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；corner_state=shifted_sideways、pose=horizontal_side_supported 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`135`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: box: exactly 1, state corner_state=centered_support,
> pose=vertical, required; floor: one declared collective group, required. Counts and identities stay
> fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] one visible bottom
> corner of the tan box moves sideways happens BEFORE tall silhouette rotates toward gray floor.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸箱（box）: 恰好1个，状态 corner_state=centered_support, pose=vertical，必需; 地面（floor）:
> 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 棕褐色纸箱的一个可见底角向侧面移动先于高大轮廓朝灰色地面旋转发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`138`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: box: exactly 1, state corner_state=centered_support,
> pose=vertical, required; floor: one declared collective group, required. Counts and identities stay
> fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] tall silhouette
> rotates toward gray floor is already present BEFORE one visible bottom corner of the tan box moves
> sideways.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸箱（box）: 恰好1个，状态 corner_state=centered_support, pose=vertical，必需; 地面（floor）:
> 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 在棕褐色纸箱的一个可见底角向侧面移动发生之前，高大轮廓朝灰色地面旋转已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c2`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`137`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: box: exactly 1, state corner_state=centered_support,
> pose=vertical, transitioning; floor: one declared collective group, transitioning. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] one
> visible bottom corner of the tan box moves sideways happens BEFORE tall silhouette rotates toward
> gray floor.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸箱（box）: 恰好1个，状态 corner_state=centered_support, pose=vertical，正在过渡; 地面（floor）:
> 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 棕褐色纸箱的一个可见底角向侧面移动先于高大轮廓朝灰色地面旋转发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`138`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c2`

> [STAGE ENTITY INVENTORY] Present now: box: exactly 1, state corner_state=centered_support,
> pose=vertical, transitioning; floor: one declared collective group, transitioning. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> tall silhouette rotates toward gray floor happens BEFORE one visible bottom corner of the tan box
> moves sideways.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸箱（box）: 恰好1个，状态 corner_state=centered_support, pose=vertical，正在过渡; 地面（floor）:
> 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 高大轮廓朝灰色地面旋转先于棕褐色纸箱的一个可见底角向侧面移动发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`141`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: box: exactly 1, state corner_state=shifted_sideways,
> pose=horizontal_side_supported, required; floor: one declared collective group, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The
> coupled changes occur together: bottom support moves laterally WHILE tall silhouette rotates toward
> gray floor.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸箱（box）: 恰好1个，状态 corner_state=shifted_sideways, pose=horizontal_side_supported，必需;
> 地面（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 耦合变化同时发生：底部支撑横向移动，同时高大轮廓朝灰色地面旋转。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`148`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3`

> [STAGE ENTITY INVENTORY] Present now: box: exactly 1, state corner_state=shifted_sideways,
> pose=horizontal_side_supported, required; floor: one declared collective group, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> Only one coupled change occurs while the other remains unchanged: bottom support moves laterally;
> tall silhouette rotates toward gray floor.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸箱（box）: 恰好1个，状态 corner_state=shifted_sideways, pose=horizontal_side_supported，必需;
> 地面（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 只有一个耦合变化发生，另一个保持不变：底部支撑横向移动；高大轮廓朝灰色地面旋转。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`143`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: box: exactly 1, state corner_state=shifted_sideways,
> pose=horizontal_side_supported, transitioning; floor: one declared collective group, transitioning.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, unlisted object exists. All globally protected identity, shape, material, scene, and
> camera attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID
> RELATION] The coupled changes occur together: bottom support moves laterally WHILE tall silhouette
> rotates toward gray floor.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸箱（box）: 恰好1个，状态 corner_state=shifted_sideways, pose=horizontal_side_supported，正在过渡;
> 地面（floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 耦合变化同时发生：底部支撑横向移动，同时高大轮廓朝灰色地面旋转。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`136`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3, c4`

> [STAGE ENTITY INVENTORY] Present now: box: exactly 1, state corner_state=shifted_sideways,
> pose=horizontal_side_supported, transitioning; floor: one declared collective group, transitioning.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, unlisted object exists. All globally protected identity, shape, material, scene, and
> camera attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED
> RELATION] The declared cause occurs BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸箱（box）: 恰好1个，状态 corner_state=shifted_sideways, pose=horizontal_side_supported，正在过渡;
> 地面（floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`175`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: box: exactly 1, state corner_state=shifted_sideways,
> pose=horizontal_side_supported, required; floor: one declared collective group, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The
> terminal facts pose=horizontal_side_supported REMAIN through the final frames. [AND] The same
> logical slot has only its valid state; mutually exclusive facts pose=vertical,
> pose=horizontal_side_supported do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸箱（box）: 恰好1个，状态 corner_state=shifted_sideways, pose=horizontal_side_supported，必需;
> 地面（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实 pose=horizontal_side_supported（水平侧躺支撑） 在最后几帧持续保持。 [且]
> 同一逻辑槽位只具有其有效状态；互斥事实 pose=vertical（竖直）与 pose=horizontal_side_supported（水平侧躺支撑） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`170`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: box: exactly 1, state corner_state=shifted_sideways,
> pose=horizontal_side_supported, required; floor: one declared collective group, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> The valid terminal state returns to its mutually exclusive earlier state. [AND] The same logical
> slot has only its valid state; mutually exclusive facts pose=vertical,
> pose=horizontal_side_supported do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸箱（box）: 恰好1个，状态 corner_state=shifted_sideways, pose=horizontal_side_supported，必需;
> 地面（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 pose=vertical（竖直）与
> pose=horizontal_side_supported（水平侧躺支撑） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`161`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: box: exactly 1, state corner_state=shifted_sideways,
> pose=horizontal_side_supported, required; floor: one declared collective group, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> The terminal facts pose=horizontal_side_supported REMAIN through the final frames. [AND] Copies of
> the same logical slot appear in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸箱（box）: 恰好1个，状态 corner_state=shifted_sideways, pose=horizontal_side_supported，必需;
> 地面（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实 pose=horizontal_side_supported（水平侧躺支撑） 在最后几帧持续保持。 [且]
> 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_corner_shift 必须先于 tan_box_tip。 |
| `c2` | `requires` | 因果依赖：tan_box_tip 只有在 pictured_corner_shift 之后才能发生。 |
| `c3` | `couples` | 耦合约束：bottom_shift, tan_box_tip 必须共同变化；关系为“底部偏移之后，朝地面的旋转逐渐增大”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 pose=horizontal_side_supported 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 pose=vertical, pose=horizontal_side_supported 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "tan_box_tip",
    "before": "pictured_corner_shift",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "cause": "pictured_corner_shift",
    "effect": "tan_box_tip",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "type": "requires"
  },
  {
    "effects": [
      "bottom_shift",
      "tan_box_tip"
    ],
    "id": "c3",
    "relation": "bottom shift is followed by increasing rotation toward floor",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "pose=horizontal_side_supported"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "pose=vertical",
      "pose=horizontal_side_supported"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=2个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"box":"unresolved_static_envelope","floor":"unresolved_static_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P08</strong> — continuous；setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5</summary>

### P08 概要

- 时间路由类型：`continuous`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`无`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains ice: exactly 1; meltwater: exactly 1; plate: exactly 1.
> The reference first frame fixes these identities and initial layout. [INITIAL STATE]
> ice_size=large_sharp_cube, water_area=absent_or_tiny. [PHYSICAL EVENT] The single clear cubic ice
> piece centered on the large white plate shrinks as a transparent puddle spreads around it.
> [PERSISTENT PROTECTION] exactly one clear ice piece; same large shallow white plate and centered
> placement; wood tabletop, warm lighting, three-quarter camera, and plate position remain unchanged.
> [CLOSED WORLD] No additional dropper, external liquid stream, external source, extra copy, hand,
> person, pipette, tool, unlisted object enters or appears; unoccupied scene regions remain empty.
> [CONSERVATION] Source loss and result gain stay coupled: tracked logical entity slots remain
> consistent; centered clear solid shrinks as transparent plate footprint grows. No result appears
> from an undeclared external source. [TERMINAL STATE] The same logical entity slots persist without
> duplicates; ice_size=smaller_receded_ice, water_area=wide_puddle remains visible through the final
> frames.

#### 中文解释

> [实体清单] 封闭场景包含 ice：恰好 1 个；meltwater：恰好 1 个；plate：恰好 1 个。参考首帧固定这些身份和初始布局。[初始状态]
> ice_size=large_sharp_cube、water_area=absent_or_tiny。[物理事件] 大白盘中央的单个透明立方体冰块缩小，同时其周围的透明水洼扩散。[持续保护]
> 恰好一个透明冰块；保持同一大号浅白盘及居中位置；木桌面、暖色照明、四分之三视角相机和盘子位置保持不变。[封闭世界]
> 不允许新增滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；中央透明固体缩小，同时盘上的透明覆盖区增大。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；ice_size=smaller_receded_ice、water_area=wide_puddle 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`172`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ice: exactly 1, state ice_size=large_sharp_cube, required;
> meltwater: exactly 0, forbidden; plate: exactly 1, state water_area=absent_or_tiny, required. Counts
> and identities stay fixed; no duplicate state copy is created. No additional dropper, external
> liquid stream, external source, extra copy, hand, person, pipette, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VALID RELATION] first visible cube-edge recession and
> wet rim happens BEFORE transparent footprint expands over white plate.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 冰块（ice）: 恰好1个，状态 ice_size=large_sharp_cube，必需; 融水（meltwater）: 恰好0个，禁止出现; 盘子（plate）:
> 恰好1个，状态 water_area=absent_or_tiny，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [有效关系] 首次看见立方体边缘退缩并出现湿润边缘先于透明覆盖区在白盘上扩大发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`175`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: ice: exactly 1, state ice_size=large_sharp_cube, required;
> meltwater: exactly 0, forbidden; plate: exactly 1, state water_area=absent_or_tiny, required. Counts
> and identities stay fixed; no duplicate state copy is created. No additional dropper, external
> liquid stream, external source, extra copy, hand, person, pipette, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VIOLATED RELATION] transparent footprint expands over
> white plate is already present BEFORE first visible cube-edge recession and wet rim.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 冰块（ice）: 恰好1个，状态 ice_size=large_sharp_cube，必需; 融水（meltwater）: 恰好0个，禁止出现; 盘子（plate）:
> 恰好1个，状态 water_area=absent_or_tiny，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [违规关系] 在首次看见立方体边缘退缩并出现湿润边缘发生之前，透明覆盖区在白盘上扩大已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`173`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ice: exactly 1, state ice_size=large_sharp_cube,
> transitioning; meltwater: exactly 1, transitioning; plate: exactly 1, state
> water_area=absent_or_tiny, transitioning. Counts and identities stay fixed; no duplicate state copy
> is created. No additional dropper, external liquid stream, external source, extra copy, hand,
> person, pipette, tool, unlisted object exists. All globally protected identity, shape, material,
> scene, and camera attributes stay unchanged. Declared source loss and result gain remain coupled.
> [VALID RELATION] first visible cube-edge recession and wet rim happens BEFORE transparent footprint
> expands over white plate.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 冰块（ice）: 恰好1个，状态 ice_size=large_sharp_cube，正在过渡; 融水（meltwater）: 恰好1个，正在过渡; 盘子（plate）:
> 恰好1个，状态 water_area=absent_or_tiny，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [有效关系] 首次看见立方体边缘退缩并出现湿润边缘先于透明覆盖区在白盘上扩大发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`176`
- 权重：`1.0`；violation：`early_effect`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: ice: exactly 1, state ice_size=large_sharp_cube,
> transitioning; meltwater: exactly 1, transitioning; plate: exactly 1, state
> water_area=absent_or_tiny, transitioning. Counts and identities stay fixed; no duplicate state copy
> is created. No additional dropper, external liquid stream, external source, extra copy, hand,
> person, pipette, tool, unlisted object exists. All globally protected identity, shape, material,
> scene, and camera attributes stay unchanged. Declared source loss and result gain remain coupled.
> [VIOLATED RELATION] transparent footprint expands over white plate is already present BEFORE first
> visible cube-edge recession and wet rim.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 冰块（ice）: 恰好1个，状态 ice_size=large_sharp_cube，正在过渡; 融水（meltwater）: 恰好1个，正在过渡; 盘子（plate）:
> 恰好1个，状态 water_area=absent_or_tiny，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [违规关系] 在首次看见立方体边缘退缩并出现湿润边缘发生之前，透明覆盖区在白盘上扩大已经出现。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`164`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ice: exactly 1, state ice_size=smaller_receded_ice, required;
> meltwater: exactly 1, required; plate: exactly 1, state water_area=wide_puddle, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional dropper, external liquid
> stream, external source, extra copy, hand, person, pipette, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VALID RELATION] The declared change clear solid
> boundary contracts inward occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 冰块（ice）: 恰好1个，状态 ice_size=smaller_receded_ice，必需; 融水（meltwater）: 恰好1个，必需; 盘子（plate）:
> 恰好1个，状态 water_area=wide_puddle，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [有效关系] 已声明的变化“透明固体边界向内收缩”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`165`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3`

> [STAGE ENTITY INVENTORY] Present now: ice: exactly 1, state ice_size=smaller_receded_ice, required;
> meltwater: exactly 1, required; plate: exactly 1, state water_area=wide_puddle, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional dropper, external liquid
> stream, external source, extra copy, hand, person, pipette, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VIOLATED RELATION] Only one coupled change occurs while
> the other remains unchanged: .

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 冰块（ice）: 恰好1个，状态 ice_size=smaller_receded_ice，必需; 融水（meltwater）: 恰好1个，必需; 盘子（plate）:
> 恰好1个，状态 water_area=wide_puddle，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [违规关系] 只有一个耦合变化发生，而另一个保持不变：（源编译文本此处为空）。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`167`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ice: exactly 1, state ice_size=smaller_receded_ice,
> transitioning; meltwater: exactly 1, transitioning; plate: exactly 1, state water_area=wide_puddle,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> dropper, external liquid stream, external source, extra copy, hand, person, pipette, tool, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The declared change
> clear solid boundary contracts inward occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 冰块（ice）: 恰好1个，状态 ice_size=smaller_receded_ice，正在过渡; 融水（meltwater）: 恰好1个，正在过渡;
> 盘子（plate）: 恰好1个，状态 water_area=wide_puddle，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [有效关系] 已声明的变化“透明固体边界向内收缩”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`166`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3, c4`

> [STAGE ENTITY INVENTORY] Present now: ice: exactly 1, state ice_size=smaller_receded_ice,
> transitioning; meltwater: exactly 1, transitioning; plate: exactly 1, state water_area=wide_puddle,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> dropper, external liquid stream, external source, extra copy, hand, person, pipette, tool, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] The declared
> cause occurs BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 冰块（ice）: 恰好1个，状态 ice_size=smaller_receded_ice，正在过渡; 融水（meltwater）: 恰好1个，正在过渡;
> 盘子（plate）: 恰好1个，状态 water_area=wide_puddle，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [违规关系] 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`228`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ice: exactly 1, state ice_size=smaller_receded_ice, required;
> meltwater: exactly 1, required; plate: exactly 1, state water_area=wide_puddle, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional dropper, external liquid
> stream, external source, extra copy, hand, person, pipette, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VALID RELATION] The terminal facts
> ice_size=smaller_receded_ice, water_area=wide_puddle REMAIN through the final frames. [AND] The same
> logical slot has only its valid state; mutually exclusive facts ice_size=large_sharp_cube,
> ice_size=smaller_receded_ice do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 冰块（ice）: 恰好1个，状态 ice_size=smaller_receded_ice，必需; 融水（meltwater）: 恰好1个，必需; 盘子（plate）:
> 恰好1个，状态 water_area=wide_puddle，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [有效关系] 最终事实 ice_size=smaller_receded_ice（缩小退缩的冰）、water_area=wide_puddle（宽水洼） 在最后几帧持续保持。 [且]
> 同一逻辑槽位只具有其有效状态；互斥事实 ice_size=large_sharp_cube（大而棱角分明的立方体）与 ice_size=smaller_receded_ice（缩小退缩的冰）
> 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`210`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: ice: exactly 1, state ice_size=smaller_receded_ice, required;
> meltwater: exactly 1, required; plate: exactly 1, state water_area=wide_puddle, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional dropper, external liquid
> stream, external source, extra copy, hand, person, pipette, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VIOLATED RELATION] The valid terminal state returns to
> its mutually exclusive earlier state. [AND] The same logical slot has only its valid state; mutually
> exclusive facts ice_size=large_sharp_cube, ice_size=smaller_receded_ice do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 冰块（ice）: 恰好1个，状态 ice_size=smaller_receded_ice，必需; 融水（meltwater）: 恰好1个，必需; 盘子（plate）:
> 恰好1个，状态 water_area=wide_puddle，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [违规关系] 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 ice_size=large_sharp_cube（大而棱角分明的立方体）与
> ice_size=smaller_receded_ice（缩小退缩的冰） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`203`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: ice: exactly 1, state ice_size=smaller_receded_ice, required;
> meltwater: exactly 1, required; plate: exactly 1, state water_area=wide_puddle, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional dropper, external liquid
> stream, external source, extra copy, hand, person, pipette, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VIOLATED RELATION] The terminal facts
> ice_size=smaller_receded_ice, water_area=wide_puddle REMAIN through the final frames. [AND] Copies
> of the same logical slot appear in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 冰块（ice）: 恰好1个，状态 ice_size=smaller_receded_ice，必需; 融水（meltwater）: 恰好1个，必需; 盘子（plate）:
> 恰好1个，状态 water_area=wide_puddle，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [违规关系] 最终事实 ice_size=smaller_receded_ice（缩小退缩的冰）、water_area=wide_puddle（宽水洼） 在最后几帧持续保持。 [且]
> 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_first_melt 必须先于 white_plate_puddle。 |
| `c2` | `changes` | 方向变化：转移 e1 中的效果 pictured_ice_recede 必须按声明方向发生。 |
| `c3` | `couples` | 耦合约束：pictured_ice_recede, white_plate_puddle 必须共同变化；关系为“中央透明固体缩小，同时盘上的透明覆盖区增大”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 ice_size=smaller_receded_ice, water_area=wide_puddle 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 ice_size=large_sharp_cube, ice_size=smaller_receded_ice 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "white_plate_puddle",
    "before": "pictured_first_melt",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "effect": "pictured_ice_recede",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "transition": "e1",
    "type": "changes"
  },
  {
    "effects": [
      "pictured_ice_recede",
      "white_plate_puddle"
    ],
    "id": "c3",
    "relation": "centered clear solid shrinks as transparent plate footprint grows",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "ice_size=smaller_receded_ice",
      "water_area=wide_puddle"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "ice_size=large_sharp_cube",
      "ice_size=smaller_receded_ice"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=2个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"ice":"unresolved_static_envelope","plate":"derived_motion_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P09</strong> — continuous；setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5</summary>

### P09 概要

- 时间路由类型：`continuous`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`无`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains butter: exactly 1; liquid butter: one declared
> collective group; pan: exactly 1. The reference first frame fixes these identities and initial
> layout. [INITIAL STATE] edge_shape=sharp_rectangular, footprint=small_rectangle. [PHYSICAL EVENT]
> The pale-yellow rectangular butter pat near the center of the stainless pan rounds, recedes, and
> spreads into a yellow liquid pool. [PERSISTENT PROTECTION] same pale-yellow butter material; same
> large stainless pan, rim, rivets, and handle; black stovetop, lighting, close three-quarter camera,
> and pan position remain unchanged. [CLOSED WORLD] No additional dropper, external liquid stream,
> external source, extra copy, pipette, unlisted object enters or appears; unoccupied scene regions
> remain empty. [CONSERVATION] Source loss and result gain stay coupled: tracked logical entity slots
> remain consistent; pale solid rounds and lowers while yellow pan footprint expands. No result
> appears from an undeclared external source. [TERMINAL STATE] The same logical entity slots persist
> without duplicates; edge_shape=rounded_receded, footprint=broad_liquid_pool remains visible through
> the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 butter：恰好 1 个；liquid_butter：一个已声明的集合组；pan：恰好 1 个。参考首帧固定这些身份和初始布局。[初始状态]
> edge_shape=sharp_rectangular、footprint=small_rectangle。[物理事件]
> 不锈钢平底锅中央附近的浅黄色长方形黄油块逐渐变圆、退缩，并摊开成黄色液体池。[持续保护]
> 保持同一浅黄色黄油材质；保持同一大型不锈钢锅、锅沿、铆钉和手柄；黑色炉面、照明、近距离四分之三视角相机和锅的位置保持不变。[封闭世界]
> 不允许新增滴管、外部液流、外部来源、额外副本、移液管、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；浅色固体变圆并降低，同时锅中的黄色覆盖区扩大。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；edge_shape=rounded_receded、footprint=broad_liquid_pool 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`162`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: butter: exactly 1, state edge_shape=sharp_rectangular,
> required; liquid butter: exactly 0, forbidden; pan: exactly 1, state footprint=small_rectangle,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> dropper, external liquid stream, external source, extra copy, pipette, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VALID RELATION] first visible rounding of a pale-yellow
> butter corner happens BEFORE yellow footprint expands over stainless base.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 黄油（butter）: 恰好1个，状态 edge_shape=sharp_rectangular，必需; 液态黄油（liquid butter）: 恰好0个，禁止出现;
> 平底锅（pan）: 恰好1个，状态 footprint=small_rectangle，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、移液管、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 浅黄色黄油块的一个角首次明显变圆先于黄色覆盖区在不锈钢锅底上扩大发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`165`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: butter: exactly 1, state edge_shape=sharp_rectangular,
> required; liquid butter: exactly 0, forbidden; pan: exactly 1, state footprint=small_rectangle,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> dropper, external liquid stream, external source, extra copy, pipette, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VIOLATED RELATION] yellow footprint expands over
> stainless base is already present BEFORE first visible rounding of a pale-yellow butter corner.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 黄油（butter）: 恰好1个，状态 edge_shape=sharp_rectangular，必需; 液态黄油（liquid butter）: 恰好0个，禁止出现;
> 平底锅（pan）: 恰好1个，状态 footprint=small_rectangle，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、移液管、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 在浅黄色黄油块的一个角首次明显变圆发生之前，黄色覆盖区在不锈钢锅底上扩大已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`164`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: butter: exactly 1, state edge_shape=sharp_rectangular,
> transitioning; liquid butter: one declared collective group, transitioning; pan: exactly 1, state
> footprint=small_rectangle, transitioning. Counts and identities stay fixed; no duplicate state copy
> is created. No additional dropper, external liquid stream, external source, extra copy, pipette,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION]
> first visible rounding of a pale-yellow butter corner happens BEFORE yellow footprint expands over
> stainless base.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 黄油（butter）: 恰好1个，状态 edge_shape=sharp_rectangular，正在过渡; 液态黄油（liquid butter）:
> 1个已声明集合组，正在过渡; 平底锅（pan）: 恰好1个，状态 footprint=small_rectangle，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、移液管、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 浅黄色黄油块的一个角首次明显变圆先于黄色覆盖区在不锈钢锅底上扩大发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`167`
- 权重：`1.0`；violation：`early_effect`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: butter: exactly 1, state edge_shape=sharp_rectangular,
> transitioning; liquid butter: one declared collective group, transitioning; pan: exactly 1, state
> footprint=small_rectangle, transitioning. Counts and identities stay fixed; no duplicate state copy
> is created. No additional dropper, external liquid stream, external source, extra copy, pipette,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> yellow footprint expands over stainless base is already present BEFORE first visible rounding of a
> pale-yellow butter corner.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 黄油（butter）: 恰好1个，状态 edge_shape=sharp_rectangular，正在过渡; 液态黄油（liquid butter）:
> 1个已声明集合组，正在过渡; 平底锅（pan）: 恰好1个，状态 footprint=small_rectangle，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、移液管、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 在浅黄色黄油块的一个角首次明显变圆发生之前，黄色覆盖区在不锈钢锅底上扩大已经出现。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`158`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: butter: exactly 1, state edge_shape=rounded_receded, required;
> liquid butter: one declared collective group, required; pan: exactly 1, state
> footprint=broad_liquid_pool, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional dropper, external liquid stream, external source, extra copy, pipette,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The
> declared change pale-yellow corners round and solid height lowers occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 黄油（butter）: 恰好1个，状态 edge_shape=rounded_receded，必需; 液态黄油（liquid butter）: 1个已声明集合组，必需;
> 平底锅（pan）: 恰好1个，状态 footprint=broad_liquid_pool，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、移液管、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 已声明的变化“浅黄色边角变圆且固体高度降低”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`155`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3`

> [STAGE ENTITY INVENTORY] Present now: butter: exactly 1, state edge_shape=rounded_receded, required;
> liquid butter: one declared collective group, required; pan: exactly 1, state
> footprint=broad_liquid_pool, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional dropper, external liquid stream, external source, extra copy, pipette,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> Only one coupled change occurs while the other remains unchanged: .

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 黄油（butter）: 恰好1个，状态 edge_shape=rounded_receded，必需; 液态黄油（liquid butter）: 1个已声明集合组，必需;
> 平底锅（pan）: 恰好1个，状态 footprint=broad_liquid_pool，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、移液管、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 只有一个耦合变化发生，而另一个保持不变：（源编译文本此处为空）。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`161`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: butter: exactly 1, state edge_shape=rounded_receded,
> transitioning; liquid butter: one declared collective group, transitioning; pan: exactly 1, state
> footprint=broad_liquid_pool, transitioning. Counts and identities stay fixed; no duplicate state
> copy is created. No additional dropper, external liquid stream, external source, extra copy,
> pipette, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The
> declared change pale-yellow corners round and solid height lowers occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 黄油（butter）: 恰好1个，状态 edge_shape=rounded_receded，正在过渡; 液态黄油（liquid butter）:
> 1个已声明集合组，正在过渡; 平底锅（pan）: 恰好1个，状态 footprint=broad_liquid_pool，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、移液管、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 已声明的变化“浅黄色边角变圆且固体高度降低”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`156`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3, c4`

> [STAGE ENTITY INVENTORY] Present now: butter: exactly 1, state edge_shape=rounded_receded,
> transitioning; liquid butter: one declared collective group, transitioning; pan: exactly 1, state
> footprint=broad_liquid_pool, transitioning. Counts and identities stay fixed; no duplicate state
> copy is created. No additional dropper, external liquid stream, external source, extra copy,
> pipette, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> The declared cause occurs BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 黄油（butter）: 恰好1个，状态 edge_shape=rounded_receded，正在过渡; 液态黄油（liquid butter）:
> 1个已声明集合组，正在过渡; 平底锅（pan）: 恰好1个，状态 footprint=broad_liquid_pool，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、移液管、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`205`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: butter: exactly 1, state edge_shape=rounded_receded, required;
> liquid butter: one declared collective group, required; pan: exactly 1, state
> footprint=broad_liquid_pool, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional dropper, external liquid stream, external source, extra copy, pipette,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The
> terminal facts edge_shape=rounded_receded, footprint=broad_liquid_pool REMAIN through the final
> frames. [AND] The same logical slot has only its valid state; mutually exclusive facts
> footprint=small_rectangle, footprint=broad_liquid_pool do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 黄油（butter）: 恰好1个，状态 edge_shape=rounded_receded，必需; 液态黄油（liquid butter）: 1个已声明集合组，必需;
> 平底锅（pan）: 恰好1个，状态 footprint=broad_liquid_pool，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、移液管、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实
> edge_shape=rounded_receded（圆化退缩）、footprint=broad_liquid_pool（宽阔液体池） 在最后几帧持续保持。 [且]
> 同一逻辑槽位只具有其有效状态；互斥事实 footprint=small_rectangle（小长方形）与 footprint=broad_liquid_pool（宽阔液体池） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`191`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: butter: exactly 1, state edge_shape=rounded_receded, required;
> liquid butter: one declared collective group, required; pan: exactly 1, state
> footprint=broad_liquid_pool, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional dropper, external liquid stream, external source, extra copy, pipette,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> The valid terminal state returns to its mutually exclusive earlier state. [AND] The same logical
> slot has only its valid state; mutually exclusive facts footprint=small_rectangle,
> footprint=broad_liquid_pool do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 黄油（butter）: 恰好1个，状态 edge_shape=rounded_receded，必需; 液态黄油（liquid butter）: 1个已声明集合组，必需;
> 平底锅（pan）: 恰好1个，状态 footprint=broad_liquid_pool，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、移液管、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 footprint=small_rectangle（小长方形）与
> footprint=broad_liquid_pool（宽阔液体池） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`189`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: butter: exactly 1, state edge_shape=rounded_receded, required;
> liquid butter: one declared collective group, required; pan: exactly 1, state
> footprint=broad_liquid_pool, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional dropper, external liquid stream, external source, extra copy, pipette,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> The terminal facts edge_shape=rounded_receded, footprint=broad_liquid_pool REMAIN through the final
> frames. [AND] Copies of the same logical slot appear in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 黄油（butter）: 恰好1个，状态 edge_shape=rounded_receded，必需; 液态黄油（liquid butter）: 1个已声明集合组，必需;
> 平底锅（pan）: 恰好1个，状态 footprint=broad_liquid_pool，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、移液管、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实
> edge_shape=rounded_receded（圆化退缩）、footprint=broad_liquid_pool（宽阔液体池） 在最后几帧持续保持。 [且]
> 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_softening 必须先于 steel_pan_spread。 |
| `c2` | `changes` | 方向变化：转移 e1 中的效果 pictured_edges_round 必须按声明方向发生。 |
| `c3` | `couples` | 耦合约束：pictured_edges_round, steel_pan_spread 必须共同变化；关系为“浅色固体变圆并降低，同时锅中的黄色覆盖区扩大”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 edge_shape=rounded_receded, footprint=broad_liquid_pool 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 footprint=small_rectangle, footprint=broad_liquid_pool 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "steel_pan_spread",
    "before": "pictured_softening",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "effect": "pictured_edges_round",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "transition": "e1",
    "type": "changes"
  },
  {
    "effects": [
      "pictured_edges_round",
      "steel_pan_spread"
    ],
    "id": "c3",
    "relation": "pale solid rounds and lowers while yellow pan footprint expands",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "edge_shape=rounded_receded",
      "footprint=broad_liquid_pool"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "footprint=small_rectangle",
      "footprint=broad_liquid_pool"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=2个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"butter":"unresolved_static_envelope","pan":"derived_motion_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P10</strong> — multi_transition；setup→c1; onset→c1; evolution→c2; completion→c5; terminal→c6,c7</summary>

### P10 概要

- 时间路由类型：`multi_transition`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c2; completion→c5; terminal→c6,c7`
- 同时进入 Global 守恒文本的 couples 约束：`无`
- 仅校验、未进入生成张量的约束：`c3,c4`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains balloon: exactly 1. The reference first frame fixes
> these identities and initial layout. [INITIAL STATE] size=small_pear_shape, surface=wrinkled_loose,
> integrity=intact. [PHYSICAL EVENT] The single small tied green rubber balloon centered on the plain
> background inflates, bursts, and leaves green collapsed fragments in place. [PERSISTENT PROTECTION]
> same green rubber color through balloon and fragments; same tied-neck material identity; plain pale
> background, soft shadow, lighting, central location, and static camera remain unchanged; no
> inflation tool or person is added. [CLOSED WORLD] No additional external source, extra copy, hand,
> person, tool, unlisted object enters or appears; unoccupied scene regions remain empty.
> [CONSERVATION] Source loss and result gain stay coupled: tracked logical entity slots remain
> consistent. No result appears from an undeclared external source. [TERMINAL STATE] The same logical
> entity slots persist without duplicates; size=collapsed_green_pieces, surface=deflated,
> integrity=ruptured_fragments remains visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 balloon：恰好 1 个。参考首帧固定这些身份和初始布局。[初始状态]
> integrity=intact、size=small_pear_shape、surface=wrinkled_loose。[物理事件]
> 纯色背景中央的单个小型系结绿色橡胶气球膨胀、爆裂，并在原处留下瘪塌的绿色碎片。[持续保护]
> 从气球到碎片均保持同一绿色橡胶颜色；保持同一系结颈部的材质身份；纯浅色背景、柔和阴影、照明、中央位置和静止相机保持不变；不添加充气工具或人物。[封闭世界]
> 不允许新增外部来源、额外副本、手、人物、工具、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；integrity=ruptured_fragments、size=collapsed_green_pieces、surface=deflated
> 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`138`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: balloon: exactly 1, state size=small_pear_shape,
> surface=wrinkled_loose, integrity=intact, required. Counts and identities stay fixed; no duplicate
> state copy is created. No additional external source, extra copy, hand, person, tool, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] first visible
> expansion of the small green silhouette happens BEFORE centered green outline expands.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 气球（balloon）: 恰好1个，状态 size=small_pear_shape, surface=wrinkled_loose,
> integrity=intact，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 小型绿色轮廓首次明显扩张先于居中的绿色外轮廓扩大发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`141`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c3`

> [STAGE ENTITY INVENTORY] Present now: balloon: exactly 1, state size=small_pear_shape,
> surface=wrinkled_loose, integrity=intact, required. Counts and identities stay fixed; no duplicate
> state copy is created. No additional external source, extra copy, hand, person, tool, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] centered green
> outline expands is already present BEFORE first visible expansion of the small green silhouette.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 气球（balloon）: 恰好1个，状态 size=small_pear_shape, surface=wrinkled_loose,
> integrity=intact，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 在小型绿色轮廓首次明显扩张发生之前，居中的绿色外轮廓扩大已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`139`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: balloon: exactly 1, state size=small_pear_shape,
> surface=wrinkled_loose, integrity=intact, transitioning. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, hand, person, tool,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION]
> first visible expansion of the small green silhouette happens BEFORE centered green outline expands.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 气球（balloon）: 恰好1个，状态 size=small_pear_shape, surface=wrinkled_loose,
> integrity=intact，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 小型绿色轮廓首次明显扩张先于居中的绿色外轮廓扩大发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`140`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c3, c4`

> [STAGE ENTITY INVENTORY] Present now: balloon: exactly 1, state size=small_pear_shape,
> surface=wrinkled_loose, integrity=intact, transitioning. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, hand, person, tool,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> centered green outline expands happens BEFORE first visible expansion of the small green silhouette.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 气球（balloon）: 恰好1个，状态 size=small_pear_shape, surface=wrinkled_loose,
> integrity=intact，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 居中的绿色外轮廓扩大先于小型绿色轮廓首次明显扩张发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`135`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: balloon: exactly 1, state size=large_taut_shape,
> surface=smooth_taut, integrity=intact, required. Counts and identities stay fixed; no duplicate
> state copy is created. No additional external source, extra copy, hand, person, tool, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The declared change
> centered green outline expands occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 气球（balloon）: 恰好1个，状态 size=large_taut_shape, surface=smooth_taut, integrity=intact，必需.
> 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 已声明的变化“居中的绿色外轮廓扩大”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`137`
- 权重：`1.0`；violation：`wrong_direction`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2`

> [STAGE ENTITY INVENTORY] Present now: balloon: exactly 1, state size=large_taut_shape,
> surface=smooth_taut, integrity=intact, required. Counts and identities stay fixed; no duplicate
> state copy is created. No additional external source, extra copy, hand, person, tool, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] The same
> declared property changes in the opposite or out-of-range direction.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 气球（balloon）: 恰好1个，状态 size=large_taut_shape, surface=smooth_taut, integrity=intact，必需.
> 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 同一项已声明属性朝相反方向变化，或变化超出规定范围。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c5`
- violation 中声明的候选约束：`c5, c6`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`138`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: balloon: exactly 1, state size=collapsed_green_pieces,
> surface=deflated, integrity=ruptured_fragments, transitioning. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, hand, person, tool,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The
> declared change green envelope separates into fragments occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 气球（balloon）: 恰好1个，状态 size=collapsed_green_pieces, surface=deflated,
> integrity=ruptured_fragments，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 已声明的变化“绿色外包络分裂成碎片”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`137`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5, c6`

> [STAGE ENTITY INVENTORY] Present now: balloon: exactly 1, state size=collapsed_green_pieces,
> surface=deflated, integrity=ruptured_fragments, transitioning. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, hand, person, tool,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> The declared cause occurs BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 气球（balloon）: 恰好1个，状态 size=collapsed_green_pieces, surface=deflated,
> integrity=ruptured_fragments，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c6, c7`
- violation 中声明的候选约束：`c6, c7`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`185`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: balloon: exactly 1, state size=collapsed_green_pieces,
> surface=deflated, integrity=ruptured_fragments, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, hand, person, tool,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The
> terminal facts size=collapsed_green_pieces, integrity=ruptured_fragments REMAIN through the final
> frames. [AND] The same logical slot has only its valid state; mutually exclusive facts
> integrity=intact, integrity=ruptured_fragments do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 气球（balloon）: 恰好1个，状态 size=collapsed_green_pieces, surface=deflated,
> integrity=ruptured_fragments，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实
> size=collapsed_green_pieces（瘪塌绿色碎片）、integrity=ruptured_fragments（破裂碎片） 在最后几帧持续保持。 [且]
> 同一逻辑槽位只具有其有效状态；互斥事实 integrity=intact（完整）与 integrity=ruptured_fragments（破裂碎片） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`171`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c6`；violation 声明的候选约束：`c6`

> [STAGE ENTITY INVENTORY] Present now: balloon: exactly 1, state size=collapsed_green_pieces,
> surface=deflated, integrity=ruptured_fragments, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, hand, person, tool,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> The valid terminal state returns to its mutually exclusive earlier state. [AND] The same logical
> slot has only its valid state; mutually exclusive facts integrity=intact,
> integrity=ruptured_fragments do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 气球（balloon）: 恰好1个，状态 size=collapsed_green_pieces, surface=deflated,
> integrity=ruptured_fragments，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实
> integrity=intact（完整）与 integrity=ruptured_fragments（破裂碎片） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`172`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c7`；violation 声明的候选约束：`c7`

> [STAGE ENTITY INVENTORY] Present now: balloon: exactly 1, state size=collapsed_green_pieces,
> surface=deflated, integrity=ruptured_fragments, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, hand, person, tool,
> unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> The terminal facts size=collapsed_green_pieces, integrity=ruptured_fragments REMAIN through the
> final frames. [AND] Copies of the same logical slot appear in mutually exclusive states at the same
> time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 气球（balloon）: 恰好1个，状态 size=collapsed_green_pieces, surface=deflated,
> integrity=ruptured_fragments，必需. 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实
> size=collapsed_green_pieces（瘪塌绿色碎片）、integrity=ruptured_fragments（破裂碎片） 在最后几帧持续保持。 [且]
> 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：green_expansion_onset 必须先于 green_inflate。 |
| `c2` | `changes` | 方向变化：转移 e1 中的效果 green_inflate 必须按声明方向发生。 |
| `c3` | `precedes` | 前置顺序：green_burst 必须先于 green_rupture。 |
| `c4` | `requires` | 因果依赖：green_rupture 只有在 green_burst 之后才能发生。 |
| `c5` | `changes` | 方向变化：转移 e2 中的效果 green_rupture 必须按声明方向发生。 |
| `c6` | `persists` | 持续约束：状态 z2 的事实 size=collapsed_green_pieces, integrity=ruptured_fragments 必须维持到最终帧。 |
| `c7` | `excludes` | 互斥约束：事实 integrity=intact, integrity=ruptured_fragments 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "green_inflate",
    "before": "green_expansion_onset",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "effect": "green_inflate",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "transition": "e1",
    "type": "changes"
  },
  {
    "after": "green_rupture",
    "before": "green_burst",
    "id": "c3",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "cause": "green_burst",
    "effect": "green_rupture",
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "type": "requires"
  },
  {
    "effect": "green_rupture",
    "id": "c5",
    "source_refs": [
      "origin"
    ],
    "transition": "e2",
    "type": "changes"
  },
  {
    "facts": [
      "size=collapsed_green_pieces",
      "integrity=ruptured_fragments"
    ],
    "id": "c6",
    "source_refs": [
      "origin"
    ],
    "state": "z2",
    "type": "persists"
  },
  {
    "facts": [
      "integrity=intact",
      "integrity=ruptured_fragments"
    ],
    "id": "c7",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=1个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"balloon":"derived_motion_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P11</strong> — triggered；setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5</summary>

### P11 概要

- 时间路由类型：`triggered`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`c2`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains bottle: exactly 1; tile floor: one declared collective
> group. The reference first frame fixes these identities and initial layout. [INITIAL STATE]
> position=upper_left_support, integrity=intact_horizontal_bottle, configuration=single_bottle.
> [PHYSICAL EVENT] The clear horizontal bottle leaves the upper-left beige support, falls across the
> pale wall, and shatters on the beige tile floor. [PERSISTENT PROTECTION] same clear colorless glass
> material; upper-left beige support and vertical leg remain fixed; beige tile grid, pale wall,
> baseboard, lighting, and static camera remain unchanged; no release hand or tool is added. [CLOSED
> WORLD] No additional external source, extra copy, hand, person, tool, unlisted object enters or
> appears; unoccupied scene regions remain empty. [CONSERVATION] Source loss and result gain stay
> coupled: tracked logical entity slots remain consistent; beige-tile arrival produces clear fracture
> and local spread. No result appears from an undeclared external source. [TERMINAL STATE] The same
> logical entity slots persist without duplicates; position=beige_tile,
> integrity=fractured_clear_glass, configuration=scattered_floor_fragments remains visible through the
> final frames.

#### 中文解释

> [实体清单] 封闭场景包含 bottle：恰好 1 个；tile_floor：一个已声明的集合组。参考首帧固定这些身份和初始布局。[初始状态]
> configuration=single_bottle、integrity=intact_horizontal_bottle、position=upper_left_support。[物理事件]
> 水平放置的透明瓶子离开左上方米色支撑，沿浅色墙面前方下落，并在米色瓷砖地面上摔碎。[持续保护]
> 保持同一透明无色玻璃材质；左上方米色支撑及其竖直支腿保持固定；米色瓷砖网格、浅色墙、踢脚线、照明和静止相机保持不变；不添加用于释放瓶子的手或工具。[封闭世界]
> 不允许新增外部来源、额外副本、手、人物、工具、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；到达米色瓷砖会产生透明玻璃破裂和局部散开。不允许任何结果由未声明的外部来源产生。[终止状态] 相同的逻辑实体槽位持续存在且没有重复副本；con
> figuration=scattered_floor_fragments、integrity=fractured_clear_glass、position=beige_tile 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`151`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: bottle: exactly 1, state position=upper_left_support,
> integrity=intact_horizontal_bottle, configuration=single_bottle, required; tile floor: one declared
> collective group, required. Counts and identities stay fixed; no duplicate state copy is created. No
> additional external source, extra copy, hand, person, tool, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VALID RELATION] the clear bottle reaches the beige tile
> surface happens BEFORE transparent bottle body breaks.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 恰好1个，状态 position=upper_left_support, integrity=intact_horizontal_bottle,
> configuration=single_bottle，必需; 瓷砖地面（tile floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 透明瓶子抵达米色瓷砖表面先于透明瓶身破裂发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`154`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: bottle: exactly 1, state position=upper_left_support,
> integrity=intact_horizontal_bottle, configuration=single_bottle, required; tile floor: one declared
> collective group, required. Counts and identities stay fixed; no duplicate state copy is created. No
> additional external source, extra copy, hand, person, tool, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VIOLATED RELATION] transparent bottle body breaks is already
> present BEFORE the clear bottle reaches the beige tile surface.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 恰好1个，状态 position=upper_left_support, integrity=intact_horizontal_bottle,
> configuration=single_bottle，必需; 瓷砖地面（tile floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 在透明瓶子抵达米色瓷砖表面发生之前，透明瓶身破裂已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c2`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`153`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: bottle: exactly 1, state position=upper_left_support,
> integrity=intact_horizontal_bottle, configuration=single_bottle, transitioning; tile floor: one
> declared collective group, transitioning. Counts and identities stay fixed; no duplicate state copy
> is created. No additional external source, extra copy, hand, person, tool, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VALID RELATION] the clear bottle reaches the
> beige tile surface happens BEFORE transparent bottle body breaks.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 恰好1个，状态 position=upper_left_support, integrity=intact_horizontal_bottle,
> configuration=single_bottle，正在过渡; 瓷砖地面（tile floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 透明瓶子抵达米色瓷砖表面先于透明瓶身破裂发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`154`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c2`

> [STAGE ENTITY INVENTORY] Present now: bottle: exactly 1, state position=upper_left_support,
> integrity=intact_horizontal_bottle, configuration=single_bottle, transitioning; tile floor: one
> declared collective group, transitioning. Counts and identities stay fixed; no duplicate state copy
> is created. No additional external source, extra copy, hand, person, tool, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] transparent bottle body
> breaks happens BEFORE the clear bottle reaches the beige tile surface.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 恰好1个，状态 position=upper_left_support, integrity=intact_horizontal_bottle,
> configuration=single_bottle，正在过渡; 瓷砖地面（tile floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 透明瓶身破裂先于透明瓶子抵达米色瓷砖表面发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`166`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: bottle: exactly 1, state position=beige_tile,
> integrity=fractured_clear_glass, configuration=scattered_floor_fragments, required; tile floor: one
> declared collective group, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, hand, person, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VALID RELATION] The coupled changes occur together:
> clear bottle descends through open wall corridor WHILE transparent bottle body breaks WHILE clear
> pieces spread over beige tile.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 恰好1个，状态 position=beige_tile, integrity=fractured_clear_glass,
> configuration=scattered_floor_fragments，必需; 瓷砖地面（tile floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 耦合变化同时发生：透明瓶子沿墙前开放通道下降，同时透明瓶身破裂，透明碎片散布到米色瓷砖上。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`172`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3`

> [STAGE ENTITY INVENTORY] Present now: bottle: exactly 1, state position=beige_tile,
> integrity=fractured_clear_glass, configuration=scattered_floor_fragments, required; tile floor: one
> declared collective group, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, hand, person, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VIOLATED RELATION] Only one coupled change occurs while
> the other remains unchanged: clear bottle descends through open wall corridor; transparent bottle
> body breaks; clear pieces spread over beige tile.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 恰好1个，状态 position=beige_tile, integrity=fractured_clear_glass,
> configuration=scattered_floor_fragments，必需; 瓷砖地面（tile floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 只有一个耦合变化发生，另一个保持不变：透明瓶子沿墙前开放通道下降；透明瓶身破裂；透明碎片散布到米色瓷砖上。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`168`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: bottle: exactly 1, state position=beige_tile,
> integrity=fractured_clear_glass, configuration=scattered_floor_fragments, transitioning; tile floor:
> one declared collective group, transitioning. Counts and identities stay fixed; no duplicate state
> copy is created. No additional external source, extra copy, hand, person, tool, unlisted object
> exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The coupled changes
> occur together: clear bottle descends through open wall corridor WHILE transparent bottle body
> breaks WHILE clear pieces spread over beige tile.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 恰好1个，状态 position=beige_tile, integrity=fractured_clear_glass,
> configuration=scattered_floor_fragments，正在过渡; 瓷砖地面（tile floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 耦合变化同时发生：透明瓶子沿墙前开放通道下降，同时透明瓶身破裂，透明碎片散布到米色瓷砖上。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`151`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3, c4`

> [STAGE ENTITY INVENTORY] Present now: bottle: exactly 1, state position=beige_tile,
> integrity=fractured_clear_glass, configuration=scattered_floor_fragments, transitioning; tile floor:
> one declared collective group, transitioning. Counts and identities stay fixed; no duplicate state
> copy is created. No additional external source, extra copy, hand, person, tool, unlisted object
> exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] The declared
> cause occurs BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 恰好1个，状态 position=beige_tile, integrity=fractured_clear_glass,
> configuration=scattered_floor_fragments，正在过渡; 瓷砖地面（tile floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`207`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: bottle: exactly 1, state position=beige_tile,
> integrity=fractured_clear_glass, configuration=scattered_floor_fragments, required; tile floor: one
> declared collective group, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, hand, person, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VALID RELATION] The terminal facts
> integrity=fractured_clear_glass, configuration=scattered_floor_fragments REMAIN through the final
> frames. [AND] The same logical slot has only its valid state; mutually exclusive facts
> integrity=intact_horizontal_bottle, integrity=fractured_clear_glass do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 恰好1个，状态 position=beige_tile, integrity=fractured_clear_glass,
> configuration=scattered_floor_fragments，必需; 瓷砖地面（tile floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实
> integrity=fractured_clear_glass（破碎透明玻璃）、configuration=scattered_floor_fragments（地面散落碎片） 在最后几帧持续保持。
> [且] 同一逻辑槽位只具有其有效状态；互斥事实 integrity=intact_horizontal_bottle（完整水平瓶）与
> integrity=fractured_clear_glass（破碎透明玻璃） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`191`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: bottle: exactly 1, state position=beige_tile,
> integrity=fractured_clear_glass, configuration=scattered_floor_fragments, required; tile floor: one
> declared collective group, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, hand, person, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VIOLATED RELATION] The valid terminal state returns to
> its mutually exclusive earlier state. [AND] The same logical slot has only its valid state; mutually
> exclusive facts integrity=intact_horizontal_bottle, integrity=fractured_clear_glass do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 恰好1个，状态 position=beige_tile, integrity=fractured_clear_glass,
> configuration=scattered_floor_fragments，必需; 瓷砖地面（tile floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 integrity=intact_horizontal_bottle（完整水平瓶）与
> integrity=fractured_clear_glass（破碎透明玻璃） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`187`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: bottle: exactly 1, state position=beige_tile,
> integrity=fractured_clear_glass, configuration=scattered_floor_fragments, required; tile floor: one
> declared collective group, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, hand, person, tool, unlisted object exists. All
> globally protected identity, shape, material, scene, and camera attributes stay unchanged. Declared
> source loss and result gain remain coupled. [VIOLATED RELATION] The terminal facts
> integrity=fractured_clear_glass, configuration=scattered_floor_fragments REMAIN through the final
> frames. [AND] Copies of the same logical slot appear in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 恰好1个，状态 position=beige_tile, integrity=fractured_clear_glass,
> configuration=scattered_floor_fragments，必需; 瓷砖地面（tile floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：外部来源、额外副本、手、人物、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实
> integrity=fractured_clear_glass（破碎透明玻璃）、configuration=scattered_floor_fragments（地面散落碎片） 在最后几帧持续保持。
> [且] 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_tile_impact 必须先于 clear_fracture。 |
| `c2` | `requires` | 因果依赖：clear_fracture 只有在 pictured_tile_impact 之后才能发生。 |
| `c3` | `couples` | 耦合约束：pictured_descent, clear_fracture, tile_scatter 必须共同变化；关系为“到达米色瓷砖会产生透明玻璃破裂和局部散开”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 integrity=fractured_clear_glass, configuration=scattered_floor_fragments 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 integrity=intact_horizontal_bottle, integrity=fractured_clear_glass 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "clear_fracture",
    "before": "pictured_tile_impact",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "cause": "pictured_tile_impact",
    "effect": "clear_fracture",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "type": "requires"
  },
  {
    "effects": [
      "pictured_descent",
      "clear_fracture",
      "tile_scatter"
    ],
    "id": "c3",
    "relation": "beige-tile arrival produces clear fracture and local spread",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "integrity=fractured_clear_glass",
      "configuration=scattered_floor_fragments"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "integrity=intact_horizontal_bottle",
      "integrity=fractured_clear_glass"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=2个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"bottle":"derived_motion_envelope","tile_floor":"unresolved_static_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P12</strong> — continuous；setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5</summary>

### P12 概要

- 时间路由类型：`continuous`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`无`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains dandelion: exactly 1; seeds: one declared collective
> group; drift region: one declared collective group. The reference first frame fixes these identities
> and initial layout. [INITIAL STATE] attachment=densely_attached, seed_position=on_left_head,
> fullness=full_round_head. [PHYSICAL EVENT] Seeds detach from the full white dandelion at left and
> drift rightward across the blue-sky background as the head depletes. [PERSISTENT PROTECTION] same
> yellow-green stem and brown central head; white seed identity remains consistent; blue sky, blurred
> green field, daylight, focus, and static camera remain unchanged. [CLOSED WORLD] No additional
> external source, extra copy, unlisted object enters or appears; unoccupied scene regions remain
> empty. [CONSERVATION] Source loss and result gain stay coupled: tracked logical entity slots remain
> consistent; rightward white-seed drift accompanies decreasing white head coverage. No result appears
> from an undeclared external source. [TERMINAL STATE] The same logical entity slots persist without
> duplicates; attachment=detached, seed_position=rightward_airborne, fullness=sparse_depleted_head
> remains visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 dandelion：恰好 1 个；seeds：一个已声明的集合组；drift_region：一个已声明的集合组。参考首帧固定这些身份和初始布局。[初始状态]
> attachment=densely_attached、fullness=full_round_head、seed_position=on_left_head。[物理事件]
> 种子从左侧饱满的白色蒲公英上脱离，沿蓝天背景向右飘移，同时花头逐渐稀疏。[持续保护] 保持同一黄绿色茎和棕色中央花头；白色种子的身份保持一致；蓝天、虚化绿地、日光、焦点和静止相机保持不变。[封闭世界]
> 不允许新增外部来源、额外副本、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；白色种子向右飘移，同时花头的白色覆盖减少。不允许任何结果由未声明的外部来源产生。[终止状态] 相同的逻辑实体槽位持续存在且没有重复副本；att
> achment=detached、fullness=sparse_depleted_head、seed_position=rightward_airborne 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`160`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: dandelion: exactly 1, state fullness=full_round_head,
> required; seeds: one declared collective group, state attachment=densely_attached,
> seed_position=on_left_head, required; drift region: one declared collective group, required. Counts
> and identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION]
> first white seeds leave the left-side head happens BEFORE white seeds travel right across blue sky.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 蒲公英（dandelion）: 恰好1个，状态 fullness=full_round_head，必需; 种子（seeds）: 1个已声明集合组，状态
> attachment=densely_attached, seed_position=on_left_head，必需; 飘移区域（drift region）: 1个已声明集合组，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 第一批白色种子离开左侧花头先于白色种子横穿蓝天向右移动发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`163`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: dandelion: exactly 1, state fullness=full_round_head,
> required; seeds: one declared collective group, state attachment=densely_attached,
> seed_position=on_left_head, required; drift region: one declared collective group, required. Counts
> and identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> white seeds travel right across blue sky is already present BEFORE first white seeds leave the
> left-side head.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 蒲公英（dandelion）: 恰好1个，状态 fullness=full_round_head，必需; 种子（seeds）: 1个已声明集合组，状态
> attachment=densely_attached, seed_position=on_left_head，必需; 飘移区域（drift region）: 1个已声明集合组，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 在第一批白色种子离开左侧花头发生之前，白色种子横穿蓝天向右移动已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`163`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: dandelion: exactly 1, state fullness=full_round_head,
> transitioning; seeds: one declared collective group, state attachment=densely_attached,
> seed_position=on_left_head, transitioning; drift region: one declared collective group,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] first white seeds leave the left-side head happens BEFORE white seeds
> travel right across blue sky.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 蒲公英（dandelion）: 恰好1个，状态 fullness=full_round_head，正在过渡; 种子（seeds）: 1个已声明集合组，状态
> attachment=densely_attached, seed_position=on_left_head，正在过渡; 飘移区域（drift region）: 1个已声明集合组，正在过渡.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 第一批白色种子离开左侧花头先于白色种子横穿蓝天向右移动发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`164`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: dandelion: exactly 1, state fullness=full_round_head,
> transitioning; seeds: one declared collective group, state attachment=densely_attached,
> seed_position=on_left_head, transitioning; drift region: one declared collective group,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] white seeds travel right across blue sky happens BEFORE first white
> seeds leave the left-side head.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 蒲公英（dandelion）: 恰好1个，状态 fullness=full_round_head，正在过渡; 种子（seeds）: 1个已声明集合组，状态
> attachment=densely_attached, seed_position=on_left_head，正在过渡; 飘移区域（drift region）: 1个已声明集合组，正在过渡.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 白色种子横穿蓝天向右移动先于第一批白色种子离开左侧花头发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`153`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: dandelion: exactly 1, state fullness=sparse_depleted_head,
> required; seeds: one declared collective group, state attachment=detached,
> seed_position=rightward_airborne, required; drift region: one declared collective group, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, unlisted object exists. All globally protected identity, shape, material, scene, and
> camera attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID
> RELATION] The declared change white seed connections release occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 蒲公英（dandelion）: 恰好1个，状态 fullness=sparse_depleted_head，必需; 种子（seeds）: 1个已声明集合组，状态
> attachment=detached, seed_position=rightward_airborne，必需; 飘移区域（drift region）: 1个已声明集合组，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 已声明的变化“白色种子的连接解除”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`157`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3`

> [STAGE ENTITY INVENTORY] Present now: dandelion: exactly 1, state fullness=sparse_depleted_head,
> required; seeds: one declared collective group, state attachment=detached,
> seed_position=rightward_airborne, required; drift region: one declared collective group, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, unlisted object exists. All globally protected identity, shape, material, scene, and
> camera attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED
> RELATION] Only one coupled change occurs while the other remains unchanged: .

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 蒲公英（dandelion）: 恰好1个，状态 fullness=sparse_depleted_head，必需; 种子（seeds）: 1个已声明集合组，状态
> attachment=detached, seed_position=rightward_airborne，必需; 飘移区域（drift region）: 1个已声明集合组，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 只有一个耦合变化发生，而另一个保持不变：（源编译文本此处为空）。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`156`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: dandelion: exactly 1, state fullness=sparse_depleted_head,
> transitioning; seeds: one declared collective group, state attachment=detached,
> seed_position=rightward_airborne, transitioning; drift region: one declared collective group,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] The declared change white seed connections release occurs in its stated
> direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 蒲公英（dandelion）: 恰好1个，状态 fullness=sparse_depleted_head，正在过渡; 种子（seeds）: 1个已声明集合组，状态
> attachment=detached, seed_position=rightward_airborne，正在过渡; 飘移区域（drift region）: 1个已声明集合组，正在过渡.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 已声明的变化“白色种子的连接解除”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`158`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3, c4`

> [STAGE ENTITY INVENTORY] Present now: dandelion: exactly 1, state fullness=sparse_depleted_head,
> transitioning; seeds: one declared collective group, state attachment=detached,
> seed_position=rightward_airborne, transitioning; drift region: one declared collective group,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] The declared cause occurs BUT the required target change does not
> become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 蒲公英（dandelion）: 恰好1个，状态 fullness=sparse_depleted_head，正在过渡; 种子（seeds）: 1个已声明集合组，状态
> attachment=detached, seed_position=rightward_airborne，正在过渡; 飘移区域（drift region）: 1个已声明集合组，正在过渡.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`204`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: dandelion: exactly 1, state fullness=sparse_depleted_head,
> required; seeds: one declared collective group, state attachment=detached,
> seed_position=rightward_airborne, required; drift region: one declared collective group, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, unlisted object exists. All globally protected identity, shape, material, scene, and
> camera attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID
> RELATION] The terminal facts attachment=detached, fullness=sparse_depleted_head REMAIN through the
> final frames. [AND] The same logical slot has only its valid state; mutually exclusive facts
> attachment=densely_attached, attachment=detached do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 蒲公英（dandelion）: 恰好1个，状态 fullness=sparse_depleted_head，必需; 种子（seeds）: 1个已声明集合组，状态
> attachment=detached, seed_position=rightward_airborne，必需; 飘移区域（drift region）: 1个已声明集合组，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 最终事实 attachment=detached（已脱离）、fullness=sparse_depleted_head（稀疏耗尽的花头） 在最后几帧持续保持。 [且]
> 同一逻辑槽位只具有其有效状态；互斥事实 attachment=densely_attached（密集附着）与 attachment=detached（已脱离） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`191`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: dandelion: exactly 1, state fullness=sparse_depleted_head,
> required; seeds: one declared collective group, state attachment=detached,
> seed_position=rightward_airborne, required; drift region: one declared collective group, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, unlisted object exists. All globally protected identity, shape, material, scene, and
> camera attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED
> RELATION] The valid terminal state returns to its mutually exclusive earlier state. [AND] The same
> logical slot has only its valid state; mutually exclusive facts attachment=densely_attached,
> attachment=detached do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 蒲公英（dandelion）: 恰好1个，状态 fullness=sparse_depleted_head，必需; 种子（seeds）: 1个已声明集合组，状态
> attachment=detached, seed_position=rightward_airborne，必需; 飘移区域（drift region）: 1个已声明集合组，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 attachment=densely_attached（密集附着）与
> attachment=detached（已脱离） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`190`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: dandelion: exactly 1, state fullness=sparse_depleted_head,
> required; seeds: one declared collective group, state attachment=detached,
> seed_position=rightward_airborne, required; drift region: one declared collective group, required.
> Counts and identities stay fixed; no duplicate state copy is created. No additional external source,
> extra copy, unlisted object exists. All globally protected identity, shape, material, scene, and
> camera attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED
> RELATION] The terminal facts attachment=detached, fullness=sparse_depleted_head REMAIN through the
> final frames. [AND] Copies of the same logical slot appear in mutually exclusive states at the same
> time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 蒲公英（dandelion）: 恰好1个，状态 fullness=sparse_depleted_head，必需; 种子（seeds）: 1个已声明集合组，状态
> attachment=detached, seed_position=rightward_airborne，必需; 飘移区域（drift region）: 1个已声明集合组，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 最终事实 attachment=detached（已脱离）、fullness=sparse_depleted_head（稀疏耗尽的花头） 在最后几帧持续保持。 [且]
> 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_first_detachment 必须先于 blue_sky_drift。 |
| `c2` | `changes` | 方向变化：转移 e1 中的效果 white_detach 必须按声明方向发生。 |
| `c3` | `couples` | 耦合约束：white_detach, blue_sky_drift, pictured_depletion 必须共同变化；关系为“白色种子向右飘移，同时花头的白色覆盖减少”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 attachment=detached, fullness=sparse_depleted_head 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 attachment=densely_attached, attachment=detached 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "blue_sky_drift",
    "before": "pictured_first_detachment",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "effect": "white_detach",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "transition": "e1",
    "type": "changes"
  },
  {
    "effects": [
      "white_detach",
      "blue_sky_drift",
      "pictured_depletion"
    ],
    "id": "c3",
    "relation": "rightward white-seed drift accompanies decreasing white head coverage",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "attachment=detached",
      "fullness=sparse_depleted_head"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "attachment=densely_attached",
      "attachment=detached"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=3个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"dandelion":"unresolved_static_envelope","drift_region":"unresolved_static_envelope","seeds":"derived_motion_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P13</strong> — continuous；setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5</summary>

### P13 概要

- 时间路由类型：`continuous`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`无`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains pitcher: exactly 1; stream: one declared collective
> group; glass: exactly 1. The reference first frame fixes these identities and initial layout.
> [INITIAL STATE] source_level=about_half_full, stream_state=absent, sink_level=empty. [PHYSICAL
> EVENT] The large clear half-filled pitcher at left pours water into the empty clear tumbler at
> right. [PERSISTENT PROTECTION] same clear handled pitcher and clear straight tumbler; container
> transparency, gray tabletop, pale background, lighting, and static camera remain unchanged; water
> does not spill outside the glass. [CLOSED WORLD] No additional external source, extra copy, unlisted
> object enters or appears; unoccupied scene regions remain empty. [CONSERVATION] Source loss and
> result gain stay coupled: tracked logical entity slots remain consistent; left waterline falls
> through clear stream while right waterline rises. No result appears from an undeclared external
> source. [TERMINAL STATE] The same logical entity slots persist without duplicates;
> source_level=visibly_lower, stream_state=continuous_clear_stream, sink_level=visibly_higher remains
> visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 pitcher：恰好 1 个；stream：一个已声明的集合组；glass：恰好 1 个。参考首帧固定这些身份和初始布局。[初始状态]
> sink_level=empty、source_level=about_half_full、stream_state=absent。[物理事件]
> 左侧大型透明半满水壶向右侧空透明玻璃杯中倒水。[持续保护] 保持同一透明带柄水壶和透明直筒玻璃杯；容器透明度、灰色桌面、浅色背景、照明和静止相机保持不变；水不溢出玻璃杯。[封闭世界]
> 不允许新增外部来源、额外副本、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；左侧水位经透明水流下降，同时右侧水位上升。不允许任何结果由未声明的外部来源产生。[终止状态] 相同的逻辑实体槽位持续存在且没有重复副本；sin
> k_level=visibly_higher、source_level=visibly_lower、stream_state=continuous_clear_stream 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`160`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: pitcher: exactly 1, state source_level=about_half_full,
> required; stream: exactly 0, state stream_state=absent, forbidden; glass: exactly 1, state
> sink_level=empty, required. Counts and identities stay fixed; no duplicate state copy is created. No
> additional external source, extra copy, unlisted object exists. All globally protected identity,
> shape, material, scene, and camera attributes stay unchanged. Declared source loss and result gain
> remain coupled. [VALID RELATION] clear water first connects the pitcher spout to the tumbler opening
> happens BEFORE right tumbler waterline rises.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（pitcher）: 恰好1个，状态 source_level=about_half_full，必需; 水流（stream）: 恰好0个，状态
> stream_state=absent，禁止出现; 玻璃杯（glass）: 恰好1个，状态 sink_level=empty，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 清水首次从壶嘴连续连接至玻璃杯开口先于右侧玻璃杯的水位上升发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`163`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: pitcher: exactly 1, state source_level=about_half_full,
> required; stream: exactly 0, state stream_state=absent, forbidden; glass: exactly 1, state
> sink_level=empty, required. Counts and identities stay fixed; no duplicate state copy is created. No
> additional external source, extra copy, unlisted object exists. All globally protected identity,
> shape, material, scene, and camera attributes stay unchanged. Declared source loss and result gain
> remain coupled. [VIOLATED RELATION] right tumbler waterline rises is already present BEFORE clear
> water first connects the pitcher spout to the tumbler opening.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（pitcher）: 恰好1个，状态 source_level=about_half_full，必需; 水流（stream）: 恰好0个，状态
> stream_state=absent，禁止出现; 玻璃杯（glass）: 恰好1个，状态 sink_level=empty，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 在清水首次从壶嘴连续连接至玻璃杯开口发生之前，右侧玻璃杯的水位上升已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`162`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: pitcher: exactly 1, state source_level=about_half_full,
> transitioning; stream: one declared collective group, state stream_state=absent, transitioning;
> glass: exactly 1, state sink_level=empty, transitioning. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VALID RELATION] clear water first connects the
> pitcher spout to the tumbler opening happens BEFORE right tumbler waterline rises.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（pitcher）: 恰好1个，状态 source_level=about_half_full，正在过渡; 水流（stream）: 1个已声明集合组，状态
> stream_state=absent，正在过渡; 玻璃杯（glass）: 恰好1个，状态 sink_level=empty，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 清水首次从壶嘴连续连接至玻璃杯开口先于右侧玻璃杯的水位上升发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`163`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: pitcher: exactly 1, state source_level=about_half_full,
> transitioning; stream: one declared collective group, state stream_state=absent, transitioning;
> glass: exactly 1, state sink_level=empty, transitioning. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] right tumbler waterline
> rises happens BEFORE clear water first connects the pitcher spout to the tumbler opening.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（pitcher）: 恰好1个，状态 source_level=about_half_full，正在过渡; 水流（stream）: 1个已声明集合组，状态
> stream_state=absent，正在过渡; 玻璃杯（glass）: 恰好1个，状态 sink_level=empty，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 右侧玻璃杯的水位上升先于清水首次从壶嘴连续连接至玻璃杯开口发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`157`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: pitcher: exactly 1, state source_level=visibly_lower,
> required; stream: one declared collective group, state stream_state=continuous_clear_stream,
> required; glass: exactly 1, state sink_level=visibly_higher, required. Counts and identities stay
> fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The declared change
> left pitcher waterline descends occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（pitcher）: 恰好1个，状态 source_level=visibly_lower，必需; 水流（stream）: 1个已声明集合组，状态
> stream_state=continuous_clear_stream，必需; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 已声明的变化“左侧水壶的水位下降”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`158`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3`

> [STAGE ENTITY INVENTORY] Present now: pitcher: exactly 1, state source_level=visibly_lower,
> required; stream: one declared collective group, state stream_state=continuous_clear_stream,
> required; glass: exactly 1, state sink_level=visibly_higher, required. Counts and identities stay
> fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] Only one coupled
> change occurs while the other remains unchanged: .

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（pitcher）: 恰好1个，状态 source_level=visibly_lower，必需; 水流（stream）: 1个已声明集合组，状态
> stream_state=continuous_clear_stream，必需; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 只有一个耦合变化发生，而另一个保持不变：（源编译文本此处为空）。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`160`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: pitcher: exactly 1, state source_level=visibly_lower,
> transitioning; stream: one declared collective group, state stream_state=continuous_clear_stream,
> transitioning; glass: exactly 1, state sink_level=visibly_higher, transitioning. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The
> declared change left pitcher waterline descends occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（pitcher）: 恰好1个，状态 source_level=visibly_lower，正在过渡; 水流（stream）: 1个已声明集合组，状态
> stream_state=continuous_clear_stream，正在过渡; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，正在过渡.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 已声明的变化“左侧水壶的水位下降”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`159`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3, c4`

> [STAGE ENTITY INVENTORY] Present now: pitcher: exactly 1, state source_level=visibly_lower,
> transitioning; stream: one declared collective group, state stream_state=continuous_clear_stream,
> transitioning; glass: exactly 1, state sink_level=visibly_higher, transitioning. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> The declared cause occurs BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（pitcher）: 恰好1个，状态 source_level=visibly_lower，正在过渡; 水流（stream）: 1个已声明集合组，状态
> stream_state=continuous_clear_stream，正在过渡; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，正在过渡.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`209`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: pitcher: exactly 1, state source_level=visibly_lower,
> required; stream: exactly 0, state stream_state=continuous_clear_stream, forbidden; glass: exactly
> 1, state sink_level=visibly_higher, required. Counts and identities stay fixed; no duplicate state
> copy is created. No additional external source, extra copy, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VALID RELATION] The terminal facts source_level=visibly_lower,
> sink_level=visibly_higher REMAIN through the final frames. [AND] The same logical slot has only its
> valid state; mutually exclusive facts sink_level=empty, sink_level=visibly_higher do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（pitcher）: 恰好1个，状态 source_level=visibly_lower，必需; 水流（stream）: 恰好0个，状态
> stream_state=continuous_clear_stream，禁止出现; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 最终事实 source_level=visibly_lower（源水位明显降低）、sink_level=visibly_higher（目标水位明显升高） 在最后几帧持续保持。 [且]
> 同一逻辑槽位只具有其有效状态；互斥事实 sink_level=empty（空）与 sink_level=visibly_higher（明显升高） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`194`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: pitcher: exactly 1, state source_level=visibly_lower,
> required; stream: exactly 0, state stream_state=continuous_clear_stream, forbidden; glass: exactly
> 1, state sink_level=visibly_higher, required. Counts and identities stay fixed; no duplicate state
> copy is created. No additional external source, extra copy, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VIOLATED RELATION] The valid terminal state returns to its
> mutually exclusive earlier state. [AND] The same logical slot has only its valid state; mutually
> exclusive facts sink_level=empty, sink_level=visibly_higher do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（pitcher）: 恰好1个，状态 source_level=visibly_lower，必需; 水流（stream）: 恰好0个，状态
> stream_state=continuous_clear_stream，禁止出现; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 sink_level=empty（空）与 sink_level=visibly_higher（明显升高）
> 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`194`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: pitcher: exactly 1, state source_level=visibly_lower,
> required; stream: exactly 0, state stream_state=continuous_clear_stream, forbidden; glass: exactly
> 1, state sink_level=visibly_higher, required. Counts and identities stay fixed; no duplicate state
> copy is created. No additional external source, extra copy, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VIOLATED RELATION] The terminal facts
> source_level=visibly_lower, sink_level=visibly_higher REMAIN through the final frames. [AND] Copies
> of the same logical slot appear in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（pitcher）: 恰好1个，状态 source_level=visibly_lower，必需; 水流（stream）: 恰好0个，状态
> stream_state=continuous_clear_stream，禁止出现; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 最终事实 source_level=visibly_lower（源水位明显降低）、sink_level=visibly_higher（目标水位明显升高） 在最后几帧持续保持。 [且]
> 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_stream_onset 必须先于 pictured_sink_rise。 |
| `c2` | `changes` | 方向变化：转移 e1 中的效果 pictured_source_fall 必须按声明方向发生。 |
| `c3` | `couples` | 耦合约束：pictured_source_fall, clear_carrier, pictured_sink_rise 必须共同变化；关系为“左侧水位经透明水流下降，同时右侧水位上升”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 source_level=visibly_lower, sink_level=visibly_higher 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 sink_level=empty, sink_level=visibly_higher 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "pictured_sink_rise",
    "before": "pictured_stream_onset",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "effect": "pictured_source_fall",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "transition": "e1",
    "type": "changes"
  },
  {
    "effects": [
      "pictured_source_fall",
      "clear_carrier",
      "pictured_sink_rise"
    ],
    "id": "c3",
    "relation": "left waterline falls through clear stream while right waterline rises",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "source_level=visibly_lower",
      "sink_level=visibly_higher"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "sink_level=empty",
      "sink_level=visibly_higher"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=2个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"glass":"derived_motion_envelope","pitcher":"derived_motion_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P14</strong> — continuous；setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5</summary>

### P14 概要

- 时间路由类型：`continuous`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`无`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains bottle: one declared collective group; stream: one
> declared collective group; glass: exactly 1. The reference first frame fixes these identities and
> initial layout. [INITIAL STATE] source_level=high, stream_state=absent, sink_level=empty. [PHYSICAL
> EVENT] The tall clear bottle of bright orange juice at left pours into the empty clear tumbler at
> right. [PERSISTENT PROTECTION] same tall clear bottle, tapered clear tumbler, and bright orange
> juice; wood tabletop, pale gray background, lighting, and static camera remain unchanged; orange
> juice does not spill onto the table. [CLOSED WORLD] No additional dropper, external liquid stream,
> external source, extra copy, hand, person, pipette, tool, unlisted object enters or appears;
> unoccupied scene regions remain empty. [CONSERVATION] Source loss and result gain stay coupled:
> tracked logical entity slots remain consistent; left orange level falls through bright stream while
> right level rises. No result appears from an undeclared external source. [TERMINAL STATE] The same
> logical entity slots persist without duplicates; source_level=visibly_lower,
> stream_state=continuous_orange, sink_level=visibly_higher remains visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 bottle：一个已声明的集合组；stream：一个已声明的集合组；glass：恰好 1 个。参考首帧固定这些身份和初始布局。[初始状态]
> sink_level=empty、source_level=high、stream_state=absent。[物理事件] 左侧装有亮橙色果汁的高透明瓶向右侧空透明玻璃杯中倾倒果汁。[持续保护]
> 保持同一高透明瓶、锥形透明玻璃杯和亮橙色果汁；木桌面、浅灰背景、照明和静止相机保持不变；橙汁不洒到桌上。[封闭世界]
> 不允许新增滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；左侧橙色液位经亮色液流下降，同时右侧液位上升。不允许任何结果由未声明的外部来源产生。[终止状态] 相同的逻辑实体槽位持续存在且没有重复副本；s
> ink_level=visibly_higher、source_level=visibly_lower、stream_state=continuous_orange 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`169`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: bottle: one declared collective group, state
> source_level=high, required; stream: exactly 0, state stream_state=absent, forbidden; glass: exactly
> 1, state sink_level=empty, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional dropper, external liquid stream, external source, extra copy, hand, person,
> pipette, tool, unlisted object exists. All globally protected identity, shape, material, scene, and
> camera attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID
> RELATION] orange juice first connects the bottle mouth to the glass rim happens BEFORE orange line
> rises in tumbler.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 1个已声明集合组，状态 source_level=high，必需; 液流（stream）: 恰好0个，状态
> stream_state=absent，禁止出现; 玻璃杯（glass）: 恰好1个，状态 sink_level=empty，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [有效关系] 橙汁首次从瓶口连续连接至玻璃杯杯沿先于玻璃杯中的橙色液面上升发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`172`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: bottle: one declared collective group, state
> source_level=high, required; stream: exactly 0, state stream_state=absent, forbidden; glass: exactly
> 1, state sink_level=empty, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional dropper, external liquid stream, external source, extra copy, hand, person,
> pipette, tool, unlisted object exists. All globally protected identity, shape, material, scene, and
> camera attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED
> RELATION] orange line rises in tumbler is already present BEFORE orange juice first connects the
> bottle mouth to the glass rim.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 1个已声明集合组，状态 source_level=high，必需; 液流（stream）: 恰好0个，状态
> stream_state=absent，禁止出现; 玻璃杯（glass）: 恰好1个，状态 sink_level=empty，必需. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [违规关系] 在橙汁首次从瓶口连续连接至玻璃杯杯沿发生之前，玻璃杯中的橙色液面上升已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`171`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: bottle: one declared collective group, state
> source_level=high, transitioning; stream: one declared collective group, state stream_state=absent,
> transitioning; glass: exactly 1, state sink_level=empty, transitioning. Counts and identities stay
> fixed; no duplicate state copy is created. No additional dropper, external liquid stream, external
> source, extra copy, hand, person, pipette, tool, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VALID RELATION] orange juice first connects the bottle mouth to the
> glass rim happens BEFORE orange line rises in tumbler.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 1个已声明集合组，状态 source_level=high，正在过渡; 液流（stream）: 1个已声明集合组，状态
> stream_state=absent，正在过渡; 玻璃杯（glass）: 恰好1个，状态 sink_level=empty，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [有效关系] 橙汁首次从瓶口连续连接至玻璃杯杯沿先于玻璃杯中的橙色液面上升发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`172`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: bottle: one declared collective group, state
> source_level=high, transitioning; stream: one declared collective group, state stream_state=absent,
> transitioning; glass: exactly 1, state sink_level=empty, transitioning. Counts and identities stay
> fixed; no duplicate state copy is created. No additional dropper, external liquid stream, external
> source, extra copy, hand, person, pipette, tool, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VIOLATED RELATION] orange line rises in tumbler happens BEFORE orange
> juice first connects the bottle mouth to the glass rim.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 1个已声明集合组，状态 source_level=high，正在过渡; 液流（stream）: 1个已声明集合组，状态
> stream_state=absent，正在过渡; 玻璃杯（glass）: 恰好1个，状态 sink_level=empty，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。
> [违规关系] 玻璃杯中的橙色液面上升先于橙汁首次从瓶口连续连接至玻璃杯杯沿发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`171`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: bottle: one declared collective group, state
> source_level=visibly_lower, required; stream: one declared collective group, state
> stream_state=continuous_orange, required; glass: exactly 1, state sink_level=visibly_higher,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> dropper, external liquid stream, external source, extra copy, hand, person, pipette, tool, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The declared change
> bright orange line descends in bottle occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 1个已声明集合组，状态 source_level=visibly_lower，必需; 液流（stream）: 1个已声明集合组，状态
> stream_state=continuous_orange，必需; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，必需.
> 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 已声明的变化“瓶内亮橙色液面下降”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`172`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3`

> [STAGE ENTITY INVENTORY] Present now: bottle: one declared collective group, state
> source_level=visibly_lower, required; stream: one declared collective group, state
> stream_state=continuous_orange, required; glass: exactly 1, state sink_level=visibly_higher,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> dropper, external liquid stream, external source, extra copy, hand, person, pipette, tool, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] Only one coupled
> change occurs while the other remains unchanged: .

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 1个已声明集合组，状态 source_level=visibly_lower，必需; 液流（stream）: 1个已声明集合组，状态
> stream_state=continuous_orange，必需; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，必需.
> 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 只有一个耦合变化发生，而另一个保持不变：（源编译文本此处为空）。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`174`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: bottle: one declared collective group, state
> source_level=visibly_lower, transitioning; stream: one declared collective group, state
> stream_state=continuous_orange, transitioning; glass: exactly 1, state sink_level=visibly_higher,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> dropper, external liquid stream, external source, extra copy, hand, person, pipette, tool, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The declared change
> bright orange line descends in bottle occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 1个已声明集合组，状态 source_level=visibly_lower，正在过渡; 液流（stream）: 1个已声明集合组，状态
> stream_state=continuous_orange，正在过渡; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，正在过渡.
> 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 已声明的变化“瓶内亮橙色液面下降”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`173`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3, c4`

> [STAGE ENTITY INVENTORY] Present now: bottle: one declared collective group, state
> source_level=visibly_lower, transitioning; stream: one declared collective group, state
> stream_state=continuous_orange, transitioning; glass: exactly 1, state sink_level=visibly_higher,
> transitioning. Counts and identities stay fixed; no duplicate state copy is created. No additional
> dropper, external liquid stream, external source, extra copy, hand, person, pipette, tool, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] The declared
> cause occurs BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 1个已声明集合组，状态 source_level=visibly_lower，正在过渡; 液流（stream）: 1个已声明集合组，状态
> stream_state=continuous_orange，正在过渡; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，正在过渡.
> 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`223`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: bottle: one declared collective group, state
> source_level=visibly_lower, required; stream: exactly 0, state stream_state=continuous_orange,
> forbidden; glass: exactly 1, state sink_level=visibly_higher, required. Counts and identities stay
> fixed; no duplicate state copy is created. No additional dropper, external liquid stream, external
> source, extra copy, hand, person, pipette, tool, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VALID RELATION] The terminal facts source_level=visibly_lower,
> sink_level=visibly_higher REMAIN through the final frames. [AND] The same logical slot has only its
> valid state; mutually exclusive facts sink_level=empty, sink_level=visibly_higher do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 1个已声明集合组，状态 source_level=visibly_lower，必需; 液流（stream）: 恰好0个，状态
> stream_state=continuous_orange，禁止出现; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，必需.
> 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实
> source_level=visibly_lower（源液位明显降低）、sink_level=visibly_higher（目标液位明显升高） 在最后几帧持续保持。 [且]
> 同一逻辑槽位只具有其有效状态；互斥事实 sink_level=empty（空）与 sink_level=visibly_higher（明显升高） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`208`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: bottle: one declared collective group, state
> source_level=visibly_lower, required; stream: exactly 0, state stream_state=continuous_orange,
> forbidden; glass: exactly 1, state sink_level=visibly_higher, required. Counts and identities stay
> fixed; no duplicate state copy is created. No additional dropper, external liquid stream, external
> source, extra copy, hand, person, pipette, tool, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VIOLATED RELATION] The valid terminal state returns to its mutually
> exclusive earlier state. [AND] The same logical slot has only its valid state; mutually exclusive
> facts sink_level=empty, sink_level=visibly_higher do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 1个已声明集合组，状态 source_level=visibly_lower，必需; 液流（stream）: 恰好0个，状态
> stream_state=continuous_orange，禁止出现; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，必需.
> 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 sink_level=empty（空）与
> sink_level=visibly_higher（明显升高） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`208`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: bottle: one declared collective group, state
> source_level=visibly_lower, required; stream: exactly 0, state stream_state=continuous_orange,
> forbidden; glass: exactly 1, state sink_level=visibly_higher, required. Counts and identities stay
> fixed; no duplicate state copy is created. No additional dropper, external liquid stream, external
> source, extra copy, hand, person, pipette, tool, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VIOLATED RELATION] The terminal facts source_level=visibly_lower,
> sink_level=visibly_higher REMAIN through the final frames. [AND] Copies of the same logical slot
> appear in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 瓶子（bottle）: 1个已声明集合组，状态 source_level=visibly_lower，必需; 液流（stream）: 恰好0个，状态
> stream_state=continuous_orange，禁止出现; 玻璃杯（glass）: 恰好1个，状态 sink_level=visibly_higher，必需.
> 数量和身份保持固定；不创建重复的状态副本。 以下额外对象均不存在：滴管、外部液流、外部来源、额外副本、手、人物、移液管、工具、未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实
> source_level=visibly_lower（源液位明显降低）、sink_level=visibly_higher（目标液位明显升高） 在最后几帧持续保持。 [且]
> 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_orange_onset 必须先于 pictured_glass_rise。 |
| `c2` | `changes` | 方向变化：转移 e1 中的效果 pictured_bottle_fall 必须按声明方向发生。 |
| `c3` | `couples` | 耦合约束：pictured_bottle_fall, pictured_orange_stream, pictured_glass_rise 必须共同变化；关系为“左侧橙色液位经亮色液流下降，同时右侧液位上升”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 source_level=visibly_lower, sink_level=visibly_higher 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 sink_level=empty, sink_level=visibly_higher 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "pictured_glass_rise",
    "before": "pictured_orange_onset",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "effect": "pictured_bottle_fall",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "transition": "e1",
    "type": "changes"
  },
  {
    "effects": [
      "pictured_bottle_fall",
      "pictured_orange_stream",
      "pictured_glass_rise"
    ],
    "id": "c3",
    "relation": "left orange level falls through bright stream while right level rises",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "source_level=visibly_lower",
      "sink_level=visibly_higher"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "sink_level=empty",
      "sink_level=visibly_higher"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=2个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"bottle":"derived_motion_envelope","glass":"unresolved_static_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P15</strong> — continuous；setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5</summary>

### P15 概要

- 时间路由类型：`continuous`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`无`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains cup: one declared collective group; sand stream: one
> declared collective group; pile region: exactly 1; hand: exactly 1. The reference first frame fixes
> these identities and initial layout. [INITIAL STATE] source_amount=filled_to_rim,
> stream_state=absent, pile_size=bare_wood. [PHYSICAL EVENT] The left hand holds the tilted clear cup
> of pale dry sand while sand pours onto the bare wooden table and forms a cone. [PERSISTENT
> PROTECTION] same transparent cup and pale dry sand; same visible left hand and its hold on the cup;
> wood tabletop, gray background, lighting, and static camera remain unchanged; sand stays within the
> cup-stream-pile corridor. [CLOSED WORLD] No additional external source, extra copy, unlisted object
> enters or appears; unoccupied scene regions remain empty. [CONSERVATION] Source loss and result gain
> stay coupled: tracked logical entity slots remain consistent; clear-cup contents decrease through
> pale stream while table cone grows. No result appears from an undeclared external source. [TERMINAL
> STATE] The same logical entity slots persist without duplicates; source_amount=nearly_empty,
> stream_state=continuous_pale_grains, pile_size=grown_cone remains visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 cup：一个已声明的集合组；sand_stream：一个已声明的集合组；pile_region：恰好 1 个；hand：恰好 1
> 个。参考首帧固定这些身份和初始布局。[初始状态] pile_size=bare_wood、source_amount=filled_to_rim、stream_state=absent。[物理事件]
> 左手握着倾斜的透明杯，杯中浅色干沙倾倒到裸露木桌上并形成锥形沙堆。[持续保护]
> 保持同一透明杯和浅色干沙；保持同一只可见左手及其对杯子的握持；木桌面、灰色背景、照明和静止相机保持不变；沙子保持在杯子—沙流—沙堆通道内。[封闭世界]
> 不允许新增外部来源、额外副本、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；透明杯中内容物经浅色沙流减少，同时桌面锥形沙堆增大。不允许任何结果由未声明的外部来源产生。[终止状态] 相同的逻辑实体槽位持续存在且没有重复副
> 本；pile_size=grown_cone、source_amount=nearly_empty、stream_state=continuous_pale_grains 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`170`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: cup: one declared collective group, state
> source_amount=filled_to_rim, required; sand stream: exactly 0, state stream_state=absent, forbidden;
> pile region: exactly 1, state pile_size=bare_wood, required; hand: exactly 1, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION]
> first pale grains leave the tilted clear cup happens BEFORE pale pile rises and widens on table.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 杯子（cup）: 1个已声明集合组，状态 source_amount=filled_to_rim，必需; 沙流（sand stream）: 恰好0个，状态
> stream_state=absent，禁止出现; 沙堆区域（pile region）: 恰好1个，状态 pile_size=bare_wood，必需; 手（hand）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 第一批浅色沙粒离开倾斜的透明杯先于浅色沙堆在桌面上升高并变宽发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`173`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: cup: one declared collective group, state
> source_amount=filled_to_rim, required; sand stream: exactly 0, state stream_state=absent, forbidden;
> pile region: exactly 1, state pile_size=bare_wood, required; hand: exactly 1, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> pale pile rises and widens on table is already present BEFORE first pale grains leave the tilted
> clear cup.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 杯子（cup）: 1个已声明集合组，状态 source_amount=filled_to_rim，必需; 沙流（sand stream）: 恰好0个，状态
> stream_state=absent，禁止出现; 沙堆区域（pile region）: 恰好1个，状态 pile_size=bare_wood，必需; 手（hand）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 在第一批浅色沙粒离开倾斜的透明杯发生之前，浅色沙堆在桌面上升高并变宽已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`172`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: cup: one declared collective group, state
> source_amount=filled_to_rim, transitioning; sand stream: one declared collective group, state
> stream_state=absent, transitioning; pile region: exactly 1, state pile_size=bare_wood,
> transitioning; hand: exactly 1, required. Counts and identities stay fixed; no duplicate state copy
> is created. No additional external source, extra copy, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VALID RELATION] first pale grains leave the tilted clear cup
> happens BEFORE pale pile rises and widens on table.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 杯子（cup）: 1个已声明集合组，状态 source_amount=filled_to_rim，正在过渡; 沙流（sand stream）: 1个已声明集合组，状态
> stream_state=absent，正在过渡; 沙堆区域（pile region）: 恰好1个，状态 pile_size=bare_wood，正在过渡; 手（hand）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 第一批浅色沙粒离开倾斜的透明杯先于浅色沙堆在桌面上升高并变宽发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`173`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: cup: one declared collective group, state
> source_amount=filled_to_rim, transitioning; sand stream: one declared collective group, state
> stream_state=absent, transitioning; pile region: exactly 1, state pile_size=bare_wood,
> transitioning; hand: exactly 1, required. Counts and identities stay fixed; no duplicate state copy
> is created. No additional external source, extra copy, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VIOLATED RELATION] pale pile rises and widens on table happens
> BEFORE first pale grains leave the tilted clear cup.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 杯子（cup）: 1个已声明集合组，状态 source_amount=filled_to_rim，正在过渡; 沙流（sand stream）: 1个已声明集合组，状态
> stream_state=absent，正在过渡; 沙堆区域（pile region）: 恰好1个，状态 pile_size=bare_wood，正在过渡; 手（hand）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 浅色沙堆在桌面上升高并变宽先于第一批浅色沙粒离开倾斜的透明杯发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`168`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: cup: one declared collective group, state
> source_amount=nearly_empty, required; sand stream: one declared collective group, state
> stream_state=continuous_pale_grains, required; pile region: exactly 1, state pile_size=grown_cone,
> required; hand: exactly 1, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VALID RELATION] The declared change visible pale sand inside cup
> decreases occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 杯子（cup）: 1个已声明集合组，状态 source_amount=nearly_empty，必需; 沙流（sand stream）: 1个已声明集合组，状态
> stream_state=continuous_pale_grains，必需; 沙堆区域（pile region）: 恰好1个，状态 pile_size=grown_cone，必需; 手（hand）:
> 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 已声明的变化“杯内可见浅色沙量减少”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`169`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3`

> [STAGE ENTITY INVENTORY] Present now: cup: one declared collective group, state
> source_amount=nearly_empty, required; sand stream: one declared collective group, state
> stream_state=continuous_pale_grains, required; pile region: exactly 1, state pile_size=grown_cone,
> required; hand: exactly 1, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VIOLATED RELATION] Only one coupled change occurs while the other
> remains unchanged: .

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 杯子（cup）: 1个已声明集合组，状态 source_amount=nearly_empty，必需; 沙流（sand stream）: 1个已声明集合组，状态
> stream_state=continuous_pale_grains，必需; 沙堆区域（pile region）: 恰好1个，状态 pile_size=grown_cone，必需; 手（hand）:
> 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 只有一个耦合变化发生，而另一个保持不变：（源编译文本此处为空）。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`171`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: cup: one declared collective group, state
> source_amount=nearly_empty, transitioning; sand stream: one declared collective group, state
> stream_state=continuous_pale_grains, transitioning; pile region: exactly 1, state
> pile_size=grown_cone, transitioning; hand: exactly 1, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VALID RELATION] The declared change visible
> pale sand inside cup decreases occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 杯子（cup）: 1个已声明集合组，状态 source_amount=nearly_empty，正在过渡; 沙流（sand stream）: 1个已声明集合组，状态
> stream_state=continuous_pale_grains，正在过渡; 沙堆区域（pile region）: 恰好1个，状态 pile_size=grown_cone，正在过渡;
> 手（hand）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 已声明的变化“杯内可见浅色沙量减少”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`170`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3, c4`

> [STAGE ENTITY INVENTORY] Present now: cup: one declared collective group, state
> source_amount=nearly_empty, transitioning; sand stream: one declared collective group, state
> stream_state=continuous_pale_grains, transitioning; pile region: exactly 1, state
> pile_size=grown_cone, transitioning; hand: exactly 1, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] The declared cause occurs
> BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 杯子（cup）: 1个已声明集合组，状态 source_amount=nearly_empty，正在过渡; 沙流（sand stream）: 1个已声明集合组，状态
> stream_state=continuous_pale_grains，正在过渡; 沙堆区域（pile region）: 恰好1个，状态 pile_size=grown_cone，正在过渡;
> 手（hand）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`220`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: cup: one declared collective group, state
> source_amount=nearly_empty, required; sand stream: exactly 0, state
> stream_state=continuous_pale_grains, forbidden; pile region: exactly 1, state pile_size=grown_cone,
> required; hand: exactly 1, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VALID RELATION] The terminal facts source_amount=nearly_empty,
> pile_size=grown_cone REMAIN through the final frames. [AND] The same logical slot has only its valid
> state; mutually exclusive facts pile_size=bare_wood, pile_size=grown_cone do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 杯子（cup）: 1个已声明集合组，状态 source_amount=nearly_empty，必需; 沙流（sand stream）: 恰好0个，状态
> stream_state=continuous_pale_grains，禁止出现; 沙堆区域（pile region）: 恰好1个，状态 pile_size=grown_cone，必需;
> 手（hand）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实 source_amount=nearly_empty（源杯近空）、pile_size=grown_cone（增大的锥形沙堆）
> 在最后几帧持续保持。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 pile_size=bare_wood（裸木桌面）与 pile_size=grown_cone（增大的锥形沙堆） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`206`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: cup: one declared collective group, state
> source_amount=nearly_empty, required; sand stream: exactly 0, state
> stream_state=continuous_pale_grains, forbidden; pile region: exactly 1, state pile_size=grown_cone,
> required; hand: exactly 1, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VIOLATED RELATION] The valid terminal state returns to its mutually
> exclusive earlier state. [AND] The same logical slot has only its valid state; mutually exclusive
> facts pile_size=bare_wood, pile_size=grown_cone do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 杯子（cup）: 1个已声明集合组，状态 source_amount=nearly_empty，必需; 沙流（sand stream）: 恰好0个，状态
> stream_state=continuous_pale_grains，禁止出现; 沙堆区域（pile region）: 恰好1个，状态 pile_size=grown_cone，必需;
> 手（hand）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 pile_size=bare_wood（裸木桌面）与
> pile_size=grown_cone（增大的锥形沙堆） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`204`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: cup: one declared collective group, state
> source_amount=nearly_empty, required; sand stream: exactly 0, state
> stream_state=continuous_pale_grains, forbidden; pile region: exactly 1, state pile_size=grown_cone,
> required; hand: exactly 1, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VIOLATED RELATION] The terminal facts source_amount=nearly_empty,
> pile_size=grown_cone REMAIN through the final frames. [AND] Copies of the same logical slot appear
> in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 杯子（cup）: 1个已声明集合组，状态 source_amount=nearly_empty，必需; 沙流（sand stream）: 恰好0个，状态
> stream_state=continuous_pale_grains，禁止出现; 沙堆区域（pile region）: 恰好1个，状态 pile_size=grown_cone，必需;
> 手（hand）: 恰好1个，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实 source_amount=nearly_empty（源杯近空）、pile_size=grown_cone（增大的锥形沙堆）
> 在最后几帧持续保持。 [且] 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_first_grains 必须先于 pictured_cone_grow。 |
| `c2` | `changes` | 方向变化：转移 e1 中的效果 pictured_cup_empty 必须按声明方向发生。 |
| `c3` | `couples` | 耦合约束：pictured_cup_empty, pictured_grain_stream, pictured_cone_grow 必须共同变化；关系为“透明杯中内容物经浅色沙流减少，同时桌面锥形沙堆增大”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 source_amount=nearly_empty, pile_size=grown_cone 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 pile_size=bare_wood, pile_size=grown_cone 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "pictured_cone_grow",
    "before": "pictured_first_grains",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "effect": "pictured_cup_empty",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "transition": "e1",
    "type": "changes"
  },
  {
    "effects": [
      "pictured_cup_empty",
      "pictured_grain_stream",
      "pictured_cone_grow"
    ],
    "id": "c3",
    "relation": "clear-cup contents decrease through pale stream while table cone grows",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "source_amount=nearly_empty",
      "pile_size=grown_cone"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "pile_size=bare_wood",
      "pile_size=grown_cone"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=3个; grounding.protected_boxes=1个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"cup":"unresolved_static_envelope","hand":"unresolved_static_envelope","pile_region":"unresolved_static_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P16</strong> — continuous；setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5</summary>

### P16 概要

- 时间路由类型：`continuous`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`无`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains dye: exactly 1; water: one declared collective group.
> The reference first frame fixes these identities and initial layout. [INITIAL STATE]
> dye_position=above_center_surface, colored_area=none, water_state=fully_clear. [PHYSICAL EVENT] The
> single compact dark-blue drop above the centered water surface enters the clear rectangular glass
> vessel and spreads outward. [PERSISTENT PROTECTION] same dark-blue dye color; same transparent
> rectangular glass vessel and horizontal waterline; pale background, tabletop reflection, lighting,
> centered framing, and static camera remain unchanged; blue color stays inside the vessel. [CLOSED
> WORLD] No additional external source, extra copy, unlisted object enters or appears; unoccupied
> scene regions remain empty. [CONSERVATION] Source loss and result gain stay coupled: tracked logical
> entity slots remain consistent; center entry precedes outward blue growth within clear vessel. No
> result appears from an undeclared external source. [TERMINAL STATE] The same logical entity slots
> persist without duplicates; dye_position=inside_water, colored_area=expanded,
> water_state=locally_blue remains visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 dye：恰好 1 个；water：一个已声明的集合组。参考首帧固定这些身份和初始布局。[初始状态]
> colored_area=none、dye_position=above_center_surface、water_state=fully_clear。[物理事件]
> 水面中央上方单个紧实的深蓝色液滴进入透明长方形玻璃容器，并向外扩散。[持续保护]
> 保持同一深蓝色染料颜色；保持同一透明长方形玻璃容器和水平水线；浅色背景、桌面反射、照明、居中构图和静止相机保持不变；蓝色保持在容器内部。[封闭世界]
> 不允许新增外部来源、额外副本、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；中央进入动作先于透明容器内蓝色向外扩张。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；colored_area=expanded、dye_position=inside_water、water_state=locally_blue
> 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`153`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: dye: exactly 1, state dye_position=above_center_surface,
> required; water: one declared collective group, state colored_area=none, water_state=fully_clear,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] the dark-blue drop crosses the centered horizontal waterline happens
> BEFORE blue boundary widens inside vessel.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 染料（dye）: 恰好1个，状态 dye_position=above_center_surface，必需; 水（water）: 1个已声明集合组，状态
> colored_area=none, water_state=fully_clear，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 深蓝色液滴穿过居中的水平水线先于蓝色边界在容器内变宽发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`156`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: dye: exactly 1, state dye_position=above_center_surface,
> required; water: one declared collective group, state colored_area=none, water_state=fully_clear,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] blue boundary widens inside vessel is already present BEFORE the
> dark-blue drop crosses the centered horizontal waterline.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 染料（dye）: 恰好1个，状态 dye_position=above_center_surface，必需; 水（water）: 1个已声明集合组，状态
> colored_area=none, water_state=fully_clear，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 在深蓝色液滴穿过居中的水平水线发生之前，蓝色边界在容器内变宽已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`155`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: dye: exactly 1, state dye_position=above_center_surface,
> transitioning; water: one declared collective group, state colored_area=none,
> water_state=fully_clear, transitioning. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VALID RELATION] the dark-blue drop crosses the centered horizontal
> waterline happens BEFORE blue boundary widens inside vessel.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 染料（dye）: 恰好1个，状态 dye_position=above_center_surface，正在过渡; 水（water）: 1个已声明集合组，状态
> colored_area=none, water_state=fully_clear，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 深蓝色液滴穿过居中的水平水线先于蓝色边界在容器内变宽发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`156`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: dye: exactly 1, state dye_position=above_center_surface,
> transitioning; water: one declared collective group, state colored_area=none,
> water_state=fully_clear, transitioning. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VIOLATED RELATION] blue boundary widens inside vessel happens BEFORE
> the dark-blue drop crosses the centered horizontal waterline.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 染料（dye）: 恰好1个，状态 dye_position=above_center_surface，正在过渡; 水（water）: 1个已声明集合组，状态
> colored_area=none, water_state=fully_clear，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 蓝色边界在容器内变宽先于深蓝色液滴穿过居中的水平水线发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`144`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: dye: exactly 1, state dye_position=inside_water, required;
> water: one declared collective group, state colored_area=expanded, water_state=locally_blue,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] The declared change blue boundary widens inside vessel occurs in its
> stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 染料（dye）: 恰好1个，状态 dye_position=inside_water，必需; 水（water）: 1个已声明集合组，状态
> colored_area=expanded, water_state=locally_blue，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 已声明的变化“蓝色边界在容器内变宽”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`145`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3`

> [STAGE ENTITY INVENTORY] Present now: dye: exactly 1, state dye_position=inside_water, required;
> water: one declared collective group, state colored_area=expanded, water_state=locally_blue,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] Only one coupled change occurs while the other remains unchanged: .

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 染料（dye）: 恰好1个，状态 dye_position=inside_water，必需; 水（water）: 1个已声明集合组，状态
> colored_area=expanded, water_state=locally_blue，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 只有一个耦合变化发生，而另一个保持不变：（源编译文本此处为空）。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`146`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: dye: exactly 1, state dye_position=inside_water,
> transitioning; water: one declared collective group, state colored_area=expanded,
> water_state=locally_blue, transitioning. Counts and identities stay fixed; no duplicate state copy
> is created. No additional external source, extra copy, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VALID RELATION] The declared change blue boundary widens
> inside vessel occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 染料（dye）: 恰好1个，状态 dye_position=inside_water，正在过渡; 水（water）: 1个已声明集合组，状态
> colored_area=expanded, water_state=locally_blue，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 已声明的变化“蓝色边界在容器内变宽”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`145`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3, c4`

> [STAGE ENTITY INVENTORY] Present now: dye: exactly 1, state dye_position=inside_water,
> transitioning; water: one declared collective group, state colored_area=expanded,
> water_state=locally_blue, transitioning. Counts and identities stay fixed; no duplicate state copy
> is created. No additional external source, extra copy, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VIOLATED RELATION] The declared cause occurs BUT the required
> target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 染料（dye）: 恰好1个，状态 dye_position=inside_water，正在过渡; 水（water）: 1个已声明集合组，状态
> colored_area=expanded, water_state=locally_blue，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`203`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: dye: exactly 1, state dye_position=inside_water, required;
> water: one declared collective group, state colored_area=expanded, water_state=locally_blue,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] The terminal facts dye_position=inside_water, colored_area=expanded,
> water_state=locally_blue REMAIN through the final frames. [AND] The same logical slot has only its
> valid state; mutually exclusive facts water_state=fully_clear, water_state=locally_blue do not
> coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 染料（dye）: 恰好1个，状态 dye_position=inside_water，必需; 水（water）: 1个已声明集合组，状态
> colored_area=expanded, water_state=locally_blue，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实
> dye_position=inside_water（染料位于水中）、colored_area=expanded（着色区域扩大）、water_state=locally_blue（水局部变蓝）
> 在最后几帧持续保持。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 water_state=fully_clear（完全透明）与 water_state=locally_blue（局部变蓝）
> 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`181`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: dye: exactly 1, state dye_position=inside_water, required;
> water: one declared collective group, state colored_area=expanded, water_state=locally_blue,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] The valid terminal state returns to its mutually exclusive earlier
> state. [AND] The same logical slot has only its valid state; mutually exclusive facts
> water_state=fully_clear, water_state=locally_blue do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 染料（dye）: 恰好1个，状态 dye_position=inside_water，必需; 水（water）: 1个已声明集合组，状态
> colored_area=expanded, water_state=locally_blue，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实
> water_state=fully_clear（完全透明）与 water_state=locally_blue（局部变蓝） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`187`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: dye: exactly 1, state dye_position=inside_water, required;
> water: one declared collective group, state colored_area=expanded, water_state=locally_blue,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] The terminal facts dye_position=inside_water, colored_area=expanded,
> water_state=locally_blue REMAIN through the final frames. [AND] Copies of the same logical slot
> appear in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 染料（dye）: 恰好1个，状态 dye_position=inside_water，必需; 水（water）: 1个已声明集合组，状态
> colored_area=expanded, water_state=locally_blue，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实
> dye_position=inside_water（染料位于水中）、colored_area=expanded（着色区域扩大）、water_state=locally_blue（水局部变蓝）
> 在最后几帧持续保持。 [且] 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_surface_entry 必须先于 pictured_blue_expand。 |
| `c2` | `changes` | 方向变化：转移 e1 中的效果 pictured_blue_expand 必须按声明方向发生。 |
| `c3` | `couples` | 耦合约束：pictured_entry, pictured_blue_expand, pictured_water_color 必须共同变化；关系为“中央进入动作先于透明容器内蓝色向外扩张”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 dye_position=inside_water, colored_area=expanded, water_state=locally_blue 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 water_state=fully_clear, water_state=locally_blue 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "pictured_blue_expand",
    "before": "pictured_surface_entry",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "effect": "pictured_blue_expand",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "transition": "e1",
    "type": "changes"
  },
  {
    "effects": [
      "pictured_entry",
      "pictured_blue_expand",
      "pictured_water_color"
    ],
    "id": "c3",
    "relation": "center entry precedes outward blue growth within clear vessel",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "dye_position=inside_water",
      "colored_area=expanded",
      "water_state=locally_blue"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "water_state=fully_clear",
      "water_state=locally_blue"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=2个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"dye":"derived_motion_envelope","water":"derived_motion_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P17</strong> — multi_transition；setup→c1; onset→c1; evolution→c5; completion→c6; terminal→c7,c8</summary>

### P17 概要

- 时间路由类型：`multi_transition`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c5; completion→c6; terminal→c7,c8`
- 同时进入 Global 守恒文本的 couples 约束：`无`
- 仅校验、未进入生成张量的约束：`c2,c3,c4`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains hand: exactly 1; sponge: exactly 1; surface: one
> declared collective group. The reference first frame fixes these identities and initial layout.
> [INITIAL STATE] hand_state=hovering_above, thickness=full_rectangular. [PHYSICAL EVENT] The open
> left hand presses the rectangular yellow sponge on the white marble surface, lifts, and the sponge
> recovers. [PERSISTENT PROTECTION] same left hand skin appearance; same rectangular yellow porous dry
> sponge; white marble surface, pale background, lighting, side camera, and framing remain unchanged.
> [CLOSED WORLD] No additional external source, extra copy, unlisted object enters or appears;
> unoccupied scene regions remain empty. [CONSERVATION] Source loss and result gain stay coupled:
> tracked logical entity slots remain consistent. No result appears from an undeclared external
> source. [TERMINAL STATE] The same logical entity slots persist without duplicates;
> hand_state=lifted_above, thickness=recovered_rectangular remains visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 hand：恰好 1 个；sponge：恰好 1 个；surface：一个已声明的集合组。参考首帧固定这些身份和初始布局。[初始状态]
> hand_state=hovering_above、thickness=full_rectangular。[物理事件]
> 张开的左手按压白色大理石表面上的长方形黄色海绵，随后抬起，海绵恢复原状。[持续保护]
> 保持同一左手皮肤外观；保持同一长方形黄色多孔干海绵；白色大理石表面、浅色背景、照明、侧视相机和构图保持不变。[封闭世界]
> 不允许新增外部来源、额外副本、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒] 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；hand_state=lifted_above、thickness=recovered_rectangular 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`146`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: hand: exactly 1, state hand_state=hovering_above, required;
> sponge: exactly 1, state thickness=full_rectangular, required; surface: one declared collective
> group, required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] the open palm reaches the yellow sponge top happens BEFORE yellow vertical
> extent decreases.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 手（hand）: 恰好1个，状态 hand_state=hovering_above，必需; 海绵（sponge）: 恰好1个，状态
> thickness=full_rectangular，必需; 表面（surface）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 张开的手掌触及黄色海绵顶部先于黄色物体的竖直高度减小发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`149`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c3`

> [STAGE ENTITY INVENTORY] Present now: hand: exactly 1, state hand_state=hovering_above, required;
> sponge: exactly 1, state thickness=full_rectangular, required; surface: one declared collective
> group, required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] yellow vertical extent decreases is already present BEFORE the open
> palm reaches the yellow sponge top.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 手（hand）: 恰好1个，状态 hand_state=hovering_above，必需; 海绵（sponge）: 恰好1个，状态
> thickness=full_rectangular，必需; 表面（surface）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 在张开的手掌触及黄色海绵顶部发生之前，黄色物体的竖直高度减小已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c2, c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`148`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: hand: exactly 1, state hand_state=hovering_above,
> transitioning; sponge: exactly 1, state thickness=full_rectangular, transitioning; surface: one
> declared collective group, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VALID RELATION] the open palm reaches the yellow sponge top happens
> BEFORE yellow vertical extent decreases.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 手（hand）: 恰好1个，状态 hand_state=hovering_above，正在过渡; 海绵（sponge）: 恰好1个，状态
> thickness=full_rectangular，正在过渡; 表面（surface）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 张开的手掌触及黄色海绵顶部先于黄色物体的竖直高度减小发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`149`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c2, c3, c4`

> [STAGE ENTITY INVENTORY] Present now: hand: exactly 1, state hand_state=hovering_above,
> transitioning; sponge: exactly 1, state thickness=full_rectangular, transitioning; surface: one
> declared collective group, required. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VIOLATED RELATION] yellow vertical extent decreases happens BEFORE the
> open palm reaches the yellow sponge top.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 手（hand）: 恰好1个，状态 hand_state=hovering_above，正在过渡; 海绵（sponge）: 恰好1个，状态
> thickness=full_rectangular，正在过渡; 表面（surface）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 黄色物体的竖直高度减小先于张开的手掌触及黄色海绵顶部发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c5`
- violation 中声明的候选约束：`c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`134`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: hand: exactly 1, state hand_state=pressing, required; sponge:
> exactly 1, state thickness=flat, required; surface: one declared collective group, required. Counts
> and identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The
> declared change yellow vertical extent decreases occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 手（hand）: 恰好1个，状态 hand_state=pressing，必需; 海绵（sponge）: 恰好1个，状态 thickness=flat，必需;
> 表面（surface）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 已声明的变化“黄色物体的竖直高度减小”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`135`
- 权重：`1.0`；violation：`effect_missing`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: hand: exactly 1, state hand_state=pressing, required; sponge:
> exactly 1, state thickness=flat, required; surface: one declared collective group, required. Counts
> and identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> The declared cause occurs BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 手（hand）: 恰好1个，状态 hand_state=pressing，必需; 海绵（sponge）: 恰好1个，状态 thickness=flat，必需;
> 表面（surface）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c6`
- violation 中声明的候选约束：`c6, c7`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`145`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: hand: exactly 1, state hand_state=lifted_above, transitioning;
> sponge: exactly 1, state thickness=recovered_rectangular, transitioning; surface: one declared
> collective group, required. Counts and identities stay fixed; no duplicate state copy is created. No
> additional external source, extra copy, unlisted object exists. All globally protected identity,
> shape, material, scene, and camera attributes stay unchanged. Declared source loss and result gain
> remain coupled. [VALID RELATION] The declared change yellow vertical extent rises gradually occurs
> in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 手（hand）: 恰好1个，状态 hand_state=lifted_above，正在过渡; 海绵（sponge）: 恰好1个，状态
> thickness=recovered_rectangular，正在过渡; 表面（surface）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 已声明的变化“黄色物体的竖直高度逐渐回升”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`145`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c6`；violation 声明的候选约束：`c6, c7`

> [STAGE ENTITY INVENTORY] Present now: hand: exactly 1, state hand_state=lifted_above, transitioning;
> sponge: exactly 1, state thickness=recovered_rectangular, transitioning; surface: one declared
> collective group, required. Counts and identities stay fixed; no duplicate state copy is created. No
> additional external source, extra copy, unlisted object exists. All globally protected identity,
> shape, material, scene, and camera attributes stay unchanged. Declared source loss and result gain
> remain coupled. [VIOLATED RELATION] The declared cause occurs BUT the required target change does
> not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 手（hand）: 恰好1个，状态 hand_state=lifted_above，正在过渡; 海绵（sponge）: 恰好1个，状态
> thickness=recovered_rectangular，正在过渡; 表面（surface）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c7, c8`
- violation 中声明的候选约束：`c7, c8`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`191`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: hand: exactly 1, state hand_state=lifted_above, required;
> sponge: exactly 1, state thickness=recovered_rectangular, required; surface: one declared collective
> group, required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] The terminal facts hand_state=lifted_above,
> thickness=recovered_rectangular REMAIN through the final frames. [AND] The same logical slot has
> only its valid state; mutually exclusive facts thickness=flat, thickness=recovered_rectangular do
> not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 手（hand）: 恰好1个，状态 hand_state=lifted_above，必需; 海绵（sponge）: 恰好1个，状态
> thickness=recovered_rectangular，必需; 表面（surface）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实
> hand_state=lifted_above（手已抬起）、thickness=recovered_rectangular（恢复长方形厚度） 在最后几帧持续保持。 [且]
> 同一逻辑槽位只具有其有效状态；互斥事实 thickness=flat（压扁）与 thickness=recovered_rectangular（恢复长方形厚度） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`177`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c7`；violation 声明的候选约束：`c7`

> [STAGE ENTITY INVENTORY] Present now: hand: exactly 1, state hand_state=lifted_above, required;
> sponge: exactly 1, state thickness=recovered_rectangular, required; surface: one declared collective
> group, required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] The valid terminal state returns to its mutually exclusive earlier
> state. [AND] The same logical slot has only its valid state; mutually exclusive facts
> thickness=flat, thickness=recovered_rectangular do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 手（hand）: 恰好1个，状态 hand_state=lifted_above，必需; 海绵（sponge）: 恰好1个，状态
> thickness=recovered_rectangular，必需; 表面（surface）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 有效的终止状态恢复成与其互斥的较早状态。
> [且] 同一逻辑槽位只具有其有效状态；互斥事实 thickness=flat（压扁）与 thickness=recovered_rectangular（恢复长方形厚度） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`179`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c8`；violation 声明的候选约束：`c8`

> [STAGE ENTITY INVENTORY] Present now: hand: exactly 1, state hand_state=lifted_above, required;
> sponge: exactly 1, state thickness=recovered_rectangular, required; surface: one declared collective
> group, required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] The terminal facts hand_state=lifted_above,
> thickness=recovered_rectangular REMAIN through the final frames. [AND] Copies of the same logical
> slot appear in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 手（hand）: 恰好1个，状态 hand_state=lifted_above，必需; 海绵（sponge）: 恰好1个，状态
> thickness=recovered_rectangular，必需; 表面（surface）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实
> hand_state=lifted_above（手已抬起）、thickness=recovered_rectangular（恢复长方形厚度） 在最后几帧持续保持。 [且]
> 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_hand_contact 必须先于 pictured_flatten。 |
| `c2` | `requires` | 因果依赖：pictured_flatten 只有在 pictured_hand_contact 之后才能发生。 |
| `c3` | `precedes` | 前置顺序：pictured_release 必须先于 pictured_recover。 |
| `c4` | `requires` | 因果依赖：pictured_recover 只有在 pictured_release 之后才能发生。 |
| `c5` | `changes` | 方向变化：转移 e1 中的效果 pictured_flatten 必须按声明方向发生。 |
| `c6` | `changes` | 方向变化：转移 e2 中的效果 pictured_recover 必须按声明方向发生。 |
| `c7` | `persists` | 持续约束：状态 z2 的事实 hand_state=lifted_above, thickness=recovered_rectangular 必须维持到最终帧。 |
| `c8` | `excludes` | 互斥约束：事实 thickness=flat, thickness=recovered_rectangular 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "pictured_flatten",
    "before": "pictured_hand_contact",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "cause": "pictured_hand_contact",
    "effect": "pictured_flatten",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "type": "requires"
  },
  {
    "after": "pictured_recover",
    "before": "pictured_release",
    "id": "c3",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "cause": "pictured_release",
    "effect": "pictured_recover",
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "type": "requires"
  },
  {
    "effect": "pictured_flatten",
    "id": "c5",
    "source_refs": [
      "origin"
    ],
    "transition": "e1",
    "type": "changes"
  },
  {
    "effect": "pictured_recover",
    "id": "c6",
    "source_refs": [
      "origin"
    ],
    "transition": "e2",
    "type": "changes"
  },
  {
    "facts": [
      "hand_state=lifted_above",
      "thickness=recovered_rectangular"
    ],
    "id": "c7",
    "source_refs": [
      "origin"
    ],
    "state": "z2",
    "type": "persists"
  },
  {
    "facts": [
      "thickness=flat",
      "thickness=recovered_rectangular"
    ],
    "id": "c8",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=3个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"hand":"derived_motion_envelope","sponge":"unresolved_static_envelope","surface":"unresolved_static_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P18</strong> — triggered；setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5</summary>

### P18 概要

- 时间路由类型：`triggered`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c3; completion→c3; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`c2`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains ball: exactly 1; floor: one declared collective group.
> The reference first frame fixes these identities and initial layout. [INITIAL STATE]
> height_state=upper_center, motion_state=falling, peak_envelope=initial_gap. [PHYSICAL EVENT] The
> single red rubber ball drops through the centered vertical corridor onto the smooth gray floor,
> bounces lower, and rests. [PERSISTENT PROTECTION] same single red rubber sphere; smooth gray floor,
> pale wall, white baseboard, lighting, centered corridor, and static side camera remain unchanged.
> [CLOSED WORLD] No additional external source, extra copy, unlisted object enters or appears;
> unoccupied scene regions remain empty. [CONSERVATION] Source loss and result gain stay coupled:
> tracked logical entity slots remain consistent; gray-floor contacts yield successively lower
> red-ball peaks until rest. No result appears from an undeclared external source. [TERMINAL STATE]
> The same logical entity slots persist without duplicates; height_state=resting_at_floor,
> motion_state=resting, peak_envelope=decreasing_to_zero remains visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 ball：恰好 1 个；floor：一个已声明的集合组。参考首帧固定这些身份和初始布局。[初始状态]
> height_state=upper_center、motion_state=falling、peak_envelope=initial_gap。[物理事件]
> 单个红色橡胶球沿中央竖直通道落到光滑灰色地面，反弹高度逐渐降低，最后静止。[持续保护] 保持同一个红色橡胶球；光滑灰色地面、浅色墙、白色踢脚线、照明、中央通道和静止侧视相机保持不变。[封闭世界]
> 不允许新增外部来源、额外副本、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；每次接触灰色地面后红球峰值依次降低，直至静止。不允许任何结果由未声明的外部来源产生。[终止状态] 相同的逻辑实体槽位持续存在且没有重复副本；h
> eight_state=resting_at_floor、motion_state=resting、peak_envelope=decreasing_to_zero 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`148`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state height_state=upper_center,
> motion_state=falling, peak_envelope=initial_gap, required; floor: one declared collective group,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] the red ball first reaches the smooth gray floor happens BEFORE red-ball
> rebounds decay to stillness.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 球（ball）: 恰好1个，状态 height_state=upper_center, motion_state=falling,
> peak_envelope=initial_gap，必需; 地面（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 红球首次抵达光滑灰色地面先于红球反弹逐渐衰减至静止发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`151`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state height_state=upper_center,
> motion_state=falling, peak_envelope=initial_gap, required; floor: one declared collective group,
> required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] red-ball rebounds decay to stillness is already present BEFORE the red
> ball first reaches the smooth gray floor.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 球（ball）: 恰好1个，状态 height_state=upper_center, motion_state=falling,
> peak_envelope=initial_gap，必需; 地面（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 在红球首次抵达光滑灰色地面发生之前，红球反弹逐渐衰减至静止已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1, c2`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`150`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state height_state=upper_center,
> motion_state=falling, peak_envelope=initial_gap, transitioning; floor: one declared collective
> group, transitioning. Counts and identities stay fixed; no duplicate state copy is created. No
> additional external source, extra copy, unlisted object exists. All globally protected identity,
> shape, material, scene, and camera attributes stay unchanged. Declared source loss and result gain
> remain coupled. [VALID RELATION] the red ball first reaches the smooth gray floor happens BEFORE
> red-ball rebounds decay to stillness.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 球（ball）: 恰好1个，状态 height_state=upper_center, motion_state=falling,
> peak_envelope=initial_gap，正在过渡; 地面（floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 红球首次抵达光滑灰色地面先于红球反弹逐渐衰减至静止发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`151`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1, c2`

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state height_state=upper_center,
> motion_state=falling, peak_envelope=initial_gap, transitioning; floor: one declared collective
> group, transitioning. Counts and identities stay fixed; no duplicate state copy is created. No
> additional external source, extra copy, unlisted object exists. All globally protected identity,
> shape, material, scene, and camera attributes stay unchanged. Declared source loss and result gain
> remain coupled. [VIOLATED RELATION] red-ball rebounds decay to stillness happens BEFORE the red ball
> first reaches the smooth gray floor.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 球（ball）: 恰好1个，状态 height_state=upper_center, motion_state=falling,
> peak_envelope=initial_gap，正在过渡; 地面（floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 红球反弹逐渐衰减至静止先于红球首次抵达光滑灰色地面发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`170`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state height_state=resting_at_floor,
> motion_state=resting, peak_envelope=decreasing_to_zero, required; floor: one declared collective
> group, required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] The coupled changes occur together: red-ball vertical path converges to
> floor WHILE red-ball rebounds decay to stillness WHILE each wall-referenced peak is lower.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 球（ball）: 恰好1个，状态 height_state=resting_at_floor, motion_state=resting,
> peak_envelope=decreasing_to_zero，必需; 地面（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 耦合变化同时发生：红球的竖直轨迹收敛到地面，同时反弹逐渐衰减至静止，且每个以墙面为参照的峰值都更低。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`176`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3`

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state height_state=resting_at_floor,
> motion_state=resting, peak_envelope=decreasing_to_zero, required; floor: one declared collective
> group, required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] Only one coupled change occurs while the other remains unchanged:
> red-ball vertical path converges to floor; red-ball rebounds decay to stillness; each
> wall-referenced peak is lower.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 球（ball）: 恰好1个，状态 height_state=resting_at_floor, motion_state=resting,
> peak_envelope=decreasing_to_zero，必需; 地面（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 只有一个耦合变化发生，另一个保持不变：红球的竖直轨迹收敛到地面；反弹逐渐衰减至静止；每个以墙面为参照的峰值都更低。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`172`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state height_state=resting_at_floor,
> motion_state=resting, peak_envelope=decreasing_to_zero, transitioning; floor: one declared
> collective group, transitioning. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VALID RELATION] The coupled changes occur together: red-ball vertical
> path converges to floor WHILE red-ball rebounds decay to stillness WHILE each wall-referenced peak
> is lower.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 球（ball）: 恰好1个，状态 height_state=resting_at_floor, motion_state=resting,
> peak_envelope=decreasing_to_zero，正在过渡; 地面（floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 耦合变化同时发生：红球的竖直轨迹收敛到地面，同时反弹逐渐衰减至静止，且每个以墙面为参照的峰值都更低。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`149`
- 权重：`1.0`；violation：`envelope_broken`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3, c4`

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state height_state=resting_at_floor,
> motion_state=resting, peak_envelope=decreasing_to_zero, transitioning; floor: one declared
> collective group, transitioning. Counts and identities stay fixed; no duplicate state copy is
> created. No additional external source, extra copy, unlisted object exists. All globally protected
> identity, shape, material, scene, and camera attributes stay unchanged. Declared source loss and
> result gain remain coupled. [VIOLATED RELATION] The same declared property changes in the opposite
> or out-of-range direction.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 球（ball）: 恰好1个，状态 height_state=resting_at_floor, motion_state=resting,
> peak_envelope=decreasing_to_zero，正在过渡; 地面（floor）: 1个已声明集合组，正在过渡. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 同一项已声明属性朝相反方向变化，或变化超出规定范围。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`206`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state height_state=resting_at_floor,
> motion_state=resting, peak_envelope=decreasing_to_zero, required; floor: one declared collective
> group, required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VALID RELATION] The terminal facts height_state=resting_at_floor, motion_state=resting,
> peak_envelope=decreasing_to_zero REMAIN through the final frames. [AND] The same logical slot has
> only its valid state; mutually exclusive facts motion_state=bouncing, motion_state=resting do not
> coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 球（ball）: 恰好1个，状态 height_state=resting_at_floor, motion_state=resting,
> peak_envelope=decreasing_to_zero，必需; 地面（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实 height_state=res
> ting_at_floor（静止于地面）、motion_state=resting（静止）、peak_envelope=decreasing_to_zero（峰值包络递减至零） 在最后几帧持续保持。
> [且] 同一逻辑槽位只具有其有效状态；互斥事实 motion_state=bouncing（弹跳）与 motion_state=resting（静止） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`180`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state height_state=resting_at_floor,
> motion_state=resting, peak_envelope=decreasing_to_zero, required; floor: one declared collective
> group, required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] The valid terminal state returns to its mutually exclusive earlier
> state. [AND] The same logical slot has only its valid state; mutually exclusive facts
> motion_state=bouncing, motion_state=resting do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 球（ball）: 恰好1个，状态 height_state=resting_at_floor, motion_state=resting,
> peak_envelope=decreasing_to_zero，必需; 地面（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 有效的终止状态恢复成与其互斥的较早状态。
> [且] 同一逻辑槽位只具有其有效状态；互斥事实 motion_state=bouncing（弹跳）与 motion_state=resting（静止） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`193`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: ball: exactly 1, state height_state=resting_at_floor,
> motion_state=resting, peak_envelope=decreasing_to_zero, required; floor: one declared collective
> group, required. Counts and identities stay fixed; no duplicate state copy is created. No additional
> external source, extra copy, unlisted object exists. All globally protected identity, shape,
> material, scene, and camera attributes stay unchanged. Declared source loss and result gain remain
> coupled. [VIOLATED RELATION] The terminal facts height_state=resting_at_floor, motion_state=resting,
> peak_envelope=decreasing_to_zero REMAIN through the final frames. [AND] Copies of the same logical
> slot appear in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 球（ball）: 恰好1个，状态 height_state=resting_at_floor, motion_state=resting,
> peak_envelope=decreasing_to_zero，必需; 地面（floor）: 1个已声明集合组，必需. 数量和身份保持固定；不创建重复的状态副本。
> 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实 height_state=res
> ting_at_floor（静止于地面）、motion_state=resting（静止）、peak_envelope=decreasing_to_zero（峰值包络递减至零） 在最后几帧持续保持。
> [且] 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_first_impact 必须先于 pictured_bounce_decay。 |
| `c2` | `requires` | 因果依赖：pictured_bounce_decay 只有在 pictured_first_impact 之后才能发生。 |
| `c3` | `couples` | 耦合约束：pictured_height_finish, pictured_bounce_decay, pictured_peaks_decay 必须共同变化；关系为“每次接触灰色地面后红球峰值依次降低，直至静止”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 height_state=resting_at_floor, motion_state=resting, peak_envelope=decreasing_to_zero 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 motion_state=bouncing, motion_state=resting 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "pictured_bounce_decay",
    "before": "pictured_first_impact",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "cause": "pictured_first_impact",
    "effect": "pictured_bounce_decay",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "type": "requires"
  },
  {
    "effects": [
      "pictured_height_finish",
      "pictured_bounce_decay",
      "pictured_peaks_decay"
    ],
    "id": "c3",
    "relation": "gray-floor contacts yield successively lower red-ball peaks until rest",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "height_state=resting_at_floor",
      "motion_state=resting",
      "peak_envelope=decreasing_to_zero"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "motion_state=bouncing",
      "motion_state=resting"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=2个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"ball":"unresolved_static_envelope","floor":"unresolved_static_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P19</strong> — continuous；setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5</summary>

### P19 概要

- 时间路由类型：`continuous`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c2; completion→c2; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`无`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains paper: exactly 1. The reference first frame fixes these
> identities and initial layout. [INITIAL STATE] tear_extent=small_top_notch,
> integrity=one_wide_sheet, separation=connected. [PHYSICAL EVENT] The wide white horizontal sheet
> tears downward and outward from its single small top-edge notch and separates into two pieces.
> [PERSISTENT PROTECTION] same plain white paper material and rectangular outer proportions; single
> centered top-edge notch remains the tear origin; gray background, lighting, frontal camera, and
> framing remain unchanged; no pulling hands or tools are added. [CLOSED WORLD] No additional external
> source, extra copy, unlisted object enters or appears; unoccupied scene regions remain empty.
> [CONSERVATION] Source loss and result gain stay coupled: tracked logical entity slots remain
> consistent; centered gray slit lengthens as white sides separate. No result appears from an
> undeclared external source. [TERMINAL STATE] The same logical entity slots persist without
> duplicates; tear_extent=full_vertical_tear, integrity=two_white_pieces, separation=apart remains
> visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 paper：恰好 1 个。参考首帧固定这些身份和初始布局。[初始状态]
> integrity=one_wide_sheet、separation=connected、tear_extent=small_top_notch。[物理事件]
> 宽大的白色水平纸张从顶部边缘的单个小缺口向下并向外撕裂，分成两片。[持续保护]
> 保持同一纯白纸张材质和长方形外部比例；顶部边缘中央的单个缺口保持为撕裂起点；灰色背景、照明、正视相机和构图保持不变；不添加拉纸的手或工具。[封闭世界]
> 不允许新增外部来源、额外副本、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；中央灰色裂缝延长，同时白色两侧分离。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；integrity=two_white_pieces、separation=apart、tear_extent=full_vertical_tear
> 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`139`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: paper: exactly 1, state tear_extent=small_top_notch,
> integrity=one_wide_sheet, separation=connected, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VALID RELATION] the centered top-edge notch
> first lengthens downward happens BEFORE white rectangle divides into two regions.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸张（paper）: 恰好1个，状态 tear_extent=small_top_notch, integrity=one_wide_sheet,
> separation=connected，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 顶部边缘中央的缺口首次向下延长先于白色长方形分成两个区域发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`142`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: paper: exactly 1, state tear_extent=small_top_notch,
> integrity=one_wide_sheet, separation=connected, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] white rectangle divides
> into two regions is already present BEFORE the centered top-edge notch first lengthens downward.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸张（paper）: 恰好1个，状态 tear_extent=small_top_notch, integrity=one_wide_sheet,
> separation=connected，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 在顶部边缘中央的缺口首次向下延长发生之前，白色长方形分成两个区域已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`140`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: paper: exactly 1, state tear_extent=small_top_notch,
> integrity=one_wide_sheet, separation=connected, transitioning. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VALID RELATION] the centered top-edge notch
> first lengthens downward happens BEFORE white rectangle divides into two regions.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸张（paper）: 恰好1个，状态 tear_extent=small_top_notch, integrity=one_wide_sheet,
> separation=connected，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系] 顶部边缘中央的缺口首次向下延长先于白色长方形分成两个区域发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`141`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: paper: exactly 1, state tear_extent=small_top_notch,
> integrity=one_wide_sheet, separation=connected, transitioning. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] white rectangle divides
> into two regions happens BEFORE the centered top-edge notch first lengthens downward.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸张（paper）: 恰好1个，状态 tear_extent=small_top_notch, integrity=one_wide_sheet,
> separation=connected，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。
> 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系] 白色长方形分成两个区域先于顶部边缘中央的缺口首次向下延长发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`132`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: paper: exactly 1, state tear_extent=full_vertical_tear,
> integrity=two_white_pieces, separation=apart, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VALID RELATION] The declared change gray slit
> lengthens downward occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸张（paper）: 恰好1个，状态 tear_extent=full_vertical_tear, integrity=two_white_pieces,
> separation=apart，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 已声明的变化“灰色裂缝向下延长”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`134`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3`

> [STAGE ENTITY INVENTORY] Present now: paper: exactly 1, state tear_extent=full_vertical_tear,
> integrity=two_white_pieces, separation=apart, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] Only one coupled change
> occurs while the other remains unchanged: .

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸张（paper）: 恰好1个，状态 tear_extent=full_vertical_tear, integrity=two_white_pieces,
> separation=apart，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 只有一个耦合变化发生，而另一个保持不变：（源编译文本此处为空）。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`133`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: paper: exactly 1, state tear_extent=full_vertical_tear,
> integrity=two_white_pieces, separation=apart, transitioning. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VALID RELATION] The declared change gray slit
> lengthens downward occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸张（paper）: 恰好1个，状态 tear_extent=full_vertical_tear, integrity=two_white_pieces,
> separation=apart，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 已声明的变化“灰色裂缝向下延长”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`133`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3, c4`

> [STAGE ENTITY INVENTORY] Present now: paper: exactly 1, state tear_extent=full_vertical_tear,
> integrity=two_white_pieces, separation=apart, transitioning. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] The declared cause occurs
> BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸张（paper）: 恰好1个，状态 tear_extent=full_vertical_tear, integrity=two_white_pieces,
> separation=apart，正在过渡. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`179`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: paper: exactly 1, state tear_extent=full_vertical_tear,
> integrity=two_white_pieces, separation=apart, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VALID RELATION] The terminal facts
> integrity=two_white_pieces, separation=apart REMAIN through the final frames. [AND] The same logical
> slot has only its valid state; mutually exclusive facts integrity=one_wide_sheet,
> integrity=two_white_pieces do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸张（paper）: 恰好1个，状态 tear_extent=full_vertical_tear, integrity=two_white_pieces,
> separation=apart，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [有效关系] 最终事实 integrity=two_white_pieces（两片白纸）、separation=apart（彼此分离） 在最后几帧持续保持。 [且]
> 同一逻辑槽位只具有其有效状态；互斥事实 integrity=one_wide_sheet（一张宽纸）与 integrity=two_white_pieces（两片白纸） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`170`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: paper: exactly 1, state tear_extent=full_vertical_tear,
> integrity=two_white_pieces, separation=apart, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] The valid terminal state
> returns to its mutually exclusive earlier state. [AND] The same logical slot has only its valid
> state; mutually exclusive facts integrity=one_wide_sheet, integrity=two_white_pieces do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸张（paper）: 恰好1个，状态 tear_extent=full_vertical_tear, integrity=two_white_pieces,
> separation=apart，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实
> integrity=one_wide_sheet（一张宽纸）与 integrity=two_white_pieces（两片白纸） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`163`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: paper: exactly 1, state tear_extent=full_vertical_tear,
> integrity=two_white_pieces, separation=apart, required. Counts and identities stay fixed; no
> duplicate state copy is created. No additional external source, extra copy, unlisted object exists.
> All globally protected identity, shape, material, scene, and camera attributes stay unchanged.
> Declared source loss and result gain remain coupled. [VIOLATED RELATION] The terminal facts
> integrity=two_white_pieces, separation=apart REMAIN through the final frames. [AND] Copies of the
> same logical slot appear in mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 纸张（paper）: 恰好1个，状态 tear_extent=full_vertical_tear, integrity=two_white_pieces,
> separation=apart，必需. 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。
> 声明的来源减少与结果增加保持耦合。 [违规关系] 最终事实 integrity=two_white_pieces（两片白纸）、separation=apart（彼此分离） 在最后几帧持续保持。 [且]
> 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_notch_opens 必须先于 pictured_sheet_split。 |
| `c2` | `changes` | 方向变化：转移 e1 中的效果 pictured_tear_grow 必须按声明方向发生。 |
| `c3` | `couples` | 耦合约束：pictured_tear_grow, pictured_sheet_split, pictured_pieces_apart 必须共同变化；关系为“中央灰色裂缝延长，同时白色两侧分离”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 integrity=two_white_pieces, separation=apart 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 integrity=one_wide_sheet, integrity=two_white_pieces 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "pictured_sheet_split",
    "before": "pictured_notch_opens",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "effect": "pictured_tear_grow",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "transition": "e1",
    "type": "changes"
  },
  {
    "effects": [
      "pictured_tear_grow",
      "pictured_sheet_split",
      "pictured_pieces_apart"
    ],
    "id": "c3",
    "relation": "centered gray slit lengthens as white sides separate",
    "source_refs": [
      "origin"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "integrity=two_white_pieces",
      "separation=apart"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "integrity=one_wide_sheet",
      "integrity=two_white_pieces"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=1个; grounding.protected_boxes=0个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"paper":"derived_motion_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>

<details>
<summary><strong>P20</strong> — continuous；setup→c1; onset→c1; evolution→c2; completion→c3; terminal→c4,c5</summary>

### P20 概要

- 时间路由类型：`continuous`
- 阶段文本实际选中约束：`setup→c1; onset→c1; evolution→c2; completion→c3; terminal→c4,c5`
- 同时进入 Global 守恒文本的 couples 约束：`c3`
- 仅校验、未进入生成张量的约束：`无`
- 实际编码文本键：
  - `global_semantic`
  - `stage.completion.negative.0`
  - `stage.completion.positive`
  - `stage.evolution.negative.0`
  - `stage.evolution.positive`
  - `stage.onset.negative.0`
  - `stage.onset.positive`
  - `stage.setup.negative.0`
  - `stage.setup.positive`
  - `stage.terminal.negative.0`
  - `stage.terminal.negative.1`
  - `stage.terminal.positive`
  - `cfg_negative(空字符串)`

### Global Semantic（实际编码）

#### 英文原文

> [ENTITY INVENTORY] The closed scene contains kettle: exactly 1; coffee stream: one declared
> collective group; cup: exactly 1; campfire: exactly 1. The reference first frame fixes these
> identities and initial layout. [INITIAL STATE] source_amount=more, stream_state=absent,
> cup_level=lower. [PHYSICAL EVENT] The stainless kettle at right tilts its left-pointing spout toward
> the gray cup and pours coffee into it while the campfire at left keeps flickering. [PERSISTENT
> PROTECTION] same stainless kettle, black handle, lid, and left-pointing spout; same small gray cup
> below-left of the kettle; campfire, logs, stones, and flames remain in their left-side region;
> campfire keeps flickering without reacting to the coffee transfer; snowy ground, framing, and
> daylight remain unchanged; no coffee spills onto the snow or campfire. [CLOSED WORLD] No additional
> external source, extra copy, unlisted object enters or appears; unoccupied scene regions remain
> empty. [CONSERVATION] Source loss and result gain stay coupled: tracked logical entity slots remain
> consistent; visible spout outflow crosses one stream path and raises the dark surface inside the
> gray cup. No result appears from an undeclared external source. [TERMINAL STATE] The same logical
> entity slots persist without duplicates; source_amount=less, stream_state=continuous,
> cup_level=higher remains visible through the final frames.

#### 中文解释

> [实体清单] 封闭场景包含 kettle：恰好 1 个；coffee_stream：一个已声明的集合组；cup：恰好 1 个；campfire：恰好 1
> 个。参考首帧固定这些身份和初始布局。[初始状态] cup_level=lower、source_amount=more、stream_state=absent。[物理事件]
> 右侧不锈钢水壶倾斜，使朝左的壶嘴对准灰色杯子并向其中倒入咖啡，同时左侧营火持续闪烁。[持续保护] 保持同一不锈钢水壶、黑色手柄、壶盖和朝左壶嘴；保持水壶左下方同一个小灰杯；营火、木柴、石头和火焰保持在
> 左侧区域；营火持续闪烁且不对咖啡转移作出反应；雪地、构图和日光保持不变；咖啡不洒到雪地或营火上。[封闭世界] 不允许新增外部来源、额外副本、未列出的物体进入或出现；未占用的场景区域保持为空。[守恒]
> 来源减少与结果增加保持耦合：被跟踪的逻辑实体槽位保持一致；可见壶嘴流出物沿单一液流路径跨过间隙，并抬高灰杯内的深色液面。不允许任何结果由未声明的外部来源产生。[终止状态]
> 相同的逻辑实体槽位持续存在且没有重复副本；cup_level=higher、source_amount=less、stream_state=continuous 在最后几帧保持可见。

### 各阶段实际使用的正、负提示词

#### setup / 准备/初始阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.setup.positive`；T5 token 数：`170`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: kettle: exactly 1, state source_amount=more, required; coffee
> stream: exactly 0, state stream_state=absent, forbidden; cup: exactly 1, state cup_level=lower,
> required; campfire: exactly 1, required. Counts and identities stay fixed; no duplicate state copy
> is created. No additional external source, extra copy, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VALID RELATION] coffee first leaves the left-pointing spout
> and reaches the gray cup happens BEFORE dark coffee surface rises inside gray rim.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（kettle）: 恰好1个，状态 source_amount=more，必需; 咖啡流（coffee stream）: 恰好0个，状态
> stream_state=absent，禁止出现; 杯子（cup）: 恰好1个，状态 cup_level=lower，必需; 营火（campfire）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 咖啡首次离开朝左的壶嘴并抵达灰色杯子先于深色咖啡液面在灰色杯沿内上升发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.setup.negative.0`；T5 token 数：`173`
- 权重：`1.0`；violation：`outcome_preset`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: kettle: exactly 1, state source_amount=more, required; coffee
> stream: exactly 0, state stream_state=absent, forbidden; cup: exactly 1, state cup_level=lower,
> required; campfire: exactly 1, required. Counts and identities stay fixed; no duplicate state copy
> is created. No additional external source, extra copy, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VIOLATED RELATION] dark coffee surface rises inside gray rim
> is already present BEFORE coffee first leaves the left-pointing spout and reaches the gray cup.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（kettle）: 恰好1个，状态 source_amount=more，必需; 咖啡流（coffee stream）: 恰好0个，状态
> stream_state=absent，禁止出现; 杯子（cup）: 恰好1个，状态 cup_level=lower，必需; 营火（campfire）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 在咖啡首次离开朝左的壶嘴并抵达灰色杯子发生之前，深色咖啡液面在灰色杯沿内上升已经出现。

#### onset / 起始/触发阶段

- 实际用于构造 Prompt 的约束：`c1`
- violation 中声明的候选约束：`c1`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.onset.positive`；T5 token 数：`172`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: kettle: exactly 1, state source_amount=more, transitioning;
> coffee stream: one declared collective group, state stream_state=absent, transitioning; cup: exactly
> 1, state cup_level=lower, transitioning; campfire: exactly 1, required. Counts and identities stay
> fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] coffee first leaves
> the left-pointing spout and reaches the gray cup happens BEFORE dark coffee surface rises inside
> gray rim.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（kettle）: 恰好1个，状态 source_amount=more，正在过渡; 咖啡流（coffee stream）: 1个已声明集合组，状态
> stream_state=absent，正在过渡; 杯子（cup）: 恰好1个，状态 cup_level=lower，正在过渡; 营火（campfire）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 咖啡首次离开朝左的壶嘴并抵达灰色杯子先于深色咖啡液面在灰色杯沿内上升发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.onset.negative.0`；T5 token 数：`173`
- 权重：`1.0`；violation：`wrong_order`
- 实际用于构造该负提示词的约束：`c1`；violation 声明的候选约束：`c1`

> [STAGE ENTITY INVENTORY] Present now: kettle: exactly 1, state source_amount=more, transitioning;
> coffee stream: one declared collective group, state stream_state=absent, transitioning; cup: exactly
> 1, state cup_level=lower, transitioning; campfire: exactly 1, required. Counts and identities stay
> fixed; no duplicate state copy is created. No additional external source, extra copy, unlisted
> object exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] dark coffee
> surface rises inside gray rim happens BEFORE coffee first leaves the left-pointing spout and reaches
> the gray cup.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（kettle）: 恰好1个，状态 source_amount=more，正在过渡; 咖啡流（coffee stream）: 1个已声明集合组，状态
> stream_state=absent，正在过渡; 杯子（cup）: 恰好1个，状态 cup_level=lower，正在过渡; 营火（campfire）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 深色咖啡液面在灰色杯沿内上升先于咖啡首次离开朝左的壶嘴并抵达灰色杯子发生。

#### evolution / 演化阶段

- 实际用于构造 Prompt 的约束：`c2`
- violation 中声明的候选约束：`c2, c3`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.evolution.positive`；T5 token 数：`159`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: kettle: exactly 1, state source_amount=less, required; coffee
> stream: one declared collective group, state stream_state=continuous, required; cup: exactly 1,
> state cup_level=higher, required; campfire: exactly 1, required. Counts and identities stay fixed;
> no duplicate state copy is created. No additional external source, extra copy, unlisted object
> exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The declared change
> coffee visibly exits the metal spout occurs in its stated direction.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（kettle）: 恰好1个，状态 source_amount=less，必需; 咖啡流（coffee stream）: 1个已声明集合组，状态
> stream_state=continuous，必需; 杯子（cup）: 恰好1个，状态 cup_level=higher，必需; 营火（campfire）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 已声明的变化“咖啡明显流出金属壶嘴”按规定方向发生。

**负提示词 0（实际编码完整文本）**

- 键：`stage.evolution.negative.0`；T5 token 数：`158`
- 权重：`1.0`；violation：`coupling_broken`
- 实际用于构造该负提示词的约束：`c2`；violation 声明的候选约束：`c2, c3`

> [STAGE ENTITY INVENTORY] Present now: kettle: exactly 1, state source_amount=less, required; coffee
> stream: one declared collective group, state stream_state=continuous, required; cup: exactly 1,
> state cup_level=higher, required; campfire: exactly 1, required. Counts and identities stay fixed;
> no duplicate state copy is created. No additional external source, extra copy, unlisted object
> exists. All globally protected identity, shape, material, scene, and camera attributes stay
> unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION] Only one coupled
> change occurs while the other remains unchanged: .

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（kettle）: 恰好1个，状态 source_amount=less，必需; 咖啡流（coffee stream）: 1个已声明集合组，状态
> stream_state=continuous，必需; 杯子（cup）: 恰好1个，状态 cup_level=higher，必需; 营火（campfire）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 只有一个耦合变化发生，而另一个保持不变：（源编译文本此处为空）。

#### completion / 完成阶段

- 实际用于构造 Prompt 的约束：`c3`
- violation 中声明的候选约束：`c3, c4`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.completion.positive`；T5 token 数：`183`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: kettle: exactly 1, state source_amount=less, transitioning;
> coffee stream: one declared collective group, state stream_state=continuous, transitioning; cup:
> exactly 1, state cup_level=higher, transitioning; campfire: exactly 1, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VALID RELATION] The
> coupled changes occur together: coffee visibly exits the metal spout WHILE coffee bridges the gap
> from spout to cup WHILE dark coffee surface rises inside gray rim.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（kettle）: 恰好1个，状态 source_amount=less，正在过渡; 咖啡流（coffee stream）: 1个已声明集合组，状态
> stream_state=continuous，正在过渡; 杯子（cup）: 恰好1个，状态 cup_level=higher，正在过渡; 营火（campfire）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 耦合变化同时发生：咖啡明显流出金属壶嘴，同时咖啡跨越壶嘴到杯子的间隙，并且深色咖啡液面在灰色杯沿内上升。

**负提示词 0（实际编码完整文本）**

- 键：`stage.completion.negative.0`；T5 token 数：`159`
- 权重：`1.0`；violation：`incomplete_transition`
- 实际用于构造该负提示词的约束：`c3`；violation 声明的候选约束：`c3, c4`

> [STAGE ENTITY INVENTORY] Present now: kettle: exactly 1, state source_amount=less, transitioning;
> coffee stream: one declared collective group, state stream_state=continuous, transitioning; cup:
> exactly 1, state cup_level=higher, transitioning; campfire: exactly 1, required. Counts and
> identities stay fixed; no duplicate state copy is created. No additional external source, extra
> copy, unlisted object exists. All globally protected identity, shape, material, scene, and camera
> attributes stay unchanged. Declared source loss and result gain remain coupled. [VIOLATED RELATION]
> The declared cause occurs BUT the required target change does not become visible.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（kettle）: 恰好1个，状态 source_amount=less，正在过渡; 咖啡流（coffee stream）: 1个已声明集合组，状态
> stream_state=continuous，正在过渡; 杯子（cup）: 恰好1个，状态 cup_level=higher，正在过渡; 营火（campfire）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 声明的原因已经发生，但所要求的目标变化没有变得可见。

#### terminal / 终止/保持阶段

- 实际用于构造 Prompt 的约束：`c4, c5`
- violation 中声明的候选约束：`c4, c5`
- minimal-pair 质量：`minimal_verified`
- 正提示词键：`stage.terminal.positive`；T5 token 数：`201`

**正提示词（实际编码完整文本）**

> [STAGE ENTITY INVENTORY] Present now: kettle: exactly 1, state source_amount=less, required; coffee
> stream: exactly 0, state stream_state=continuous, forbidden; cup: exactly 1, state cup_level=higher,
> required; campfire: exactly 1, required. Counts and identities stay fixed; no duplicate state copy
> is created. No additional external source, extra copy, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VALID RELATION] The terminal facts source_amount=less,
> cup_level=higher REMAIN through the final frames. [AND] The same logical slot has only its valid
> state; mutually exclusive facts cup_level=lower, cup_level=higher do not coexist.

**正提示词中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（kettle）: 恰好1个，状态 source_amount=less，必需; 咖啡流（coffee stream）: 恰好0个，状态
> stream_state=continuous，禁止出现; 杯子（cup）: 恰好1个，状态 cup_level=higher，必需; 营火（campfire）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [有效关系]
> 最终事实 source_amount=less（源中咖啡减少）、cup_level=higher（杯内液位更高） 在最后几帧持续保持。 [且] 同一逻辑槽位只具有其有效状态；互斥事实
> cup_level=lower（较低）与 cup_level=higher（较高） 不会共存。

**负提示词 0（实际编码完整文本）**

- 键：`stage.terminal.negative.0`；T5 token 数：`192`
- 权重：`0.5`；violation：`reversal`
- 实际用于构造该负提示词的约束：`c4`；violation 声明的候选约束：`c4`

> [STAGE ENTITY INVENTORY] Present now: kettle: exactly 1, state source_amount=less, required; coffee
> stream: exactly 0, state stream_state=continuous, forbidden; cup: exactly 1, state cup_level=higher,
> required; campfire: exactly 1, required. Counts and identities stay fixed; no duplicate state copy
> is created. No additional external source, extra copy, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VIOLATED RELATION] The valid terminal state returns to its
> mutually exclusive earlier state. [AND] The same logical slot has only its valid state; mutually
> exclusive facts cup_level=lower, cup_level=higher do not coexist.

**负提示词 0 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（kettle）: 恰好1个，状态 source_amount=less，必需; 咖啡流（coffee stream）: 恰好0个，状态
> stream_state=continuous，禁止出现; 杯子（cup）: 恰好1个，状态 cup_level=higher，必需; 营火（campfire）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 有效的终止状态恢复成与其互斥的较早状态。 [且] 同一逻辑槽位只具有其有效状态；互斥事实 cup_level=lower（较低）与 cup_level=higher（较高） 不会共存。

**负提示词 1（实际编码完整文本）**

- 键：`stage.terminal.negative.1`；T5 token 数：`188`
- 权重：`0.5`；violation：`exclusion_broken`
- 实际用于构造该负提示词的约束：`c5`；violation 声明的候选约束：`c5`

> [STAGE ENTITY INVENTORY] Present now: kettle: exactly 1, state source_amount=less, required; coffee
> stream: exactly 0, state stream_state=continuous, forbidden; cup: exactly 1, state cup_level=higher,
> required; campfire: exactly 1, required. Counts and identities stay fixed; no duplicate state copy
> is created. No additional external source, extra copy, unlisted object exists. All globally
> protected identity, shape, material, scene, and camera attributes stay unchanged. Declared source
> loss and result gain remain coupled. [VIOLATED RELATION] The terminal facts source_amount=less,
> cup_level=higher REMAIN through the final frames. [AND] Copies of the same logical slot appear in
> mutually exclusive states at the same time.

**负提示词 1 中文翻译（仅供阅读，不参与编码）**

> [阶段实体清单] 当前存在： 水壶（kettle）: 恰好1个，状态 source_amount=less，必需; 咖啡流（coffee stream）: 恰好0个，状态
> stream_state=continuous，禁止出现; 杯子（cup）: 恰好1个，状态 cup_level=higher，必需; 营火（campfire）: 恰好1个，必需.
> 数量和身份保持固定；不创建重复的状态副本。 不存在额外的外部来源、额外副本或未列出的物体。 所有受全局保护的身份、形状、材质、场景和相机属性保持不变。 声明的来源减少与结果增加保持耦合。 [违规关系]
> 最终事实 source_amount=less（源中咖啡减少）、cup_level=higher（杯内液位更高） 在最后几帧持续保持。 [且] 同一逻辑槽位的多个副本同时以互斥状态出现。

### 约束定义

| ID | 类型 | 中文解释 |
|---|---|---|
| `c1` | `precedes` | 前置顺序：pictured_stream_onset 必须先于 pictured_cup_increase。 |
| `c2` | `changes` | 方向变化：转移 e1 中的效果 pictured_source_decrease 必须按声明方向发生。 |
| `c3` | `couples` | 耦合约束：pictured_source_decrease, pictured_stream_form, pictured_cup_increase 必须共同变化；关系为“可见壶嘴流出物沿单一液流路径跨过间隙，并抬高灰杯内的深色液面”。 |
| `c4` | `persists` | 持续约束：状态 z1 的事实 source_amount=less, cup_level=higher 必须维持到最终帧。 |
| `c5` | `excludes` | 互斥约束：事实 cup_level=lower, cup_level=higher 不得在同一时间共存。 |

<details>
<summary>展开全部约束原始 JSON</summary>

```json
[
  {
    "after": "pictured_cup_increase",
    "before": "pictured_stream_onset",
    "id": "c1",
    "source_refs": [
      "origin"
    ],
    "type": "precedes"
  },
  {
    "effect": "pictured_source_decrease",
    "id": "c2",
    "source_refs": [
      "origin"
    ],
    "transition": "e1",
    "type": "changes"
  },
  {
    "effects": [
      "pictured_source_decrease",
      "pictured_stream_form",
      "pictured_cup_increase"
    ],
    "id": "c3",
    "relation": "visible spout outflow crosses one stream path and raises the dark surface inside the gray cup",
    "source_refs": [
      "origin",
      "reference_input"
    ],
    "type": "couples"
  },
  {
    "facts": [
      "source_amount=less",
      "cup_level=higher"
    ],
    "id": "c4",
    "source_refs": [
      "origin"
    ],
    "state": "z1",
    "type": "persists"
  },
  {
    "facts": [
      "cup_level=lower",
      "cup_level=higher"
    ],
    "id": "c5",
    "scope": "same_time",
    "source_refs": [
      "origin"
    ],
    "type": "excludes"
  }
]
```

</details>

### 空间控制与未直接作为 Prompt 的字段

- 参与空间控制的 JSON：grounding.entity_boxes=3个; grounding.protected_boxes=1个; stages[].affected_roles; stages[].support_kind; predicates[].attribute; transitions[].effects[].direction/from/to
- 空间控制编译结果：mode=planned; source=plan_boxes_and_tracks; target_provenance={"campfire":"unresolved_static_envelope","cup":"unresolved_static_envelope","kettle":"unresolved_static_envelope"}
- 没有原样作为 Prompt 使用的字段：stages[].positive; stages[].violations[].text; stages[].observable_check; predicates[].observable; events[].observable; source_refs; assumptions; warnings; generation_basis; source_files

</details>
