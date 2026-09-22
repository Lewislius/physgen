# P0 人话分阶段提示词使用报告（中英双语、紧凑版）

> 来源：`prompt/p0_direct_prompts_p01_p20.json` 与 P01–P20 当前 `plan/planimg/i0.json`。英文原文逐字读取；中文仅供阅读，不参与模型编码。
> 这是一份对当前实现的静态模拟报告，不代表已经运行了 20 个新视频。

## 当前 P0 到底把什么送入模型

| 部分 | M3 / I2V | M4 / T2V | 是否改写 |
|---|---|---|---|
| 首帧图像 | `planimg.source_files.reference_input` 指定的 1280×704 JPG | 不使用 | 否 |
| 全局正向文本 | `global_semantic`，整个去噪过程持续生效 | `global_semantic`，整个去噪过程持续生效 | 否，逐字传入 |
| CFG 负向文本 | `cfg_negative`，实际值为一个 U+0020 空格 | 同左 | 否，逐字传入 |
| 五阶段 c+ | 五条 `stages[].positive` 分别编码并注入各自时间 token | 同左 | 否，逐字传入 |
| 五阶段 c- | 不设置；共用 `cfg_negative` 的空格上下文 | 同左 | 无负面物体或反事实句 |
| 时间路由 | 直接读取 `temporal_route.weights` 的 5×25 权重 | 同左 | 不从文本推断 |
| `initial_condition/entities/boxes` | 用于数据说明和首帧审查；P0 不注入 DiT | 同左 | 不参与生成 |
| 旧约束、violation、模板 c+/c− | 不存在 | 不存在 | 不参与生成 |

因此，每个时间段实际使用的是“始终存在的 `global_semantic` + 当前阶段的人话 c+”。
阶段 c+ 不会拼进 Global Semantic，而是单独编码后只作用于该阶段的 DiT 时间 token；
相邻阶段用两个 token 平滑交接。阶段 c- 为空，不再把不希望出现的物体名称喂给模型。

### 三种固定时间分配

97 个输出帧在 Wan latent 中对应 25 个时间 token，编号为 0–24。箭头位置各含两个
过渡 token：第一个权重为“前阶段 0.740741 / 后阶段 0.259259”，第二个反过来。

| 路由 | 五阶段核心 token（setup / onset / evolution / completion / terminal） | 两-token 过渡位置 |
|---|---|---|
| `triggered` | 0–2 / 5–6 / 9–14 / 17–19 / 22–24 | 3–4、7–8、15–16、20–21 |
| `continuous` | 0–1 / 4–5 / 8–15 / 18–19 / 22–24 | 2–3、6–7、16–17、20–21 |
| `multi_transition` | 0–2 / 5–7 / 10–15 / 18–20 / 23–24 | 3–4、8–9、16–17、21–22 |

## 总览

| P | 初始原因 | I2V 首帧 | 25-token 路由 | 阶段 c- |
|---|---|---:|---|---|
| P01 | `initial_momentum` | P01-i0-1280x704.jpg | `triggered` | 空 |
| P02 | `initial_momentum` | P02-i0-1280x704.jpg | `triggered` | 空 |
| P03 | `visible_actuator` | P03-i0-1280x704.jpg | `triggered` | 空 |
| P04 | `visible_actuator` | P04-i0-1280x704.jpg | `triggered` | 空 |
| P05 | `gravity_or_instability` | P05-i0-1280x704.jpg | `multi_transition` | 空 |
| P06 | `visible_actuator` | P06-i0-1280x704.jpg | `multi_transition` | 空 |
| P07 | `visible_actuator` | P07-i0-1280x704.jpg | `triggered` | 空 |
| P08 | `environmental_field` | P08-i0-1280x704.jpg | `continuous` | 空 |
| P09 | `environmental_field` | P09-i0-1280x704.jpg | `continuous` | 空 |
| P10 | `visible_actuator` | P10-i0-1280x704.jpg | `multi_transition` | 空 |
| P11 | `gravity_or_instability` | P11-i0-1280x704.jpg | `triggered` | 空 |
| P12 | `environmental_field` | P12-i0-1280x704.jpg | `continuous` | 空 |
| P13 | `visible_actuator` | P13-i0-1280x704.jpg | `continuous` | 空 |
| P14 | `visible_actuator` | P14-i0-1280x704.jpg | `continuous` | 空 |
| P15 | `visible_actuator` | P15-i0-1280x704.jpg | `continuous` | 空 |
| P16 | `gravity_or_instability` | P16-i0-1280x704.jpg | `continuous` | 空 |
| P17 | `visible_actuator` | P17-i0-1280x704.jpg | `multi_transition` | 空 |
| P18 | `initial_momentum` | P18-i0-1280x704.jpg | `triggered` | 空 |
| P19 | `visible_actuator` | P19-i0-1280x704.jpg | `continuous` | 空 |
| P20 | `visible_actuator` | P20-i0-1280x704.jpg | `continuous` | 空 |

## 每个 P 的完整提示词

<details>
<summary><strong>P01</strong> — initial_momentum；全局语义 + 五阶段人话 c+</summary>

### P01 使用概要

- M3/I2V 首帧：`P01-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：The scooter is already rolling toward the trash can when the clip begins.
- 初始原因中文：滑板车在视频开始时已经朝垃圾桶滚动。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed side-view photograph on an uncluttered gray street. One black stand-up electric kick scooter is upright at the left, facing right and already aligned on a straight path toward one upright black slatted metal trash can at the right. Leave a clear gap before contact and open space beside the can for the scooter to fall. Both objects are fully visible, large in frame, with no collision or tipping yet. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定侧视街景：一辆直立黑色电动滑板车位于左侧并正对右侧垃圾桶；碰撞前留有清楚间距，垃圾桶旁留出倒地空间，尚未接触或倾倒。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Static side view of one black stand-up electric kick scooter already rolling right toward one upright black slatted trash can on a gray street. Its front wheel hits the can once, the same scooter tips onto its side and stops beside it, and the can stays upright.

#### 中文解释

> 固定侧视：一辆黑色电动滑板车已经向右滚向一个直立黑色条板垃圾桶。前轮只撞击一次；同一辆车随后侧倒并停在桶旁，垃圾桶保持直立。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The upright scooter is already rolling right and the clear gap to the trash can narrows. | 直立滑板车已向右滚动，与垃圾桶的间距持续缩小。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The scooter's front wheel touches the trash can for the first and only impact. | 滑板车前轮第一次也是唯一一次接触垃圾桶。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The same scooter slows and rotates down onto its right side beside the can. | 同一辆滑板车减速并向右侧旋倒在桶旁。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The scooter finishes falling and stops while the trash can remains upright. | 滑板车完成倒地并停止，垃圾桶仍然直立。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | One scooter lies still beside one upright trash can through the final frames. | 最后画面中只有一辆倒地滑板车和一个直立垃圾桶。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P02</strong> — initial_momentum；全局语义 + 五阶段人话 c+</summary>

### P02 使用概要

- M3/I2V 首帧：`P02-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：The red ball is already rolling toward the stationary blue ball when the clip begins.
- 初始原因中文：红球在视频开始时已经朝静止蓝球滚动。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed side-view photograph of a clean green billiard table. One glossy red billiard ball is at the left and one glossy blue billiard ball is farther right on the same straight horizontal path. The red ball is rolling right toward the stationary blue ball, with a clearly visible gap before contact and ample empty table beyond the blue ball. Both balls are fully visible, identical in size, and sharply separated. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定侧视台球桌：一个红球在左、一个蓝球在右，位于同一直线路径上；红球正在向右运动，二者尚未接触，蓝球右侧保留运动空间。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed side view of one glossy red billiard ball already rolling right toward one stationary glossy blue ball on green cloth. The red ball hits the blue ball once. Both balls remain visible; afterward the blue ball rolls ahead to the right while the red ball follows more slowly.

#### 中文解释

> 固定侧视：一个有光泽的红色台球已经向右滚向静止蓝球。红球只碰蓝球一次；两个球始终可见，之后蓝球向右滚在前面，红球以较慢速度跟随。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The red ball rolls right while the blue ball waits motionless ahead and the gap closes. | 红球向右滚动，蓝球静止等待，两球间距缩小。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The red ball makes one visible edge-to-edge contact with the blue ball. | 红球与蓝球发生一次清楚的边缘接触。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The blue ball begins rolling right as the same red ball continues right more slowly. | 蓝球开始向右滚动，同一个红球继续向右但速度更慢。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The moving blue ball opens a growing gap ahead of the slower red ball. | 蓝球在前方运动，与较慢红球之间的距离增大。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | Both original balls remain visible, with the blue ball farther right than the red ball. | 两个原始球都保持可见，蓝球位于红球右侧更远处。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P03</strong> — visible_actuator；全局语义 + 五阶段人话 c+</summary>

### P03 使用概要

- M3/I2V 首帧：`P03-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：A visible hand is already swinging the racket toward the approaching ball.
- 初始原因中文：一只可见的手已经挥动球拍迎向来球。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed side-view sports photograph against a simple dark court background. At the left, one visible hand grips one black tennis racket and swings its strings toward the right. At the right, one yellow tennis ball approaches the racket along the same clear path, with a visible gap before impact and open space behind the ball for its return. Hand, racket, and ball are fully visible and sharply separated. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定侧视运动场景：左侧手握黑色网球拍向右挥动，右侧黄色网球沿同一路径接近，击球前仍有间距，球后方留有反弹空间。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed side view of one visible hand swinging one black racket toward one yellow tennis ball approaching from the right. The strings hit the ball once, and the same ball reverses direction and travels right away from the racket while the camera stays still.

#### 中文解释

> 固定侧视：一只手挥动黑色球拍迎向从右侧接近的黄色网球。拍线只击球一次，同一个球反向后向右远离球拍，相机保持静止。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The hand swings the racket rightward while the tennis ball approaches from the right with a gap. | 手向右挥拍，网球从右侧接近，球拍和球之间仍有间距。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The approaching ball visibly meets the center of the racket strings once. | 来球只与球拍网面中心接触一次。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The same ball compresses briefly and reverses its travel direction. | 同一个球短暂受压并反转运动方向。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The ball moves right away from the racket and the separation grows. | 球向右远离球拍，二者间距扩大。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | The original ball remains intact to the right while the hand and racket remain at the left. | 原来的球完整地留在右侧，手和球拍留在左侧。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P04</strong> — visible_actuator；全局语义 + 五阶段人话 c+</summary>

### P04 使用概要

- M3/I2V 首帧：`P04-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：A visible fingertip is touching the first domino to start the chain.
- 初始原因中文：一根可见手指正在推动第一块多米诺骨牌。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed three-quarter-view photograph of one straight row of upright pale wooden dominoes on a plain tabletop, running from the lower left toward the upper right. A single visible fingertip approaches only the first domino at the left end, just before pushing it. Every domino is upright, individually visible, evenly spaced, and fully inside the frame, with clear room along the entire fall direction. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定三分之四视角：一排浅色木质多米诺骨牌全部直立，一根手指只靠近左端第一块；整排及完整倒下方向均在画面内，尚无骨牌倒下。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed three-quarter view of one fingertip pushing the first piece in one straight row of upright pale wooden dominoes. Each domino knocks down its next neighbor from left to right, the fall reaches the last piece once, and the whole row remains down.

#### 中文解释

> 固定三分之四视角：一根手指推动直立木质多米诺骨牌队列的第一块。每块依次撞倒下一块，倒下过程从左到右到达最后一块，整排最后保持倒下。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The fingertip pushes only the first upright domino at the left end of the row. | 手指只推动队列左端第一块直立骨牌。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The first domino falls into the second and starts a single chain reaction. | 第一块倒向第二块并启动一次连锁反应。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The falling front advances domino by domino from left to right. | 倒下前沿逐块从左向右推进。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The chain reaches the last domino and that final piece falls. | 连锁反应到达最后一块，最后一块随之倒下。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | The same row remains fully down and motionless on the tabletop. | 同一排骨牌全部倒下并静止在桌面。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P05</strong> — gravity_or_instability；全局语义 + 五阶段人话 c+</summary>

### P05 使用概要

- M3/I2V 首帧：`P05-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：The block begins on the upper ramp so gravity can pull it down the slope.
- 初始原因中文：木块位于斜坡上部，重力可以使其沿坡下滑。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed side-view photograph of one light-wood rectangular block near the upper left portion of one wooden ramp that slopes down toward the right. The block is aligned with the slope and just beginning to slide under gravity. The ramp ends above a broad rough gray floor that continues far to the right, leaving a complete visible path for slowing and stopping. Block, ramp, and floor are fully visible. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定侧视：一个浅色木块位于向右下倾斜的木坡上部并刚开始下滑；坡底连接宽阔粗糙灰色地面，完整减速和停止路径都在画面内。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed side view of one light-wood block beginning to slide right under gravity from the upper part of one wooden ramp. The same block follows the slope onto the rough gray floor, slows continuously across the floor, and remains stopped to the right of the ramp.

#### 中文解释

> 固定侧视：一个浅色木块在重力作用下从木坡上部开始向右滑动。同一个木块沿斜坡进入粗糙灰色地面，持续减速，最后停在斜坡右侧。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The wooden block begins sliding right down the upper part of the ramp under gravity. | 木块在重力作用下从斜坡上部开始向右下滑。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The same block reaches the ramp's lower end and crosses onto the rough floor. | 同一个木块到达坡底并进入粗糙地面。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The block continues right on the floor with visibly smaller displacement each moment. | 木块继续向右，但每一时刻的位移明显减小。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The block loses its remaining speed and comes to rest right of the ramp. | 木块失去剩余速度并停在斜坡右侧。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | One intact wooden block remains motionless on the rough floor. | 一个完整木块静止留在粗糙地面上。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P06</strong> — visible_actuator；全局语义 + 五阶段人话 c+</summary>

### P06 使用概要

- M3/I2V 首帧：`P06-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：A visible hand pushes the book toward the open end of the shelf.
- 初始原因中文：一只可见的手把书推向书架开放边缘。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed wide side-view photograph of one closed blue-edged book near the right end of a wooden shelf. One visible hand at the left is pushing the book rightward toward the shelf edge, with the book still fully supported and closed. Show the empty vertical space below the edge and the wooden floor in the same frame, leaving a clear uninterrupted fall path. Shelf, hand, book, and landing area are fully visible. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定宽侧视：一只手把一本闭合的蓝边书推向木架右端；书仍完整支撑且闭合，架边到地板的下落空间和落点均清楚可见。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Static side view of one hand pushing one closed blue-edged book rightward from the end of a wooden shelf. The same book clears the shelf, falls continuously through the open space, hits the wooden floor, opens once, and remains open there.

#### 中文解释

> 静止侧视：一只手把一本闭合蓝边书从木架末端向右推出。同一本书离开书架后连续下落，撞到木地板并只打开一次，最后保持摊开。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The hand pushes the one closed book rightward while the shelf still supports it. | 手向右推动唯一一本闭合书，书架仍在支撑它。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The same closed book clears the shelf edge and begins one continuous fall. | 同一本闭合书越过书架边缘并开始一次连续下落。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The book descends through the open gap while the shelf above becomes empty. | 书穿过空旷间隙向下运动，上方书架变空。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The book contacts the wooden floor and opens once at the landing point. | 书接触木地板，并在落点只打开一次。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | The single original book remains open and still on the floor. | 唯一的原书摊开并静止留在地板上。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P07</strong> — visible_actuator；全局语义 + 五阶段人话 c+</summary>

### P07 使用概要

- M3/I2V 首帧：`P07-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：A visible hand pulls the lower corner so the tall box loses support and tips.
- 初始原因中文：一只可见的手横向拉动高纸箱底角，使其失去支撑并倾倒。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed side-view photograph of one tall tan cardboard box standing upright on a smooth gray floor. A visible hand at the lower left grips and pulls the box's bottom right corner sideways toward the right. The box is still upright with its base touching the floor, but its tall shape and low pull make the impending tip physically clear. Leave broad empty floor to the right for the fall. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定侧视：一个高的棕色纸箱直立在光滑地面，手从低处向右拉箱体右下角；箱体尚未倾倒，右侧留有完整倒地空间。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed side view of one hand pulling the lower right corner of one tall tan cardboard box across a smooth gray floor. The same box loses its support, tips to the right onto one broad side, stops rotating, and remains lying there.

#### 中文解释

> 固定侧视：一只手在光滑灰地面上拉动高棕色纸箱的右下角。同一个箱子失去支撑后向右倾倒到一个宽侧面，停止转动并保持躺倒。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The hand pulls the box's lower right corner while the tall box is still upright. | 手拉纸箱右下角，高纸箱仍然直立。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The box's weight shifts beyond its base and the upper edge starts moving right. | 纸箱重心越过底面，上沿开始向右移动。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The same box rotates continuously toward its broad right side. | 同一个纸箱连续旋转倒向右侧宽面。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The broad side contacts the floor, rotation stops, and the hand releases. | 宽面接触地板，旋转停止，手松开。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | One cardboard box remains lying motionless on its right side. | 一个纸箱静止躺在右侧面上。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P08</strong> — environmental_field；全局语义 + 五阶段人话 c+</summary>

### P08 使用概要

- M3/I2V 首帧：`P08-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：Warm ambient sunlight supplies the heat that melts the ice.
- 初始原因中文：温暖阳光提供使冰块融化的环境热量。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed close photograph of one clear, sharply edged ice cube centered on one dry shallow white plate in a warm sunlit room. The cube is fully solid and large enough to track, with crisp corners and no visible meltwater yet. Soft sunlight and warm reflections make the heat source visually credible without changing the cube's initial state. Keep the plate and all of the future puddle area inside one uncluttered frame. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定近景：一个边缘清晰的完整透明冰块放在干燥白盘中央；阳光使温暖环境可信，但此时还没有融水，盘内保留水迹扩展区域。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed close view of one clear ice cube on a dry white plate in a warm sunlit room. Ambient heat gradually shrinks the same solid cube while a transparent meltwater puddle grows from its edges, and the smaller ice remnant and larger puddle remain on the plate.

#### 中文解释

> 固定近景：温暖阳光房间中，一个透明冰块放在干燥白盘上。环境热量使同一固体冰块逐渐缩小，同时透明融水从边缘形成并扩展；最后小冰块和更大水洼都留在盘中。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | One sharp solid ice cube rests on a dry white plate in warm sunlight. | 一个边角清晰的固体冰块位于温暖阳光下的干燥白盘。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The cube's lower edges round and the first thin wet rim appears around its base. | 冰块底边开始变圆，底部出现第一圈薄水迹。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The same ice cube becomes smaller while its connected water puddle spreads wider. | 同一冰块变小，连通的透明水洼扩展变宽。 | 本行英文原文就是 c+ |
| `completion` / 完成 | A small ice remnant sits inside a visibly larger transparent puddle. | 较小冰块残余位于明显更大的透明水洼中。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | The smaller ice remnant and its meltwater remain together on the plate. | 较小冰块和由它融成的水一起留在盘中。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P09</strong> — environmental_field；全局语义 + 五阶段人话 c+</summary>

### P09 使用概要

- M3/I2V 首帧：`P09-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：The visibly heated pan supplies the heat that softens and melts the butter.
- 初始原因中文：可见的热锅提供软化和融化黄油的热量。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed close photograph of one firm rectangular pat of pale-yellow butter at the center of one clean stainless-steel pan. The butter has straight sides and sharp edges, with no liquid pool yet. Gentle heat shimmer and warm reflections on the pan make the heating condition clear while the initial butter remains solid. Show the entire pan center with open room for the material to spread. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定近景：一块边缘锐利的淡黄色固体黄油位于不锈钢锅中央；锅体暖反光表现加热条件，但尚未出现液体池，并为扩散留出空间。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed close view of one firm rectangular pat of pale-yellow butter in one heated stainless-steel pan. Heat softens the same butter, rounds and lowers its edges, and steadily spreads its material into one connected yellow liquid pool that remains near the pan center.

#### 中文解释

> 固定近景：一块坚实矩形淡黄色黄油位于加热不锈钢锅中。热量使同一黄油软化，边缘变圆变低，并持续铺展成一个连通黄色液体池，最后留在锅中央附近。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | One firm rectangular butter pat with sharp edges rests in the heated pan. | 一块边缘清晰的坚实矩形黄油放在热锅中。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The same pat softens first at its lower corners and a glossy liquid rim appears. | 同一黄油先从底部角落软化，并出现有光泽液体边缘。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The solid butter lowers and shrinks while one connected yellow pool spreads outward. | 固体部分变低变小，一个连通黄色液体池向外扩展。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The remaining soft center settles into the wider liquid butter pool. | 剩余柔软中心沉入更宽的液态黄油池。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | The melted butter remains as one connected pool near the pan center. | 融化黄油以一个连通液体池留在锅中央附近。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P10</strong> — visible_actuator；全局语义 + 五阶段人话 c+</summary>

### P10 使用概要

- M3/I2V 首帧：`P10-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：The balloon is visibly attached to an air-pump nozzle that supplies the inflating air.
- 初始原因中文：气球清楚连接空气泵嘴，由泵嘴提供充气空气。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed medium photograph of one small unknotted green rubber balloon securely attached to one black air-pump nozzle from below. The balloon is loose, wrinkled, and only slightly filled, with its complete outline visible against a plain background. Center the nozzle and balloon with broad empty space around them for expansion, and show the physical connection clearly. The rubber is intact and there are no fragments yet. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定中景：一个小而松弛、有褶皱的绿色气球连接黑色泵嘴；连接处清楚，周围留出膨胀空间，气球完整且尚无碎片。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed medium view of one small unknotted green rubber balloon attached to one black air-pump nozzle. The pump steadily inflates the same balloon from loose and wrinkled to large and taut; it then bursts once, and its green rubber fragments remain near the nozzle.

#### 中文解释

> 固定中景：一个小的未打结绿色橡胶气球连接黑色空气泵嘴。泵持续把同一气球从松弛褶皱充到又大又紧，随后只爆裂一次，绿色橡胶碎片留在泵嘴附近。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The small wrinkled green balloon is intact and attached to the pump nozzle. | 小而褶皱的绿色气球完整连接在泵嘴上。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | Air from the nozzle begins enlarging the same balloon while its surface stays intact. | 泵嘴送入空气，同一气球开始变大且表面仍完整。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The balloon grows steadily larger and its green rubber becomes smooth and taut. | 气球稳定增大，绿色橡胶表面变得平滑绷紧。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The taut balloon bursts once and its stretched rubber separates into fragments. | 绷紧气球只爆裂一次，橡胶分成碎片。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | The green rubber fragments remain near the same pump nozzle. | 绿色橡胶碎片留在同一个泵嘴附近。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P11</strong> — gravity_or_instability；全局语义 + 五阶段人话 c+</summary>

### P11 使用概要

- M3/I2V 首帧：`P11-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：The bottle is already slipping beyond a shelf edge so gravity can pull it down.
- 初始原因中文：玻璃瓶已经滑出架边，重力会继续把它向下拉。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed wide side-view photograph of one intact clear glass bottle leaning beyond the right edge of one low shelf and just beginning to slip. The bottle is fully visible above a hard gray tile floor, with a clear vertical gap and the complete landing area in frame. Its glass surface is unbroken and the floor is empty before impact. Keep the shelf, fall path, and landing point visible together. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定宽侧视：一个完整透明玻璃瓶倾斜越过低架右边缘并开始滑落；硬瓷砖地面、完整下落路径和落点同时可见，瓶子尚未破碎。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed side view of one intact clear glass bottle already slipping from the edge of one low shelf under gravity. The same bottle falls through the open space, strikes the hard tile floor once, shatters, and its transparent fragments settle and remain around the impact point.

#### 中文解释

> 固定侧视：一个完整透明玻璃瓶在重力下从低架边缘滑落。同一个瓶子连续下落，只撞击硬瓷砖地面一次并破碎，透明碎片沉降后留在撞击点周围。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The intact bottle begins slipping past the shelf edge under gravity. | 完整玻璃瓶在重力下开始滑过架边。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The same bottle clears the shelf completely and starts a continuous fall. | 同一个瓶子完全离开架面并开始连续下落。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The intact bottle descends through the open gap toward the tile floor. | 完整瓶子穿过空旷间隙向瓷砖地面下降。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The bottle strikes the floor once and breaks into transparent fragments. | 瓶子只撞地一次并碎成透明碎片。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | The fragments from the original bottle remain settled around one impact point. | 原瓶产生的碎片静止留在一个撞击点周围。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P12</strong> — environmental_field；全局语义 + 五阶段人话 c+</summary>

### P12 使用概要

- M3/I2V 首帧：`P12-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：A visible left-to-right gust supplies the force that removes the dandelion seeds.
- 初始原因中文：从左向右的可见阵风提供吹走蒲公英种子的力。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed close side-view photograph of one full dry white dandelion seed head on its stem at the left of a blue-sky frame. The complete round seed head is intact and dense before seeds detach. Fine grass and nearby wisps bend gently toward the right to make a left-to-right breeze visible. Leave most of the right side as clear open sky for drifting seeds. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定近侧视：左侧一个完整饱满的白色蒲公英头，附近草叶向右弯曲以显示风向；种子尚未脱落，右侧天空留作漂移区域。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed close side view of one full white dandelion at the left of a blue-sky frame. A steady left-to-right gust detaches its seeds progressively; the same seeds drift right while the original head becomes visibly sparse, and the depleted head remains on its stem.

#### 中文解释

> 固定近侧视：蓝天画面左侧有一个完整白色蒲公英。一阵稳定左向右风逐步吹落种子；同一批种子向右漂移，原花头明显变稀，稀疏花头仍留在茎上。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The full round seed head is intact at the left as the breeze begins moving right. | 完整圆形种子头位于左侧，微风开始向右。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The first seeds detach from the right-facing side of the same dandelion. | 第一批种子从同一蒲公英迎风侧脱离。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | More original seeds drift right while the attached seed head becomes progressively sparse. | 更多原有种子向右漂移，仍连接的花头逐渐稀疏。 | 本行英文原文就是 c+ |
| `completion` / 完成 | A loose seed cloud has moved right and only a depleted head remains on the stem. | 松散种子云移到右侧，茎上只剩稀疏花头。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | The sparse original head stays on its stem while its detached seeds continue rightward. | 稀疏原花头留在茎上，脱落种子继续向右。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P13</strong> — visible_actuator；全局语义 + 五阶段人话 c+</summary>

### P13 使用概要

- M3/I2V 首帧：`P13-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：A visible hand holds the pitcher so it can tilt and control the pour.
- 初始原因中文：一只可见的手持握水壶，以便倾斜并控制倒水。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed side-view photograph on a clean tabletop. One visible hand holds one upright transparent pitcher half filled with clear water at the left, with its spout aimed toward one empty clear glass at the right. The pitcher has not tilted and no stream exists yet. Both waterlines, the spout-to-glass gap, the whole glass, and the controlling hand are unobstructed and fully visible. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定侧视：手握半满透明水壶直立在左，壶嘴对准右侧空玻璃杯；此时没有水流，壶内水线、杯口和二者间隙均清楚可见。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed side view of one hand tilting one half-filled transparent pitcher toward one empty clear glass. A clear water stream enters the glass, the pitcher waterline falls while the glass waterline rises, then the hand stops pouring and both new levels remain visible.

#### 中文解释

> 固定侧视：一只手把半满透明水壶倾向空玻璃杯。透明水流进入杯中，水壶水位下降同时杯中水位上升；随后手停止倒水，两个新水位保持可见。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The hand holds the half-filled pitcher upright beside the empty glass, with no stream. | 手把半满水壶直立放在空杯旁，此时没有水流。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The hand tilts the pitcher and one clear stream begins entering the glass. | 手倾斜水壶，一股透明水流开始进入玻璃杯。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The pitcher waterline falls as the receiving glass waterline rises. | 水壶水位下降，接水杯水位同时上升。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The hand returns the pitcher upright and the water stream stops. | 手把水壶恢复直立，水流停止。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | The pitcher holds less water and the glass holds more, with both levels still and no stream. | 水壶剩水更少、杯中水更多，两个水位静止且没有水流。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P14</strong> — visible_actuator；全局语义 + 五阶段人话 c+</summary>

### P14 使用概要

- M3/I2V 首帧：`P14-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：A visible hand holds the juice bottle so it can tilt and control the pour.
- 初始原因中文：一只可见的手持握果汁瓶，以便倾斜并控制倒汁。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed side-view photograph on a clean table. One visible hand holds one upright tall clear bottle partly filled with bright orange juice at the left, with its mouth aimed toward one empty clear tumbler at the right. The bottle has not tilted and no stream is present. The orange liquid level, receiving glass, bottle mouth, connecting gap, and hand are fully visible with no obstruction. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定侧视：手握装有橙汁的高透明瓶直立在左，瓶口对准右侧空玻璃杯；尚无水流，瓶内液面、杯口和间隙清楚可见。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed side view of one hand tilting one tall clear bottle of bright orange juice toward one empty clear tumbler. One orange stream enters the tumbler, the bottle level falls while the tumbler level rises, then pouring stops and both new levels remain visible.

#### 中文解释

> 固定侧视：一只手把一瓶亮橙色果汁倾向空透明杯。一股橙色液流进入杯中，瓶内液面下降同时杯内液面上升；倒汁随后停止，两个新液面保持可见。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The hand holds the partly filled juice bottle upright beside the empty tumbler, with no stream. | 手把部分装满的果汁瓶直立放在空杯旁，此时没有液流。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The hand tilts the bottle and one orange stream begins entering the tumbler. | 手倾斜瓶子，一股橙色液流开始进入杯中。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The bottle's orange level falls as the receiving tumbler's level rises. | 瓶内橙汁液面下降，接收杯内液面同时上升。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The hand stops the pour and the orange stream disappears. | 手停止倒汁，橙色液流消失。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | The bottle holds less juice and the tumbler holds more, with both levels still and no stream. | 瓶内果汁更少、杯中果汁更多，两个液面静止且没有液流。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P15</strong> — visible_actuator；全局语义 + 五阶段人话 c+</summary>

### P15 使用概要

- M3/I2V 首帧：`P15-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：A visible hand holds the sand cup so it can tilt and control the grain flow.
- 初始原因中文：一只可见的手持握沙杯，以便倾斜并控制沙粒流。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed side-view photograph of one visible hand holding one upright clear cup filled with pale dry sand above a bare wooden table. The cup is centered over an empty landing area, still upright before pouring, and every grain remains inside with no stream or pile yet. Show the cup, hand, vertical drop path, and a broad tabletop area clearly and at useful scale. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定侧视：手把装满浅色干沙的透明杯直立悬在空木桌上；所有沙粒仍在杯中，尚无沙流或沙堆，完整下落路径清楚可见。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed side view of one hand tilting one clear cup of pale dry sand above a bare wooden table. A continuous grain stream leaves the cup, the cup empties while one conical pile grows directly below, then the stream stops and the pile remains settled.

#### 中文解释

> 固定侧视：一只手在空木桌上方倾斜装有浅色干沙的透明杯。连续沙粒流离开杯子；杯子变空同时下方圆锥沙堆长大，最后沙流停止，沙堆保持稳定。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The hand holds the full sand cup upright above a bare table, with no grain stream. | 手把装满沙的杯子直立悬在空桌上，没有沙流。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The hand tilts the cup and the first continuous grain stream begins falling. | 手倾斜杯子，第一股连续沙粒开始下落。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The sand level inside the cup falls while one conical pile grows directly below. | 杯内沙面下降，下方一个圆锥沙堆同时长大。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The cup becomes nearly empty and the final grains join the pile as the stream stops. | 杯子接近空，最后沙粒加入沙堆，沙流停止。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | One settled conical sand pile remains on the table beneath the emptied cup. | 一个稳定圆锥沙堆留在几乎倒空的杯子下方。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P16</strong> — gravity_or_instability；全局语义 + 五阶段人话 c+</summary>

### P16 使用概要

- M3/I2V 首帧：`P16-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：The compact dye drop is already falling under gravity toward the water surface.
- 初始原因中文：紧凑蓝色染料滴已经在重力下向水面下落。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed close side-view photograph of one compact dark-blue dye drop suspended just above the flat surface of still clear water inside one rectangular glass vessel. The drop is separate from the water and already falling downward under gravity, with no blue color inside the water yet. Show the complete vessel, waterline, drop, entry point, and broad clear water volume without obstruction. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定近侧视：一个紧凑深蓝染料滴悬在矩形玻璃容器的平静清水水面上方并正在下落；水中尚无蓝色，完整水体和入水点均可见。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed close side view of one compact dark-blue dye drop falling into still clear water inside one rectangular glass vessel. The same drop crosses the water surface and spreads outward through the water, forming one expanding blue region that remains inside the vessel.

#### 中文解释

> 固定近侧视：一个紧凑深蓝染料滴落入矩形玻璃容器内的平静清水。同一滴染料穿过水面后向外扩散，形成不断扩大的蓝色区域并留在容器内。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The compact blue drop falls toward the flat surface while the water remains clear. | 紧凑蓝色液滴向平坦水面下落，水仍然透明。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The same drop crosses the water surface at one visible entry point. | 同一液滴从一个清楚入水点穿过水面。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | Blue dye spreads outward and downward from that entry point through the clear water. | 蓝色染料从入水点向外、向下扩散到清水中。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The colored region grows broad while remaining continuous with the original dye. | 染色区域扩大，同时保持与原染料连续。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | One expanded blue region remains distributed inside the same glass vessel. | 一个扩大的蓝色区域分布并留在同一玻璃容器内。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P17</strong> — visible_actuator；全局语义 + 五阶段人话 c+</summary>

### P17 使用概要

- M3/I2V 首帧：`P17-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：A visible hand above the sponge supplies the downward pressing force.
- 初始原因中文：海绵上方一只可见的手提供向下按压力。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed three-quarter side-view photograph of one dry yellow rectangular kitchen sponge resting at full thickness on a clean white marble surface. One open visible hand is centered just above the sponge, aligned to press straight downward but not yet touching it. The sponge is uncompressed, fully visible, and surrounded by open tabletop space, with its top and side profile easy to compare later. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定三分之四侧视：一个干燥黄色矩形海绵以完整厚度放在白色大理石上，一只张开的手对准海绵但尚未接触，顶部和侧面均清楚可见。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed three-quarter side view of one hand pressing one dry yellow sponge straight down on a white marble surface. The same sponge becomes flatter under the palm; after the hand lifts, it gradually regains its original thickness and remains resting on the table.

#### 中文解释

> 固定三分之四侧视：一只手把干燥黄色海绵直压在白色大理石面上。同一海绵在掌下变扁；手抬起后它逐渐恢复原厚度并留在桌面。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The uncompressed sponge rests at full thickness while the open hand moves down above it. | 未压缩海绵以完整厚度静置，张开的手从上方下移。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The palm first contacts the sponge's top surface and begins pressing downward. | 手掌第一次接触海绵顶面并开始下压。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The same sponge becomes visibly flatter under the continuing hand pressure. | 同一海绵在持续手压下明显变扁。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The hand lifts clear and the sponge expands upward toward its original thickness. | 手完全抬起，海绵向上恢复到原厚度。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | The recovered sponge remains resting at full thickness on the marble surface. | 恢复后的海绵以完整厚度留在大理石表面。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P18</strong> — initial_momentum；全局语义 + 五阶段人话 c+</summary>

### P18 使用概要

- M3/I2V 首帧：`P18-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：The ball is already falling vertically toward the hard floor when the clip begins.
- 初始原因中文：红球在视频开始时已经沿竖直路径向硬地板下落。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed wide side-view photograph of one intact red rubber ball high above one hard gray floor, already moving straight downward along a clear vertical path. Keep the ball fully inside the upper part of frame and the entire floor and future bounce area visible below. The ball has not touched the floor, remains round, and is large enough to track through several progressively lower bounces. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定宽侧视：一个完整红色橡胶球位于硬灰地板上方并沿清楚竖直路径向下运动；尚未触地，完整地板和多次反弹区域都在画面内。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed side view of one red rubber ball already falling down a clear vertical path toward one hard gray floor. The same ball contacts the floor and rebounds several times to progressively lower heights, then stops at the impact area and retains its size.

#### 中文解释

> 固定侧视：一个红色橡胶球已经沿清楚竖直路径向硬灰地板下落。同一个球触地后反弹多次，每次高度逐渐降低，最后停在撞击区域并保持大小。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The intact red ball is already descending vertically toward the hard floor. | 完整红球已经竖直向硬地板下降。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The same ball reaches the floor for its first visible contact. | 同一个球第一次清楚接触地板。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The ball rebounds repeatedly, with each new peak lower than the previous peak. | 球多次反弹，每次新高点都低于前一次。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The final small rebound ends and the ball returns to the impact area. | 最后一次小反弹结束，球回到撞击区域。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | One round red ball remains at rest on the floor without changing size. | 一个圆形红球保持原大小并静止在地板上。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P19</strong> — visible_actuator；全局语义 + 五阶段人话 c+</summary>

### P19 使用概要

- M3/I2V 首帧：`P19-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：Two visible hands pull opposite edges of the notched paper to start and extend one tear.
- 初始原因中文：两只可见的手反向拉纸张两侧，从已有缺口开始一次撕裂。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed front-view photograph of one intact white paper sheet held taut by two visible hands at its left and right edges. A single small centered notch is already cut into the top edge, but the tear has not extended below that notch. Keep the full sheet, both gripping hands, the notch, and clear separation space inside one uncluttered frame so the future tear path remains visible. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定正视：两只手分别拉住一张完整白纸左右边缘；纸张上边中央只有一个小缺口，裂口尚未向下扩展，完整撕裂路径在画面内。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed front view of two hands pulling the left and right edges of one notched white paper sheet apart. A tear starts at the centered top notch, travels continuously to the opposite edge, and the same sheet becomes two pieces that remain separated.

#### 中文解释

> 固定正视：两只手向两侧拉一张带缺口白纸。裂口从上边中央缺口开始并连续延伸到相对边缘，同一张纸最终成为两片并保持分开。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | Both hands hold the intact sheet taut with only the small top notch visible. | 两只手拉紧完整纸张，此时只存在顶部小缺口。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The opposite pulls extend one tear downward from the existing notch. | 反向拉力使一条裂口从已有缺口向下延伸。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The same tear lengthens continuously as the two sides move farther apart. | 同一裂口连续变长，纸张两侧距离增大。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The tear reaches the opposite edge and the original sheet becomes two pieces. | 裂口到达相对边缘，原纸张成为两片。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | The two paper pieces remain visibly separated in the two hands. | 两片纸分别留在两只手中并保持明显分开。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

<details>
<summary><strong>P20</strong> — visible_actuator；全局语义 + 五阶段人话 c+</summary>

### P20 使用概要

- M3/I2V 首帧：`P20-i0-1280x704.jpg`（1280×704）。
- M4/T2V 首帧：不使用。
- 初始原因原文：A visible gloved hand holds the kettle so it can tilt and control the coffee pour.
- 初始原因中文：一只可见的戴手套手持握水壶，以便倾斜并控制咖啡流。
- 实际编码文本键：`global_semantic`、五条 `stages[].positive`、`cfg_negative` 与 `temporal_route.weights`。
- Global Semantic 全程生效；setup/onset/evolution/completion/terminal 分别切换到下表对应的 c+，相邻阶段用两个时间 token 平滑过渡。

### 首帧图像生成提示词（只用于生成/挑选 I0，不进入 Wan 视频 T5）

#### 英文原文

> A realistic fixed three-quarter-view winter photograph on snowy ground. At the right, one visible gloved hand holds one upright stainless-steel kettle containing dark coffee beside one empty gray camping cup, with the spout aimed at the cup but no stream yet. At the left, one small campfire is already flickering. Show the hand, kettle, cup interior, spout gap, and fire clearly in one uncluttered composition. One continuous 1280x704 landscape frame.

#### 中文解释

> 固定冬季三分之四视角：雪地右侧手持直立不锈钢水壶，壶嘴对准空灰杯但尚无液流；左侧小营火已在闪动，手、壶、杯口和火焰均清楚可见。

### Global Semantic（Wan 实际编码，全程基础条件）

#### 英文原文

> Fixed three-quarter view on snowy ground beside one flickering campfire. One gloved hand tilts one stainless-steel kettle over one empty gray cup; dark coffee flows into the cup, the cup level rises, pouring stops, and the same fire continues flickering at the left.

#### 中文解释

> 固定三分之四视角：雪地上一团营火持续闪动。一只戴手套的手把不锈钢水壶倾向空灰杯；深色咖啡流入杯中，杯内液面上升，倒液停止，而左侧同一团火继续闪动。

### CFG Negative（Wan 实际编码，同时作为空阶段 c-）

> `␠`

实际 JSON 值是一个 U+0020 空格，没有物体名词，也没有反事实句。这里用 `␠` 只是让不可见字符可读；运行时传入的仍是原始空格。

### 五阶段 c+（Wan 逐句编码并按时间注入）

| 阶段 | 英文原文 | 中文解释 | 该阶段实际局部文本 |
|---|---|---|---|
| `setup` / 准备/初始 | The gloved hand holds the kettle upright beside the empty cup while the campfire flickers. | 戴手套的手把水壶直立放在空杯旁，营火持续闪动。 | 本行英文原文就是 c+ |
| `onset` / 起始/触发 | The hand tilts the kettle and one dark coffee stream begins entering the cup. | 手倾斜水壶，一股深色咖啡流开始进入杯中。 | 本行英文原文就是 c+ |
| `evolution` / 演化 | The coffee level rises inside the cup while the same fire continues flickering at the left. | 杯内咖啡液面上升，左侧同一团火继续闪动。 | 本行英文原文就是 c+ |
| `completion` / 完成 | The hand returns the kettle upright and the coffee stream stops. | 手把水壶恢复直立，咖啡流停止。 | 本行英文原文就是 c+ |
| `terminal` / 终止/保持 | The filled cup remains on the snow beside the continuously flickering campfire. | 装有咖啡的杯子留在雪地上，旁边营火继续闪动。 | 本行英文原文就是 c+ |

### 不作为文本编码的辅助字段

- `initial_condition`：说明首帧为什么能发生运动。
- `entities`：给人和后续 StateBridge 数据准备使用；P0 不编码。
- `grounding.entity_boxes`：planimg 中保留的人工框；P0 只使用整张首帧，不注入框。
- `temporal_route`：直接使用 JSON 中的 5×25 数值权重，不从文字推断。
- `cplus-cminus.json`：旧长格式已停用；当前 c+ 就是 plan/planimg 中的 `stages[].positive`。

</details>

## 与 phase5 旧报告的关键差别

- 旧版确实为每个阶段分别提供 c+ 和 c-；但两边都塞入很长的实体清单、关系标签和反事实句。
- 当前保留“每阶段一个 c+”的时间结构，c+ 改为上表中的短英文人话；阶段 c- 为空。
- 旧版 setup/onset、evolution/completion 可能重复；当前五条 c+ 分别描述不同的可见进度。
- Global Semantic 继续全程说明完整事件，阶段 c+ 单独编码并局部注入，不会被长实体清单淹没。
- 过渡只使用 JSON 已保存的 5×25 数值权重；没有字符串匹配、模板扩写或 Python 文本改写。
