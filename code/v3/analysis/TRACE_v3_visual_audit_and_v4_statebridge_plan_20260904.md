# TRACE Writer v3 生成审计与下一轮最小化迭代方案

日期：2026-09-04  
范围：`v3/outputs/trace_writer` 中 3 组运行、P01–P20、共 60 个视频。  
结论先行：**目前首要问题不是物理约束“不够多”，而是把内部约束写成了过长、互相冲突、且含大量禁用名词的自然语言，再让 T5/DiT 自己猜这些逻辑。** 下一轮不应继续加提示词模板、负面句子、cap 或独立网络；建议改成“短而自然的语义提示 + 内部连续状态轨迹 + 一个共享的 StateBridge”。StateBridge 同时承担轻量写入和状态读出；Corrector 只是基于读出误差调节同一写入门，不是第三个网络。训练目标只增加一个统一的状态重建损失。

---

## 1. 审计方式与证据边界

本次逐视频检查了以下三组输出：

| 截图行 | 运行目录 | 模式 | 实际 CFG | 说明 |
|---|---|---:|---:|---|
| M3 / I2V | `trace-v3-m3-phase5` | 首帧条件视频 | 5.0 | 由 `reader.audit.jsonl` 中实际运行记录确认 |
| M4 / T2V run A | `trace-v3-m4-phase5` | 文生视频 | 3.5 | 同 seed=42 |
| M4 / T2V run B | `trace-v3-m4-phase5-2` | 文生视频 | 5.0 | 同 seed=42 |

每个视频为 1280×704、97 帧、24 FPS，约 4.04 秒。截图每隔 6 帧取一张，共 17 张，覆盖 frame 0–96；三次运行纵向对齐，因此可以直接观察动作是否缺段、重复、逆转或突变。完整截图位于 [`visual_audit_20260904/by_sample`](visual_audit_20260904/by_sample/)。生成脚本为 [`render_dense_contact_sheets.sh`](../tools/render_dense_contact_sheets.sh)。

注意两点：

1. 三组结果都只有 seed=42，下面是**逐帧证据审计**，不能冒充统计显著性结论。
2. 60 个 `trace.verify.json` 全部写着 `pass`，但当前 verifier 只检查张量有限性、相邻帧 MAD 和二阶差分，并明确写着语义检查未执行。因此这个 `pass` 不能说明对象、因果关系或物理过程正确，详见 [`verifier.py:42`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/verifier.py:42)。

---

## 2. 最关键的截图证据

### 2.1 P02：禁用词不是没有生效，而是被放进了正向语义通道

P02 原始 I0 只有红球、蓝球和球桌，[可查看原图](../../v1/demo/P02/P02-i0-1280x704.png)。但 M3 的生成 frame 0 已出现两根球杆状物；红球在约 frame 42 接触蓝球后于 frame 48 消失，frame 54 起手和人物进入。也就是说，球杆和人不是首帧继承误差，而是在生成时被引入。

![P02 三组运行逐帧对照](visual_audit_20260904/by_sample/P02_all_runs.jpg)

直接原因是 `cue, cue stick, hand, person, tool...` 并没有被隔离在一个安全的“负向通道”里：它们被写入全局正向 anchor、五个阶段的正向文本、以及每条违反文本的共享前缀。P02 的这些禁用词在实际编码文本中合计出现 **121 次**。模型未必可靠解析 `No additional ...` 的逻辑作用，却一定能读到被反复强调的名词。当前 CFG negative 实际为空字符串，因此不能把责任归结为传统意义上的 negative prompt；这是**正向上下文污染**。

### 2.2 P06：全局终态泄漏 + 后续旧状态重放，直接诱发“第二本书”

M3 中第一本书约在 frame 24 已经落地并打开；随后架子上又出现一本闭合书，约 frame 54–72 再次下落。最终同时保留落地书与后来出现的书。这与用户观察完全一致，不是轻微纹理漂移，而是同一逻辑实体被模型实现成多个时相副本。

![P06 三组运行逐帧对照](visual_audit_20260904/by_sample/P06_all_runs.jpg)

其机制很明确：全局提示从第一帧起就写入“书最终打开并留在地板”，而固定阶段路由稍后又注入“闭合书在空中下降”。当模型同时满足全局终态和局部旧状态时，最容易的视觉解就是保留已完成的书，再生成一个闭合书继续执行。现有五阶段不是在跟踪同一本书的连续状态，而是在轮流激活五段文字。

### 2.3 P01：初始图方向错误，之后又出现旧状态与新状态共存

P01 原图中踏板车前轮/车把在左、垃圾桶在右，车实际背向垃圾桶，[可查看原图](../../v1/demo/P01/P01-i0-1280x704.png)。任务本来只要求直行碰撞，I0 却额外制造了“先掉头”的难题。M3 后半段又出现一辆仍直立的旧车和一辆倒下的新车，说明状态转换被实现成了状态复制。

![P01 三组运行逐帧对照](visual_audit_20260904/by_sample/P01_all_runs.jpg)

M4 还把 `scooter` 多次理解成坐式摩托踏板车，而非图中的站立式电动滑板车，说明仅靠一个歧义名词不足以固定类别。下一版应写成 `black stand-up electric kick scooter`，I0 则直接生成车头朝向垃圾桶的侧视布局。

### 2.4 P14：一个字符串匹配错误，能在画面里留下可见痕迹

P14 的橙汁任务被自动加入 `dropper, pipette, external liquid stream, hand, person, tool` 等禁用类别。原因不是计划真的提到了冰，而是代码用 `"ice" in text` 做子串判断，`juice` 的结尾正好包含 `ice`，详见 [`entity_ledger.py:99`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/entity_ledger.py:99)。这些词又在实际上下文中合计出现 **109 次**。M3 随后出现蓝色滴管/工具；M4-A 则直接产生两个杯子、预装液体和滴管式交互。

![P14 三组运行逐帧对照](visual_audit_20260904/by_sample/P14_all_runs.jpg)

这组证据很重要：它表明“禁用词反向增强”不是纯理论猜测，而是从字符串规则、编码文本到可见生成物之间存在完整因果链。

---

## 3. P01–P20 全量逐帧诊断

下表中的“好”只表示该次运行在当前视觉审查下较完整，可作为回归保护，不表示已经通过严格物理测量。

| 样例 | M3 / I2V | M4-A / T2V 3.5 | M4-B / T2V 5.0 | 核心判断 |
|---|---|---|---|---|
| P01 踏板车碰桶 | I0 背向目标；约 f30–48 生成新车状态；末尾直立车与倒地车并存 | 生成坐式摩托和人物腿，未完成侧倒 | 类别仍错；发生碰靠但不是清晰侧倒 | 初始几何错误 + 类别歧义 + 状态复制 |
| P02 台球碰撞 | f0 已有球杆；约 f42 接触，f48 红球消失，f54 后手/人进入 | 手、球杆从开头存在，运动传递很弱 | 两球开头已接触，手/杆进入，无清楚碰撞过程 | 禁用词污染、结果预置、实体消失、缺少合法初始动量表述 |
| P03 网球与球拍 | 球进入拍网后像被黏住，反弹轨迹弱，球拍漂移 | 球拍约 f54 才出现；约 f78 同时有空中球和地面球 | 有球拍与运动，但接触和离开路径仍不清楚 | 交互物晚出现、中段复制、接触后响应不足 |
| P04 多米诺 | 手出现；骨牌主要靠消失而非依次倒下，末尾还剩一块直立 | 前几块倾斜，后排基本不传播，背景有人 | **较好**：f36–54 连续传播，倒下状态保持 | M4-B 是应保护的正例；需要合法触发而非禁止手 |
| P05 方块下坡 | 方块 f18–48 消失，f54 在坡上重现后才滑 | 方块缺失、坡道裁切，后段有人脚 | 坡道消失，方块约 f42 在地面凭空出现 | 典型“删除中段—目标处复活”，非连续滑动 |
| P06 书本掉落 | 第一本约 f24 已打开落地，架上又生成闭合书并再次掉落 | 落下并打开较完整，但后段人物腿进入 | 能落下但基本保持闭合 | 全局终态泄漏 + 旧状态重放，是阶段设计失败的代表 |
| P07 箱子倾倒 | **较好**：单一箱体连续倒下并保持 | 像悬浮/倚靠，未真正稳定落地 | 倒下尚可，但 f72 后又抬起/回摆 | M3 是正例；M4-B 有终态逆转 |
| P08 冰块融化 | f18 后出现滴管并生成液池；冰几乎不融，液体又回缩 | 全程外部水柱，手臂进入，冰不融 | 开头就是两块冰且一块在盘外，几乎不融 | 用“倒液体”替代相变；禁用词恰好提示了错误捷径 |
| P09 黄油融化 | 固体快速变成过大的黄色液池；早期有细外流 | 固体和预存液体共存，后段出现外流 | 外部液流浇在黄油上，固体大体保留 | 相变被偷换成外部加液；源/结果面积也不守恒 |
| P10 气球充爆 | 膨胀很弱，约 f36 突然破裂；缺少充气原因和连续膨胀 | 手进入后变成灰色碎块并出现额外气球 | 破成大块布/枕状物，无可信膨胀 | I0 是系口气球却要求继续充气，任务条件自身矛盾 |
| P11 瓶子坠落破碎 | **较好**：落下、撞击、碎片保留；但完整瓶与碎片短暂共存 | 开头瓶已在地面，破后人物处理碎片 | 瓶从顶部出现，落地倾斜但多数保持完整 | M3 可作碎裂正例；T2V 初始布局与相变不稳 |
| P12 蒲公英飞散 | 大量种子右移，但花头几乎仍饱满，源不减少 | 只散少量种子，花头仍满 | 从 f0 起有两个花头，且几乎不耗散 | 源端没有随输出减少；M4-B 还发生对象复制 |
| P13 倒水 | 手进入后倾倒与杯中水位上升较好；末帧仍在持续倒，源水位下降偏弱 | 壶到 f78 左右才出现并开始倒，动作太晚 | f0 已经倾斜倒水，缺少 setup；末尾仍不停 | 正确 transfer 雏形已有，但阶段起止和终止条件错误 |
| P14 倒橙汁 | 倾倒/液位上升可见，但出现蓝色滴管且液流末段变形 | f0 已有两个杯子且都有液体，又由瓶子和滴管操作 | 同样两个杯子，其中一个预先装满 | `juice`→`ice` 子串误判；对象复制与结果预置严重 |
| P15 倒沙 | 原始 I0 无沙流，但生成 f0 已经在倒；沙堆增长，动作不结束 | 杯子竖插在预成沙堆里，后续出现壶状物 | **较好**：有连续沙流、沙堆增长、源约 f60 离开且终态稳定 | M4-B 是 transfer/沉积正例；M3 有 outcome preset |
| P16 染料扩散 | **较好**：液滴过水面后蓝色扩散；略快且深色滴残留偏久 | 玻璃容器消失，变成宏观蓝色二维场/悬挂流 | 同样缺失容器并语义崩坏 | M3 是扩散正例；M4 表明 T2V 布局约束尚不可靠 |
| P17 海绵压缩 | 手只轻压，随后把海绵整个提离桌面，没有回弹终态 | 海绵基本不变，后段有额外粉色手状物 | 压缩/恢复相对清楚，但开头竖放、末态尺寸偏小 | 作用—形变—释放—恢复链只在 M4-B 较完整 |
| P18 球反弹 | 球下落后直接停住，没有衰减反弹 | 前半空白，f48 才凭空出现球，后段落下再升起 | 下落过程中持续缩成小点 | 中段动力学缺失；T2V 有物体迟生与尺度/身份丢失 |
| P19 撕纸 | 无执行者，裂缝自行张开，未形成两片分离 | 纸只翻动/压平，没有撕裂 | 纸先转成侧面，再突然跳成两片，无撕裂中间态 | I0 没有手或夹具；原因缺失迫使模型跳变 |
| P20 壶向杯中倒液体 | 壶无支撑悬浮倾倒；火焰局部保持较好，但倒水不结束 | 有手持壶，杯和火较稳，整体较好但从开头就在倒且无结束 | f0 已有两个杯子，填充关系不清，壶约 f48 离开 | 原 I0 缺少持壶者；M4-A 提示“允许并声明执行者”优于禁用它 |

完整对照图可逐项展开：

<details><summary>P01–P05</summary>

![P01](visual_audit_20260904/by_sample/P01_all_runs.jpg)

![P02](visual_audit_20260904/by_sample/P02_all_runs.jpg)

![P03](visual_audit_20260904/by_sample/P03_all_runs.jpg)

![P04](visual_audit_20260904/by_sample/P04_all_runs.jpg)

![P05](visual_audit_20260904/by_sample/P05_all_runs.jpg)

</details>

<details><summary>P06–P10</summary>

![P06](visual_audit_20260904/by_sample/P06_all_runs.jpg)

![P07](visual_audit_20260904/by_sample/P07_all_runs.jpg)

![P08](visual_audit_20260904/by_sample/P08_all_runs.jpg)

![P09](visual_audit_20260904/by_sample/P09_all_runs.jpg)

![P10](visual_audit_20260904/by_sample/P10_all_runs.jpg)

</details>

<details><summary>P11–P15</summary>

![P11](visual_audit_20260904/by_sample/P11_all_runs.jpg)

![P12](visual_audit_20260904/by_sample/P12_all_runs.jpg)

![P13](visual_audit_20260904/by_sample/P13_all_runs.jpg)

![P14](visual_audit_20260904/by_sample/P14_all_runs.jpg)

![P15](visual_audit_20260904/by_sample/P15_all_runs.jpg)

</details>

<details><summary>P16–P20</summary>

![P16](visual_audit_20260904/by_sample/P16_all_runs.jpg)

![P17](visual_audit_20260904/by_sample/P17_all_runs.jpg)

![P18](visual_audit_20260904/by_sample/P18_all_runs.jpg)

![P19](visual_audit_20260904/by_sample/P19_all_runs.jpg)

![P20](visual_audit_20260904/by_sample/P20_all_runs.jpg)

</details>

必须保留的正向基线是：P04 M4-B 的连续多米诺、P07 M3 的单箱倾倒、P11 M3 的坠落破碎、P13 M3 的液体传递主体、P15 M4-B 的沙流沉积、P16 M3 的入水扩散，以及 P20 M3 对火焰区域的局部保持。新方法如果只修坏例却破坏这些正例，也不应接受。

---

## 4. 提示词问题不是主观感受：长度、重复和通道位置都可量化

使用项目本地的 UMT5 tokenizer 对 P01–P20 的实际文本重新计数：

| 项目 | 原始计划的人话文本 | v3 实际编码文本 | 变化 |
|---|---:|---:|---:|
| 全局 semantic，20 例均值 | 28.15 tokens | 283.90 tokens | 每例膨胀倍数均值 10.18× |
| 阶段 positive，100 段均值 | 21.20 tokens | 162.72 tokens | 约 7.68× |
| 阶段共享 inventory/protection | — | 126.29 tokens | 占完整阶段 positive 的平均 78.37% |

其他可复现事实：

- 20 个原始全局提示范围仅 22–36 tokens；编译后为 245–349 tokens。
- 100 个阶段槽位中，有 **35 个是同一样例内已有关系文本的额外重复**。P01/P02 等都出现 setup≈onset、evolution≈completion。
- P02 禁用类别相关词在全部实际编码上下文中合计 121 次；P08 和被误判的 P14 都是 109 次。
- `c1`–`c5` 主要留在 `violation_constraint_ids` 元数据里，并没有直接作为大段 `c1 c2...` 喂给 T5；真正污染 T5 的是这些编号被展开后的通用模板、重复 inventory 和禁用名词。用户对“机器味”的感受是对的，但应修正准确对象。

更关键的是：计划文件本来已经有人话阶段描述，v3 却在严格 minimal-pair 路径中把它丢掉。例如 P06 原始五段依次是“书靠近架边—越过架边—同一本闭合书连续下降—接触地板时打开—保持打开”，非常清楚且互不重复；但 [`minimal_pair.py:121`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/minimal_pair.py:121) 只有在找不到约束或报错时才回退使用 `stage.positive`，正常路径反而调用 [`minimal_pair.py:40`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/minimal_pair.py:40) 的通用句式，最终得到：

> The declared cause occurs BUT the required target change does not become visible.

这不是可视场景描述，也没有说“哪一个物体在什么位置发生什么变化”。它更像给符号验证器看的错误标签，不应作为视频模型的生成文本。

### 当前链路中的具体错误

| 代码位置 | 实际行为 | 生成后果 |
|---|---|---|
| [`semantic_anchor.py:74`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/semantic_anchor.py:74) | 把 inventory、初态、整个事件、保护项、禁用名词、conservation、终态串成一个全时段正向 prompt | 主动作被十倍长的附加文本淹没；终态从第一帧起泄漏 |
| [`semantic_anchor.py:110`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/semantic_anchor.py:110) | 每一阶段再次复制 inventory、对象计数、禁用名词与通用保护句 | 共享文本占约 78%，阶段真正差异很弱 |
| [`minimal_pair.py:40`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/minimal_pair.py:40) | 用 `declared change / terminal facts / required target` 等模板替代原有人话阶段 | 模型收到抽象逻辑句，不是可直接成像的动作 |
| [`minimal_pair.py:141`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/minimal_pair.py:141) | 正/反事实都拼接同一个巨大 shared anchor | “负分支相减”无法可靠消除共享名词，反而形成纠缠残差 |
| [`entity_ledger.py:72`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/entity_ledger.py:72) | 五阶段状态固定映射成 `source, source, middle, target, target` | 只有两态时 middle 就已经等于 target，evolution 提前跳到完成态 |
| [`entity_ledger.py:91`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/entity_ledger.py:91) | 用宽泛子串推导禁用对象；`ice` 会命中 `juice` | P14 被错误注入滴管、工具、人等概念 |
| [`entity_ledger.py:232`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/entity_ledger.py:232) | 未受影响的 carrier 被设为 count=0/forbidden，但仍附上状态事实 | P13 terminal 同时说 stream 不存在且 `stream_state=continuous_clear_stream`，内部自相矛盾 |
| [`entity_ledger.py:254`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/entity_ledger.py:254) | 所有任务默认 `external_source_allowed=false` | 手、风、热、初始冲量等合法原因被一刀切禁止，任务本身失去因果来源 |
| [`dynamic_support.py:14`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/dynamic_support.py:14) | 从英文词猜固定 ±0.18 位移；M4 无框时回退全图/正向算子显著性 | support 不是实体轨迹；被污染的文本还会把显著性引向错误概念 |
| [`temporal.py:36`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/temporal.py:36) | 按 transition 数和固定模板划 25 个 latent frame，只做相邻 crossfade | 不观察视频是否已完成；完成后仍会按时钟重放旧阶段 |
| [`controllers.py:9`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/controllers.py:9) | `AuditReader` 只记录范数、cosine 和 cap | 它不是读取对象/接触/进度的视觉 Reader，不能纠正重复或漏执行 |
| [`verifier.py:42`](/home/liuzhirui/Project/physGen/code/v3/trace_writer_v3/verifier.py:42) | 只测像素变化统计并无条件返回 `pass` | 60/60 pass 掩盖了可见的语义与物理失败 |

### Reader 日志也说明：继续叠加 cap 不是方向

| 运行 | 正/反 cross-attn cosine 均值 | `||正-反|| / ||正||` | 阶段边界 cosine 均值 / 最小值 | token cap 触发率 | layer/group cap | 最终 CFG 残差保留比例 |
|---|---:|---:|---:|---:|---:|---:|
| M3 / CFG 5.0 | 0.9946 | 0.0972 | 0.2727 / -0.9949 | 0.00014% | 1.0 / 1.0 | 16.46% |
| M4-A / CFG 3.5 | 0.9941 | 0.1010 | 0.2408 / -0.9978 | 0.00046% | 1.0 / 1.0 | 39.64% |
| M4-B / CFG 5.0 | 0.9943 | 0.1003 | 0.2361 / -0.9973 | 0.00041% | 1.0 / 1.0 | 28.50% |

正反上下文 cosine 接近 0.995，说明二者绝大部分都是相同 anchor；相减只剩一个小而高度上下文化的差，不是干净的“物理关系方向”。token/layer/group cap 几乎完全没触发，真正限制结果的是最后的 CFG cap。阶段边界最小 cosine 接近 -1，说明有些相邻操作方向近乎相反，简单 crossfade 不能把矛盾变成连续动作。与其再加第四种 cap 或额外 loss，应该先改变条件表示。

此外，产物 manifest 只保存了 `config.manifest()`，没有保存实际 `guide_scale`、solver、steps、shift 等采样参数，见 [`infer_trace_writer.py:432`](/home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer.py:432)。本次只能从 Reader 日志还原 M3=5.0、M4-A=3.5、M4-B=5.0。下一轮必须把完整采样参数写进 manifest，否则相同目录标签和 seed 也无法公平复现。

---

## 5. 先修任务与首帧的因果可行性，不让模型替我们补原因

“没有外部东西”不是物理规律。更合理的是把原因分成四类，并明确到每个任务：

1. **继承的初始动量**：视频开始时对象已经在运动，之前的施力可以发生在镜头外。适合 P01、P02、P03。提示词应明确 `already rolling/moving when the clip begins`，而不是一边说静止画面、一边禁止一切外力。
2. **重力与支撑释放**：对象已处于不稳定位置，随后受重力运动。适合 P05、P06、P07、P11、P18。I0 必须让重心、边缘和路径合理。
3. **环境场**：风、热、扩散等不一定需要画出人或工具，但必须在事件语义里声明。适合 P08、P09、P12、P16。
4. **可见执行者/装置**：手、夹具、泵嘴、持壶者不是“额外错误物体”，而是事件需要的因果实体。适合 P04、P10、P13、P14、P15、P17、P19、P20。若不想出现人，就应在 I0 中提供机械夹具或让动作已合法开始，而不是简单写 `no hand/person/tool`。

建议在 I0 进入视频模型前做一个**规则式可行性闸门**，不新增训练模块，只检查：

- 运动主体是否朝向目标，是否有足够直线路径和画面空间；
- 必需的触发者、初始动量、重力不稳定性或环境场是否至少有一种成立；
- I0 是否已经出现本应后续才发生的接触、液流、破碎或结果；
- 实体数量、类别与计划是否一致；
- 4 秒时长内是否能完成事件并留出稳定终态。

当前应直接返工的首帧包括：P01（方向相反）、P10（气球已经系口却要继续充气，且无泵嘴）、P19（纸自行撕裂，无手/夹具）、P20（壶自行悬浮倾倒）。P04/P13/P14/P15/P17 也应在“明确允许可见执行者”与“把动作设置为合法的初始进行态”之间二选一。P02 则不需要球杆出镜，只需明确红球在视频开始时已经具有向右初速度。

---

## 6. 推荐 v4：Semantic–State Split + 一个共享 StateBridge

### 6.1 只保留两条条件通道

```text
人话语义（T5，短） ───────────────────────────────► Wan 原生语义 cross-attention

内部状态轨迹（非文本） ─► shared StateBridge ─────► 局部、时序门控的 DiT residual
                              │
                              └─ 同一特征上的 State Reader
                                      │
                                      └─ 确定性误差门控（Corrector，不是新网络）
```

**语义通道**只回答“画面是什么、动作是什么、摄影机怎样”，建议 35–60 T5 tokens：

- 不放 `[ENTITY INVENTORY]`、`[VALID RELATION]`、`c1–c5`、`gap=contact`、`red_motion=slower_right`；
- 不列举 `cue/hand/person/tool...` 等禁用名词；
- 不从第一帧就写完整终态清单；
- 必要的稳定性用简短正向闭合句表达，例如 “Both balls remain visible throughout” 和 “The camera stays fixed”。

**状态通道**不再是英文 prose，而是每个逻辑实体随 latent time 的小型数值轨迹：

```text
entity slot = {
  identity prototype, presence,
  coarse box/tube,
  role,
  normalized event progress p*(t),
  start/target state vector,
  relation edge: contact / support / source→sink
}
```

其中 `identity prototype` 在 M3 中直接从 I0 的已有 entity box 内池化早期 DiT/VAE 特征；同一 slot 从头到尾只对应同一实例。`presence` 通常固定为 1，破碎类则仍是同一父 slot 的形态变化，而不是创建“完整瓶 + 碎片”两个独立对象。旧状态和新状态在一个 slot 内插值，模型没有机会把它们当成两件物体。

五段文字不再直接控制模型。界面上仍可显示 setup/onset/evolution/completion/terminal 方便人审，但内部由真实 transition 数生成连续 `p*(t)`：

- 两态任务：初态 → 连续过渡 → 终态；
- 三态任务（如闭书→下落→打开、未充气→膨胀→破裂、压缩→恢复）：按两条有序 transition 形成两段单调轨迹；
- terminal 时间段只接收终态向量，绝不重新激活 setup/evolution 文本。

这比固定 `source, source, middle, target, target` 更贴近事件，也直接解决“动作没做完就切段”和“做完又重做”。

### 6.2 一个模块，三种职责，而不是 Writer/Reader/Corrector 三套网络

StateBridge 在少量中层 DiT block 复用同一组轻量权重。对实体 `e`、时间 `t`、位置 `(x,y)`：

```text
Δh_l(t,x,y) = α_l · M_e(t,x,y) · G(p*(t), error) · Bridge(h_l, z_state[e,t])
```

- `M_e` 是实体 tube 的软掩码；I2V 从 I0 的实体原型在保守运动走廊内做特征亲和传播，不再用“被提示词污染的 positive cross-attention norm”找目标。
- Writer 是 `Bridge(...)` 的局部残差写入。
- Reader 从同一个 `M_e ⊙ h_l` 池化特征预测一个归一化状态向量 `ŝ(e,t)`，包含 presence confidence、位置/尺度、事件进度、接触或 source→sink 进度。
- Corrector 只是 `error = p*(t) - p̂(t)` 等读出误差经过 EMA、迟滞和单一幅度截断后调节 `G`；没有新参数、没有新 loss。Reader 未校准前关闭反馈，只运行 open-loop Writer。

这里应特别避免把“视频时间推进”和“扩散 denoise step 推进”混为一谈：`p*(t)` 沿 25 个视频 latent frame 单调，Reader 在每个噪声步读取整段视频的逐时状态；它不再像全局 stage clock 那样在 denoise 过程中切换整段提示词。

### 6.3 只增加一个辅助损失

冻结 Wan 主干，先训练 StateBridge 与共享 Reader head：

```text
L_total = L_diffusion + λ_state · Huber(ŝ(e,t), s*(e,t))
```

所有可读状态先归一化成同一个数值向量，因此只有一个 mask-weighted Huber 状态损失；identity 由固定 slot/prototype 和单实例 tube 结构保证，局部性由 `M_e` 架构保证，Corrector 不另设损失。不要再加“对象数损失 + 接触损失 + 时序损失 + 守恒损失 + 背景损失”的组合。

训练最多分两步：

1. **Teacher-forced open loop**：真实/合成短视频提供实体 tube 与状态轨迹，随机噪声步训练 Bridge；先以碰撞、掉落、倾倒、相变、弹性恢复等简单物理原语覆盖当前 20 类任务。
2. **少量 rollout 校准 Reader**：加入当前常见失败——中段消失、旧状态复制、终态逆转、原因缺失、source/sink 不同步——但把它们标成状态轨迹误差，不写成负面名词 prompt。只有 held-out 校准通过后才开启确定性反馈门。

相关工作说明“事件链、显式物理信号、token/局部条件”是合理方向，但本方案刻意压缩复杂度：Wan 的基础架构见 [Wan 技术报告](https://arxiv.org/abs/2503.20314)；事件中心链式表示可参考 [CoECT](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Chain_of_Event-Centric_Causal_Thought_for_Physically_Plausible_Video_Generation_CVPR_2026_paper.html)；用合成物理原语与显式目标信号训练可参考 [Goal Force](https://openaccess.thecvf.com/content/CVPR2026/html/Gillman_Goal_Force_Teaching_Video_Models_To_Accomplish_Physics-Conditioned_Goals_CVPR_2026_paper.html)。[ProPhy](https://arxiv.org/abs/2512.05564) 采用更复杂的 token-level expert 路线，本项目当前没有必要照搬多专家结构。

### 6.4 物理约束应成为类型化数据边，不再成为长句

只需少量事件类型，每类字段不同，不要把一切都写成 `source loss and result gain`：

| 事件类型 | 内部需要的关系 | 不应再写的通用话术 |
|---|---|---|
| 刚体碰撞 | 同实例、刚性形状、无穿透、接触先于响应、动量方向 | “declared cause / required target” |
| 倾倒/坠落 | 支撑解除、连续轨迹、接触地面、终态不反转 | “source loss / result gain” |
| 液体/颗粒传递 | source level↓、短时 stream、sink level↑、结束后 stream=0 | 在 terminal 同时写 stream=0 和 continuous stream |
| 相变 | 同一材料 slot，固态比例↓、液态比例↑，没有外部液流 | 列举 dropper/pipette 作为禁用词 |
| 破碎/撕裂 | 父 slot 完整度↓、碎片/两片状态↑、完整体不重现 | 把碎片误判成额外对象 |
| 弹性形变 | 施力→压缩，卸力→恢复，尺寸变化局限于对象 tube | 五段固定时钟反复施力 |

保护背景也不必每阶段重复 “camera/background unchanged”。在实体 tube 外不施加 StateBridge residual，原生 Wan 语义通道只用一句 `Static camera`，即可把“局部优化”落实为结构而非口号。

---

## 7. 人话提示词示例：P01、P02、P06

这些阶段句不再一次性拼进全局提示词。P0-R2 的最终决定是：保留短全局句作为全程基础，
再把五句分别作为 setup/onset/evolution/completion/terminal 的 c+ 单独编码；每次只在对应的
DiT 时间 token 上局部注入，相邻阶段用两个 token 过渡，阶段 c- 为空。下面是设计示例；
P01–P20 实际逐字使用的最终原文见
[P0_direct_prompt_usage_bilingual_20260904.md](/home/liuzhirui/Project/physGen/code/v3/analysis/P0_direct_prompt_usage_bilingual_20260904.md)。

### P01

**全局：**

> Static side-view shot of one black stand-up electric kick scooter and one upright black slatted trash can on a gray street. The scooter starts on the left with its front wheel pointing toward the can. It rolls straight into the can, tips onto its side, and stays still. The can remains upright.

**五个可读检查点：**

1. The upright scooter rolls straight toward the can, steadily closing the road gap.
2. Its front wheel touches the can; only then does the same scooter begin to tip.
3. The scooter rotates onto its side while its forward motion slows.
4. It finishes falling and settles beside the unchanged upright can.
5. One scooter lies still beside one upright can; neither object is replaced or copied.

### P02

**全局：**

> Fixed side view of one glossy red billiard ball already rolling right toward one stationary glossy blue ball on green cloth. The red ball hits the blue ball once. Both balls remain visible; afterward the blue ball rolls ahead to the right while the red ball follows more slowly.

**五个可读检查点：**

1. The red ball rolls right across the cloth while the blue ball stays still.
2. The two ball surfaces touch once.
3. At contact, the blue ball starts moving right as the red ball slows.
4. A green gap opens again with the blue ball leading.
5. Both original balls remain visible and separated, moving right at different speeds.

这里完全不需要写 `no cue, no cue stick, no hand, no person, no tool`。初始速度已经给出了合法原因，闭合句只强调“原来的两球持续可见”。

### P06

**全局：**

> Static side view of one closed blue-edged book at the end of a wooden shelf. The same book slides off, falls continuously through the open space, hits the wooden floor, opens once, and remains open there.

**五个可读检查点：**

1. The single closed book slides toward the shelf edge.
2. The same book clears the shelf before it starts descending.
3. It falls continuously through the empty space; no book remains on the shelf.
4. On floor contact, that same book opens and settles.
5. The single book remains open and still on the floor.

注意最后一句中的“no book remains on the shelf”是对已声明实体的正向去占用关系，不是罗列一串新物体名词。更稳妥的主控制仍应来自 slot 的 presence/tube，而不是依赖 T5 理解否定。

---

## 8. 最小实验矩阵与停止条件

不要直接训练一个复杂闭环然后只看新视频。建议四格实验，其中只有 E2 需要新增可训练参数：

| 实验 | 人话 semantic | v3 正/反长文本 Writer | StateBridge | Reader 反馈 | 目的 |
|---|---:|---:|---:|---:|---|
| E0 当前 v3 | 否 | 是 | 否 | 否；只有 audit | 现有参考 |
| E1 Clean baseline | 是 | 关闭 | 否 | 否 | 单独测量当前长提示/残差是否在伤害 Wan |
| E2 Open-loop | 是 | 关闭 | 是 | 只读不反馈 | 验证结构化状态注入本身是否有效 |
| E3 Closed-loop | 是 | 关闭 | 与 E2 同一个 | 校准合格后开门控 | 验证纠偏是否额外改善且不破坏画面 |

实验纪律：

- 第一轮只主攻 M3/I2V；它有 I0 和 entity boxes，能把“状态控制能力”与“T2V 布局生成失败”分开。M4 先作为压力测试，不能与 M3 混成一个结论。
- 每格至少 3 个固定 seed，并把 CFG、steps、solver、shift、模型哈希完整写入 manifest。
- 有效 I0 与返工 I0 分开测试：固定原图比较 E0–E3；另开一列比较原 I0 与因果可行 I0，避免把首帧改善错误归功于 StateBridge。
- 首轮重点样例：P01、P02、P06（身份/阶段）、P07（正例保护）、P13（source→sink）、P16（扩散正例）、P20（局部火焰与合法执行者）。

离线评估只需要一套 evaluator，不进入生成模块：实体存活/复制、事件顺序与中间态覆盖、终态保持/逆转、对象 tube 外背景变化、人工物理可读性。当前 MAD verifier 只能保留作数值健康检查，不能再命名为物理通过。

建议的 go/no-go：

1. 若 E1 明显优于 E0，先完成提示词清理，不要急着训练更多东西。
2. 若 E2 不优于 E1，说明 StateBridge 的表示或注入位置无效；停下修单模块，不加第二个模块补救。
3. Reader 在 held-out 视频上的归一化 progress MAE 未低于 0.10、presence/状态检测未稳定跨 seed 前，E3 一律关闭。
4. E3 必须减少 P01/P02/P06 的复制、消失和重放，同时不能使 P04/P07/P11/P13/P15/P16 的正例成功率下降超过 5%；否则反馈门回退为只读。

---

## 9. 实施优先级

### P0：无需训练，先做正确

> 2026-09-04 已完成 P0-R2 人话分阶段直接条件基线。实际改动、验证结果和仍然不能保证的部分见
> [P0_direct_json_changes_20260904.md](/home/liuzhirui/Project/physGen/code/v3/analysis/P0_direct_json_changes_20260904.md)。

- T5 使用一条人话全局 prompt 和五条互不重复的人话阶段 c+；删除所有 bracket 模板、
  状态键值串和禁用名词清单。
- 直接使用 plan/planimg 中的 `stage.positive` 英文原文，不再用 generic minimal-pair prose
  替换它；阶段 c- 为空。
- 恢复 25 个 DiT 时间 token 的五阶段分配与每个边界两个过渡 token，但权重直接存入 JSON，
  Python 不根据文字重新推断。
- 修正 `ice`/`juice` 子串问题；更重要的是停止从关键词自动推导执行者禁用表。
- 增加 cause/actuator 类型和 I0 可行性闸门；重做 P01/P10/P19/P20 首帧。
- manifest 保存实际采样参数；verifier 的 `pass` 改名为 `numeric_pass`。

### P1：唯一的训练增量

- 实现并训练共享 StateBridge + State Reader head；一个 `L_state`。
- 先跑 E2 open-loop，证明同一实体 slot、连续进度和局部 tube 有效。

### P2：有条件开启

- Reader 校准通过后，开启同一 Bridge 的确定性误差门控。
- M3 稳定后才解决 M4 的无首帧实体原型和布局初始化问题。

明确不做：不继续扩写 `[STAGE ENTITY INVENTORY]`；不设计五个阶段各自的网络；不建立 Writer/Reader/Corrector 三套独立参数；不增加对象数、接触、守恒、背景、时序五个 loss；不继续依赖负面名词提示；不靠更多 cap 掩盖表示问题。

---

## 最终建议

下一轮的核心不是“让提示词更会讲逻辑”，而是承认 T5 提示词不适合承载实体账本、互斥状态、守恒关系和连续进度。自然语言只负责让模型看懂场景与动作；物理状态通过一个共享 StateBridge 在 DiT 内部按实体、时间和空间局部注入。Reader 读取同一状态向量，Corrector 只调同一写入门。这样既正面回应了 P02 的禁用词反向增强、P06 的状态留存/动作重放、P01 的首帧布局错误，也把复杂度严格控制在**一个新增组件、一个新增损失、一个可选的无参数闭环**。
