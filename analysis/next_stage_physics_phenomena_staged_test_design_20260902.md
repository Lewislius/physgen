# 下一阶段物理现象分阶段测试与 Conditioning 判定方案

日期：2026-09-02  
适用代码：`Project/physGen/code/v1`  
主线方法：M3（I2V）；M4（T2V）仅在 M3 得到稳定结论后做迁移验证  
目标：先判定 semantic conditioning 的真实贡献，再按物理复杂度验证输出时间分阶段、局部作用域和守恒关系；不能用画面清晰或单个 seed 的成功代替物理过程成立。

## 1. 结论与决策

对当前 `conditioning.py` 的直接判断是：**元数据是需要的，但当前 `compiled` semantic 的完整拼接形式不应直接作为已确认的最终方案。**

需要保留的是元数据表达的事件契约：实体身份、初态、变化变量、保持项、正确因果关系和最近反事实。尚未被证明需要的是把这些内容全部写成一个长的、全时域共享的 semantic prompt。建议立即冻结继续扩写，先做严格消融。

当前结论的证据强度如下：

| 判断 | 置信度 | 依据 |
|---|---:|---|
| 当前 full compiled semantic 没有被单独证明有效 | 高 | 新旧实验同时改变了 semantic、c+/c- 构造、CFG negative、lambda、cap、层窗口和 step schedule，无法归因 |
| full compiled semantic 存在影响生成质量的现实风险 | 中高 | M3 semantic 从原始平均 25.4 token 增至平均 179.3 token，约 7.1 倍；同一长文本对所有输出时间和空间 token 生效 |
| 风险不是 512-token 截断 | 高 | 当前 20 个 M3 样本为 158–213 token，均低于 Wan 的 512-token 上限 |
| 元数据本身仍然有价值 | 高 | 它是生成前编译、时间/空间路由、自动评测和失败归因所需的结构化来源 |
| 下一步不应继续增加全局文字约束 | 高 | 现有主要失败是结果预置、中间态删除、错误对象、源—流—汇断裂和终态回生，属于时空作用域问题 |

下一阶段采用以下决策顺序：

1. 先在完全相同的推理链路中比较 `original`、`compact` 和当前 `compiled_full`，决定 semantic 应保留多少信息。
2. 选定 semantic 后，只增加输出视频时间上的 `start / trigger / evolve / settle` 条件路由；不得把 denoising step 当作输出帧时间。
3. 时间分阶段通过后，再对错误作用对象明显的样本加入最小 ROI；未通过时不直接提高 CFG、lambda 或 cap。
4. M3 通过某一物理阶段后，才允许 M4 在该阶段的少量代表样本上验证迁移性。

本方案的成功标准不是“更多视频动起来”，而是：**同一实体从正确初态出发，在正确触发之后经过可见中间态，到达并保持物理上方向一致的终态，同时不增加复制、消失、形变和外来施力者。**

## 2. Conditioning 诊断和建议

当前 `compile_conditioning(..., mode="compiled", use_initial_image=True)` 会依次加入：原始 prompt、必需实体、初态、状态变化、允许变化变量和保持项。以 P01 为例，原始 semantic 为 24 token，compiled semantic 为 158 token；20 个样本的原始 semantic 为 20–32 token，compiled semantic 为 158–213 token。

当前实现有四个需要通过实验而不是直觉解决的问题：

| 问题 | 当前表现 | 可能后果 |
|---|---|---|
| 语义重复 | 原 prompt 已描述实体和变化，`required_visible_entities`、`changed_state_variables`、`preserved_context` 又重复实体和结果 | 主动作词的相对权重下降，模型更倾向复述外观或终态 |
| 自然语言分布偏移 | `Required visible entities:`、`These event variables are allowed to change:` 等是规格标签，不是自然视频 caption | UMT5 可编码这些词，但生成模型未必把它们理解为硬约束 |
| 时间状态混写 | 初态、变化过程和终态同时出现在一个全局 context 中 | 早帧提前出现结果、完整态与碎片态并存、模型跳过中间态 |
| 软约束冒充硬约束 | 数量、身份、相机、守恒只以文本出现 | 文本不能可靠替代对象跟踪、ROI、守恒检查或生成后 reject |

因此将 conditioning 分成三个职责层：

| 层 | 进入模型的方式 | 保留内容 | 不应承担的内容 |
|---|---|---|---|
| 全局 semantic | 全视频共享的主 cross-attention | 主体、主动作、关键身份；必要时一条简短相机/场景条件 | 逐阶段状态、长列表、reject 规则、重复的实体计数 |
| phase causal pair | 只在相应输出时间窗注入的 c+/c- | 一个可见阶段的正确关系与最近失败关系 | 背景、画质、相机、无关实体、完整原 prompt |
| evaluation contract | 不直接拼入 prompt | 数量、身份、初态、时序、守恒、终态保持、reject_if | 不参与 cross-attention，不以“写进 prompt”视为已满足 |

新增一个 `compact` profile，作为推荐候选，但在消融完成前不能替换默认值。建议规则如下：

- M3/I2V 上限建议为 70 个 UMT5 token，超过 80 token 直接 warning；M4/T2V 上限建议为 100 token。
- 使用自然、按事件发生顺序书写的英文 caption，不使用 `Required...`、`At the beginning...`、`Preserve throughout...` 等列表标签。
- M3 已有首帧，不重复写完整实体列表、布局和全部初态；只在身份极易漂移时保留一个短的不变量子句。
- 删除 `These event variables are allowed to change`，因为它通常只是再次复述 `changed_state_variables`。
- `protected_scene_properties` 与 `preserved_context` 先做语义级去重，只保留与本次失败相关的一项；其余进入评测契约。
- 不在全局 semantic 中同时展开全部初态和终态；阶段状态交给 phase causal pair。

P01 的 compact 候选可以写为：

```text
Static side view of one black scooter rolling toward a stationary metal trash can. The same scooter makes visible contact, then tilts, slows, and remains stopped beside the can.
```

这不是预设答案，而是待验证的第三个 profile。公平比较必须保留同一首帧、同一 seed、同一 CFG、同一 scheduler、同一 ACE 参数和 relation-only c+/c-；`initial_v1` 不能作为 conditioning 单因素对照。

## 3. 公平对照与统一实验协议

统一生成协议固定为：Wan2.2-TI2V-5B、1280×704、97 帧、24 FPS、50 UniPC steps、shift 5、CFG 3.5、空 unconditional、同一张已审核 I0、相同视频 latent seed。CFG 5 只作为压力测试，不进入主结论。所有实验保存 resolved config、semantic/c+/c-/phase 文本及 token、输入 hash、视频 hash和逐层 residual diagnostics。

**Conditioning 因子实验。** 首轮选择 P01、P06、P13、P18，分别代表接触顺序、落体与冲击、液体守恒、重复衰减；使用 seed `7,42`，执行完整 `3 × 2` 因子设计：

| 因子 | 水平 | 说明 |
|---|---|---|
| semantic | `original` / `compact` / `compiled_full` | 原始短 prompt、建议的自然短扩写、当前完整编译 |
| ACE | `off` / `global_safe` | `lambda0=0` 与当前 M3 safe residual；其余完全一致 |

这里的 `compiled_full` 是实验报告中的名称，对应现有代码的 `mode="compiled"`；`compact` 需要在 W0 中新增，避免把实验标签误当成已经可用的 CLI mode。

首轮共 `4 样本 × 3 semantic × 2 ACE × 2 seeds = 48` 个视频。随后在 P01、P04、P06、P08、P11、P13、P17、P18 上，用胜出的两个 semantic、seed `7,42,123` 做 48 个确认视频。若 `lambda0=0` 的统一 pipeline 与同 semantic 的 M1 不能在数值容差内复现，先修对照链路，不进行主实验。

保留 semantic profile 的判定规则：

- `compact` 或 `compiled_full` 必须在阶段完整度上相对 `original` 至少提高 5 个百分点，或在相同完整度下显著降低实体/初态失败。
- 身份、数量、构图或严重伪影失败率不得比 `original` 增加 10 个百分点以上。
- 如果 `compiled_full` 不优于 `compact`，默认采用更短的 `compact`；如果二者都不优于 `original`，M3 回到 `original`。
- 如果收益只在 ACE on 出现，记录为 semantic×ACE 交互，不能宣称长 prompt 本身有效。

**TRACE-Time 对照。** Conditioning 决策完成后，固定 semantic，仅比较：

| 组别 | 全局 semantic | causal residual | 输出时间 gate |
|---|---|---|---|
| B0 | 选定 semantic | 关闭 | 无 |
| B1 | 同上 | 当前单对 global ACE | 全输出时间 |
| T1 | 同上 | 每阶段一对 c+/c- | 3 或 4 个重叠输出时间窗 |
| S1 | 同上 | 每阶段一对 c+/c- | 输出时间窗 + 最小事件 ROI，仅在 T1 通过后运行 |

三段模板用于没有独立瞬时触发的连续变化：`start 0–0.30`、`evolve 0.20–0.80`、`settle 0.70–1.00`。四段模板用于可见触发后才发生响应的现象：`start 0–0.25`、`trigger 0.18–0.45`、`evolve 0.35–0.80`、`settle 0.70–1.00`。区间是归一化输出视频时间，可重叠平滑过渡；它们不是去噪 step 区间。

每个 phase 的正反句为 12–26 个英文词，使用相同实体词和尽量相同句法，只改变一个可见关系。反事实用肯定句描述最近失败，例如“倾斜发生在接触前”“碎片出现时完整瓶仍存在”，不使用泛化的 `bad physics` 或长篇解释。

实验必须配对盲评：评审界面隐藏方法名和 seed；同一输入的候选随机左右排列。自动量测和人工判断分别保存，不能先看 diagnostics 再给视频打分。每个阶段先用 seed `7,42,123` 筛选，通过后用 `23,77` 补至 5 seeds 做确认。

## 4. 阶段一：原子运动与可逆形变

本阶段先验证一个主体、一个主要变化变量、较少实体交互的现象。目的不是覆盖所有物理类别，而是确认选定 semantic 与 TRACE-Time 不会破坏基础运动连续性，并能表达可观察的速度、姿态或形变进度。

| 样本 | 物理现象 | 输出阶段设计 | 核心量测 | 最近失败反事实 |
|---|---|---|---|---|
| P05 | 斜坡加速后在粗糙面减速 | start：块位于坡上；evolve：越过边界且速度连续下降；settle：停在粗糙面 | 质心轨迹、分段速度、末 20% 位移 | 离开坡后保持恒速或突然停止 |
| P07 | 偏心侧向拉力导致刚体倾倒 | start：箱体直立；trigger：下角侧移且支撑变不稳；evolve：连续转动；settle：侧躺保持 | 下角位移、箱体主轴角、形状保持 | 未见下角位移便倾倒或箱体软化折叠 |
| P17 | 海绵压缩—释放—恢复 | start：同一海绵原厚；trigger：手接触压平；evolve：手离开后厚度恢复；settle：同一海绵接近原形 | 厚度曲线、手—海绵距离、实例身份 | 抬走一个海绵且桌面留下另一个 |

每个现象增加两个只改一个因素的输入变体，用于判断方法是否学习关系而非记住构图：P05 改变坡向或块的初始水平位置；P07 改变拉动方向但保持箱体外观；P17 改变海绵颜色或手从画面另一侧进入。变体必须重新审核 I0，但相同方法之间仍共用同一张 I0。

本阶段运行 B0、B1、T1，3 个原样本和每个样本 1 个变体，共 6 个输入，首轮 3 seeds，共 54 个视频。只有在原样本和变体都满足以下条件时进入阶段二：

- 至少 70% 的视频阶段顺序正确，且 T1 相对 B1 的阶段完整度提高至少 10 个百分点。
- 末态保持率至少 80%；P05 末段近似静止、P07 不重新竖起、P17 不复制或被替换。
- 实体身份与严重伪影失败不高于 B1；若时间分段只增加运动但使主体形变，停止提高强度并检查空间作用域。

## 5. 阶段二：触发、碰撞、传播与不可逆转变

本阶段验证“触发必须先可见，响应随后发生”，以及完整态不能与结果态错误共存。默认采用四段模板；P04 和 P18 的重复过程不按每次接触无限增加 phase，而是用一个传播/衰减 regime 表达。

| 样本 | 物理现象 | start / trigger / evolve / settle 检查 | 关键失败判据 |
|---|---|---|---|
| P01 | 接触后倾倒并停止 | 分离直立 / 可见接触 / 倾斜减速 / 倾倒后停在桶旁 | 接触前倾斜；未触桶；结尾继续滑行 |
| P02 | 双球碰撞与动量转移 | 红动蓝静 / 两球接触 / 蓝球离开且红球减速 / 保持两球身份 | 第三球、手或球杆；蓝球接触前运动 |
| P03 | 碰撞后方向反转 | 球接近 / 球拍接触 / 同一球反向 / 球持续离开 | 球未接触即反向；接触后出现新球 |
| P04 | 多米诺接触传播 | 全直立 / 首块受推 / 倒下前沿依次传播 / 全部或大部稳定倒下 | 整排同时倒；中间状态跳跃；行列复制 |
| P06 | 失去支撑、落地后打开 | 闭书受支撑 / 越过架边 / 下落与撞地后打开 / 同一本书落地保持 | 开场已打开；未落地便出现第二本打开书 |
| P10 | 累积充气后的阈值破裂 | 小而松 / 持续变大变紧 / 单次破裂 / 仅碎片留在原处 | 未充气便爆；完整气球和碎片共存 |
| P11 | 撞击触发玻璃破碎 | 完整瓶悬空 / 瓶接触地面 / 从撞击区形成碎片 / 无完整瓶回生 | 碎片先出现；碎后完整瓶继续下落 |
| P18 | 重复碰撞与能量衰减 | 球下落 / 首次触地 / 反弹峰值逐次降低 / 末段静止 | 峰值上升或不降；结尾仍持续弹跳 |
| P19 | 缺口起裂并不可逆分离 | 一张带小缺口的纸 / 双侧拉力使缺口增长 / 裂纹向外扩展 / 两片保持分开 | 无拉力突裂；从错误位置裂；重新连接 |

扩展变体遵循“单一关系改变”：改变碰撞方向、触发侧、物体颜色或左右布局，但不同时改变材质、相机和背景。P02/P03 必须保持对象数量不变；P10/P11/P19 必须把“完整态消失”与“结果态出现”作为同一身份转移评测，而不是只检查结果存在。

首轮运行 9 个原样本的 B1 与 T1，3 seeds，共 54 个视频。优先在 P01、P04、P11、P18 上补至 5 seeds；它们分别代表时序、传播、排他状态转移和重复衰减。若 T1 已解决时序但仍有错误对象/额外实体，只在这四个样本上运行 S1，ROI 分为 actor、trigger/contact region、result material 和 background 四类，不为每个像素生成新文本。

阶段二进入下一阶段的硬门槛：

- `trigger_time < response_time < settle_time` 在至少 70% 视频成立；缺失任一时间点视为失败，不能以终态正确抵消。
- P04 传播前沿和 P18 峰值序列的方向正确率至少 70%。
- P10/P11/P19 的完整态—结果态错误共存率低于 20%。
- T1/S1 相比 B1 不增加第三实体、外来手、错误工具或背景重构。

## 6. 阶段三：连续相变、物质转移与多物理

本阶段处理文本最容易“画出结果”、但最难满足来源、过程和守恒的现象。连续过程默认三段；存在可见入水、起流等触发时可用四段。评测必须同时检查 source decrease 与 sink/result increase，不能只给“出现水池/沙堆/颜色”打成功。

| 样本 | 物理现象 | 耦合量与阶段 | 关键失败判据 |
|---|---|---|---|
| P08 | 固体融化 | 冰可见面积单调下降；同一盘内水域单调增加；尾段保持 | 外部水源；水增而冰不减；出现第二块冰 |
| P09 | 软化与液化铺展 | 固态黄油面积/边缘锐度下降；液池面积增加 | 液池新增但固态块不减少；画外液体进入 |
| P12 | 风致脱落与源耗尽 | 附着种子数下降；离散种子从花头附近开始漂移；花头变稀 | 画外粒子出现；花头不耗尽；整头消失 |
| P13 | 透明水的源—流—汇 | 壶液位下降；连续水流连接壶嘴和杯口；杯液位上升 | 流从错误位置出现；只涨杯不降壶 |
| P14 | 有色液体转移 | 瓶倾斜和液位下降；橙色流连续；同一玻璃杯液位上升 | 瓶直立但画外倒入；额外杯子；颜色改变 |
| P15 | 颗粒流与堆积 | 杯内沙量下降；沙流连接源与落点；锥形堆持续增长 | 开场已有沙堆；杯内不减少；沙从画外出现 |
| P16 | 扩散 | 单个染料滴进入；着色区域连续扩张；局部浓度逐步分散 | 出现悬垂固体蓝团；整杯瞬间均匀变蓝 |
| P20 | 液体转移与局部火焰并行 | 咖啡只在壶—杯区域转移；火焰只在营火 ROI 内闪烁；雪地保持 | 多出容器；咖啡影响火焰；火焰/雪地大范围重构 |

每类增加一个方向或材质外观变体：冰块/黄油改变颜色背景对比但不改过程；P13/P14 改变倒入方向；P15 改变杯子左右位置；P16 改变染料颜色；P20 交换壶杯在营火左右的位置。目的在于验证 source–stream–sink 和局部性，而不是记忆固定坐标。

首轮运行 8 个原样本的 B1 与 T1，3 seeds，共 48 个视频。P13、P15、P20 再运行 S1，因为它们分别代表液体、颗粒和多物理空间隔离。只有在 T1 已经改善时间进度、但 causal residual 仍明显泄漏到背景或错误对象时才启用 ROI。

阶段三完成条件：

- P08/P09/P12/P16 至少有一个自动可测进度量在 70% 视频中方向单调，且人工确认没有通过实体消失伪造进度。
- P13/P14/P15 至少 70% 视频同时满足 `source decrease + connected stream + sink increase` 三项；只满足一项不得算成功。
- P20 的两个事件都可见，且各自变化主要位于对应区域；额外容器率不高于 B1。
- 通过后才在 P01、P04、P13、P16、P18、P20 上运行 M4 的 B0/B1/T1、3 seeds，共 54 个迁移视频。M4 若实体/布局失败明显高于 M3，不提高 residual，结论记为缺少 I0 grounding。

## 7. 评测门槛、排期与交付物

每个视频产生一条统一评测记录，至少包含以下字段：

```json
{
  "sample_id": "P01",
  "method_profile": "T1",
  "semantic_profile": "compact",
  "seed": 42,
  "start_ok": true,
  "trigger_frame": 31,
  "response_frame": 39,
  "settle_frame": 78,
  "evolve_visible": true,
  "terminal_ok": true,
  "terminal_persistence": 0.84,
  "identity_ok": true,
  "entity_count_ok": true,
  "source_decrease": null,
  "sink_increase": null,
  "artifact_severity": 1,
  "human_pairwise_preference": "left",
  "failure_codes": []
}
```

统一核心指标为：

| 指标 | 定义 | 用途 |
|---|---|---|
| Phase Completion Score | `start + trigger(若适用) + evolve + terminal` 的完成比例 | 判断是否生成完整过程，而非只看终态 |
| Order Accuracy | 所有可见阶段时间戳严格按因果顺序出现的比例 | 判断结果预置和 cause-effect reversal |
| Terminal Persistence | 终态首次成立后到视频结束保持正确的帧比例 | 判断回生、重新竖起、继续运动 |
| Identity/Count Accuracy | 主体身份、数量和同一性同时正确 | 判断复制、消失、完整态与结果态共存 |
| Directional Physics Score | 速度、峰值、源量、汇量、面积或浓度的变化方向正确 | 判断是否只是生成“像物理”的纹理 |
| Artifact Safety | 严重形变、外来手/工具、背景重构、出框的失败率 | 确保物理提升不是以画质和身份为代价 |

建议排期和计算预算如下；每一阶段都是前一阶段通过后的条件任务，不要求一次性全部提交：

| 工作包 | 视频数 | 产出 | Go/No-Go |
|---|---:|---|---|
| W0：compiler 与评测准备 | 0 | `compact` profile、phase schema、token/重复检查、单元测试、盲评表 | artifact 可复现、同输入 hash 配对一致 |
| W1：conditioning 因子筛选与确认 | 48 + 48 | 选定 `original/compact/full`，量化 ACE 交互 | 选出一个不降低安全性的 semantic |
| W2：阶段一原子运动 | 54 | 速度、姿态、形变的 phase 证据 | T1 完整度提升且不伤身份 |
| W3：阶段二触发与不可逆过程 | 54，外加最多 24 个确认/ROI 视频 | 接触时序、传播、排他状态、衰减 | 触发顺序和结果保持达标 |
| W4：阶段三守恒与多物理 | 48，外加最多 18 个 ROI 视频 | source–stream–sink、单调进度、局部隔离 | 守恒方向与局部性达标 |
| W5：M4 迁移 | 最多 54 | 无 I0 条件下的能力边界 | 只报告迁移，不反向改写 M3 结论 |

每个工作包交付：冻结的 YAML/环境变量、sample/seed 清单、`conditioning.compiled.json` 或 phase artifact、逐视频评测 JSON、聚合 CSV、盲评结果、失败样例 contact sheet、代表视频和一页决策记录。所有结论必须报告分母、seed 和配对关系。

停止或回退规则：

- `compiled_full` 没有稳定优于 `compact` 时立即停止继续扩写 semantic。
- T1 未改善 Order Accuracy 时，不提高 lambda/cap；先检查 phase 文本是否真正只改变一个关系，以及 gate 是否作用于输出时间 token。
- T1 改善时序但增加错误对象时，进入最小 ROI；S1 仍失败则判断为 grounding 缺失，不继续堆 prompt。
- 自动量测与盲评不一致时，保留两者并人工复核量测定义，不挑选对方法有利的一项。
- 任一阶段只在单个 seed 成功，不进入下一阶段；至少完成 3-seed 筛选和关键样本 5-seed 确认。

最终应回答三个可证伪问题：`compact/full semantic 是否独立改善生成`、`输出时间分阶段是否修复触发—过程—终态`、`空间局部化是否在不损害身份和画质的前提下修复错误对象与多物理泄漏`。只有这三个问题按顺序得到肯定结果，才值得进入学习型 router、更多物理类别或更大规模 benchmark。
