# TRACE-Writer v2 iter2 代码详细设计

日期：2026-09-02  
目标代码根目录：`/home/liuzhirui/Project/physGen/code/v2`  
依据文档：

- `/home/liuzhirui/Project/physGen/code/v2/v2-m3-iter1-问题解析.md`
- `/home/liuzhirui/Project/physGen/analysis/trace_writer_staged_dynamics_rwc_extensible_design_plan_20260902.md`

## 1. 目标、范围与关键决定

iter2 的目标不是增加更多自由 prompt 或提高 TRACE 强度，而是让已有结构化因果信息真正进入生成链路，并限制它只能在正确实体、正确时空区域和受控总预算内起作用。

本轮实现以下能力：

1. 按第 7.1 节恢复持续生效的全局不可变语义锚点；
2. 按第 7.2 节把阶段正反例重编译为共享锚点的最小关系对；
3. 每个阶段显式写出当前存在的实体、逻辑实例数和状态实例数；
4. 将 `protected_predicates` 分类并注入主 semantic context、阶段共享锚点、空间保护和生成后检查；
5. 增加封闭世界、实体生命周期、状态互斥和守恒耦合；
6. 用时变轨迹/运动走廊替代 M3 的静态首帧框，用受限显著性支持替代 M4 的全画面 stage residual；
7. 按第 7.5 节增加 layer-group、active-support 和 CFG 后总预算；
8. 按第 7.6 节增加 C1 边界插值、Boundary-Cosine Trust Region、时间导数限幅和单调事件时钟；
9. 补齐可定位到 step/layer/stage/token 的诊断和验收测试。

本轮明确保留当前正反例计算：

\[
D_j=A(Q,C_j^+)-\sum_k\rho_{jk}A(Q,C_{jk}^-),\qquad \sum_k\rho_{jk}=1.
\]

iter2 的实现范围不包含 `positive-only` 开关，不删除或绕过 violation context。改动重点是让 `C_j^+` 与 `C_{jk}^-` 除被验证的单一关系外保持一致，避免身份、数量、场景和关系同时相减。

固定的仍是五个职责：`SETUP → ONSET → EVOLUTION → COMPLETION → TERMINAL`。iter2 不增加自由阶段数，不把视频拆成五段生成，也不让 MLLM 自由输出 dense mask 或任意时间边界。

## 2. 当前实现与 iter2 的对应关系

| 问题 | 当前实现 | iter2 处理 |
|---|---|---|
| semantic 信息弱 | `global_semantic` 通常只有 20–32 tokens | 编译 60–120 token 的 M3 anchor、100–180 token 的 M4 anchor |
| 保护信息未进入模型 | `protected_predicates` 只解析/保存 | 分类后进入 global anchor，并派生背景保护和 verifier 规则 |
| 正反例污染 | positive/violation 是分别自由写的完整句子 | 共用完全相同的实体/镜头/保护前缀，只改变一个关系槽位 |
| 阶段实体不明确 | stage 只有 `affected_roles` | 增加 `stage_inventory`，明确每个实体槽位、数量、存在状态 |
| 首帧框失配 | `[stage,H,W]` 静态 mask | 改为 `[stage,F,H,W]` 的轨迹、走廊、目标区域和 source/sink 支持 |
| M4 全画面写入 | `spatial_mode=time` 等于 100% mask | 以计划布局或 probe-layer 显著性构造受限动态支持 |
| 边界突变 | 1 token 的 `0.5/0.5` | 默认 2 token C1 插值、余弦信赖域和时间导数 cap |
| 事件回退 | 开环固定边界 | 单调 stage clock、hysteresis、最小驻留和最大偏移 |
| SafeCap 局部化 | 每层、每步、CFG 前 token/global cap | 增加 block-group 能量账本和 CFG 后 noise-prediction trust region |
| 实体/质量不守恒 | 无运行时实体账本 | closed-world ledger、lifecycle、exclusion、conservation 与 verifier |

## 3. 总体数据流

```text
PXX-V2-plan[img].json
    │
    ├─ TracePlan v1 loader（保持兼容）
    │
    ├─ ControlNormalizer
    │    ├─ EntityLedger
    │    ├─ ProtectedPredicateCompiler
    │    ├─ LifecycleCompiler
    │    ├─ ConservationCompiler
    │    └─ provenance: explicit / derived / default / unresolved
    │
    ├─ SemanticAnchorCompiler
    │    ├─ global_anchor
    │    └─ stage_inventory[5]
    │
    ├─ MinimalPairCompiler
    │    ├─ shared_stage_anchor[5]
    │    ├─ positive_relation[5]
    │    └─ violation_relation[5][1..2]
    │
    ├─ TemporalScheduleCompiler
    │    └─ stage weights [5,F]
    │
    ├─ DynamicSupportCompiler
    │    └─ allowed supports [5,F,H,W]
    │
    └─ CompiledTraceV2 + ContextBundle + ConstraintBundle
             │
             ├─ T5 一次编码并缓存所有 context
             ├─ 正反例凸平均差分
             ├─ Boundary-Cosine morph
             ├─ dynamic support gate
             ├─ token/layer/group SafeCap
             ├─ conditional prediction
             ├─ unconditional prediction
             ├─ base-conditional shadow prediction（strict 模式）
             ├─ CFG 后 trust region
             └─ scheduler step
                    │
                    ├─ video.mp4
                    ├─ trace.contexts.compiled.json
                    ├─ trace.constraints.compiled.json
                    ├─ trace.route.compiled.json
                    ├─ trace.cap.audit.jsonl
                    └─ trace.verify.json
```

## 4. 现有 JSON 是否足够

### 4.1 已经足够直接提取的内容

现有 `PXX-V2-planimg.json` 和 `PXX-V2-plan.json` 通常已经包含：

- `entities[].id/mention/role`；
- `global_semantic`；
- 初态、目标态和 predicate；
- transition、event、effect 和 constraint；
- 五阶段 positive、typed violations 和 `observable_check`；
- `protected_predicates`；
- M3 的首帧 entity boxes。

这些内容足以确定性构建：增强 global anchor、阶段实体清单、初版 lifecycle、共享锚点最小关系对、保护规则以及初版后验检查。

### 4.2 不能从当前 JSON 可靠恢复的内容

以下信息不能由现有文本无歧义推出：

- domino、碎片、液滴等集合实体的精确个体数；
- 运动实体的中间轨迹和 terminal target box；
- M4 无首帧时的真实空间位置；
- 相变中的面积/体积与质量之间的标定；
- 触发、接触、破裂等事件在实际生成视频中的真实发生 token；
- 哪些未声明实体类别必须显式禁止，例如 `hand/cue/dropper/tool/external stream`。

因此不能把这些内容“猜成真值”。`ControlNormalizer` 必须为每个字段记录来源：

```python
Literal["explicit", "derived", "default", "unresolved"]
```

`unresolved` 不阻止 20 个样本继续生成，但会进入 validation 和 manifest；涉及精确守恒、精确群体计数或严格动态轨迹的正式结论，只能使用 `explicit` 或人工批准的 `derived` 数据。

### 4.3 兼容策略

不批量改写 `/home/liuzhirui/Project/physGen/code/v1/demo` 中的源 JSON。输入仍接受 `schema_version=trace-writer-v1`，运行时规范化为：

```text
input schema: trace-writer-v1
compiled schema: trace-writer-v2
design revision: fixed5-causal-r4
```

额外人工信息采用可选 sidecar：

```text
/home/liuzhirui/Project/physGen/code/v2/control_overlays/PXX/PXX-m3-control.json
/home/liuzhirui/Project/physGen/code/v2/control_overlays/PXX/PXX-m4-control.json
```

优先级为 `sidecar explicit > input JSON explicit > deterministic derived > versioned default`。冲突时停止该样本并报告，不做静默覆盖。

## 5. iter2 数据结构

建议新增 `/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_control_schema.py`，避免把兼容解析和新执行结构混在同一个 dataclass 中。

### 5.1 实体账本

```python
@dataclass(frozen=True)
class Cardinality:
    kind: Literal["exact", "range", "group", "unknown"]
    exact: int | None = None
    minimum: int | None = None
    maximum: int | None = None

@dataclass(frozen=True)
class EntitySlot:
    entity_id: str
    canonical_mention: str
    role: str
    cardinality: Cardinality
    persistent_identity: bool
    mutable_attributes: tuple[str, ...]
    invariant_attributes: tuple[str, ...]
    aliases: tuple[str, ...]
    provenance: str

@dataclass(frozen=True)
class StagePresence:
    entity_id: str
    stage_id: str
    logical_count: Cardinality
    visible: Literal["required", "allowed", "forbidden", "transitioning"]
    state_facts: tuple[str, ...]
```

`logical_count` 和视觉状态必须分开。例如一本书从 closed 变为 open，逻辑实例始终为 1；不能把 closed book 和 open book 当作两个实体。气球爆裂则可表示为 parent slot 退场、fragment group 入场，并仅在明确的 transition window 中允许短时共存。

### 5.2 封闭世界、生命周期和守恒

```python
@dataclass(frozen=True)
class ClosedWorldSpec:
    allowed_entity_ids: tuple[str, ...]
    forbidden_new_categories: tuple[str, ...]
    empty_regions: tuple[str, ...]
    external_source_allowed: bool

@dataclass(frozen=True)
class LifecycleTransition:
    entity_id: str
    from_stage: str
    to_stage: str
    from_state: str
    to_state: str
    identity_preserved: bool
    mutually_exclusive_states: tuple[tuple[str, str], ...]

@dataclass(frozen=True)
class ConservationRule:
    rule_id: str
    quantity: Literal[
        "instance_count", "material_proxy", "area_proxy",
        "volume_proxy", "motion_transfer"
    ]
    sources: tuple[str, ...]
    sinks: tuple[str, ...]
    relation: Literal["constant", "transfer", "coupled_monotonic"]
    tolerance: float
    observable_proxy: str
    provenance: str
```

约束分三级执行：

1. **文本级：**进入 persistent anchor 和每阶段共享实体清单；
2. **路由级：**只允许 residual 写入声明实体的轨迹/source/sink 区域，保护补集；
3. **验证级：**生成后检测数量、身份连续、外部实体、source/sink 联动和 terminal persistence。

不声称文本和 mask 能严格保证质量守恒。像素面积只能作为单调 proxy，只有经过标定后才能近似物质量。

### 5.3 动态支持与事件时钟

```python
@dataclass(frozen=True)
class TrackKeyframe:
    latent_frame: int
    box: tuple[float, float, float, float]
    confidence: float
    provenance: str

@dataclass(frozen=True)
class EntityTrack:
    entity_id: str
    keyframes: tuple[TrackKeyframe, ...]
    interpolation: Literal["linear_center_log_size", "swept_union"]
    max_dilation: float

@dataclass(frozen=True)
class EventClockSpec:
    initial_boundaries: tuple[int, int, int, int]
    max_shift_tokens: int
    min_core_tokens: tuple[int, int, int, int, int]
    unlock_checks: tuple[str, ...]
    hysteresis_high: float
    hysteresis_low: float
```

所有 box 仍是少量规范化几何原语，直接在 `25×22×40` token grid 上生成，不创建高分辨率自由 mask。

## 6. 第 7.1 节：全局不可变语义锚点

### 6.1 编译顺序

新增 `/home/liuzhirui/Project/physGen/code/v2/ace_router/semantic_anchor.py`：

```text
1. ENTITY INVENTORY
2. INITIAL SCENE AND LAYOUT
3. EVENT AND ALLOWED CHANGES
4. IDENTITY / APPEARANCE / GEOMETRY INVARIANTS
5. CAMERA / BACKGROUND INVARIANTS
6. CLOSED-WORLD AND EXTERNAL-SOURCE RULES
7. TERMINAL PERSISTENCE
```

主 context 不再直接使用原始 `plan.global_semantic`，而使用：

```python
global_anchor = SemanticAnchorCompiler.compile(plan, control_bundle, mode)
```

它仍通过 Wan 原有 semantic cross-attention 在全部层、全部去噪步和全部空间 token 上生效；不把 anchor 再作为额外 TRACE residual 重复注入。

### 6.2 `protected_predicates` 的分类注入

新增 `ProtectedPredicateCompiler`，使用确定性模式和实体 ID 将保护项分成：

| 类别 | 例子 | 注入位置 |
|---|---|---|
| identity/count | exactly one red ball | global anchor、stage inventory、count verifier |
| appearance | color/size/material unchanged | global anchor、identity verifier |
| rigid geometry | roundness/proportions unchanged | global anchor、geometry verifier |
| camera/background | static camera/unchanged cloth | global anchor、background-protect complement |
| closed world | no hand/tool/external source | global anchor、forbidden detector |
| terminal persistence | final state remains | global anchor、terminal verifier |
| unclassified | 其他自由文本 | global anchor 的 preserve 句并发出 warning |

这里的字段名以现有 schema 的 `protected_predicates` 为准；用户描述中的 `protecter_predicates` 视为同一字段的口头写法，不新增第二个拼写。

规则：

- 所有保护项至少进入 `global_anchor`，不能只留在 manifest；
- 与当前阶段相关的实体身份、数量和不变量还要进入正反例共享前缀；
- count/identity 文本必须在 token 压缩时最后删除；
- 背景保护 mask 只约束 TRACE 写入，不冻结 Wan 基座；因此还必须保留文本和后验检查；
- 同义保护项先规范化和去重，避免长 prompt 重复争夺 attention。

### 6.3 阶段实体清单

每个阶段编译如下固定前缀，正例和全部反例逐字相同：

```text
[STAGE ENTITY INVENTORY]
Present now: exactly 1 glossy red ball (red_ball#1),
exactly 1 glossy blue ball (blue_ball#1), and the same billiard table.
Logical entity count is unchanged. No duplicate ball, person, hand,
cue, tool, or external source enters. Both ball identities, colors,
sizes, and round shapes remain unchanged.
```

“重点标注”通过固定位置、固定标题、数字和 canonical ID 实现，不依赖自由散文中的偶然提及。实际送入 T5 的仍是自然语言；方括号标题用于稳定结构和审计，不代表特殊 tokenizer token。

对每个 stage 保存机器可读清单：

```json
{
  "stage_id": "onset",
  "entities": [
    {"id": "red_ball", "logical_count": 1, "visible": "required"},
    {"id": "blue_ball", "logical_count": 1, "visible": "required"}
  ],
  "forbidden_new_categories": ["person", "hand", "cue", "tool"]
}
```

### 6.4 P02 的 global anchor 示例

```text
A locked side-view camera shows exactly one glossy red billiard ball
on the left and exactly one glossy blue billiard ball on the right,
both fully visible from the first frame, on unchanged green billiard
cloth. The scene contains only these two balls and the table; the empty
space above the table remains empty, with no person, hand, cue, tool,
extra ball, or external source entering. Both balls keep the same
identity, count, round shape, size, and color. Only their positions and
velocities may change: the red ball moves right, visibly contacts the
blue ball, slows, and only then the blue ball moves right. The resulting
separated post-contact motion persists through the final frames.
```

M3 anchor 目标为 60–120 个实际 T5 tokens；M4 因没有首帧约束，目标为 100–180 tokens，并强制包含 `from the first frame`、相对布局、完整入镜、运动平面和空区域。

### 6.5 Token 预算

新增 `PromptBudgeter`，必须使用运行时同一个 T5 tokenizer 计算，不用字符数近似。

优先级：

```text
P0  entity ID / exact count / lifecycle / closed-world
P1  protected identity / geometry / camera / background
P2  cause-order-effect / allowed changes / terminal persistence
P3  nonessential visual adjectives / redundant observable wording
```

超预算时只压缩 P3、再去重 P2；P0/P1 发生截断则 preflight 失败。每条 context 最长仍为 512 tokens，但 iter2 不应靠接近 512 的长文本解决结构问题。

## 7. 第 7.2 节：保留正反例差分的严格最小关系对

### 7.1 Context 结构

新增 `/home/liuzhirui/Project/physGen/code/v2/ace_router/minimal_pair.py`。每个 stage 编译为：

```python
positive_text = shared_stage_anchor + positive_relation
negative_text[k] = shared_stage_anchor + violation_relation[k]
```

其中 `shared_stage_anchor` 包含相同的实体、数量、状态、相机、背景和无关不变量。relation clause 只能改变被该 violation 引用的一条 constraint。

运行公式保持不变：

\[
P_j=A(Q,\text{shared}_j\oplus r_j^+),
\]

\[
N_j=\sum_k\rho_{jk}A(Q,\text{shared}_j\oplus r_{jk}^-),
\]

\[
D_j=P_j-N_j,\qquad \rho_{jk}=1/K_j.
\]

增加 violation 不改变总 negative 权重。禁止把数量、颜色、对象类别、镜头或背景只写在一侧。

### 7.2 关系模板

| constraint / violation | positive 槽位 | negative 槽位 |
|---|---|---|
| `precedes/wrong_order` | A BEFORE B | B BEFORE A |
| `requires/early_effect` | B STARTS ONLY AFTER A | B STARTS BEFORE A |
| `requires/trigger_missing` | A OCCURS AND THEN B STARTS | B STARTS WITHOUT A |
| `changes/wrong_direction` | X CHANGES FROM u TO v | X CHANGES FROM v TO u |
| `couples/coupling_broken` | A DECREASES WHILE B INCREASES | B INCREASES WHILE A STAYS UNCHANGED |
| `persists/reversal` | STATE v REMAINS | STATE v RETURNS TO u |
| `excludes/exclusion_broken` | ONE SLOT HAS ONLY STATE v | COPIES IN STATES u AND v COEXIST |

模板所需名词从 event description、effect direction、predicate observable 和 canonical mention 中确定性获得。模板编译失败时：

- `compat` 模式保留原 positive/violation，但标记 `pair_quality=legacy_nonminimal`；
- `strict` 模式停止该样本；
- iter2 正式实验只统计 `pair_quality=minimal_verified` 的样本。

这保证当前 20 个旧 JSON 可继续运行，同时不会把非最小对伪装成已经满足 7.2。

### 7.3 P02 ONSET 示例

共享前缀：

```text
[STAGE ENTITY INVENTORY] Present now: exactly one red_ball#1 and exactly
one blue_ball#1; both are fully visible and keep the same color, size,
round shape, and identity. The camera, table, and empty upper region are
unchanged. No extra ball, person, hand, cue, or tool exists.
```

positive relation：

```text
[RELATION] The same red ball contacts the same blue ball BEFORE the blue
ball starts moving.
```

wrong-order relation：

```text
[RELATION] The same blue ball starts moving BEFORE the red ball contacts
the same blue ball.
```

两侧实体清单完全相同；只交换 contact 与 motion-start 的先后关系。

### 7.4 编译验证

`MinimalPairValidator` 必须检查：

- positive/negative 的 entity ID multiset 完全一致；
- 数字和 count phrase 完全一致；
- protected attribute multiset 完全一致；
- 每个 negative 只引用 1 条主要 constraint；
- 除 relation span 外，token-level LCS 覆盖率至少 0.80；
- stage inventory 与 lifecycle 推导一致；
- 所有 violation 的凸权重和为 1。

LCS 只做结构审计，不作为语义正确性的充分条件。

## 8. 封闭世界、生命周期与守恒约束

### 8.1 EntityLedger 构建

新增 `/home/liuzhirui/Project/physGen/code/v2/ace_router/entity_ledger.py`：

1. 每个 `entities[].id` 建立一个逻辑槽位；
2. 默认单数 canonical entity 的 `logical_count=1`；
3. 复数/集合对象不自动猜精确数，使用 `group` 或 `unknown`；
4. 从 `states + transitions` 推导每阶段状态；
5. 从 `persists/excludes` 推导 terminal persistence 和状态互斥；
6. 从 `protected_predicates` 提取 count、identity 和 geometry 不变量；
7. 对 sidecar 的显式 count/lifecycle 做交叉校验。

### 8.2 阶段存在约束

每一阶段都必须得到一个完整 inventory，而不是只列 `affected_roles`。未受作用的实体也要存在，但它们的 TRACE 空间支持为 0，并由 global anchor 维持。

示例：P06 书本翻开。

```text
SETUP:      book#1 count=1, state=closed, visible=required
ONSET:      book#1 count=1, state=opening, visible=required
EVOLUTION:  book#1 count=1, state=opening, visible=required
COMPLETION: book#1 count=1, state=open, visible=required
TERMINAL:   book#1 count=1, state=open, visible=required
```

不存在 `closed_book#1 + open_book#2`。状态改变不能产生第二个逻辑实例。

### 8.3 封闭世界约束

默认 allowed set 为计划中所有 entity slots 与静态场景载体。`forbidden_new_categories` 由以下来源合并：

- sidecar 显式条目；
- `protected_predicates` 中的禁止项；
- 按任务先验的版本化禁止表，例如 billiards 禁止 hand/cue/person，natural melting 禁止 dropper/tool/external stream；
- 用户输入明确允许的 agent 不得被先验表误删。

禁止表必须写入 manifest 并带来源。它不能只进入 CFG negative，因为 CFG negative 当前为空且可能引入新的组合偏差；主策略是“场景只包含白名单实体、其他区域保持空”的肯定式封闭世界描述，明确高风险类别作为补充。

### 8.4 守恒编译

守恒由 graph constraint 和 lifecycle 共同编译：

- `instance_count`：刚体对象数量恒定；
- `material_proxy`：相变中 source 减少与 sink 增加耦合；
- `motion_transfer`：碰撞后 actor 速度降低与 patient 速度增加耦合；
- `state_exclusion`：同一逻辑槽位不能同时显示互斥前态/后态副本。

P09 示例：

```text
source: butter_solid#1
sink: melted_butter_region#1
law: coupled_monotonic
checks:
  solid_area(t+1) <= solid_area(t) + tolerance
  liquid_area(t+1) >= liquid_area(t) - tolerance
  liquid_growth requires simultaneous solid decrease
  no connected liquid component enters from image boundary
```

路由时为 source 和 sink 分别建立 mask，但共享同一个 `conservation_group_id` 和总 residual budget；不能让 sink 获得完整预算后 source 再获得一份完整预算。

### 8.5 生成后 verifier

新增 `/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_verifier.py`，第一版只读、不反向修改当前视频：

- entity count 与额外类别检测；
- tracker identity continuity；
- lifecycle 前态/后态共存时间；
- terminal reversal；
- source/sink 面积单调性与耦合相关；
- 边界进入的外部 connected component；
- mask 外背景 SSIM/LPIPS/光流；
- `observable_check` 派生的任务规则。

若每样本生成多个 seed，使用预先登记的规则做 seed 排序；不能事后按主观观感更换指标。

## 9. 动态空间支持：解决首帧框失配

### 9.1 API 变化

当前：

```python
build_stage_spatial_masks(...) -> Tensor[5, H, W]
```

iter2：

```python
build_stage_spatiotemporal_masks(...) -> Tensor[5, F, H, W]
```

`flatten_stage_gates` 直接接收 `[stage,F,H,W]`，不再把同一二维框复制到整个时间轴。

### 9.2 M3 计划轨迹

M3 使用首帧框作为 `initial_box`，但 terminal 不再回退到 initial `role_union`：

```text
initial_box
  → stage-specific swept corridor
  → contact/interface region
  → expected target region
  → terminal support around target region
```

优先级：

1. sidecar 显式 track keyframes；
2. JSON 中显式 source/target entity boxes 与 motion direction 派生 corridor；
3. 从 `effect.direction` 和场景边界派生保守 swept union；
4. 仍无法确定时使用 initial box 的受限膨胀，并记录 `support_quality=uncertain`，不声称已解决跟踪。

box 插值使用中心线性、宽高 log-space 线性，并添加不确定性 dilation。所有插值直接落到 22×40 token grid。

### 9.3 source/sink 与多实例

- 碰撞：actor track、patient track 和 contact interface 分开；
- 融化：solid source 与 liquid sink 分开；
- domino：保留 row integrity support，并可增加代表性实例 anchors；
- balloon：膨胀阶段按 envelope 增大 support，burst 后切换到有限 fragment corridor；
- background protect：取所有允许动态 support 的时变补集，而不是人工列出每块墙和地面。

### 9.4 M4 无首帧支持

M4 不加载用户首帧，不能伪造 image-grounded box。采用两级方案：

1. 若 sidecar 有规范化布局，只把它视为计划布局先验；
2. 若无布局，在首个 Writer probe block 对当前 stage operator 计算 token-wise saliency，经时间 EMA、soft quantile、连通性过滤和面积上下限后得到动态支持。

建议约束：

```text
min area = 3% frame tokens
max area = 35% frame tokens
center shift per denoise step <= 3 latent cells
area ratio change per step <= 1.5x
EMA decay = 0.8
```

probe 只决定“允许在哪里写”，不改变正反例方向。support 为空时复用上一有效 mask；第一步无历史时使用布局先验或受限中心区域，并记录 fallback 来源，禁止静默改为 100% 全画面。

M4 动态 support 属于需单独验证的功能。若 saliency 与实体位置不相关，正式 M4 先只启用增强 global anchor、最小对、边界和 SafeCap，不把不可靠 saliency 当作有效空间控制。

## 10. 第 7.6 节：边界安全与单调事件时钟

### 10.1 C1 compact crossfade

iter2 默认 `crossfade_tokens=2`。对宽度 `w` 的边界采样：

\[
u_i=\frac{i+1}{w+1},\qquad s(u)=3u^2-2u^3.
\]

旧阶段权重为 `1-s(u)`，新阶段为 `s(u)`。仍满足：

- 任意 token 最多两个相邻阶段激活；
- 非相邻阶段严格为 0；
- 每个时间 token 的阶段权重和为 1；
- 不用 Gaussian 长尾。

兼容实验可保留旧的 `0.25/0.75 → 0.75/0.25` 表，但 `r4` 默认记录 `smoothstep_c1`。

### 10.2 Boundary-Cosine Trust Region

在相邻阶段交叠 token 上，先计算未乘 temporal weight 的 operator `D_a`、`D_b` 及 active support 内 cosine：

\[
c=\frac{\langle D_a,D_b\rangle_S}
{\|D_a\|_S\|D_b\|_S+\epsilon}.
\]

定义共同分量和冲突系数：

\[
D_{common}=\frac{D_a+D_b}{2},\qquad
\alpha(c)=\operatorname{clip}\left(\frac{c-\tau_{low}}
{\tau_{high}-\tau_{low}},0,1\right).
\]

\[
\tilde D_a=D_{common}+\alpha(D_a-D_{common}),\qquad
\tilde D_b=D_{common}+\alpha(D_b-D_{common}).
\]

最终边界 residual：

\[
D_f=w_a(f)\tilde D_a+w_b(f)\tilde D_b.
\]

初始建议 `tau_low=-0.2`、`tau_high=0.3`。同向 operator 正常插值；强冲突时主要保留共同分量。core token 不做此 morph，避免把五阶段抹平。

### 10.3 时间导数限幅

仅在边界窗口和相邻 support 的交集/软并集内约束：

\[
\Delta_f=D_f-D_{f-1},
\]

\[
\|\Delta_f\|_2\le r_{time}
\frac{\|r_{sem,f}\|_2+\|r_{sem,f-1}\|_2}{2}.
\]

超限时只缩放变化量，不放大任何 residual。初始 sweep `r_time ∈ {0.02, 0.05, 0.10}`。support 新进入区域同时使用空间 ramp，避免 mask 平移本身制造硬边。

### 10.4 事件时钟

新增 `/home/liuzhirui/Project/physGen/code/v2/ace_router/event_clock.py`。分两级实现：

**A. 默认的 plan-prior clock**

- 五阶段职责固定；
- 根据 `triggered/continuous` 和 transition 数选择版本化布局；
- 边界最多偏移 ±2 latent tokens；
- 保证每阶段最小 core 长度；
- MLLM 不能直接指定任意时间点。

**B. 实验性的 reader-hysteresis clock**

- 从 AuditReader/可见 proxy 得到 precondition、trigger、progress、completion 置信度；
- 只允许阶段索引单调增加；
- 高阈值解锁，低阈值只维持，不允许回退；
- 满足最小驻留 token 后才可进入下一阶段；
- 一旦 contact/tilt/rupture 被锁定，terminal 不允许恢复互斥前态；
- 实际边界只能相对 plan-prior 移动 ±2 tokens。

`AuditReader` 本身保持只读；闭环逻辑使用单独的 `EventClockController`，并必须作为独立实验开关。Reader proxy 未在多 seed 上证明可分前，不启用 B 作为默认。

## 11. 第 7.5 节：CFG-aware、support-aware 聚合 SafeCap

### 11.1 保留单层 cap，但改变参考区域

逐 token cap 保留。单层 global cap 从全序列 norm 改为同一 active support 内的 norm：

\[
\|D_l\|_S\le r_{layer}\|r_{sem,l}\|_S.
\]

同时限制 support 外泄漏：

\[
\|D_l\|_{\bar S}\le r_{outside}\|r_{sem,l}\|_S.
\]

`S` 为该时间 token 所有激活阶段动态 support 的并集，padding 永远不进入 norm。

### 11.2 Layer-group 能量账本

不同 block 的 hidden 表示不能简单向量相加，因此采用累积能量而不是直接求 residual 和。对 Writer blocks 14–23：

\[
E_D^2(l)=\sum_{q=14}^{l}\|D_q\|_S^2,
\qquad
E_{sem}^2(l)=\sum_{q=14}^{l}\|r_{sem,q}\|_S^2.
\]

强制：

\[
E_D(l)\le r_{group}E_{sem}(l).
\]

每个 block 只使用剩余预算：

\[
\kappa_l=\min\left(1,
\sqrt{\frac{\max(r_{group}^2E_{sem}^2(l)-E_D^2(l-1),0)}
{\|D_l\|_S^2+\epsilon}}\right).
\]

新增 `CapLedger`，作用域为一次 conditional forward 的一个 denoise step；进入下个 step 时重置。source/sink 属于同一 conservation group 时共享账本。

### 11.3 CFG 后精确 trust region

当前：

\[
\epsilon_{trace}^{cfg}=\epsilon_u+s(\epsilon_c^{trace}-\epsilon_u).
\]

为了得到真正的 TRACE 增量，strict 模式额外计算关闭 TRACE 的 base conditional：

\[
\epsilon_{base}^{cfg}=\epsilon_u+s(\epsilon_c^{base}-\epsilon_u),
\]

\[
\Delta_{cfg}=s(\epsilon_c^{trace}-\epsilon_c^{base}).
\]

对 noise-prediction 空间中的 ROI 内外分别限幅：

\[
\|\Delta_{cfg}\|_S\le r_{cfg}\|\epsilon_{base}^{cfg}\|_S,
\]

\[
\|\Delta_{cfg}\|_{\bar S}\le r_{cfg,out}
\|\epsilon_{base}^{cfg}\|_S.
\]

最后：

\[
\epsilon_{final}=\epsilon_{base}^{cfg}
+\kappa_{in}S\odot\Delta_{cfg}
+\kappa_{out}(1-S)\odot\Delta_{cfg}.
\]

动态 support 从 `[5,25,22,40]` 确定性投影到 noise prediction grid；不得把 padding 或首帧已知区域计入可写预算。

strict 模式每步从 conditional+unconditional 两次 forward 增加到 trace-conditional+base-conditional+unconditional 三次 forward，理论 denoiser 成本约增加 50%。这是获得精确 CFG 后 delta 的代价。

### 11.4 吞吐模式

`estimated` 模式不做 shadow forward，而把 layer/group cap 除以 `max(guide_scale,1)`，并将 `s·(conditional delta)` 记为 CFG delta 上界估计。该模式不是精确等价，只能在 strict pilot 证明估计稳定后用于 20 样本批量运行。

建议配置：

```text
pilot / 机制验证: cap_mode=aggregate_strict
批量候选生成:    cap_mode=aggregate_estimated
正式结论复核:    cap_mode=aggregate_strict
```

### 11.5 初始 sweep

不根据单个样本手调：

```text
lambda0          ∈ {0.03, 0.05, 0.10}
token_cap_ratio  ∈ {0.03, 0.05, 0.10}
layer_cap_ratio  ∈ {0.005, 0.01, 0.02}
group_cap_ratio  ∈ {0.005, 0.01, 0.02}
cfg_cap_ratio    ∈ {0.005, 0.01, 0.02}
```

先在 P01/P02/P04/P06/P09/P10 和 M4-P01/P02 上小规模筛选，再固定一个全局 preset 跑全部 20 个样本。

## 12. Writer 内部执行顺序

执行顺序必须固定，避免“先 cap 后 morph”导致边界处理再次放大 residual：

```python
for denoise_step in steps:
    cap_ledger.reset()
    event_clock.begin_step()

    for writer_block in blocks[14:24]:
        semantic = cross_attn(query, global_anchor_context)

        operators = []
        for stage in active_stages:
            positive = cross_attn(query_slice, stage.positive_context)
            negative = convex_mean(
                cross_attn(query_slice, context)
                for context in stage.violation_contexts
            )
            operators.append(positive - negative)

        routed = boundary_cosine_morph(operators, temporal_weights)
        routed = apply_dynamic_support(routed, support_5f_hw)
        routed = temporal_derivative_cap(routed, semantic)
        routed = lambda0 * layer_gate * step_gate * routed
        routed = token_cap(routed, semantic)
        routed = active_support_layer_cap(routed, semantic)
        routed = cap_ledger.consume(routed, semantic)

        hidden = hidden + semantic + routed

    epsilon_trace_cond = head(hidden)
    epsilon_base_cond = shadow_conditional_if_strict(...)
    epsilon_uncond = unconditional(...)
    epsilon = cfg_output_trust_region(
        epsilon_trace_cond, epsilon_base_cond, epsilon_uncond, support
    )
    latent = scheduler.step(epsilon, latent)
```

边界 operator 需要在交叠 token 上分别保留到 morph 完成，不能像当前实现一样先全部 `index_add_` 后丢失 stage 来源。

## 13. 文件级修改方案

### 13.1 新增文件

| 文件 | 职责 |
|---|---|
| `/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_control_schema.py` | iter2 normalized schema 与 provenance |
| `/home/liuzhirui/Project/physGen/code/v2/ace_router/entity_ledger.py` | 实体数量、生命周期、封闭世界、互斥状态 |
| `/home/liuzhirui/Project/physGen/code/v2/ace_router/semantic_anchor.py` | global anchor、阶段 inventory、token 预算 |
| `/home/liuzhirui/Project/physGen/code/v2/ace_router/minimal_pair.py` | constraint→严格最小关系对 |
| `/home/liuzhirui/Project/physGen/code/v2/ace_router/dynamic_support.py` | M3 track/corridor 和 M4 saliency support |
| `/home/liuzhirui/Project/physGen/code/v2/ace_router/event_clock.py` | C1 route、边界偏移、单调 clock/hysteresis |
| `/home/liuzhirui/Project/physGen/code/v2/ace_router/cap_ledger.py` | support-aware layer/group/CFG cap |
| `/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_verifier.py` | count/lifecycle/conservation/背景检查 |
| `/home/liuzhirui/Project/physGen/code/v2/tools/compile_iter2_control.py` | 从旧 JSON 编译/审计 control bundle |

### 13.2 修改文件

`/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_schema.py`

- 保持 v1 输入解析；
- 支持可选 control overlay；
- 不把缺少 iter2 字段当成 JSON parse failure。

`/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_compile.py`

- `CompiledTrace` 增加 `context_bundle/constraint_bundle/entity_ledger/event_clock/support_spec`；
- `context_texts()` 输出增强 global anchor 与最小关系对；
- manifest 记录字段来源、pair quality 和 unresolved 项；
- constraint coverage 新增 `anchor/lifecycle/conservation/verifier` 通路。

`/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_validation.py`

- 检查阶段 inventory、count/lifecycle 一致性；
- 检查 protected predicate compile coverage；
- 检查最小对结构、闭世界冲突和 conservation source/sink；
- 区分 `runnable warning` 与 `formal-eval blocking issue`。

`/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_masks.py`

- 保留固定布局函数供兼容测试；
- 新接口接收 `[5,F,H,W]`；
- 支持 keyframe interpolation、swept union、source/sink、动态 protected complement；
- 保证非相邻阶段严格为零和 partition of unity。

`/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_runtime.py`

- `StageRuntime` 增加独立 operator、dynamic support、inventory 与 conservation group；
- `RoutingRuntime` 增加 `CapLedger/EventClockController/SupportState`；
- M4 probe mask 只能通过明确状态更新。

`/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_writer.py`

- 保留 positive-minus-convex-negative；
- 延后 stage 汇总以执行 boundary cosine morph；
- 增加时间导数、support-aware 和 group cap；
- 向 reader 写入 positive/negative/difference/cosine/cap 诊断。

`/home/liuzhirui/Project/physGen/code/v2/ace_router/trace_model_adapter.py`

- 一次 model forward 内传递 `CapLedger`；
- 输出 trace conditional 和必要的 support summary；
- 保持基座权重冻结和 `lambda0=0` fast path 逐元素不变。

`/home/liuzhirui/Project/physGen/code/v2/ace_router/wan_pipeline.py`

- 编码 `context_bundle` 而不是原短 global；
- strict 模式增加 base-conditional shadow pass；
- CFG 后执行 output trust region；
- 推理结束调用 verifier 并保存新产物。

`/home/liuzhirui/Project/physGen/code/v2/ace_router/controllers.py`

- `AuditReader` 继续只读；
- 记录 stage operator cosine、ROI 内外、group budget 和 CFG delta；
- 闭环时钟使用独立 controller，不让 audit 开关改变输出。

`/home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer.py`

- 增加 iter2 preset、overlay、strict validation、cap、schedule、support 和 verifier 参数；
- preflight 生成全部 compiled artifacts；
- `--check_only` 必须无需加载模型即可检查 20 份输入。

## 14. 配置、Shell 与 Determined YAML

建议新增统一 preset：

```text
TRACE_PRESET=fixed5-causal-r4-safe
ANCHOR_MODE=compiled
MINIMAL_PAIR_MODE=strict
PROTECTED_PREDICATES=inject
ENTITY_LEDGER=on
CLOSED_WORLD_MODE=prompt_route_verify
CONSERVATION_MODE=prompt_route_verify
CROSSFADE_TOKENS=2
BOUNDARY_MORPH=cosine_trust
EVENT_CLOCK_MODE=plan_prior
CAP_MODE=aggregate_strict
READER_MODE=audit
VERIFY_MODE=audit
```

新增 CLI：

```text
--trace_preset
--control_overlay_root
--minimal_pair_mode {compat,strict}
--anchor_mode {legacy,compiled}
--dynamic_support_mode {off,planned,planned_saliency}
--event_clock_mode {fixed,plan_prior,reader_hysteresis}
--boundary_morph {linear,smoothstep,cosine_trust}
--layer_cap_ratio
--group_cap_ratio
--cfg_cap_ratio
--cfg_outside_cap_ratio
--cap_mode {legacy,aggregate_estimated,aggregate_strict}
--verify_mode {off,audit,rank}
```

M3 默认：`planned` dynamic support；M4 默认先使用 `plan_prior`，只有 saliency 验证通过后改为 `planned_saliency`。M4 仍不得加载首帧。

Shell 与 YAML 中继续使用绝对路径：

```text
/home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer_m3.sh
/home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer_m4.sh
/home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer_m3.yaml
/home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer_m4.yaml
```

YAML 必须显式记录上述新环境变量，不能只依赖 Python 默认值。正式跑全量前，先提交 `SAMPLE_IDS=P01,P02` 的 strict pilot。

## 15. 诊断产物

每个 seed 至少保存：

```text
trace.plan.input.json
trace.plan.validation.json
trace.control.normalized.json
trace.contexts.compiled.json
trace.constraints.compiled.json
trace.route.compiled.json
trace.support.compiled.json
trace.cap.audit.jsonl
reader.audit.jsonl
trace.verify.json
manifest.json
video.mp4
```

`trace.contexts.compiled.json` 必须保存真正送入 T5 的完整文本、token 数、预算优先级、是否压缩和 hash。不能只保存原 JSON prompt。

每个 step/layer/stage 记录：

- positive、convex negative、difference norm；
- positive/negative cosine；
- 相邻 stage operator cosine；
- 每个 latent time token 的 applied norm；
- ROI 内/外 residual norm；
- token/layer/group cap coefficient 和饱和比例；
- CFG 前估计 delta、CFG 后精确 delta；
- 边界 residual derivative；
- dynamic support 面积、中心、来源和更新幅度。

## 16. 测试设计

### 16.1 单元测试

新增：

```text
/home/liuzhirui/Project/physGen/code/v2/tests/test_semantic_anchor.py
/home/liuzhirui/Project/physGen/code/v2/tests/test_minimal_pair.py
/home/liuzhirui/Project/physGen/code/v2/tests/test_entity_ledger.py
/home/liuzhirui/Project/physGen/code/v2/tests/test_dynamic_support.py
/home/liuzhirui/Project/physGen/code/v2/tests/test_event_clock.py
/home/liuzhirui/Project/physGen/code/v2/tests/test_aggregate_safecap.py
```

关键断言：

- 所有 `protected_predicates` 至少编译到一种执行通路；
- global anchor 包含全部 P0/P1 项且没有被 tokenizer 截断；
- 每个 stage 的 positive/negative 实体 ID、数量和保护项一致；
- violation 权重和为 1，正反例差分没有被删除；
- 一本书状态变化时 logical count 始终为 1；
- 未知集合数量不会被自动猜成精确值；
- `[5,F,H,W]` mask 只允许相邻阶段重叠并覆盖运动 target；
- terminal support 不等于首帧 box，除非显式声明静止；
- C1 weights 和为 1，非相邻阶段严格为 0；
- operator 反向时 boundary conflict 分量被衰减而不被放大；
- group budget 在 10 个 Writer blocks 后仍满足上限；
- strict CFG cap 在 `guide_scale=3.5/5.0` 下均满足输出上限；
- reader audit on/off 不改变输出；event controller off 时逐元素一致；
- `lambda0=0` 不编码 stage contexts，继续走原 Wan fast path。

### 16.2 兼容和集成测试

1. M3 的 P01–P20 `--check_only`：读取 planimg 和对应 1280×704 首帧；
2. M4 的 P01–P20 `--check_only`：只读取 plan，确认未访问首帧；
3. 对当前旧 JSON 执行 `compat` 编译，不能因缺少 overlay 崩溃；
4. 对 formal subset 执行 `strict`，所有 `unresolved` 必须显式列出；
5. 低分辨率/少步真实模型 smoke：验证 context 投影、query slice、dynamic gate 和 strict shadow pass；
6. M3/M4 各一个 HD97 seed：验证产物、显存和时间开销。

### 16.3 视频验收指标

沿用论文设计中的因果指标，并增加实体/守恒指标：

- Premature Effect；
- Cause Visibility、Cause-before-Effect；
- Intermediate Coverage；
- Directed/Coupled Progress；
- Terminal Persistence/Exclusivity；
- Boundary Artifact 与 Stage Alignment Error；
- Entity Count Error、Unexpected Entity Rate；
- Identity Continuity；
- Source-Sink Coupling；
- Rigid Shape Drift；
- Background Leakage；
- 相邻帧 MAD/SSIM、光流 jerk。

像素变化低不等于正确。P04 式“对象删除后静止”必须由 count/lifecycle 指标判失败。

## 17. 实施顺序

### Phase A：先修正信息通路

1. `EntityLedger + ProtectedPredicateCompiler`；
2. `SemanticAnchorCompiler + PromptBudgeter`；
3. `MinimalPairCompiler + validator`；
4. 新 context/constraint artifacts；
5. 对 P01–P20 做无模型 preflight。

完成标准：真正 T5 文本中可看到实体精确数量、保护项和共享最小关系对，正反例差分仍执行。

### Phase B：动态空间与边界

1. `[5,F,H,W]` route；
2. M3 planned track/corridor/target；
3. smoothstep crossfade；
4. Boundary-Cosine morph 和 derivative cap；
5. M4 saliency support 仅做 audit，再决定是否启用。

完成标准：移动对象离开首帧框后仍位于允许 support；边界 residual derivative 明显下降，背景写入不增加。

### Phase C：聚合 SafeCap

1. active-support layer cap；
2. `CapLedger` group budget；
3. strict base-conditional shadow pass；
4. CFG 后 trust region；
5. estimated 模式与 strict 模式对齐验证。

完成标准：所有 step 的 group/CFG budget 均有可机检上界，不能只报告平均 norm。

### Phase D：生命周期、守恒与 verifier

1. lifecycle/exclusion 规则；
2. closed-world detector；
3. source/sink proxy；
4. terminal persistence；
5. 多 seed 预注册排序。

完成标准：P02/P06/P09/P10 类错误能被自动标记，且指标与逐帧人工观察一致。

### Phase E：单调事件时钟

1. 先实现 `plan_prior`；
2. audit Reader 的事件可分性；
3. 只有多 seed 稳定后启用 `reader_hysteresis`；
4. 与 fixed schedule 做独立消融。

## 18. 最小实验矩阵与 Go/No-Go

为避免同时改太多变量，建议按下列顺序：

| 组 | Anchor/最小对 | 动态支持 | SafeCap | 边界/时钟 |
|---|---|---|---|---|
| A | 当前 iter1 | 静态/全画面 | legacy | 1-token fixed |
| B | 增强 anchor + minimal pair | 同 A | legacy | 同 A |
| C | 同 B | M3 planned / M4 audit | legacy | 2-token smoothstep |
| D | 同 C | 同 C | aggregate strict | cosine trust |
| E | 同 D | 同 D | 同 D | plan-prior clock |
| F | 同 E | M4 saliency | 同 E | reader-hysteresis（仅合格后） |

正式比较保持相同 seed、CFG、分辨率、帧数和 sampler。

Go 条件：

- B 相比 A 降低身份/数量/外物错误，且动作覆盖不下降；
- C 降低背景泄漏和 target 丢失；
- D 降低突变、形变和 CFG 后 delta，同时保留可见因果动作；
- E 降低 Stage Alignment Error 和 terminal reversal；
- M4 saliency 只有在实体定位 precision/recall 达标后才进入主实验。

No-Go 条件：

- anchor 变长但实体错误不降：检查 token attention/截断，而不是继续堆文本；
- minimal pair cosine 仍异常：模板或 relation span 不合格；
- cap 降低伪影但动作消失：检查 support 覆盖和 group 饱和，禁止直接提高所有预算；
- dynamic mask 频繁追错对象：关闭 saliency 写控制，只保留 audit；
- Reader 不能跨 seed 稳定识别事件：保持 plan-prior 开环，不启用闭环时钟。

## 19. 推荐的首轮实现默认值

首轮机制验证建议：

```text
design_revision      = fixed5-causal-r4
crossfade_tokens     = 2
boundary_morph       = cosine_trust
event_clock_mode     = plan_prior
lambda0              = 0.05
token_cap_ratio      = 0.05
layer_cap_ratio      = 0.01
group_cap_ratio      = 0.01
cfg_cap_ratio        = 0.01
cfg_outside_cap      = 0.0025
cap_mode             = aggregate_strict
reader_mode          = audit
verify_mode          = audit
M3 support           = planned
M4 support           = plan_prior（saliency 先 audit）
```

这些值是安全起点，不是最终最优值；必须经过第 11.5 节的统一 sweep。不能把 M3/M4 或单个 PXX 各自手调成不同预算后再做总体优劣结论。

## 20. 最终落地原则

iter2 的核心不是“再写长一点 prompt”，而是把信息按职责放到正确通道：

- global anchor 持续负责身份、数量、场景、保护项和允许变化；
- stage shared anchor 负责明确当前实体账本；
- positive/violation 最小关系对只负责因果方向，保留凸平均差分；
- dynamic support 负责写入位置和运动后的目标区域；
- lifecycle/closed-world/conservation 负责声明同一实例如何变化、什么不能出现、哪些变化必须耦合；
- Boundary-Cosine 与 event clock 负责阶段连续和不可回退；
- aggregate SafeCap 负责限制跨层、跨 support 和 CFG 后的真实总扰动；
- verifier 负责识别文本和路由仍无法硬保证的计数、身份、几何和守恒失败。

只有当某条关键约束至少映射到 `anchor / contrast / route / cap / verifier` 中的一条可执行通路，并在 compiled artifact 中有来源和覆盖记录时，才视为“真正用于生成或验收”；仅存在于 JSON 备注中不算完成。
