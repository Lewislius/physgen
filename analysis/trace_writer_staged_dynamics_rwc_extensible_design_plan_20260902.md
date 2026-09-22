# TRACE-Writer v1（修订版）：因果状态轨迹与互斥时空路由

日期：2026-09-02  
适用工程：`Project/physGen/code/v2`  
当前基线：Wan2.2-TI2V-5B + M3 I2V + compiled/original semantic + relation-only `c+−c-` ACE residual  
计划版本：`trace-writer-v1`，设计修订 `fixed5-causal-r3`

本文是可直接拆成实现任务的设计文档。修订目标不是继续增加 phase prompt，而是用更少、更强约束的结构解决两个根本问题：

1. 原设计的 Gaussian phase gate 在全时域都非零，`start` 核心区仍可能收到 `trigger/evolve` residual；归一化和 cap 只能限制总幅度，不能消除语义冲突。
2. 一组 phase `positive/counterfactual` 只是局部文本对，不足以作为完整因果表示；前提、先后、触发、响应、持续、排他和耦合关系没有统一成为 Writer 的输入约束。

修订后的原则是：**因果图是唯一真值，各类物理现象默认统一编译为五个固定职责阶段；文本对只是执行指令，非相邻阶段严格不重叠，相邻阶段只共享 1 或 2 个边界 token。**

---

## 1. 最小目标、范围与删除项

TRACE-Writer v1 只回答一个问题：在不训练或修改 Wan checkpoint 的前提下，能否把现有 ACE 的因果差分写到正确的输出时间和事件区域，从而减少结果预置、错误顺序、中间态删除、终态回生和空间泄漏。

完整流程保持一次完成：

```text
origin prompt + I0 metadata + causal metadata + prepared I0
    -> 一次离线 MLLM：CausalTrace draft
    -> 确定性 diagnostic validator + deterministic compiler
    -> 冻结并 hash 的 input plan / compiled route / issue report
    -> 一次批量 T5 编码
    -> 一条 Wan 去噪轨迹
    -> 默认固定五阶段时间路由 + 粗空间路由的 causal residual
    -> token/global SafeCap
    -> 可选 AuditReader（只读）
    -> 一段 97 帧完整视频
```

明确不做：

- 不按阶段分别生成 clip，不做关键帧插值或视频拼接；
- 不在 denoising loop 中调用 MLLM/VLM；
- 不让 MLLM 选择数值时间点、layer、denoise step、lambda 或 cap；
- 不自由生成 dense mask；
- 不在 v1 实现 Memory/Corrector 闭环；
- 不表示任意大型因果 DAG，只支持一条主要可见因果链，最多两个宏观 transition；
- 不声称实现数值级动量、能量、质量或连续方程。

相对原稿删除或替换：

| 原设计 | 修订决定 | 原因 |
|---|---|---|
| `continuous_3 / triggered_4` 可变阶段数 | 默认统一为五个固定职责的 causal stages | 各类现象共享同一时间接口和实验预算 |
| 全时域 Gaussian phase gate | 改为 1/2-token compact crossfade，默认 1 | 避免长尾泄漏，并允许全局选择平滑度 |
| 多 phase residual 直接求和 | 改为 route 与 violation 两级凸组合 | 关系数量不能放大总预算 |
| 每 phase 固定一对 `c+/c-` | 改为每个固定 stage 一个正例和小型 typed violation bank | 分别表达提前响应、漏触发、漏响应、逆转等失败 |
| v1 中的 CorrectorProtocol/Memory | 删除，只保留 AuditReader hook | 未验证 Reader 前不引入无效抽象与错误反馈 |
| 大量未来 RWC 公式和类别 preset 讨论 | 移出 v1 | 保持实现和实验可证伪 |

保留当前 M3 安全设置作为唯一首轮 preset：

```text
writer blocks = 14..23
lambda0 = 0.10
denoise-step gates = 0.5 / 1.0 / 0.3 / 0.1
token_cap_ratio = 0.10
global_cap_ratio = 0.02
CFG = 3.5
```

---

## 2. 因果状态轨迹：不再让 prompt 充当图

### 2.1 最小表示

一条 validated trace 由四类对象组成：

```text
Entity       谁参与事件
Predicate    哪个可见状态可从视频检查
State        一组同时成立的 predicate=value
Transition   从一个 State 到下一个 State 的可见原因和状态变化
```

v1 只接受如下主链：

```text
z0 --e1--> z1 [--e2--> z2]
```

其中 `z0` 是初态，最后一个 `z` 是必须保持的终态。连续过程可以没有瞬时 trigger，但仍必须有明确的状态方向；复杂过程应先合并为一到两个可见宏观 transition。若无法无歧义地合并，validator 打印不可表达的 error；只要五阶段执行结构仍完整就照常运行，但该样本不得被记为成功的因果实验，也不因此增加更多自由 phase。

支持的因果约束只有六类：

| 类型 | 含义 | Writer 中的使用位置 |
|---|---|---|
| `requires` | effect 只有在 cause/precondition 成立后才允许出现 | `SETUP/ONSET` |
| `precedes` | A 必须先于 B | `ONSET`，构造 wrong-order violation |
| `changes` | transition 将状态从 old 沿指定方向变成 new | `EVOLUTION` |
| `persists` | 已达到的终态在后续不得回生 | `COMPLETION/TERMINAL` |
| `excludes` | 两个状态不能同时存在 | `COMPLETION/TERMINAL` violation |
| `couples` | 多个量必须同向或反向联合变化 | `EVOLUTION/COMPLETION`，如 source↓、sink↑ |

身份、材质、静态相机等不变项写入 `protected_predicates`，由 global semantic、空间保护和生成后检查承担，不再伪装成额外“物理阶段”。

每条关键因果约束都必须被 compiler 映射到 `routing`、`contrast` 或 `protected mask`；只写在 JSON 里但运行时完全不用的关键边，由 validator 作为 issue 打印。

### 2.2 推荐 schema

MLLM 输出结构化事实、来源证据和五阶段候选文本；因果图仍是事实来源，compiler 只接受带 constraint 引用的 stage contrast：

```json
{
  "schema_version": "trace-writer-v1",
  "design_revision": "fixed5-causal-r3",
  "sample_id": "P01",
  "source_hashes": {
    "origin": "...",
    "i0_metadata": "...",
    "causal_metadata": "...",
    "reference_input": "..."
  },
  "global_semantic": "the same black scooter approaches the same upright trash can",
  "entities": [
    {"id": "scooter", "mention": "the same black scooter", "role": "actor", "source_refs": ["origin"]},
    {"id": "can", "mention": "the same upright trash can", "role": "patient", "source_refs": ["origin"]}
  ],
  "predicates": [
    {"id": "gap", "subject": "scooter", "attribute": "gap_to:can", "values": ["visible", "zero"], "observable": "visible separation"},
    {"id": "pose", "subject": "scooter", "attribute": "pose", "values": ["upright", "tilted"], "observable": "body-axis angle"},
    {"id": "speed", "subject": "scooter", "attribute": "motion", "values": ["rolling", "stopped"], "observable": "frame-to-frame displacement"}
  ],
  "events": [
    {"id": "contact", "description": "visible scooter-can contact", "observable": "the visible gap reaches zero", "source_refs": ["origin"]}
  ],
  "states": [
    {"id": "z0", "facts": {"gap": "visible", "pose": "upright", "speed": "rolling"}},
    {"id": "z1", "facts": {"gap": "zero", "pose": "tilted", "speed": "stopped"}}
  ],
  "transitions": [
    {
      "id": "e1",
      "kind": "triggered",
      "from": "z0",
      "to": "z1",
      "cause_event": "contact",
      "preconditions": ["gap=visible", "pose=upright"],
      "effects": [
        {"id": "gap_close", "predicate": "gap", "from": "visible", "to": "zero", "direction": "decrease"},
        {"id": "pose_change", "predicate": "pose", "from": "upright", "to": "tilted", "direction": "increase"},
        {"id": "speed_change", "predicate": "speed", "from": "rolling", "to": "stopped", "direction": "decrease"}
      ],
      "support_kind": "motion_contact_corridor",
      "source_refs": ["origin", "causal_metadata"]
    }
  ],
  "constraints": [
    {"id": "c1", "type": "precedes", "before": "contact", "after": "pose_change"},
    {"id": "c2", "type": "requires", "cause": "contact", "effect": "pose_change"},
    {"id": "c3", "type": "changes", "transition": "e1", "effect": "pose_change"},
    {"id": "c4", "type": "changes", "transition": "e1", "effect": "speed_change"},
    {"id": "c5", "type": "persists", "state": "z1", "facts": ["pose=tilted", "speed=stopped"]},
    {"id": "c6", "type": "excludes", "facts": ["pose=upright", "pose=tilted"], "scope": "same_time"}
  ],
  "stages": [
    {
      "id": "setup",
      "positive": "...",
      "violations": [{"type": "outcome_preset", "text": "...", "constraint_ids": ["c1"]}],
      "observable_check": "...",
      "affected_roles": ["actor", "patient"],
      "support_kind": "motion_corridor",
      "constraint_ids": ["c1", "c2"]
    }
  ],
  "protected_predicates": ["can identity", "can remains upright", "static camera"],
  "grounding": {
    "coordinate_frame": "reference_input",
    "image_sha256": "...",
    "entity_boxes": [],
    "approved": false
  }
}
```

正式 schema 中 `predicate`、`state`、`transition` 和 `constraint` 都必须带 `source_refs`；上例仅为可读性省略了部分重复字段。

### 2.3 确定性校验

validator 按以下顺序执行：

1. **Schema：**严格 key、类型、枚举、长度和最多两个 transition；
2. **Provenance：**source hash/ref 存在，origin prompt 优先于其他 MLLM metadata；
3. **Entity faithfulness：**不增加原输入不存在的实体、数量、材料或施力者；
4. **Graph：**唯一初态、唯一终态、单一连通有向链、无环、transition 的 `from/to` 首尾相接；
5. **State：**effect 前值必须匹配 source state，后值必须匹配 target state；
6. **Causality：**triggered transition 必须有可见 cause、`requires` 和 `precedes`；continuous transition 必须有方向明确的 `changes/couples`；
7. **Persistence/exclusion：**不可逆或破裂任务必须有 `persists/excludes`，transfer 必须有 source/carrier/sink 的 `couples`；
8. **Compile coverage：**每条关键 constraint 都能编译到五阶段中的 route/contrast/protection；
9. **Stage completeness：**默认必须恰好生成五个 stage，每个 stage 都有可见 check、正例和至少一个 typed violation，禁止空阶段或复制相邻文本；
10. **Grounding：**hash、坐标、box 面积、角色覆盖和 protected-region 相减合法。

validator 改为纯诊断器：

```text
plan -> ValidationReport[error/warning] -> 全部打印
                                      -> 不修改 plan
                                      -> 不选择 effective_mode
                                      -> 不回退 ACE-global/trace_time
```

只要数据仍可解析和编译，发现 issue 后仍按调用者请求的 TRACE 模式继续；如果缺字段、错误 stage 顺序或缺失空间框导致运行结构根本无法构造，则打印全部已知 issue 后明确报错停止，仍不切换到其他模式。`lambda0=0` 的原 Wan fast path 是显式配置，不属于 validator 回退。pilot 阶段所有 graph 和空间框仍应人工审核并保存 issue report。

---

## 3. 从因果图默认编译为固定五阶段

### 3.1 为什么默认选择 5，而不是 3、4 或 6

在 `F=25` 且相邻阶段使用 1-token crossfade 时，可供阶段核心区使用的 token 数是 `25-(K-1)`：

| 阶段数 K | 核心 token 总数 | 平均核心长度 | 适配性判断 |
|---:|---:|---:|---|
| 3 | 23 | 7.7 | 过粗，通常只能表示初态—变化—终态，触发和完成过程被合并 |
| 4 | 22 | 5.5 | 可用于简单过程，但容易合并 onset/evolution 或 completion/terminal |
| **5** | **21** | **4.2** | **默认：保留完整因果生命周期，同时每阶段仍有充足核心区** |
| 6 | 20 | 3.3 | 表达更细，但连续/简单过程常被迫制造一个弱阶段，且比五阶段多约 20% stage context |

默认五阶段不是声称所有物理过程天然分成五份，而是选择一个跨类型稳定的 Writer 接口。相比六阶段，它只合并 `BOUNDARY + PRECONDITION` 为 `SETUP`：SETUP 同时描述可信初态和原因准备过程，并明确禁止 effect 提前出现；可见 trigger、主响应、过程完成和终态保持仍然彼此分开。

### 3.2 Universal-5 模板

所有物理现象默认使用同样的五个阶段和顺序：

| 序号 | stage | 固定职责 | 典型可见内容 |
|---:|---|---|---|
| 0 | `SETUP` | 初态与可见前提建立/演进，禁止结果提前出现 | 接近但未接触、完整物体开始下落、初始 source/sink |
| 1 | `ONSET` | 可见 trigger；无离散 trigger 时为首次可检测变化 | 接触、释放、首次破裂、第一滴/第一处融化 |
| 2 | `EVOLUTION` | 主响应、传播、转移或重复过程展开 | 倾斜、下落、传播前沿、source↓/sink↑ |
| 3 | `COMPLETION` | 过程后段完成并趋近目标状态，不预设必须减速 | 落地/打开、传播完成、衰减或剩余差值收敛 |
| 4 | `TERMINAL` | 终态保持和排他关系成立 | 停止、保持破碎/融化、无完整态回生 |

固定的是阶段职责和数量，具体内容来自因果图：

| 现象类型 | `SETUP` | `ONSET` | `EVOLUTION` | `COMPLETION` | `TERMINAL` |
|---|---|---|---|---|---|
| 接触触发 | 接近、响应未发生 | 可见接触 | 接触后响应 | 响应完成/减弱 | 结果保持 |
| 连续变化 | 初态和驱动条件 | 首次可见变化 | 单调演化 | 剩余差值收敛 | 最终状态保持 |
| 传播 | 初始局部扰动前 | 首个局部事件 | 传播前沿移动 | 前沿到达末端 | 全局结果保持 |
| source–sink | 初始源量/汇量 | 首次可见载流 | source↓、carrier、sink↑ | 转移接近完成 | 无画外流、结果保持 |
| 重复/衰减 | 初始释放/首次作用前 | 第一次撞击或峰值 | 重复响应 | 峰值包络收敛 | 静止或稳定振荡 |
| 破裂/不可逆 | 完整体运动 | 首次撞击/裂纹 | 破裂展开 | 碎片状态形成 | 完整体不回生 |

对于无独立可见 trigger 的连续过程，`ONSET` 必须写成“首次可检测变化”，不能虚构外力或工具。对于重复过程，不为每次反弹新增阶段。若输入无法给出五个不同的可见检查，validator 打印 `stage_semantic_weakness`；只要结构仍可编译就继续执行，不自动改变路由模式。

每个阶段包含：

```python
@dataclass(frozen=True)
class CausalStage:
    stage_id: Literal[
        "setup", "onset", "evolution", "completion", "terminal",
    ]
    positive: str
    violations: tuple[Violation, ...]  # each stage selects 1..2
    observable_check: str
    support_kind: str
    source_constraint_ids: tuple[str, ...]
```

允许的 violation 使用固定枚举：

| stage | violation 类型 | 表达的失败 |
|---|---|---|
| `SETUP` | `initial_state_broken`、`outcome_preset`、`precondition_missing` | 初态错误、开场已有结果、原因准备缺失 |
| `ONSET` | `early_effect`、`trigger_missing`、`wrong_order` | 响应提前、无可见触发/首次变化、顺序错误 |
| `EVOLUTION` | `effect_missing`、`wrong_direction`、`coupling_broken` | 有因无果、方向错误、源汇不联动 |
| `COMPLETION` | `envelope_broken`、`overshoot`、`incomplete_transition` | 后段包络错误、过冲或过程未完成 |
| `TERMINAL` | `reversal`、`exclusion_broken` | 终态回生或互斥状态共存 |

因果图中的 `requires/precedes` 编译到 `SETUP/ONSET`，`changes/couples` 编译到 `EVOLUTION/COMPLETION`，`persists/excludes` 编译到 `COMPLETION/TERMINAL`。阶段数固定，但因果约束不会因为合并初态和前提而丢失。

### 3.2 多 counterfactual 不做加法

对 stage `j`，正例 residual 和多个反例使用凸平均：

\[
r_j^+=CA(q,c_j^+),\qquad
r_j^-=\sum_{v\in V_j}\rho_{jv}CA(q,c_{jv}^-),
\]

\[
\rho_{jv}\ge0,\qquad \sum_v\rho_{jv}=1,\qquad
\mathcal O_j=r_j^+-r_j^-.
\]

首轮所有 violation 等权；不允许 MLLM 输出权重。增加 `wrong_order` 或 `reversal` 等关系只改变对比方向，不会按关系数量放大 residual。每条 negative 必须只违反一个已引用的 constraint，并保持实体、镜头和无关属性不变。

### 3.3 P01 的固定五阶段

| stage | positive | 主要 typed violations |
|---|---|---|
| `SETUP` | The same upright scooter approaches as the visible gap narrows, without tilting before contact. | `outcome_preset`；`precondition_missing` |
| `ONSET` | The narrowing gap reaches visible scooter-can contact before the scooter starts to tilt. | `trigger_missing`；`wrong_order` |
| `EVOLUTION` | Only after contact, the same scooter progressively tilts and slows beside the can. | `effect_missing`；`wrong_direction` |
| `COMPLETION` | The scooter's remaining motion diminishes as its tilted pose approaches rest beside the can. | `envelope_broken`；`incomplete_transition` |
| `TERMINAL` | The same scooter remains tilted and stopped beside the same upright can. | `reversal`；`exclusion_broken` |

这里不仅文本顺序固定，因果约束也决定 contrast 的位置：接触前只抑制提前倾斜，接触阶段抑制漏触发和倒序，接触后推动倾斜/减速，末段抑制不衰减和终态回生。

---

## 4. 默认五阶段的不等长固定时间路由

### 4.1 F=25 的统一布局

v1 只允许全局选择 1 或 2 个 latent temporal token 作为相邻阶段的 crossfade，默认值为 1。一个 latent token 约对应 4 个输出帧；禁止 MLLM 或单个样本选择宽度。

```text
stage count = 5
crossfade width = 1（默认）或 2 latent temporal tokens
core lengths when crossfade=1 = [4, 3, 7, 4, 3]
core lengths when crossfade=2 = [3, 2, 6, 3, 3]
schedule_version = fixed5-causal-r3
```

阶段长度不取相同值，理由如下：

| stage | 默认 core | 2-token crossfade 时 | 长度依据 |
|---|---:|---:|---|
| `SETUP` | 4 | 3 | 需要建立初态和前提，但不应占据整片 |
| `ONSET` | 3 | 2 | 接触、释放、首滴等通常是短暂可见事件 |
| `EVOLUTION` | 7 | 6 | 主运动、传播和转移需要最大时间预算 |
| `COMPLETION` | 4 | 3 | 用于完成传播、落地、衰减或收敛 |
| `TERMINAL` | 3 | 3 | 至少保留约 12 个输出帧检查终态持续性 |

默认 `crossfade=1` 布局：

```text
f=00..03  SETUP core
f=04      SETUP      <-> ONSET
f=05..07  ONSET core
f=08      ONSET      <-> EVOLUTION
f=09..15  EVOLUTION core
f=16      EVOLUTION  <-> COMPLETION
f=17..20  COMPLETION core
f=21      COMPLETION <-> TERMINAL
f=22..24  TERMINAL core
```

`crossfade=2` 布局：

```text
f=00..02  SETUP core
f=03..04  SETUP      <-> ONSET
f=05..06  ONSET core
f=07..08  ONSET      <-> EVOLUTION
f=09..14  EVOLUTION core
f=15..16  EVOLUTION  <-> COMPLETION
f=17..19  COMPLETION core
f=20..21  COMPLETION <-> TERMINAL
f=22..24  TERMINAL core
```

两套布局都恰好覆盖 25 tokens，并保持主要事件锚点大致一致。1-token 使用 `0.5/0.5`；2-token 使用对称权重 `0.75/0.25 -> 0.25/0.75`。两者都只允许相邻阶段重叠。

### 4.2 路由不变量与伪代码

1/2-token crossfade 不再增加曲线类型超参数。必须保持：

1. 每个核心区只有当前阶段权重为 1；
2. 边界 token 只有两个相邻阶段，权重按上述固定表取值；
3. 非相邻阶段严格为零；
4. 每个 token 上 `sum_j w_j(f)=1`；
5. 阶段数、核心长度和边界位置均为版本化常量。

```python
FIXED_STAGE_IDS = (
    "setup", "onset", "evolution", "completion", "terminal",
)
CORE_LENGTHS_BY_CROSSFADE = {
    1: (4, 3, 7, 4, 3),
    2: (3, 2, 6, 3, 3),
}


def fixed5_route_weights(*, frames: int = 25, crossfade_tokens: int = 1) -> torch.Tensor:
    if frames != 25:
        raise ValueError("fixed5-causal-r3 is defined for the HD97 F=25 grid")
    if crossfade_tokens not in CORE_LENGTHS_BY_CROSSFADE:
        raise ValueError("crossfade_tokens must be 1 or 2")
    weights = torch.zeros(5, frames, dtype=torch.float32)
    core_lengths = CORE_LENGTHS_BY_CROSSFADE[crossfade_tokens]
    cursor = 0
    for stage_id, core_length in enumerate(core_lengths):
        weights[stage_id, cursor:cursor + core_length] = 1.0
        cursor += core_length
        if stage_id + 1 < len(FIXED_STAGE_IDS):
            new_weights = {1: (0.5,), 2: (0.25, 0.75)}[crossfade_tokens]
            for new_weight in new_weights:
                weights[stage_id, cursor] = 1.0 - new_weight
                weights[stage_id + 1, cursor] = new_weight
                cursor += 1
    assert cursor == frames
    assert ((weights > 0).sum(0) <= 2).all()
    assert torch.allclose(weights.sum(0), torch.ones(frames), atol=1e-6)
    return weights
```

### 4.3 P01 中的实际交接

在 P01 中，`ONSET` 对应接触，`EVOLUTION` 对应接触后的倾斜/减速：

- `f=00..03` 只有 SETUP，接触和倾斜 residual 都不会提前进入；
- `f=04` 从接近状态以 `0.5/0.5` 交给可见接触；
- `f=05..07` 由 ONSET 独占，用于建立接触先于响应；
- `f=08` 才把接触交给接触后的倾斜/减速；
- EVOLUTION 与 SETUP 永不直接重叠。

### 4.4 开环限制

固定五阶段是统一的**规范化事件时间**，不是对所有物理现象真实持续时间相同的声明。v1 没有可靠 Reader，因此不能根据真实接触或融化进度动态移动边界；快速碰撞和缓慢扩散都使用同一个五阶段骨架，只改变阶段内容，不改变数量与预算。

第一轮必须报告不同现象上的 `Stage Alignment Error`。若五阶段在某类现象上持续出现实际事件早于/晚于目标 stage，下一版本应研究 Reader 驱动的时间扭曲，而不是增加默认阶段数或让 MLLM 输出任意时间点。

---

## 5. 空间路由与四轴 Writer

Wan2.2-TI2V-5B 对 97×704×1280 输出的 token grid 为：

```text
F = 25, H = 22, W = 40, valid tokens = 22,000
flatten order = f -> y -> x
i = ((f * H) + y) * W + x
```

空间 support 只保留五种固定构造：

| support | 用途 |
|---|---|
| `role_union` | 静态对象或容器内部 |
| `motion_corridor` | 起点、目标及中心线宽走廊 |
| `contact_interface` | 两对象接触边界与短 corridor |
| `motion_contact_corridor` | 接近、接触和响应共享的宽 support |
| `source_stream_sink` | source、carrier/stream、sink 的联合区域 |

mask 直接在 25×22×40 grid 上生成。soft box/corridor 在阈值 `mask_epsilon` 外显式置零，再减去 protected dynamic regions；不能先创建 HD dense mask，也不能让 MLLM 自由绘制 mask。空间 grounding 未获批准时 validator 打印问题；若调用者仍请求 `trace_st` 但缺少可构造的 box，运行时明确报错，不自动改成 `trace_time`。

对固定 stage `j` 和 token `i=(f,x,y)`：

\[
a_{ji}=w_j(f)\beta_j(x,y),\qquad
0\le\beta_j\le1,
\]

\[
D_i=\lambda_0 g_\ell g_k\sum_j a_{ji}\mathcal O_{j,i}.
\]

因为任一输出时间最多两个相邻 `w_j` 非零且 `sum_j w_j=1`：

\[
\sum_j a_{ji}\le1.
\]

这只是预算约束，不假设不同 residual 方向一致。交界处记录相邻 operator cosine、凸组合前后 norm 和 ROI 内外比值；v1 不引入未经验证的“冲突自动增益”，也不会为了抵消而放大 lambda。

### 5.1 只计算真正激活的 query

旧稿把 query slicing 放到 v1.1，会继续对全序列计算所有 phase CA。修订版把它纳入 v1：

```python
for stage in runtime.stages:
    gate = runtime.token_gates[stage.stage_id]       # [B, L, 1], exact zeros
    indices = runtime.active_indices[stage.stage_id] # [N]，同一 plan 的 batch 共享
    query_slice = query.index_select(1, indices)   # [B, N, C]

    positive = cross_attn(query_slice, stage.positive_context)
    negative = sum(
        rho * cross_attn(query_slice, context)
        for rho, context in stage.violation_contexts
    )
    local = gate.index_select(1, indices).float() * (positive.float() - negative.float())
    candidate.index_add_(1, indices, local)
```

这样做有两个目的：

- 非当前 stage 不仅 gate 为零，而且不会对该 token 执行 causal cross-attention；
- 总 query 工作量主要随覆盖 token 数变化，而不是简单按五阶段乘以 22,000。

实际实现必须验证 Wan cross-attention 对 gather 后 query 的 batch/sequence 语义；在 mock 和真实低分辨率 smoke 通过前，不得假定 slicing 与 full-query-then-mask 数值等价。

### 5.2 SafeCap

global semantic residual 始终保留，TRACE 只加 causal delta：

\[
h'=h+r_{sem}+D^{final}.
\]

先逐 token 限幅，再做全局限幅：

\[
\|D_i^{token}\|_2\le0.10\|r_{sem,i}\|_2,
\qquad
\|D^{final}\|_2\le0.02\|r_{sem}\|_2.
\]

unconditional CFG forward 始终关闭 TRACE；`lambda0=0` 必须直接走原 Wan fast path，不编码 route contexts。

---

## 6. Runtime、Reader 与产物

v1 runtime 只保存生成所需内容：

```python
@dataclass
class RoutingRuntime:
    plan: CausalTrace
    stages: tuple[CausalStage, ...]  # default fixed length = 5
    positive_contexts: tuple[torch.Tensor, ...]
    violation_contexts: tuple[tuple[torch.Tensor, ...], ...]
    violation_weights: tuple[tuple[float, ...], ...]
    token_gates: dict[str, torch.Tensor]
    reader: AuditReader | None
```

不在 v1 定义 `ControllerMemory`、`CorrectorProtocol` 或 phase gain。`AuditReader` 只记录：

- 每个 state/transition ROI 的 hidden mean/std/norm；
- 五个 stage 在输出时间上的可分性；
- crossfade 中相邻 operator cosine 和 norm；
- ROI 内外 applied residual 比；
- 相邻 denoise step 的 normalized drift。

Reader 开关必须产生逐元素相同的 Writer tensor；只有未来 Reader 能跨 seed 稳定读出五个规范化状态阶段，才另立版本设计闭环 Corrector。

最小产物：

```text
trace.plan.draft.json
trace.plan.validation.json
trace.route.compiled.json
trace.diagnostics.jsonl
reader.audit.jsonl             # reader=audit 时
trace-grounding.png            # trace_st pilot 时
```

`trace.route.compiled.json` 至少记录：plan/input hash、固定五阶段顺序、constraint coverage、固定 temporal spans、每个 token 的 active stage 摘要、空间 mask area、flatten order、请求模式和全部 validation issues；`fallback_selected` 固定为 false。

---

## 7. 实现结构与顺序

不修改 `/home/liuzhirui/model/Wan2.2`。实现放在 `Project/physGen/code/v2`，并保持 M0–M4 和历史 ACE 输出可复现。

```text
ace_router/
  trace_schema.py          # Entity/Predicate/State/Transition/Constraint
  trace_validation.py      # provenance、graph、causality、budget、grounding
  trace_compile.py         # graph-referenced stage candidates -> fixed CausalStage[5]
  trace_masks.py           # fixed5 1/2-token partition + spatial support + flatten
  trace_runtime.py         # contexts、query indices、生命周期
  trace_writer.py          # query slicing、violation 凸组合、SafeCap
  trace_model_adapter.py   # 冻结 Wan proxy 与 lambda-zero fast path
  trace_pipeline.py        # 一次 T5 batch 编码与 runtime 准备
  controllers.py           # 只读 AuditReader

tools/
  compile_trace_plan.py
  validate_trace_plan.py

tests/
  test_trace_schema_validation.py
  test_trace_compile_masks.py
  test_trace_runtime_writer.py
  test_trace_model_pipeline.py
```

公开运行参数保持最少：

```text
ROUTER_MODE=ace_global|trace_time|trace_st
TRACE_PLAN_ROOT=/.../demo
TRACE_CROSSFADE_TOKENS=1|2
TRACE_VALIDATION=print
READER_MODE=off|audit
```

layer/step gate、不等长五阶段布局、1/2-token crossfade、caps 和 mask epsilon 由 `fixed5-causal-r3` preset 固定并写入 manifest，不开放给 MLLM，也不在第一轮样本级调参。

实现顺序：

1. schema + validator，以 P01/P08 人工图验证 triggered/continuous 都能完整映射到五阶段；
2. deterministic compiler，检查默认恰好五阶段和每条关键 constraint 的 compile coverage；
3. fixed5 1/2-token partition，完成 exact-zero、邻接、权重和与固定 grid 测试；
4. spatial supports、protected subtraction 和 grounding overlay；
5. 一次批量编码 route positive/violation contexts；
6. query gather/scatter 与 full-query reference 对齐；
7. TRACE block、两级凸组合、SafeCap 和 diagnostics；
8. AuditReader neutrality、CLI、manifest 和 failure artifacts；
9. P01/P08 低分辨率 smoke，再跑 HD97 机制集。

当前 `ace_block_forward` 的 self-attention/FFN 镜像逻辑应抽为 ACE/TRACE 共用 helper，并保留 upstream hash guard，不能复制第二份长期漂移的 Wan block。

---

## 8. 必须通过的测试

| 测试 | 验收标准 |
|---|---|
| diagnostic-only validator | 所有 issue 打印，report 不含 `accepted/effective_mode`，不选择 fallback |
| graph diagnostics | 分叉、环、断链、超过两个 transition 全部形成 error issue |
| state continuity | effect 前后值与 source/target state 不一致时形成 error issue |
| causal completeness | triggered 缺 `requires/precedes`、不可逆任务缺 `persists/excludes` 时形成 issue |
| compile coverage | 每条关键 constraint 映射到 stage/contrast/protection |
| fixed stage count | triggered/continuous/repeated/transfer 默认都恰好编译为五阶段 |
| stage distinctness | 五阶段均有可见 check，且相邻 positive 不得相同或只做同义改写 |
| no Gaussian tail | `SETUP` core 中其他 stage 必须逐元素为 0 |
| adjacency | 任一 temporal token 最多两个相邻 stage 激活 |
| partition of unity | 每个 `f` 上 `sum(w)=1` |
| fixed layout | crossfade 1 的 `[4,3,7,4,3]` 与 crossfade 2 的 `[3,2,6,3,3]` 均逐项一致 |
| boundary weight | 1-token 为 `0.5/0.5`；2-token 为 `0.75/0.25 -> 0.25/0.75` |
| violation convexity | 每个 stage 的 `sum(rho)=1`，复制相同 violation 不改变结果 |
| flatten alignment | `((fH)+y)W+x` 与 Wan token 顺序一致 |
| padding/query safety | padding 和 inactive query 的 causal delta 精确为 0 |
| sliced/full equivalence | 同 gate 下 gather/scatter 与 full-query-then-mask 在容差内一致 |
| protected region | protected dynamic region 的 causal gate 为 0 |
| cap safety | token ratio ≤0.10，global ratio ≤0.02，全部 finite |
| lambda-zero | 直接调用原 Wan，不编码/执行 causal branch |
| Reader neutrality | reader on/off 不改变输出 tensor |
| deterministic artifact | 相同 plan/input/preset hash 产生相同 compiled route |

重点回归断言：P01 的 `ONSET` 在 `f=00..03` 严格为零，`EVOLUTION` 在 `f=00..08` 严格为零；五阶段权重与 violation 数量都不能改变 TRACE 的总预算。

---

## 9. 最小实验矩阵与止损条件

主实验前只做一次小型阶段数消融：在 P01（触发）、P08（连续）、P13（source–sink）、P18（重复）各一个固定 seed 上比较 3/4/5/6 阶段，所有组都保持 1-token crossfade、相同 lambda 和总 cap。3/4/6 仅通过实验开关做确定性 merge/split，不属于正式输入 schema；默认 preset 为五阶段。只有五阶段在多数类型上同时劣于同一个候选时，才允许全局改默认值，禁止按样本或物理类别分别选择阶段数。

第一轮只保留能回答设计问题的五组：

| ID | 方法 | 时间路由 | 因果表示 | 空间 |
|---|---|---|---|---|
| A | ACE-global-safe | 无 | 单对 relation | 无 |
| B | 旧 Gaussian-phase | 全时域软尾 | phase pairs | 无 |
| C | Causal-Fixed5-Time | 默认五阶段、1-token symmetric crossfade | typed graph + violation bank | `beta=1` |
| D | Causal-Fixed5-ST | 同 C | 同 C | approved support |
| E | Causal-order-shuffled | 保持五阶段和预算但打乱 `ONSET/EVOLUTION` | 同 C | 与 C/D 配对 |

核心样例：P01 接触、P04 传播、P08 连续融化、P11 破裂排他、P13 源流汇、P18 衰减反弹、P20 多物理隔离。先使用相同 seed 做低成本 smoke；机制未通过不扩到全量 P01–P20。

必须分别评估：

- `Premature Effect`：原因前结果是否已出现；
- `Cause Visibility`：trigger 是否真实可见，而非只出现响应；
- `Cause-before-Effect`：原因是否先于响应；
- `Intermediate Coverage`：是否出现可检查的中间态；
- `Directed/Coupled Progress`：方向、传播或 source/sink 联动是否正确；
- `Terminal Persistence/Exclusivity`：终态是否保持、互斥状态是否共存；
- `Boundary Artifact`：crossfade 附近的结构突变、双影、局部纹理爆裂；
- `Stage Alignment Error`：实际可见事件相对目标 stage 的提前/滞后；
- `Identity/Quality/Locality`：主体身份、画质和 ROI 外泄漏。

Go 条件：

1. C 相比 B 显著降低 cause 前 residual 泄漏和 boundary artifact，且动作幅度不明显下降；
2. C 相比 A 提升 cause-before-effect、intermediate coverage 或 terminal persistence；
3. 正确顺序 C 必须优于 shuffled E，否则不能声称因果路由有效；
4. D 相比 C 降低错误对象/来源/区域泄漏，且 protected 区域不退化；
5. 所有 exact-zero、partition、query slicing、cap 和 lambda-zero 测试通过。

止损与解释：

| 结果 | 解释 | 决策 |
|---|---|---|
| B≈C 且二者>A | 主要收益来自更多局部文本，不是 fixed5 causal route | 停止因果路由增量声明 |
| C<E 或 C≈E | Wan 未利用正确因果顺序 | 重做 graph-to-text 表示，不增加 lambda |
| C 降伪影但动作消失 | 固定 stage support、query slicing 或 cap 抑制过强 | 查 token coverage/cap，不恢复 Gaussian 长尾 |
| cause 可见但 effect 缺失 | transition operator 无控制力 | 研究 state token/projector，不继续堆 counterfactual |
| 人工图仍失败 | Writer/基座不可控 | 不进入 Reader/Corrector |
| Reader proxy 不可分 | hidden 中状态不可可靠读取 | Reader 仅保留日志，不设计闭环 |

---

## 10. 最终判断

修订后的 TRACE-Writer v1 不再把“多个阶段 prompt 的平滑叠加”当作因果建模。它使用：

```text
一条受限的可见因果状态链
    + 默认统一的 SETUP/ONSET/EVOLUTION/COMPLETION/TERMINAL 五阶段
    + 确定性 typed violation 编译
    + 默认 1-token、可选全局 2-token crossfade 与非相邻严格为零的时间路由
    + 相邻/多反例两级凸组合
    + 事件区域空间 support
    + 现有两级 SafeCap
```

这套设计不能保证开环计划与实际生成事件完全同步，也不声称“五等份时间”是自然物理定律；五阶段是不等长的默认控制接口。1-token crossfade 优先保留阶段独占作用域，2-token 是全局更平滑选项，因果图和 typed violations 承担完整关系表达。若五阶段在消融和正确顺序实验中不能胜过更简单的 3/4 阶段及 ACE-global，就应继续做减法，而不是增加 Writer、prompt 或闭环复杂度。
