# ACE Router M3/M4 完整修改建议与实施方法

日期：2026-09-01  
适用目录：`Project/physGen/code/v1`  
重点版本：M3（I2V + 因果对比残差）、M4（纯 T2V + 因果对比残差）

## 1. 最终建议

保留现有 M0–M4 方法编号以兼容旧实验，但新增两套安全配置：

- **M3-safe**：使用经过审核的 1280×704 初始图像，使用编译后的完整 semantic prompt，并注入受控的 `positive - counterfactual` 因果残差。
- **M4-safe**：完全不向模型提供初始图像，也不使用图像决定尺寸；从 `Pxx-i0.json` 和 `Pxx-cplus-cminus.json` 编译出文字化的初始场景、实体、布局和状态，再注入同样类型的安全因果残差。

研发主线仍应优先 M3-safe，因为初始图像能直接固定实体数量、身份、颜色和空间关系。M4-safe 的价值是回答另一个问题：**在没有图像锚点时，因果残差是否能给纯 T2V 带来超过完整提示词本身的收益。**

必须配套以下基线，否则无法判断收益来源：

| 待验证版本 | 必须比较的基线 | 能回答的问题 |
|---|---|---|
| M3-safe | M1-compiled | 同一初始图像和同一 semantic 下，因果残差是否有效 |
| M4-safe | M0-compiled | 同一完整 semantic 下，纯 T2V 的因果残差是否有效 |
| M3-safe 与 M4-safe | 使用同一份完整 semantic、同一因果对和完全相同参数的配对实验 | 初始图像锚点带来了多少收益 |
| M1-compiled 与 M0-compiled | 使用同一份完整 semantic、且不使用 ACE 的配对实验 | 不考虑 ACE 时，初始图像本身带来了多少收益 |

如果 M3-safe 只超过 M1-short、却没有超过 M1-compiled，改进主要来自更完整的提示词，不能归因于 ACE。M4 同理。

## 2. 当前实现与已确认问题

### 2.1 当前 M3/M4 的实际处理

现有方法映射为：

```text
M3 = I2V + semantic + λ × gate × (CA(q, c+) - CA(q, c-))
M4 = T2V + semantic + λ × gate × (CA(q, c+) - CA(q, c-))
```

M3 和 M4 使用同一个 Wan2.2-TI2V-5B 基础模型、同一组冻结的 cross-attention 权重和同一个视频 query。二者的核心差别是：

- M3 调用模型时传入首帧图像，图像被编码为 I2V 条件。
- M4 调用模型时 `image=None`，只依赖文本和初始噪声生成视频。
- M3/M4 的 counterfactual 都不是 CFG negative；它们只用于 conditional 分支内部计算差分。
- CFG unconditional 仍为空文本；不应恢复通用 negative prompt。

当前残差公式方向没有发现写反或索引错误，主要问题出在条件内容、强度控制和评测设计。

### 2.2 当前必须修正的六个问题

1. `schema.py` 把完整 original prompt 分别拼到 c+ 和 c- 前面，导致实体、外观和动作重复，差分不再只是因果关系。
2. 当前 20 对 c+/c- 的 Wan T5 token 数没有一对完全相等，最大差 11 个 token；句长和句法差也进入了 residual。
3. `Pxx-i0.json` 与 `Pxx-cplus-cminus.json` 的状态、实体、保护项、变化项和 reject 条件目前主要被校验、记录，没有系统进入生成和后验验收。
4. 当前 `lambda0=0.5`，没有相对范数上限；审计中单个采样 block 的 `scaled_delta / semantic` 最大值常在 10% 左右，且还未计算多层累积影响。
5. 当前最后 20% 去噪步骤的 ACE gate 为 0，终态形成和保持阶段没有因果修正。
6. M4 没有图像锚点，却仍使用与 M3 相同的提示结构和注入强度，因此更容易出现缺实体、错误数量、局部特写、事件从中间开始和几何形变。

还需要修复一个可复现性细节：M4 虽然可以读取 `Pxx-i0.json`，当前输出 hash 中却只在 I2V 方法下记录 `i0_metadata_json`。当 M4 开始使用其中的文字规格时，也必须把该 JSON 的路径和 SHA256 写入输入清单。

## 3. 修改后的总体流程

### 3.1 M3-safe 流程

```text
Pxx-i0.json ─────────────┐
Pxx-cplus-cminus.json ───┼─> 条件编译与冲突检查
Pxx-origin.txt ──────────┘          │
                                   ├─> semantic_i2v
Pxx-i0-1280x704.png ─> I0 审核 ────┼─> c+ / c-
                                   └─> preflight/postflight 规则

semantic_i2v + 已审核 I0 ─> Wan I2V conditional
c+ / c- ─> 相同 query 的 cross-attention 差 ─> 安全裁剪 ─> 中层/分阶段注入
生成视频 ─> 数量、身份、触发、顺序、终态、伪影检查 ─> 接受或拒绝
```

### 3.2 M4-safe 流程

```text
Pxx-i0.json ─────────────┐
Pxx-cplus-cminus.json ───┼─> 条件编译与冲突检查
Pxx-origin.txt ──────────┘          │
                                   ├─> 更完整的 semantic_t2v
                                   ├─> c+ / c-
                                   └─> 首段/全局/终态检查规则

semantic_t2v + image=None ─> Wan T2V conditional
c+ / c- ─> 相同 query 的 cross-attention 差 ─> 更保守的安全裁剪 ─> 中层/分阶段注入
生成视频 ─> 首段初态 + 数量/身份/顺序/终态/伪影检查 ─> 接受或拒绝
```

M4 中使用 `Pxx-i0.json` 不等于使用初始图像。该文件在 M4 中只作为“初始场景文字规格”；真正的 PNG/JPG 不读取、不编码、不作为尺寸参考。

## 4. 新增确定性条件编译层

建议新增：

```text
code/v1/ace_router/conditioning.py
```

它负责把两个 JSON 和 original prompt 编译成结构化条件，并保存：

```text
Pxx-compiled-conditioning.json
```

不要在 `infer_ace_router.py` 中临时拼接长字符串，也不要在每次运行时调用 LLM 改写。P01–P20 第一版可以人工整理结构化字段，编译器只做确定性渲染和验证；难以可靠解析的原始自然语言条目必须要求显式 override，不能静默猜测。

### 4.1 推荐的编译结果结构

```json
{
  "schema_version": 1,
  "compiler_version": "ace-conditioning-v1",
  "sample_id": "P02",
  "source_hashes": {
    "origin": "sha256:...",
    "i0_metadata": "sha256:...",
    "causal_metadata": "sha256:..."
  },
  "entities": [
    {"id": "red_ball", "name": "red billiard ball", "count": 1},
    {"id": "blue_ball", "name": "blue billiard ball", "count": 1}
  ],
  "initial_states": [
    {"entity": "red_ball", "attribute": "motion", "value": "approaching"},
    {"entity": "blue_ball", "attribute": "motion", "value": "stationary"}
  ],
  "transitions": [
    {
      "entity": "blue_ball",
      "attribute": "motion",
      "from": "stationary",
      "to": "moving away",
      "trigger": "contact_with:red_ball",
      "phase": "after_contact"
    }
  ],
  "invariants": [
    "one red ball and one blue ball",
    "ball colors and identities remain unchanged",
    "the table and camera remain fixed"
  ],
  "prompts": {
    "semantic_i2v": "...",
    "semantic_t2v": "...",
    "causal_positive": "...",
    "causal_counterfactual": "..."
  },
  "rules": {
    "i0_reject": ["..."],
    "early_video_reject": ["..."],
    "global_video_reject": ["..."],
    "terminal_reject": ["..."]
  },
  "validation": {
    "entity_sets_match": true,
    "single_relation_changed": true,
    "positive_t5_tokens": 18,
    "counterfactual_t5_tokens": 18,
    "token_difference": 0,
    "invariant_transition_conflicts": []
  }
}
```

### 4.2 原 JSON 字段的用途

| 原字段 | 编译后的职责 | M3 | M4 |
|---|---|---|---|
| `required_visible_entities` | 标准实体、数量、可见性规则 | I0 检查 + semantic | semantic + 视频首段检查 |
| `initial_states` | 事件发生前的状态 | I0 检查；只在提示中简述 | 必须明确写入 semantic_t2v |
| `layout_requirements` | 构图、交互空间、裁切规则 | 主要用于 I0 选择 | 必须写入 semantic_t2v |
| `protected_scene_properties` | 全程不变量 | semantic + postflight | semantic + postflight |
| `transition_variables_not_to_freeze` | 允许且要求变化的属性 | 冲突检查 + 时序评测 | 冲突检查 + 时序评测 |
| `reject_if` | 按时间作用域分类的拒绝规则 | I0/全局/终态 | 视频首段/全局/终态 |
| `changed_state_variables` | `from → to`、触发和阶段 | semantic + 评测 | semantic + 评测 |
| `preserved_context` | 身份、数量、材料来源、场景 | shared semantic | shared semantic |
| `event_type` | 选择评测器和未来时序 schedule | 使用 | 使用 |
| `causal_positive/counterfactual` | 单一关系对 | 重写并注入 | 重写并注入 |

### 4.3 时间作用域必须明确

不能把所有条件作为“全视频保持不变”的提示：

- “蓝球静止”是 P02 的初态，不是全局 invariant；视频后段蓝球必须运动。
- “杯中没有水”是倒水任务的初态，不是全局 forbidden；终态应当有水。
- “冰块完整、边界清晰”只用于融化任务的首帧；把它全程保持会直接阻止融化。

建议把规则统一标记为：

```text
i0_only       仅检查 M3 初始图像
early_video   检查 M4 前 5%–15% 帧，确认事件从初态开始
global        全视频保持，例如身份、数量、镜头和背景
terminal      最后 10%–20% 帧检查目标终态
artifact      任意时间出现严重外物、复制、拓扑伪影即失败
```

### 4.4 编译器必须执行的硬校验

1. semantic、c+、c- 均非空且不超过 Wan `text_len=512`。
2. c+ 与 c- 使用完全相同的 canonical entity 集合。
3. c+ 与 c- 使用同一个句法模板，只改变一个因果或时序关系。
4. 使用本地 Wan UMT5 tokenizer 校验；token 数差默认不超过 1，推荐优先做到完全相等。
5. invariant 与 transition 不能锁定同一个待变化属性。
6. 每个 transition 必须有实体、属性、from、to；需要触发的事件还必须有 trigger。
7. `reject_if` 必须归入明确时间作用域；无法分类时编译失败并要求人工确认。
8. 编译结果包含源文件 hash；源 JSON 改变后旧 compiled 文件必须失效。

## 5. M3-safe 的具体修改

### 5.1 初始图像要求

M3 默认只接受：

```text
Pxx-i0-1280x704.png
```

要求：

- 实际尺寸为 1280×704，RGB 可解码。
- 满足 `required_visible_entities`、`initial_states` 和 `layout_requirements`。
- 不触发任何 `i0_only reject_if`。
- 主体不能裁切，必须为后续运动保留足够空间。
- 必须存在 `Pxx-i0-selection.json`，默认 `REQUIRE_SELECTION_MANIFEST=1`。

建议 selection manifest 至少包含：

```json
{
  "schema_version": 1,
  "sample_id": "P02",
  "selected_image": "P02-i0-1280x704.png",
  "image_sha256": "...",
  "width": 1280,
  "height": 704,
  "passed": true,
  "checks": {
    "entities_and_counts": "pass",
    "initial_states": "pass",
    "layout": "pass",
    "reject_if": "pass"
  },
  "reviewer": "human_or_tool_name",
  "notes": []
}
```

若 manifest 的 SHA256 与实际图片不一致，推理必须失败，不能只报警后继续。

### 5.2 M3 semantic 的职责

M3 已有图像约束，因此 `semantic_i2v` 不必再次用很长文本描述所有像素细节，但必须包含：

- 主体身份和数量；
- 目标事件及其阶段顺序；
- 需要保持的背景、材质、镜头；
- 必须变化的状态；
- 合理终态。

不应包含：

- 原样拼接的 `reject_if`；
- 与目标变化冲突的初态锁定；
- 通用 negative prompt；
- 与 c+/c- 完全重复的第二遍事件长句。

### 5.3 P02 的 M3 示例

```text
semantic_i2v:
The same one red billiard ball and one blue billiard ball remain on the fixed
billiard table in a static side view. The red ball approaches and contacts the
stationary blue ball, then the red ball slows while the blue ball moves away.
Ball colors, identities, table appearance, and camera remain unchanged.

c+:
The red ball contacts the blue ball before the blue ball moves away.

c-:
The red ball contacts the blue ball after the blue ball moves away.
```

c+ 与 c- 只交换 `before/after`，其他实体、动作词和句法完全一致。最终措辞仍需用本地 tokenizer 校验。

### 5.4 M3 运行时要求

- `image_for_model` 必须非空。
- 保存实际送入模型的 `reference_input.png` 及 SHA256。
- `semantic_i2v`、c+、c- 来自同一份 compiled-conditioning 文件。
- 使用 `positive_minus_counterfactual`，不再使用名字含混的 `positive_only`。
- residual 仅加到 conditional block，CFG unconditional 保持空文本。
- 所有旧 M3 参数和文本都写入 resolved config，便于复现实验。

## 6. M4-safe：无初始图像因果残差版本的具体修改

### 6.1 必须定义为“纯 T2V”

M4-safe 应满足：

```text
image_for_model = None
initial_image_to_model = false
sizing_image = None
t2v_orientation = landscape
width = 1280
height = 704
```

建议在 `--method M4 --pure_t2v` 下禁止 `--t2v_orientation match_image`。即使图片只用于决定横竖方向而没有送入网络，也会造成实验间接依赖初始图像，不再是严格的纯 T2V 对照。

M4 不要求 selection manifest，因为没有待选择的输入图片。但必须验证并 hash：

- `Pxx-origin.txt`；
- `Pxx-i0.json`，作为初始场景文字规格；
- `Pxx-cplus-cminus.json`；
- `Pxx-compiled-conditioning.json`。

输出 manifest 中应明确记录：

```json
{
  "image_used_as_model_condition": false,
  "image_used_for_sizing": false,
  "image_path": null,
  "image_sha256": null,
  "i0_metadata_used_as_text_specification": true
}
```

### 6.2 M4 semantic 必须比 M3 更完整

没有 I0 时，文本必须承担场景建立职责。`semantic_t2v` 至少包含：

1. 相机视角和景别；
2. 精确实体、数量、颜色和材料；
3. 事件前布局和初态；
4. 可供动作完成的空间；
5. 事件顺序、状态变化和终态；
6. 需要全程保持的身份、背景和镜头。

这里的完整描述只放在 semantic，不复制进 c+ 和 c-。因果残差只负责“正确关系减错误关系”，不能承担建场景和精确计数的全部工作。

### 6.3 P02 的 M4 示例

```text
semantic_t2v:
A static wide side view shows exactly one red billiard ball on the left and
exactly one stationary blue billiard ball on the right on the same level green
billiard table. Several ball diameters separate them, with open table space
beyond the blue ball. The red ball rolls straight toward and contacts the blue
ball, then slows as the blue ball moves away. The two ball identities, colors,
table, framing, and camera remain unchanged.

c+:
The red ball contacts the blue ball before the blue ball moves away.

c-:
The red ball contacts the blue ball after the blue ball moves away.
```

M3 和 M4 应尽量共用同一对 c+/c-；只有 semantic_i2v 与 semantic_t2v 因是否有图像锚点而不同。这样可以减少比较中的混杂变量。

### 6.4 M4 需要额外的首段检查

M3 的初态由 I0 preflight 保证，M4 没有这道门，因此生成后必须检查前 5%–15% 帧：

- 所需实体是否已经出现且数量正确；
- 是否从规定的事件前布局开始；
- 结果是否被提前预置，例如碰撞前蓝球已经运动、倒水前杯中已满；
- 是否一开始就是局部特写或主体被裁掉；
- 镜头是否在事件开始前就发生无关切换。

如果视频没有建立可识别的初态，即使最后一帧看起来接近目标，也应判为时序失败。

### 6.5 M4 的能力边界

M4-safe 可以改善文本层面的事件顺序，但不能承诺补回以下对象级约束：

- 精确数量和身份恒常；
- 小粒子、液位、裂纹等低像素结构；
- 复杂接触轨迹；
- 物质守恒和源/汇对应；
- 稳定的空间布局。

因此 M4 的验收标准应与 M0-compiled 比较，而不应要求它直接达到 M3 的上限。若 M4-safe 仍频繁缺实体或改变身份，应优先判定为“缺少视觉锚点”，不能继续无限提高 ACE 强度。

## 7. 正反事实条件的重写规则

建议 c+/c- 使用受限模板，而不是自由改写：

```text
order:
  c+ = A happens before B.
  c- = A happens after B.

trigger:
  c+ = B begins when A occurs.
  c- = B begins before A occurs.

terminal:
  c+ = After A, entity X becomes state Y.
  c- = After A, entity X remains state Z.
```

规则：

- 每对只改变一个关系，不同时改变动作、对象、数量和结果。
- c+/c- 中实体出现次数也尽量一致。
- 不在 c- 中引入 `not/no/without/avoid` 等否定列表；优先使用肯定式的错误关系。
- 不把 c- 当作“坏画面描述”或 CFG negative。
- 不加入“extra objects、distortion、bad quality”等伪影词；这些交给 shared semantic 和 postflight。
- 若一个任务必须表达多阶段关系，拆成多个可审计 relation，但第一版每次只注入一个主关系，避免残差方向混合。

## 8. 安全因果残差的修改方法

### 8.1 推荐公式

先计算纯因果差：

\[
\Delta r = \operatorname{CA}(q,c_+) - \operatorname{CA}(q,c_-).
\]

应用 layer/step gate 和基础强度：

\[
d_0 = \lambda_0 g_l g_s \Delta r.
\]

再相对当前 semantic cross-attention residual 做非放大型裁剪：

\[
a = \min\left(1,\frac{\tau\lVert r_{sem}\rVert_2}
{\lVert d_0\rVert_2+\epsilon}\right),
\qquad d = a d_0.
\]

最终 block：

\[
h' = h + r_{sem} + d.
\]

其中：

- `tau` 是 ACE residual 相对 semantic residual 的最大比例；
- `a≤1`，只压低过强残差，不把弱残差强行放大到目标比例；
- `lambda_eff = a × lambda0`，用于诊断实际生效强度；
- 计算范数、差分和缩放时使用 FP32，最后转回 semantic dtype。

`tau=0.02` 表示单个 block 中最终 ACE 残差的全局 L2 范数不超过 semantic residual 的 2%，不是“视频只有 2% 变化”。该残差还会经过后续层并参与 CFG；当前 `guide_scale=5` 会放大 conditional 与 unconditional 的差，所以仍需保守。

### 8.2 推荐实现伪代码

在 `model_adapter.py` 增加类似函数：

```python
def relative_residual_cap(candidate, semantic, cap_ratio, eps=1e-12):
    candidate_fp32 = candidate.float()
    semantic_fp32 = semantic.float()
    reduce_dims = tuple(range(1, candidate_fp32.ndim))

    candidate_norm = torch.linalg.vector_norm(
        candidate_fp32, dim=reduce_dims, keepdim=True
    )
    semantic_norm = torch.linalg.vector_norm(
        semantic_fp32, dim=reduce_dims, keepdim=True
    )
    coefficient = torch.clamp(
        cap_ratio * semantic_norm / (candidate_norm + eps), max=1.0
    )
    coefficient = torch.where(
        semantic_norm > eps, coefficient, torch.zeros_like(coefficient)
    )
    applied = candidate_fp32 * coefficient
    return applied.to(semantic.dtype), coefficient
```

随后把当前：

```python
scaled_delta = scale * (positive - counterfactual)
```

改成：

```python
raw_delta = positive.float() - counterfactual.float()
candidate_delta = scale * raw_delta
applied_delta, clip_coefficient = relative_residual_cap(
    candidate_delta, semantic, residual_cap_ratio
)
x = x + semantic + applied_delta
```

范数应按 batch 内每个样本独立计算，不能让一个强 residual 样本压低同 batch 的其他样本。当前通常 batch=1，但实现应保持正确。

### 8.3 第一版不要默认做的处理

- 不把每个 residual 强行归一为单位向量；这会放大很弱、可能无意义的方向。
- 不默认移除 residual 在 semantic 上的平行分量；它可能同时删除有效动作分量，应作为后续独立消融。
- 不同时加入时间 mask、空间 mask、对象轨迹和新的 CFG 逻辑；第一轮应先确认 matched pair + cap 是否有效。
- 不向 unconditional 分支注入 counterfactual。

## 9. M3/M4 推荐默认参数

以下是 pilot 起点，不是已经证明的最优值。

### 9.1 共同生成参数

| 参数 | 建议值 |
|---|---:|
| 分辨率 | 1280×704 |
| 帧数 | 97（满足 Wan 常用的 `4n+1`） |
| FPS | 24 |
| sampling steps | 50 |
| solver | UniPC |
| guide scale | 5.0，第一轮保持不变 |
| shift | 5.0 |
| negative prompt | 空，不恢复通用 negative |
| diagnostics | pilot 使用 `full` |
| seeds | 至少 3 个固定 seed |

### 9.2 模式专用安全参数

| 参数 | M3-safe | M4-safe | 说明 |
|---|---:|---:|---|
| `lambda0` 默认 | 0.10 | 0.08 | M4 无图像锚点，先更保守 |
| lambda sweep | 0.05/0.10/0.20 | 0.04/0.08/0.12 | cap 开启后再扫 |
| residual cap | 0.02 | 0.015 | M4 降低几何和身份扰动风险 |
| cap sweep | 0.01/0.02/0.03 | 0.01/0.015/0.02 | 每次只扫一个变量 |
| layer preset | `mid_b`（14–23） | `mid_b`（14–23） | 减少干预层数 |
| 前 0%–30% step gate | 0.5 | 0.25 | M4 前期优先建立场景 |
| 30%–70% | 1.0 | 0.8 | 中段主注入 |
| 70%–90% | 0.3 | 0.3 | 逐步减弱 |
| 90%–100% | 0.1 | 0.1 | 保留小幅终态约束 |

若要直接比较 M3 与 M4 的“是否有初始图像”差异，必须额外运行一组共享参数：

```text
lambda0=0.10
residual_cap_ratio=0.02
layer_preset=mid_b
step_preset=safe_shared_v1（0.5/1.0/0.3/0.1）
```

不能拿 M3 的 0.10/2% 与 M4 的 0.08/1.5% 直接归因于图像差异。

## 10. 逐文件修改方法

### 10.1 新增 `ace_router/conditioning.py`

新增数据结构和函数：

```text
CompiledConditioning
CompiledEntity
CompiledTransition
CompiledRule
compile_conditioning(sample, tokenizer)
load_or_compile_conditioning(...)
validate_entity_pair(...)
validate_single_relation(...)
validate_token_balance(...)
validate_invariant_transition_conflicts(...)
```

要求：

- 编译是确定性的；同一输入产生同一 JSON 和 hash。
- 同时生成 `semantic_i2v` 与 `semantic_t2v`。
- 保留每条 compiled 内容对应的源字段，方便追踪。
- 编译失败默认阻止推理，不静默退回旧拼接方式。

### 10.2 修改 `ace_router/schema.py`

1. 保留原始 JSON 校验，但给 `AceSample` 增加 `compiled_conditioning` 和路径/hash。
2. 停止让 `positive_text`、`counterfactual_text` 自动返回 `original + causal sentence`。
3. 根据 `conditioning_mode` 明确选择：

```text
original       旧实验复现
compiled_i2v   M1/M3-safe
compiled_t2v   M0/M4-safe
```

4. 对 M3 强制图片和 selection manifest；对 M4 不要求图片。
5. M4 仍加载 i0 metadata，因为要编译场景文字，但图片路径不得进入模型条件。
6. 增加 compiled schema version 和 source hash 一致性检查。

旧属性可以暂时保留为 legacy 接口，但在 resolved config 中必须标记 `conditioning_mode=legacy_concat`，避免与新结果混淆。

### 10.3 修改 `ace_router/model_adapter.py`

1. 把 `positive_only` 的公开名称改为 `positive_minus_semantic`；保留旧名字作为兼容 alias。
2. 给 `ace_block_forward`、`ace_model_forward` 和 `AceModelProxy.forward` 增加：

```text
residual_cap_ratio: float | None
residual_cap_eps: float = 1e-12
```

3. 在 FP32 中计算 `raw_delta → candidate_delta → applied_delta`。
4. cap 按样本独立计算；semantic norm 接近 0 时 applied delta 置 0。
5. `lambda0=0`、step gate 全 0 或 layer gate 全 0 时继续直接回退原 Wan forward，保持数值等价和性能。
6. `cap=None` 仅用于复现旧实验；safe 配置必须提供正数 cap。
7. 增加 finite、非负和上限校验。

### 10.4 修改 `ace_router/pipeline.py`

1. `generate_ace` 增加 residual cap 参数，并逐层传给 proxy。
2. 接受已经编译好的三段文本，不在 pipeline 内拼接。
3. 保持 `counterfactual_as_cfg_negative=False`。
4. 保持 unconditional 空文本；上游为空字符串存在 fallback 时，继续使用与空 token 等价的最小 workaround，但 manifest 中仍记录逻辑语义为空。
5. M4 调用必须显式 `image=None`；M3 调用必须显式传入审核后的 reference image。

### 10.5 修改 `ace_router/diagnostics.py`

每条记录新增：

```text
raw_delta_l2
candidate_delta_l2
applied_delta_l2
candidate_to_semantic
applied_to_semantic
clip_coefficient
was_clipped
lambda_effective
cap_ratio
```

summary 新增：

```text
applied ratio: min / p50 / p95 / max
clip coefficient: min / p50
clip fraction
按 layer 汇总
按 step 区间汇总
全程 finite 状态
```

pilot 使用 full diagnostics。确认没有越界后，正式全量可改为 summary，但 summary 仍应基于完整运行时聚合，而不是只抽样边界点。

### 10.6 修改 `ace_router/schedules.py`

保留旧 `mvp_mid16` 和旧 step gate 用于复现，新增：

```text
layer preset:
  safe_mid10 / mid_b = blocks 14–23 为 1，其余为 0

step preset safe_i2v_v1:
  0–30%: 0.5
  30–70%: 1.0
  70–90%: 0.3
  90–100%: 0.1

step preset safe_t2v_v1:
  0–30%: 0.25
  30–70%: 0.8
  70–90%: 0.3
  90–100%: 0.1
```

把 `step_gate` 改为接收 preset 名称，不能再把唯一 schedule 写死在函数中。

### 10.7 修改 `inference/infer_ace_router.py`

新增参数：

```text
--conditioning_mode original|compiled
--residual_cap_ratio FLOAT
--residual_cap_eps FLOAT
--step_preset legacy|safe_i2v_v1|safe_t2v_v1|safe_shared_v1
--pure_t2v
--postflight_mode off|report|strict
--compiled_conditioning_policy require|rebuild|use_if_valid
```

建议行为：

- M3-safe：`conditioning_mode=compiled`，选择 `semantic_i2v`，强制图片和 selection manifest。
- M4-safe：`conditioning_mode=compiled`，选择 `semantic_t2v`，`pure_t2v=true`，禁止 match_image。
- M0/M1 也允许 compiled，用作公平基线，但不启用 residual。
- M3/M4 的 c+/c- 必须来自 compiled 文件。
- preflight 输出 token 数、实体一致性、冲突、source hash 和所有最终参数。
- M4 的 `input_hashes` 必须包含 i0 metadata 和 compiled-conditioning，即使没有图片。
- resolved config 明确区分 `metadata_used=true` 与 `image_used=false`。
- 输出目录标签可以增加 profile，例如 `M3_I2V_ACE__safe_v1`，但不要破坏 M3/M4 方法编号。

### 10.8 修改 M3/M4 shell 和 YAML

M3-safe 推荐环境变量：

```text
FRAME_NUM=97
MAX_AREA=901120
FPS=24
LAMBDA0=0.10
RESIDUAL_CAP_RATIO=0.02
LAYER_PRESET=mid_b
STEP_PRESET=safe_i2v_v1
CONDITIONING_MODE=compiled
DIAGNOSTICS=full
REQUIRE_SELECTION_MANIFEST=1
POSTFLIGHT_MODE=report
```

M4-safe 推荐环境变量：

```text
FRAME_NUM=97
MAX_AREA=901120
FPS=24
LAMBDA0=0.08
RESIDUAL_CAP_RATIO=0.015
LAYER_PRESET=mid_b
STEP_PRESET=safe_t2v_v1
CONDITIONING_MODE=compiled
PURE_T2V=1
T2V_ORIENTATION=landscape
DIAGNOSTICS=full
REQUIRE_SELECTION_MANIFEST=0
POSTFLIGHT_MODE=report
```

YAML 的 description 也应改成准确描述 compiled semantic、matched causal pair、relative cap，以及是否使用图像。

### 10.9 新增 `ace_router/quality.py`

第一版不必假设自动视觉模型能可靠判断所有物理事件。建议统一生成 `quality_audit.json`，允许每个检查来源为 `automatic`、`manual` 或 `unknown`：

```json
{
  "sample_id": "P02",
  "method": "M3_I2V_ACE__safe_v1",
  "checks": {
    "entity_count": {"status": "pass", "source": "manual"},
    "identity_persistence": {"status": "pass", "source": "manual"},
    "initial_state": {"status": "pass", "source": "automatic"},
    "trigger": {"status": "pass", "source": "manual"},
    "causal_order": {"status": "pass", "source": "manual"},
    "terminal_state": {"status": "pass", "source": "manual"},
    "artifact": {"status": "pass", "source": "manual"},
    "camera": {"status": "pass", "source": "automatic"}
  },
  "strict_pass": true
}
```

只有高置信自动规则或人工确认可以触发 strict reject；不可靠的自动判断应输出 `unknown`，不能伪装成精确分数。

## 11. 输出产物与可复现性

每个 seed 输出目录建议至少包含：

```text
video.mp4
config.resolved.json
contexts.json
compiled-conditioning.json
input_hashes.json
diagnostics.full.json 或 diagnostics.summary.json
quality_audit.json
reference_image.json
reference_input.png        # 仅 M3
run_status.json
```

M4 的 `reference_image.json` 可以保留，但内容必须明确为 `used=false`，且不应存在 `reference_input.png`。

`contexts.json` 应保存实际送入 tokenizer 的完整文本，而不是只保存源 JSON 字段。还应保存每段 T5 token 数、tokenizer 路径/hash 和文本 SHA256。

## 12. 测试修改

### 12.1 条件编译单元测试

- P02 编译结果确定且 hash 稳定。
- c+/c- 实体集合不一致时失败。
- token 数差大于 1 时失败。
- 同时改变两个关系时失败。
- 某属性同时为 invariant 和 transition 时失败。
- `reject_if` 无时间作用域时失败。
- 源 JSON 改变后 compiled hash 失效。

### 12.2 residual 单元测试

- candidate ratio 小于 cap 时完全不放大。
- candidate ratio 大于 cap 时 applied ratio 不超过 cap，允许浮点容差。
- batch 内每个样本独立裁剪。
- semantic norm 为 0 时 applied delta 为 0 且无 NaN/Inf。
- `lambda0=0` 与基础 Wan 路径一致，且不执行额外 c+/c- cross-attention。
- `cap=None` 能复现 legacy 公式。
- M3/M4 的 residual 都为 `positive - counterfactual`。

### 12.3 推理 CLI 测试

- M3-safe 缺图、图片尺寸错误、缺 selection manifest 或 hash 不一致时失败。
- M4-safe 的 `image_for_model`、sizing image 和 image hash 均为空。
- M4-safe + `match_image` 在 pure T2V 模式下失败。
- M4 compiled 输入 hash 包含 `Pxx-i0.json`。
- M0/M1 compiled 基线不执行 ACE。
- negative prompt 始终为空。
- 97 帧、1280×704 和 24 FPS 写入 resolved config。

### 12.4 集成与 smoke test

先对 P02 单 seed、低成本配置运行：

```text
M1-compiled
M3-safe with lambda0=0
M3-safe with cap
M0-compiled
M4-safe with lambda0=0
M4-safe with cap
```

验证 `lambda0=0` 的 M3/M4 分别与对应 compiled baseline 一致或在预期数值容差内，再进行完整 97 帧实验。

## 13. 推荐实验矩阵

先选：

- P02：两球碰撞，适合检查触发和先后顺序；
- P12：蒲公英粒子，适合检查拓扑伪影和数量耗尽；
- P13：倒水，适合检查来源、液位和守恒；
- P18：多次弹跳，适合检查多阶段时序和终态。

固定 3 个 seeds。

### 13.1 M3 主线：36 个视频

```text
4 samples × 3 methods × 3 seeds

M1-short
M1-compiled
M3-safe
```

### 13.2 M4 主线：36 个视频

```text
4 samples × 3 methods × 3 seeds

M0-short
M0-compiled
M4-safe
```

### 13.3 图像锚点配对实验

在完全相同的 semantic 内容范围、c+/c-、seed、lambda、cap、layer 和 step preset 下比较：

```text
M1-compiled vs M0-compiled
M3-safe-shared vs M4-safe-shared
```

这一组应让 I2V 和 T2V 都使用同一份较完整的 `semantic_t2v` 文本；I2V 端即使存在少量文字冗余，也不能换成更短的 `semantic_i2v`。这样两边唯一的条件差才是图片是否进入模型。这一组专门估计初始图像贡献，不与模式专用最优参数混合。

### 13.4 扩大到 P01–P20 的门槛

建议同时满足：

- M3-safe 在至少 3/4 pilot 任务上稳定优于 M1-compiled，而不是只在某个 seed 偶然成功。
- M4-safe 在至少 3/4 pilot 任务上优于 M0-compiled，且实体/伪影失败率不增加。
- M3/M4 所有记录的 `applied_to_semantic` 不超过对应 cap。
- clip fraction 不能接近 100%；否则说明 lambda 仍太大、实际总被 cap 接管。
- P12 不再出现明显树枝/网状拓扑过冲。
- P13 的水流来源、杯中液位和终态至少同步改善。
- 不依赖从大量 seed 中挑最好结果才成立。

若 M4 没有超过 M0-compiled，应停止提高强度，先判定纯 T2V 缺少空间/身份锚点；继续增大 residual 很可能只增加伪影。

## 14. 推荐实施顺序

### P0：先保证比较公平

1. 新增 compiled-conditioning schema 和 compiler。
2. 人工迁移并审核 P01–P20 的 canonical entities、transitions 和 rule scopes。
3. 重写匹配的 c+/c-，完成 T5 token 校验。
4. 新增 M0/M1-compiled 基线。
5. 强制 M3 selection manifest；强制 M4 pure T2V。

### P1：加入 residual 安全性

1. 实现 relative norm cap。
2. 扩展 diagnostics。
3. 新增 M3/M4 step presets。
4. 补齐单元测试和 `lambda0=0` 回退测试。

### P2：小规模实验和验收

1. 先 P02 单 seed smoke test。
2. 再运行两个 36 视频矩阵。
3. 完成人工/自动 `quality_audit.json`。
4. 仅在通过门槛后扩到 P01–P20。

### P3：后续增强，不进入第一版默认

- TRACE-Time：根据 approach/contact/terminal 阶段动态路由 residual。
- TRACE-ST：只有获得可靠对象 mask/ROI 后再限制空间 token。
- 按 event type 学习或选择 schedule。
- semantic 平行分量移除消融。
- guide scale 3/4/5 的独立 sweep。

## 15. 兼容、回滚与风险控制

- 不删除 M0–M4，不覆盖旧输出目录。
- 所有新行为由 profile/config 显式开启，旧配置可继续复现。
- compiled schema、compiler 和 residual algorithm 都写版本号。
- `cap=None + legacy schedule + legacy_concat` 可作为旧行为回滚路径，但不能作为新默认。
- 新输出路径带 `safe_v1` 或独立 run ID，禁止覆盖现有审计样本。
- 若 upstream Wan 文件 hash 变化，继续使用现有 drift guard 阻止静默运行。

主要风险与处理：

| 风险 | 处理 |
|---|---|
| c+/c- token 相等但语义差仍不纯 | 使用受限模板、实体集合校验和人工审核 |
| cap 太小导致 M3/M4 与基线几乎相同 | 先看 raw/candidate 诊断，再逐级扫 cap，不直接恢复 0.5 |
| M4 场景仍不稳定 | 判断为无视觉锚点上限，优先改 semantic/构图或回到 M3 |
| 详细 semantic 自己带来全部收益 | 用 M0/M1-compiled 隔离 |
| 自动质量评测不可靠 | 保留 manual/unknown，不把弱评测器当硬真值 |
| 97 帧显存或运行时间过高 | 先单样本 smoke test；不因资源问题改变主实验定义 |

## 16. 完成标准（Definition of Done）

满足以下条件才算本轮修改完成：

- P01–P20 都有可验证的 compiled-conditioning 文件和 source hash。
- M3 只使用审核通过的 `Pxx-i0-1280x704.png`，selection manifest 全部存在且匹配。
- M4 在代码、config 和 artifact 三处都能证明没有读取或使用图片。
- c+/c- 实体一致、单关系变化、T5 token 差不超过 1。
- M3/M4 使用 relative cap，诊断证明 applied ratio 不越界。
- `lambda0=0` 保持对应基础路径回退正确。
- M0/M1-compiled 基线可运行，能隔离提示词收益。
- 36+36 pilot 具有固定 seeds、完整 diagnostics 和 quality audit。
- M3/M4 是否有效的结论基于 paired baseline 和多 seed，而不是单个好视频。

## 17. 最简决策结论

```text
M3-safe：作为主研发版本。
  图像负责实体、布局和初态；compiled semantic 负责完整任务；
  matched c+/c- 只提供因果关系方向；2% cap 负责安全性。

M4-safe：作为严格的纯 T2V 因果残差实验。
  不读取图片，不用图片定尺寸；semantic_t2v 必须完整建立场景；
  residual 更保守，并增加视频首段初态检查。

判断 ACE 是否有用：
  M3-safe > M1-compiled；M4-safe > M0-compiled。
  只有超过各自完整提示词基线，才能把收益归因于因果残差。
```

相关文件：

- [现有残差实现](../code/v1/ace_router/model_adapter.py)
- [现有数据 schema](../code/v1/ace_router/schema.py)
- [现有诊断实现](../code/v1/ace_router/diagnostics.py)
- [现有层/步 schedule](../code/v1/ace_router/schedules.py)
- [统一推理入口](../code/v1/inference/infer_ace_router.py)
- [质量审计](../code/v1/outputs/ace_router/QUALITY_AUDIT_20260901.md)
- [上一版 6.6 优化分析](ace_router_6_6_metadata_residual_optimization_review_20260901.md)
