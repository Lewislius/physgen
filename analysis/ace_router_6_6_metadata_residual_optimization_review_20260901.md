# ACE 6.6 残差、元数据利用与默认参数优化建议

日期：2026-09-01

说明：问题中的 `Pxx-ix.json` 按当前目录结构理解为 `Pxx-i0.json`。

## 1. 结论

可以优化，而且应先改“残差方向”，再改“残差强度”。当前 ACE 数学实现本身没有发现明显写反或索引错误，但实验设计上有五个主要问题：

1. `positive_text/counterfactual_text` 都是“原始 prompt + 额外句子”，重复了实体和动作词，残差不够纯。
2. 正反事实句法和长度不匹配。20 个样本的正反事实 T5 token 数没有一对完全相等，最大相差 11 个 token。
3. `i0` 与 `cplus-cminus` 中的实体、状态、保留上下文和 reject 条件目前只被校验/记录，没有真正进入生成或后验检查。
4. ACE 仍使用 `lambda0=0.5`，没有归一化、相对范数裁剪或自适应强度；旧审计中单 block 最大残差常达到 semantic residual 的约 10%。
5. residual 是全时空 token 的全局干预，缺少事件阶段和对象区域约束；它不能独立保证数量、身份和物质守恒。

建议默认主线改成：

```text
M1-short（原始基线）
M1-compiled（完整、无重复的元数据增强 prompt）
M3-safe（与 M1-compiled 使用同一 semantic，再加匹配的正反事实残差）
```

M2 只作为消融，M4 暂不作为重点。只有 M3-safe 稳定超过 M1-compiled，才能说明收益来自 ACE，而不只是更详细的 prompt。

## 2. 当前 ACE 代码检查

### 2.1 已确认正确的部分

- semantic、positive、counterfactual 使用同一个视频 query 和同一套冻结的 Wan cross-attention 权重。
- M3/M4 的 block 公式确实是：

  \[
  h'=h+r_{sem}+\lambda g_l g_s(r_+-r_-).
  \]

- M2 实际是 `positive - semantic`，不是反事实差分。
- delta 在 FP32 中计算，之后再转换回 semantic dtype。
- 未启用层走原始 Wan block；`lambda0=0`、step gate 为 0 或全部 layer gate 为 0 时直接回退基础模型。
- upstream source/checkpoint 有哈希保护，降低手写 block forward 与 Wan 版本漂移的风险。
- 当前 25 个单元测试通过，包括公式、调用次数和 `lambda0=0` 回退测试。

因此，当前主要不是“公式代码坏了”，而是“给公式的文本方向不干净、强度无安全控制、验证不够”。

### 2.2 需要修正或补强的部分

| 优先级 | 问题 | 影响 | 建议 |
|---|---|---|---|
| P0 | `schema.py` 自动把 original prompt 再拼到 c+/c- 前 | 重复实体、外观和动作，delta 混入句长与纹理差 | 改成编译后的独立 semantic 与关系级 c+/c-，禁止运行时直接拼接 |
| P0 | i0/causal 元数据未进入生成 | 数量、身份、场景保护写在 JSON 中但模型不知道 | 新增 deterministic condition compiler |
| P0 | 无 norm cap/adaptive lambda | 不同 prompt 同用 0.5，容易无效或过冲 | 默认 `lambda0=0.10`，每 block 残差比例上限 2% |
| P0 | selection manifest 和 postflight 缺失 | 坏首帧和坏视频都会直接保存 | 强制 I0 审核；生成后做数量、身份、顺序和终态检查 |
| P1 | schema 只检查字段存在、词数和禁词 | 没验证实体集合一致、句法匹配或只改一个关系 | 增加 canonical entity/state 校验和 T5 token 差校验 |
| P1 | 最后 20% step gate 为 0 | 终态形成和保持阶段无 ACE | 末段保留很小 gate，而不是完全关闭 |
| P1 | residual 全局作用于所有 token | 背景、细纹理和无关实体也被扰动 | 先做 TRACE-Time；有可靠 mask 后再做 TRACE-ST |
| P1 | diagnostics 只抽样少数 layer/step | 看不到真实最大值、裁剪率和累积影响 | pilot 使用 full diagnostics，记录裁剪前后比例与有效 lambda |
| P2 | `positive_only` 名称实际表示 positive-semantic | 容易误读实验 | 重命名为 `positive_minus_semantic` |

当前 generic negative prompt 已经移除，CFG unconditional 分支使用空文本。因此审计 6.6/6.7 关于旧 negative prompt 的描述属于历史结果；不建议把通用 negative prompt 重新设为默认值。

## 3. JSON 字段值得使用，但必须按职责拆分

不能把所有列表直接拼成一段 prompt。尤其 `initial_states` 和 `reject_if` 经常只适用于第 0 帧；若作为全视频约束，会抑制本来必须发生的变化。

| 字段 | 推荐用途 | 不应怎样使用 |
|---|---|---|
| `required_visible_entities` | 编译精确实体/数量；I0 与视频 entity-count 检查 | 不把“可见走廊、空白区域”等布局项误当实体 |
| `initial_states` | I0 preflight；semantic 中的初态短语；阶段 0 检查 | 不当成全视频 invariant，例如“蓝球静止”之后必须改变 |
| `layout_requirements` | I0 生成/筛选、相机与裁切检查；未来 ROI 来源 | 不进入 causal delta 的正反差分 |
| `protected_scene_properties` | shared semantic、全局 invariant、postflight | 不能包含被允许变化的属性 |
| `transition_variables_not_to_freeze` | 标出允许且必须变化的属性；冲突检查；时序评测 | 不直接作为负向词，也不加入 invariant prompt |
| `reject_if` | 按时间作用域拆成 I0 reject、全局 artifact reject、终态 reject | 不原样注入全视频 prompt |
| `changed_state_variables` | 编译 from→to/单调方向；phase 设计；postflight | 不只保留为日志字符串 |
| `preserved_context` | 实体身份、数量、材料来源和场景 invariant | 不重复塞入 c+ 与 c- 后期待它严格抵消 |
| `event_type` | 选择评测器和以后按类型选择 temporal schedule | 第一轮不要同时为 20 类任务手调 lambda |
| `causal_positive/counterfactual` | 重写成实体一致、句法匹配的关系对 | 不再拼接整个 original prompt |
| `single_changed_relation`、`counterfactual_type` | 编译器验证“只改变一个关系” | 不能只相信 JSON 声明而不检查文本 |
| `warnings/confidence/ambiguities` | preflight 闸门、日志、人工复核 | 不能当图像或视频质量分数 |

### 3.1 为什么必须区分时间作用域

以 P13 倒水为例：`reject_if` 中“玻璃杯已经有水”只表示首帧不合格；目标视频终态恰恰应该让杯中有水。若把这条作为全视频 negative/forbidden 条件，会直接抑制任务完成。

以 P08 融冰为例：`initial_states` 中“冰块完整、边界清楚”只属于初态；`transition_variables_not_to_freeze` 又要求尺寸、边界和固相比例发生变化。两类字段必须分权，否则提示词会一边要求融化，一边锁死冰块轮廓。

### 3.2 数量与伪影限制怎样落地

建议使用两道约束，而不是只依赖文本：

1. **生成前的 affirmative semantic**：例如“画面中始终是同一个红球和同一个蓝球，固定侧视镜头，桌面保持不变”。尽量使用肯定式、简短描述。
2. **生成后的 hard postflight**：检查实体数量、身份连续性、外物、源/汇关系和终态；违反 `reject_if` 或出现 EX/ID/严重 DEF 直接 reject。

文本 cross-attention 对精确计数不可靠，所以“只写 exactly one”不能代替 postflight。

## 4. 建议新增一层条件编译

不要在推理代码里临时拼字符串。建议从原始两个 JSON 生成并保存 `Pxx-compiled-conditioning.json`：

```json
{
  "entities": [{"id": "red_ball", "count": 1, "persistence": "global"}],
  "invariants": ["camera static", "ball identities preserved"],
  "transitions": [
    {"entity": "blue_ball", "attribute": "speed", "from": "zero", "to": "moving", "order": "after_contact"}
  ],
  "semantic_prompt": "one complete natural prompt",
  "causal_positive": "one matched relation sentence",
  "causal_counterfactual": "one matched relation sentence",
  "i0_reject_rules": [],
  "video_reject_rules": [],
  "terminal_rules": []
}
```

编译流程：

1. 给实体、属性和状态建立 canonical ID。
2. 合并 `required_visible_entities + preserved_context`，得到数量和身份。
3. 合并 `protected_scene_properties`，得到 invariant。
4. 用 `changed_state_variables + transition_variables_not_to_freeze` 得到 transition；若某属性同时出现在 invariant 与 transition 中，preflight 直接报错。
5. 将 `reject_if` 按 I0、全局、终态作用域分类。
6. 编译一条无重复的 `semantic_prompt`。
7. 重写 c+/c-：实体词完全相同、句法骨架相同，只交换一个关系。
8. 用本地 Wan T5 tokenizer 检查；建议正反事实 token 数差不超过 1，实体集合必须完全相等。
9. 将编译结果与 hash 写入输出，确保实验可复现。

当前 20 对 c+/c- 的 T5 token 数均不完全相等，差值为 1–11；P10 最大相差 11。这说明现有 schema 的“每句最多 32 个英文词”远不足以保证 cross-attention 差分纯净。

### 4.1 P02 的示意

```text
semantic:
A static side-view billiard table contains the same one red ball and one blue
ball throughout. The red ball approaches, contacts the blue ball, then slows
as the blue ball moves away. The table and camera remain fixed.

c+:
The red ball contacts the blue ball before the blue ball moves; afterward the
red ball slows and the blue ball moves away.

c-:
The red ball contacts the blue ball after the blue ball moves; beforehand the
red ball slows and the blue ball moves away.
```

实际编译器还需按 T5 token 数微调措辞。关键是 semantic 承担实体、数量和场景；c+/c- 只承担关系差异。

## 5. 更安全的残差设计

先计算纯关系差：

\[
\Delta r=\operatorname{CA}(q,c_+)-\operatorname{CA}(q,c_-).
\]

再按原 layer/step gate 缩放：

\[
d_0=\lambda_0g_lg_s\Delta r.
\]

建议默认只压大、不抬小，设置相对 semantic residual 的上限：

\[
a=\min\left(1,\frac{\tau\|r_{sem}\|_2}{\|d_0\|_2+\epsilon}\right),
\qquad d=a d_0.
\]

其中默认 `tau=0.02`，即每个 block 的 ACE 残差不超过 semantic residual 范数的 2%。这种 cap 比“把所有 delta 强行归一到 2%”更稳，因为它不会放大非常弱、可能只是噪声的方向。

自适应强度可等价记录为：

\[
\lambda_{eff}=a\lambda_0.
\]

应记录 `raw ratio / unclipped ratio / clipped ratio / clip coefficient / lambda_eff`。如果方向仍混入大量原语义，可把“去除 delta 在 `r_sem` 上的平行分量”作为单独消融，但不建议第一版默认开启，因为它也可能删掉有效因果分量。

## 6. 建议的保守默认参数

这些是诊断 pilot 的默认假设，不应在小实验前直接用于全量结论。

| 参数 | 当前 | 建议 pilot 默认 | 理由 |
|---|---:|---:|---|
| 主方法 | M2/M3/M4 并列 | M3-safe，M1-compiled 必须配对 | 优先检验真正对比残差 |
| `lambda0` | 0.5 | 0.10；扫 0.05/0.10/0.20 | 旧 0.5 常产生约 10% 比例，线性粗估降至 0.1 接近 2% |
| residual ratio cap | 无 | 0.02；扫 0.01/0.02/0.03 | 给每个 block 安全上限 |
| layer preset | `mvp_mid16` | `mid_b`，仅 14–23 block | 先减少干预层数，便于解释 |
| step gate | 1/0.7/0.3/0 | 0.5/1.0/0.3/0.1 | 中段主导，末段保留小强度终态约束 |
| `guide_scale` | 5.0 | 暂保持 5.0 | 先隔离 ACE 变量；后续单独扫 3/4/5 |
| sampling steps | 50 | 50 | 保持基础 Wan 对照 |
| 分辨率/帧数 | 1280×704 / 97 | pilot 先保持 | 已改善细节和时长；先做显存 smoke test |
| negative prompt | 空 | 保持空 | 不恢复旧通用 negative；样本特定版本只能做独立消融 |
| seeds | 42 | 至少 3 个固定 seed | 报告方差与 pass@1/pass@N |
| diagnostics | summary | pilot 用 full | 验证真实最大比例和裁剪行为 |
| selection manifest | 可选 | I2V 强制 | 坏 I0 不进入视频生成 |

建议的新 step gate 分段为：

```text
0–30%   : 0.5
30–70%  : 1.0
70–90%  : 0.3
90–100% : 0.1
```

这不是理论最优值，只是比“前段最强、末段完全关闭”更保守且更适合诊断的起点。

## 7. 推荐实施顺序

### P0：先修条件和评测

1. 新增 condition compiler 和 `compiled-conditioning.json`。
2. 增加实体集合、T5 token 差、单关系变化和 invariant/transition 冲突检查。
3. 新增 M1-compiled，不加 ACE，用于判断详细 prompt 本身的收益。
4. 为 20 个 I0 建 selection manifest，并设为强制。
5. 增加 `quality_audit.json`：数量、身份、触发、顺序、终态、守恒、外物、镜头。

### P1：再修残差安全性

1. 在 `model_adapter.py` 加 relative norm cap 和 `lambda_eff`。
2. 在 `diagnostics.py` 记录裁剪前后数据和每层/step 汇总。
3. 新增保守 layer/step preset；旧 preset 保留作对照。
4. 增加测试：任何输入下 clipped ratio 不超过 cap，`lambda0=0` 仍与 Wan 完全一致。

### P2：小实验后再做时空路由

先实现 TRACE-Time，利用 transition/terminal 规则把残差分到接近、接触、终态阶段。只有获得可靠 ROI/mask 后再实现 TRACE-ST；不要从自由文本直接猜空间 mask。

## 8. 最小诊断实验

先选 P02（碰撞）、P12（细粒子）、P13（液体守恒）、P18（多次弹跳），比较：

1. M1-short；
2. M1-compiled；
3. M3-safe。

每个样本使用 3 个固定 seed，共 36 个视频。只有满足以下条件才扩到 P01–P20：

- M3-safe 跨 seed 明显超过 M1-compiled；
- entity/identity/artifact 失败率不增加；
- 每 block clipped ratio 始终不超过 1%–3%；
- 不依赖挑最好 seed 才得到好结果；
- P12 不再出现树枝/网状拓扑过冲，P13 的水流来源与液位耦合得到改善。

## 9. 总判断

现有 JSON 确实有价值，但它们目前更像“自然语言规格”，还不是可直接执行的控制信号。最合理的路线不是把所有条目继续拼进 prompt，而是：

```text
元数据 → 结构化编译 → shared semantic / matched causal pair
      → 安全残差注入 → I0 preflight + video postflight
```

先让实体、保护项和变化项各司其职，再用 2% 左右的安全残差测试 ACE 是否真的提供超出完整 prompt 的收益。否则继续用 0.5 的全局残差，只会把“文本更长”和“因果路由有效”混在一起。

相关代码：

- [schema.py](../code/v1/ace_router/schema.py)
- [model_adapter.py](../code/v1/ace_router/model_adapter.py)
- [diagnostics.py](../code/v1/ace_router/diagnostics.py)
- [schedules.py](../code/v1/ace_router/schedules.py)
- [infer_ace_router.py](../code/v1/inference/infer_ace_router.py)
- [QUALITY_AUDIT_20260901.md](../code/v1/outputs/ace_router/QUALITY_AUDIT_20260901.md)

