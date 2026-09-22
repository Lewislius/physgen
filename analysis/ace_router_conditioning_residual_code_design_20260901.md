# ACE Router 条件增强与安全残差：本轮代码实现设计

日期：2026-09-01  
实现范围：`Project/physGen/code/v1`  
目标版本：新版 M3、M4

## 1. 本轮范围

本轮只实现两部分：

1. **条件增强**：读取现有 `Pxx-i0.json`、`Pxx-cplus-cminus.json` 和 origin prompt，确定性生成 M3/M4 使用的 semantic prompt。
2. **残差优化**：对 `positive - counterfactual` 残差增加相对 semantic residual 的非放大型 L2 裁剪，并启用更保守的层/步配置。

本轮明确不实现：

- 初始图像 selection manifest 的新增或自动审核；
- `reject_if` 自动判定；
- 视频生成后的数量、身份、时序、终态和伪影 reject；
- 对象 mask、轨迹或 TRACE-ST；
- 自动重写现有 c+/c- 数据集。

因此，这是一次生成条件和注入安全性的迭代，不是完整质量控制系统。

## 2. 条件增强流程

新增 `ace_router/conditioning.py`：

```text
AceSample
  ├─ original_prompt
  ├─ i0_metadata
  │    ├─ required_visible_entities
  │    ├─ initial_states
  │    ├─ layout_requirements
  │    ├─ protected_scene_properties
  │    └─ transition_variables_not_to_freeze
  └─ causal_metadata
       ├─ changed_state_variables
       ├─ preserved_context
       ├─ causal_positive
       └─ causal_counterfactual
             │
             ▼
      compile_conditioning(...)
             │
             ├─ semantic_i2v（M3）
             ├─ semantic_t2v（M4）
             ├─ relation-only c+
             └─ relation-only c-
```

### 2.1 M3 semantic

M3 有初始图像，因此 semantic 包含：

- origin 事件描述；
- 必须出现的实体和数量；
- 事件开始时的初态；
- 必须发生的状态变化；
- 允许变化、不能被文本锁死的变量；
- 全程保持的身份、场景和材质。

布局细节不重复写入 M3 semantic，主要由初始图像提供。

### 2.2 M4 semantic

M4 不使用图像，因此在 M3 内容上额外加入：

- `pre_event_image_prompt` 描述的初始场景；
- `layout_requirements` 描述的主体位置、交互空间和构图。

M4 YAML 固定使用横向 1280×704；现有调用仍为 `image=None`。

### 2.3 c+ 和 c- 的关键改动

旧实现：

```text
positive_text       = origin prompt + causal_positive
counterfactual_text = origin prompt + causal_counterfactual
```

新实现：

```text
positive_text       = causal_positive
counterfactual_text = causal_counterfactual
```

origin 和元数据增强内容只进入 semantic；c+/c- 只承担因果关系差异。该行为对 original/compiled 两种 conditioning mode 都成立。

本轮没有自动重写 20 组 c+/c-，所以已有正反事实句的 token 长度差仍会被记录，但暂不作为阻止运行的条件。后续可以单独人工改写数据。

### 2.4 输出记录

每个视频目录新增：

```text
conditioning.compiled.json
```

其中保存：

- compiler version；
- semantic variant；
- 实际 semantic/c+/c-；
- 参与编译的元数据字段；
- `origin_prompt_prefixed=false`。

当 `conditioning_mode=compiled` 时，M4 的输入 hash 也记录 `Pxx-i0.json`，因为该文件虽然不是图像条件，却参与 semantic 编译。

## 3. 安全残差设计

当前 M3/M4 的关系差为：

\[
\Delta r = CA(q,c_+) - CA(q,c_-).
\]

先应用基础强度和 gate：

\[
d_0 = \lambda_0 g_l g_s \Delta r.
\]

然后按每个 batch 样本独立计算 semantic 相对上限：

\[
a = \min\left(1,
\frac{\tau\lVert r_{sem}\rVert_2}
{\lVert d_0\rVert_2+\epsilon}\right),
\qquad d=a d_0.
\]

最终：

\[
h'=h+r_{sem}+d.
\]

设计含义：

- `a` 最大为 1，只裁剪过强 residual，不放大弱 residual；
- 所有范数和差分使用 FP32；
- `lambda0=0`、layer gate 全 0 或 step gate 为 0 时仍回退原 Wan forward；
- `residual_cap_ratio=None` 可复现旧的无 cap 行为；
- cap 是单 block 上限，并不等于最终视频只改变相同比例。

## 4. 新版 M3/M4 参数

共同参数：

```text
1280×704
97 frames
24 fps
50 sampling steps
guide_scale=5.0
negative prompt=empty
```

M3：

```text
CONDITIONING_MODE=compiled
LAMBDA0=0.10
RESIDUAL_CAP_RATIO=0.02
LAYER_PRESET=mid_b          # blocks 14–23
STEP_PRESET=safe_i2v_v1    # 0.5 / 1.0 / 0.3 / 0.1
DIAGNOSTICS=full
```

M4：

```text
CONDITIONING_MODE=compiled
LAMBDA0=0.08
RESIDUAL_CAP_RATIO=0.015
LAYER_PRESET=mid_b
STEP_PRESET=safe_t2v_v1    # 0.25 / 0.8 / 0.3 / 0.1
DIAGNOSTICS=full
M4_ORIENTATION=landscape
```

这些是本轮 smoke/pilot 起点，不是已经验证的最优参数。

## 5. 代码修改点

### 新增

- `ace_router/conditioning.py`
  - `CompiledConditioning`
  - `compile_conditioning()`
  - original/compiled 两种模式
  - M3/M4 两种 semantic variant
- `tests/test_conditioning.py`
  - 验证 JSON 字段进入 semantic
  - 验证 c+/c- 不再以 origin 开头

### 修改

- `ace_router/schema.py`
  - `positive_text` 和 `counterfactual_text` 改为 relation-only。
- `ace_router/model_adapter.py`
  - 新增 `cap_residual_by_semantic_norm()`。
  - block/proxy forward 增加 cap 参数。
- `ace_router/pipeline.py`
  - 将 cap 参数传到 I2V/T2V conditional forward。
- `ace_router/diagnostics.py`
  - 同时记录 candidate/applied residual、裁剪系数、裁剪比例和有效 lambda。
- `ace_router/schedules.py`
  - 保留 legacy schedule。
  - 新增 `safe_i2v_v1`、`safe_t2v_v1`、`safe_shared_v1`。
- `inference/infer_ace_router.py`
  - 增加 conditioning、cap 和 step preset CLI。
  - 实际生成和 token preflight 都使用编译后的文本。
  - 保存 `conditioning.compiled.json`。
- `infer_ace_router_m3.sh/.yaml`
  - 启用 M3 compiled + safe residual 默认值。
- `infer_ace_router_m4.sh/.yaml`
  - 启用 M4 compiled + safe residual 默认值。

## 6. 诊断字段

新版 residual 记录包括：

```text
raw_delta_l2
candidate_delta_l2
applied_delta_l2
candidate_to_semantic
applied_to_semantic
clip_coefficient_min/max
was_clipped
lambda_effective_min
residual_cap_ratio
```

同时保留旧的 `scaled_delta_l2` 和 `scaled_to_semantic` 作为 applied residual 的兼容别名，避免旧审计脚本立即失效。

重点观察：

- `applied_to_semantic_max` 不应超过设置的 cap（允许浮点误差）；
- 如果 `clip_fraction` 接近 1，说明 lambda 仍偏大，几乎全程由 cap 接管；
- 如果几乎从不裁剪且 M3/M4 与基线完全相同，再逐步增加 lambda 或 cap，不能直接恢复 0.5。

## 7. 运行方式

在 `code/v1` 下先做参数解析：

```bash
PARSE_ONLY=1 bash inference/infer_ace_router_m3.sh
PARSE_ONLY=1 bash inference/infer_ace_router_m4.sh
```

再做单样本 preflight：

```bash
CHECK_ONLY=1 SAMPLE_IDS=P02 bash inference/infer_ace_router_m3.sh
CHECK_ONLY=1 SAMPLE_IDS=P02 bash inference/infer_ace_router_m4.sh
```

确认后用 YAML 提交：

```bash
det experiment create inference/infer_ace_router_m3.yaml .
det experiment create inference/infer_ace_router_m4.yaml .
```

建议第一次把 YAML 的 `SAMPLE_IDS` 改为 `P02`，确认显存、视频输出、contexts 和 diagnostics 正常后再跑全部 P01–P20。

## 8. 本轮验收标准

- M3/M4 parse-only 和 check-only 正常。
- `contexts.json` 中 c+/c- 不包含 origin 前缀。
- M3 semantic variant 为 `compiled_i2v`。
- M4 semantic variant 为 `compiled_t2v`，且图片不进入模型。
- `conditioning.compiled.json` 与实际模型文本一致。
- residual cap 单元测试证明不越界、不放大弱 residual、按样本独立裁剪。
- legacy schedule 和 cap=None 仍可运行。
- 不新增任何视频自动 reject 或流程校验行为。

相关总方案：

- [M3/M4 完整修改建议](ace_router_m3_m4_complete_modification_plan_20260901.md)
- [质量审计](../code/v1/outputs/ace_router/QUALITY_AUDIT_20260901.md)
