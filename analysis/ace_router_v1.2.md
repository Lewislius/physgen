# ACE Router 当前完整 Pipeline：以 P01 / M3 为例

日期：2026-09-01  
对应代码：`Project/physGen/code/v1`  
示例方法：M3（I2V + compiled semantic + positive−counterfactual ACE residual）

## 1. 一句话总览

P01/M3 的当前流程是：

```text
YAML 参数
→ 读取 P01 的 origin、i0 JSON、c+/c- JSON 和 1280×704 首帧
→ 校验输入并编译增强 semantic
→ c+、c- 保持为不带 origin 的纯关系文本
→ T5 分别编码 semantic / c+ / c- / empty CFG
→ VAE 编码首帧，seed 42 生成视频 latent 噪声
→ UniPC 用 shift=5 执行 50 步去噪
→ 每一步在 DiT 第 14–23 层注入受 2% cap 保护的 c+−c- 残差
→ guide scale=5 合并 conditional 与 unconditional
→ 每步重新锚定首帧 latent
→ VAE 解码为 97 帧、1280×704、24 FPS 视频
→ 保存视频、条件、参数、hash 和 residual diagnostics
```

ACE 不训练新参数，也不替换 Wan 的 cross-attention；它使用冻结 Wan block 的同一组权重，对 semantic、c+ 和 c- 做额外 cross-attention 前向并计算差值。

## 2. P01 的输入文件

P01 目录中实际使用：

```text
demo/P01/P01-origin.txt
demo/P01/P01-i0.json
demo/P01/P01-cplus-cminus.json
demo/P01/P01-i0-1280x704.png
```

### 2.1 原始任务

```text
A black scooter rolls into a metal trash can, then tilts to one side and stops beside it.
```

目标事件可以拆成：

```text
初态：滑板车直立、与垃圾桶分离；垃圾桶直立且静止
动作：滑板车向垃圾桶滚动
触发：发生可见接触
结果：滑板车倾斜、减速并停在垃圾桶旁
保持：实体数量、垃圾桶身份和材质、街道背景、固定侧视镜头
```

### 2.2 首帧元数据

`P01-i0.json` 提供：

- 必须可见：一个黑色滑板车、一个金属垃圾桶、水平街面；
- 初态：滑板车直立且未接触垃圾桶，垃圾桶直立静止；
- 可变化：滑板车的位置、速度和姿态；
- 保持：固定侧视镜头、街道背景、垃圾桶身份与材质；
- 布局和 reject 条件目前只被读取/校验，尚未变成自动质量检查。

### 2.3 因果元数据

`P01-cplus-cminus.json` 提供：

```text
event_type:
contact-triggered collision and terminal pose

single_changed_relation:
tilting and stopping occur before rather than after visible contact
```

正事实 c+：

```text
The scooter visibly contacts the trash can before tilting, slows during the tilt, and remains stopped beside the can.
```

反事实 c-：

```text
The scooter tilts and stops while still separated from the trash can; visible contact occurs afterward.
```

特别注意：**当前 c+ 和 c- 前面都不再拼接 origin prompt。**

## 3. M3 YAML 如何启动推理

当前 `infer_ace_router_m3.yaml` 的关键参数为：

| 参数 | 当前值 |
|---|---:|
| method | M3 |
| conditioning mode | compiled |
| 首帧 | `P01-i0-1280x704.png` |
| 输出分辨率 | 1280×704 |
| 帧数 | 97 |
| FPS | 24 |
| seed | 42 |
| sampling steps | 50 |
| solver | UniPC |
| shift | 5.0 |
| guide scale | 5.0 |
| lambda0 | 0.10 |
| residual cap | 0.02 |
| layer preset | `mid_b`，第 14–23 层 |
| step preset | `safe_i2v_v1` |
| diagnostics | full |
| negative prompt | 空 |
| selection manifest | 当前不强制 |

YAML 启动 shell：

```text
infer_ace_router_m3.yaml
→ infer_ace_router_m3.sh
→ python infer_ace_router.py --method M3 ...
```

shell 负责设置模型路径、缓存目录和环境变量，Python 入口负责解析样本、构造条件、加载 Wan 并执行生成。

## 4. 输入加载与 preflight

`schema.py` 首先执行：

1. 检查样本 ID 是否为 `Pxx`。
2. 检查 origin、i0 JSON、c+/c- JSON 是否存在。
3. 检查两个 JSON 中的 `original_prompt` 是否与 origin 文件完全一致。
4. 检查必需字段、字符串列表、counterfactual type 和 confidence。
5. P01 confidence 为 `0.98`，高于当前阈值 `0.7`。
6. M3 必须找到可解码首帧，优先顺序中首先选择 `P01-i0-1280x704.png`。
7. 图片实际为 1280×704 RGB PNG。

当前 `REQUIRE_SELECTION_MANIFEST=0`，因此 P01 缺少 selection manifest 时只产生：

```text
selection_manifest_missing
```

它不会阻止生成。这也是当前流程尚未实现质量校验的地方之一。

推理入口还会检查：

- frame number 满足 `4n+1`；
- 参数为有限合法数值；
- 最终尺寸能精确得到 1280×704；
- 所有文本不超过 Wan 的 512-token 上限；
- Wan 源代码和 checkpoint hash 没有发生未授权漂移。

## 5. 条件增强编译

`conditioning.py` 以：

```python
compile_conditioning(sample, mode="compiled", use_initial_image=True)
```

生成 `compiled_i2v` 条件。

### 5.1 P01 实际 semantic

```text
A black scooter rolls into a metal trash can, then tilts to one side and stops beside it. Required visible entities: one black scooter; one metal trash can; level street surface. At the beginning: scooter: upright, intact, and separated from the trash can; trash can: upright, intact, and stationary. Required state changes: scooter position: separated to beside the trash can; scooter pose: upright to tilted; scooter speed: rolling to stopped. These event variables are allowed to change: scooter position; scooter speed; scooter pose. Preserve throughout: static side-view camera; street background; trash-can identity and material; one black scooter; one metal trash can; the same street scene and entities.
```

该 semantic 的职责是同时告诉模型：

- 有哪些实体、数量是多少；
- 视频从什么状态开始；
- 哪些状态必须发生变化；
- 哪些属性不能被错误冻结；
- 哪些场景和身份需要保持。

M3 已有首帧，所以 `pre_event_image_prompt` 和 `layout_requirements` 不再重复进入 semantic；它们主要保留在元数据中。M4 没有首帧，因此它的 `compiled_t2v` 会额外加入这两部分。

### 5.2 实际送入模型的四个文本分支

| 分支 | P01 内容 | UMT5 token 数 | 用途 |
|---|---|---:|---|
| semantic | 上述 compiled semantic | 158 | 主任务和场景条件 |
| positive | 原始 c+ 关系句 | 27 | 正确因果方向 |
| counterfactual | 原始 c- 关系句 | 22 | 错误因果方向 |
| unconditional | 空文本 | 空文本编码长度 | CFG 无条件分支 |

当前 c+ 与 c- 已经不含 origin，但 P01 两者 token 数仍不完全相等。本轮会记录这一差异，但不会阻止运行，也没有自动重写 c+/c-。

编译结果会保存为：

```text
conditioning.compiled.json
```

其中明确记录：

```json
{
  "mode": "compiled",
  "semantic_variant": "compiled_i2v",
  "compiler_version": "ace-conditioning-mvp-v1",
  "causal_pair": {"origin_prompt_prefixed": false}
}
```

## 6. 首帧预处理与 I2V latent 初始化

### 6.1 图片预处理

P01 图片已经是 1280×704，因此中心裁剪/缩放后仍为 1280×704。实际送入模型的版本会另存为：

```text
reference_input.png
```

其路径、原尺寸、裁剪框、最终尺寸和 SHA256 会写入 `reference_image.json`。

### 6.2 VAE 和初始噪声

Wan VAE 的时间/空间 stride 为 `(4,16,16)`：

```text
97 个视频帧 → 25 个 latent 时间帧
704×1280 → 44×80 latent 空间尺寸
patch size=(1,2,2) → 约 22,000 个视频 token
```

随后：

1. 首帧被归一化到 `[-1,1]` 并由 VAE 编码为 `known_latent`。
2. seed 42 生成整段视频的随机 latent 噪声。
3. I2V mask 把首个 latent 时间位置替换/锚定为 `known_latent`，其他位置保留为待生成噪声。

直观理解：

```text
第一个 latent 帧：由 P01 首帧控制
后续 latent 帧：从随机噪声开始，由文本、ACE 和去噪过程生成
```

## 7. 文本编码与模型加载

模型为冻结的 Wan2.2-TI2V-5B：

- 30 个 DiT block；
- text length 上限 512；
- Wan 原模型被 `AceModelProxy` 包装；
- wrapper 不新增可训练参数；
- 如果基础模型存在可训练参数，proxy 会直接拒绝运行。

UMT5 分别编码 semantic、c+、c- 和空文本。当前 `T5_CPU=1`，所以先在 CPU 编码，再把 context tensor 移到推理设备。

文本 context 在进入每个 DiT block 前被补齐到 Wan 的 512 长度并通过原模型的 text embedding 投影。

## 8. 50 步 UniPC 去噪

### 8.1 Shift 的位置

UniPC scheduler 根据：

```text
sampling_steps=50
shift=5.0
```

生成 50 个噪声时间点。Shift 只调整去噪时间表，不直接描述“撞击”或“倾斜”。

### 8.2 M3 的 step gate

`safe_i2v_v1` 对 50 步展开为：

| step index | 去噪区间 | step gate |
|---|---|---:|
| 0–14 | 前 30% | 0.5 |
| 15–34 | 30%–70% | 1.0 |
| 35–44 | 70%–90% | 0.3 |
| 45–49 | 最后 10% | 0.1 |

### 8.3 M3 的 layer gate

`mid_b` 的 30 层分布为：

```text
block 0–13   : layer gate=0，使用普通 Wan semantic cross-attention
block 14–23  : layer gate=1，使用 ACE causal residual
block 24–29  : layer gate=0，使用普通 Wan semantic cross-attention
```

因此每个去噪 step 只有 10 个中层 block 执行三分支 cross-attention。

## 9. 一个 active ACE block 内部发生什么

以 step 20、block 16 为例：

```text
step 20 位于 30%–70% 区间，因此 step gate=1.0
block 16 位于 14–23，因此 layer gate=1.0
lambda0=0.10
```

### 9.1 先执行原 Wan self-attention

当前视频 token 先经过 block 原有的 norm、self-attention 和调制残差，ACE 不改变这一部分。

### 9.2 使用同一个 query 计算三个 cross-attention 输出

```text
r_sem = CA(q, semantic_context)
r_pos = CA(q, positive_context)
r_neg = CA(q, counterfactual_context)
```

三次计算使用：

- 同一个视频 query `q`；
- 同一个 frozen Wan cross-attention block；
- 不同的文本 context。

### 9.3 计算候选因果残差

```text
raw_delta = r_pos - r_neg
candidate_delta = lambda0 × layer_gate × step_gate × raw_delta
```

在 step 20 / block 16 中：

```text
candidate_delta = 0.10 × 1.0 × 1.0 × (r_pos-r_neg)
```

不同阶段的候选系数为：

| step 区间 | `lambda0 × step_gate` |
|---|---:|
| 0–14 | 0.05 |
| 15–34 | 0.10 |
| 35–44 | 0.03 |
| 45–49 | 0.01 |

### 9.4 应用 2% relative cap

候选残差不会直接加入 hidden state，而是先比较其 L2 范数与 semantic residual：

\[
a=\min\left(1,
\frac{0.02\lVert r_{sem}\rVert_2}
{\lVert candidate\_delta\rVert_2+\epsilon}
\right)
\]

\[
applied\_delta=a\times candidate\_delta
\]

这保证每个样本、每个 active block 中：

```text
||applied_delta||₂ / ||r_sem||₂ ≤ 0.02
```

如果候选残差本来很小，`a=1`，不会被反向放大；如果过强，则被压到 2% 上限附近。

### 9.5 更新 block hidden state

```text
x = x + r_sem + applied_delta
x = x + FFN(...)
```

所以 ACE 是叠加在原 semantic cross-attention residual 上的轻量关系方向，不替代 semantic，也不直接操作像素、对象 mask 或轨迹。

## 10. Conditional、Unconditional 与 CFG

每个去噪 step 会执行两次模型前向。

### 10.1 Conditional 前向

输入：

```text
compiled semantic + c+ + c- + 当前 latent + 当前 timestep
```

第 14–23 层启用 ACE，得到：

```text
prediction_cond
```

### 10.2 Unconditional 前向

输入空文本，并设置：

```text
step_gate=0
lambda0=0
```

`AceModelProxy` 因此直接走原 Wan forward，不计算 c+、c-，得到：

```text
prediction_uncond
```

### 10.3 Guide scale=5 合并

```text
prediction = prediction_uncond
           + 5.0 × (prediction_cond - prediction_uncond)
```

因此 ACE 先在 conditional 分支内部受到 2% cap，再随 conditional/unconditional 差一起参与 CFG。2% 是 block 内 residual 上限，不代表最终 latent 或视频只改变 2%。

## 11. Scheduler 更新与首帧重新锚定

UniPC 使用合并后的 prediction 更新 latent：

```text
latent_next = scheduler.step(prediction, timestep, latent)
```

更新后立即再次应用 I2V mask：

```text
latent = known-first-frame region + newly-denoised future region
```

P01/M3 在进入循环前锚定 1 次，并在 50 个 step 后各锚定 1 次，因此成功输出中：

```text
first_frame_reanchor_count = 51
```

这能稳定首帧，但不保证后续帧一定保持滑板车身份、数量或物理正确性。

## 12. Full diagnostics

M3 当前使用 `DIAGNOSTICS=full`。50 个 step × 10 个 active block，理论上每个样本记录 500 条 conditional ACE 诊断。

每条记录包括：

```text
step_index / layer_id
layer_gate / step_gate / lambda0
semantic_l2
raw_delta_l2
candidate_delta_l2
applied_delta_l2
candidate_to_semantic
applied_to_semantic
clip_coefficient
was_clipped
lambda_effective
positive_counterfactual_cosine
isfinite
```

重点检查：

- `applied_to_semantic` 是否始终不超过 0.02；
- `clip_fraction` 是否接近 100%；
- 是否出现 NaN/Inf；
- 哪些 step/layer 的 raw delta 特别大。

诊断只说明数值行为，不等于滑板车确实发生了正确碰撞。

## 13. VAE 解码与视频保存

完成 50 步后：

1. DiT 从 GPU offload 到 CPU（当前 `OFFLOAD_MODEL=1`）。
2. VAE 将最终 latent 解码成视频 tensor。
3. 检查 tensor 是否 finite。
4. 检查输出尺寸是否为 1280×704。
5. 检查帧数是否为 97。
6. 以 24 FPS 保存 MP4。

视频时长约为：

```text
97 / 24 ≈ 4.04 秒
```

先写入临时文件：

```text
.video.incomplete.mp4
```

确认文件存在且非空后，再原子替换为：

```text
video.mp4
```

## 14. P01/M3 的输出目录

默认结构：

```text
outputs/ace_router/<m3-run-id>/P01/M3_I2V_ACE/seed_000042/
```

成功时包含：

```text
video.mp4
manifest.json
config.resolved.json
contexts.json
conditioning.compiled.json
diagnostics.jsonl
reference_image.json
reference_input.png
```

其中：

- `config.resolved.json`：实际 frame、CFG、shift、lambda、cap、layer/step gates；
- `contexts.json`：实际送入模型的 semantic/c+/c-、token 数和文本 hash；
- `conditioning.compiled.json`：元数据如何进入 semantic；
- `diagnostics.jsonl`：每个 step/layer 的残差记录；
- `reference_input.png`：真正送入 I2V 的首帧；
- `manifest.json`：视频 hash、输入 hash、耗时、显存、诊断摘要和 warning。

失败时会删除未完成视频，保存：

```text
failure.json
diagnostics.jsonl
```

由于 `CONTINUE_ON_ERROR=1`，P01 失败后批处理仍会继续下一个样本。

## 15. 当前流程能保证什么、不能保证什么

### 已实现

- 1280×704 首帧优先使用；
- 97 帧、24 FPS；
- JSON 元数据进入 compiled semantic；
- c+/c- 不再拼接 origin；
- M3 使用 positive−counterfactual；
- 第 14–23 层和分阶段 step gate；
- 每 block 2% residual cap；
- 空 negative prompt；
- 完整参数、hash 和 residual diagnostics。

### 尚未实现

- 自动判断首帧是否真的满足 `reject_if`；
- 强制 selection manifest；
- 自动检测滑板车是否复制、变形或消失；
- 自动判断是否“先接触、再倾斜、最后停止”；
- 生成失败后的自动重试、换 seed 或调参；
- 对象 mask、接触点、速度或轨迹控制。

因此，当前 pipeline 可以说：

```text
“我们把正确的元数据和因果方向以受控方式送入了冻结 Wan。”
```

但还不能仅凭 pipeline 数值正常就说：

```text
“P01 的碰撞和终态一定物理正确。”
```

## 16. P01 单样本运行方式

本地参数和输入 preflight：

```bash
cd /home/liuzhirui/Project/physGen/code/v1
CHECK_ONLY=1 SAMPLE_IDS=P01 bash inference/infer_ace_router_m3.sh
```

实际单样本生成：

```bash
cd /home/liuzhirui/Project/physGen/code/v1
SAMPLE_IDS=P01 SEEDS=42 bash inference/infer_ace_router_m3.sh
```

Determined YAML 默认是 `SAMPLE_IDS=all`。如只想先测试 P01，可将 M3 YAML 中：

```text
SAMPLE_IDS=all
```

临时改为：

```text
SAMPLE_IDS=P01
```

然后提交：

```bash
det experiment create inference/infer_ace_router_m3.yaml .
```

## 17. 核心公式汇总

P01/M3 在 active block 中：

\[
r_{sem}=CA(q,c_{semantic})
\]

\[
\Delta r=CA(q,c_+)-CA(q,c_-)
\]

\[
d_0=0.10\times g_{layer}\times g_{step}\times\Delta r
\]

\[
d=\min\left(1,
\frac{0.02\lVert r_{sem}\rVert_2}
{\lVert d_0\rVert_2+\epsilon}
\right)d_0
\]

\[
h'=h+r_{sem}+d
\]

模型级 CFG：

\[
prediction=prediction_{uncond}
+5.0(prediction_{cond}-prediction_{uncond})
\]

最后由 UniPC 用该 prediction 更新 latent，并重新写回首帧 known latent。

相关代码：

- [M3 YAML](../code/v1/inference/infer_ace_router_m3.yaml)
- [M3 launcher](../code/v1/inference/infer_ace_router_m3.sh)
- [统一推理入口](../code/v1/inference/infer_ace_router.py)
- [条件编译](../code/v1/ace_router/conditioning.py)
- [ACE block 与 residual cap](../code/v1/ace_router/model_adapter.py)
- [I2V 去噪流程](../code/v1/ace_router/pipeline.py)
- [layer/step schedule](../code/v1/ace_router/schedules.py)
- [residual diagnostics](../code/v1/ace_router/diagnostics.py)
