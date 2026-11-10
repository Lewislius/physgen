**用真实的“烟花与烟雾”样本理解 verifier：输出、损失与回传**

适用版本：physGen v5.2.2-native，v25–v28；核对日期：2026-10-27。

verifier 读取生成视频，对四项过程判据输出三类概率。训练把“支持”类的负对数概率变成一个标量损失，再通过视频生成过程求导，更新构造条件 C 的模块。

本文使用真实缓存样本和已保存的训练日志。为了展示错误视频怎样产生梯度，另给出一组明确标注的手算示意数据。没有重新生成视频，也没有把日志中的高分当作人工核实过的物理正确性。

**1．从一个实际任务开始：烟花熄灭，已有烟雾继续演化**

样本 `source0001856` 来自现有 WISA 训练缓存。原始 caption 描述：夜空中的烟花以不同颜色和形状绽放，每次绽放后留下烟雾，烟雾逐渐消散。该缓存片段有 81 帧，源帧率 30 FPS，首末帧相隔约 2.667 秒。

对这个任务，直观的过程要求是：烟花亮点和烟雾有相应的出现顺序；亮点消退后，已经出现的烟雾仍可以继续变化；烟雾在扩散、变形或逐渐不可见的过程中保持合理的连续性。这里不要求跟踪每个烟团，也不把“变化越多”当作“越正确”。

实际输入还包括现有缓存中的 clauses。其中一条写了 `Smoke clouds form and drift downward after each explosion.`。**“向下漂移”是这个缓存附加的要求，不是烟雾的普遍物理规律；原 caption 也没有明确该方向。** 解释模型判断时，应保留这一条件来源区别。

一次训练反馈的前向过程如下：

```text
首帧 + caption + clauses
        ↓ 冻结的 Qwen 条件编码
       memory
        ↓ 可训练的 G / P / norm / alignment（Q 版另有 query seeds）
       条件 C
        ↓ Wan：固定首帧，其余部分从噪声开始，16 步去噪，CFG=5
     生成视频 latent
        ↓ VAE 解码
     81 帧视频中的 28 个观察帧
        ↓ 与时间戳、caption、clauses 一起送入 verifier
     四项判据 × 两种标签排列 × 三类 logits
```

观察索引为 `0, 3, 6, …, 78, 80`，保持 832×480。这个样本对应的观察时刻约为 `0, 0.1, 0.2, …, 2.6, 2.667` 秒。**16 步是去噪步数，81 帧是视频长度，28 帧是判别器观察数量。**

verifier 的输入是视频和原始要求，不包含生成侧的分析文字或条件张量 C。它评价的是实际生成结果对要求的支持程度。抽样帧之间发生了什么，不能仅凭这些观察完全确定。

**2．实际输出长什么样？**

对单个视频，原始输出形状为 `[1, 4, 2, 3]`。最后一维统一为：

```text
[supported, violated, unknown]
 支持          违背       证据不足
```

四项判据在这个任务中的含义如下。

| 判据 | 对烟花样本具体检查什么 |
| --- | --- |
| requirements：要求完成 | 请求的烟花绽放与后续烟雾是否在可见片段中实现 |
| response：条件响应 | 可见烟花、烟雾及其变化之间是否呈现合理的过程关系 |
| persistence：持续与终止 | 早、中、晚期的出现、延续和消退是否合理；亮点消退不要求已有烟雾同步消失 |
| continuity：连续性 | 烟雾的材料外观和来源在变形、扩散、遮挡、消散过程中是否连贯 |

每项判据使用两种字母映射分别读取 Qwen 的原生答案 logits：

| 标签排列 | A | B | C |
| --- | --- | --- | --- |
| 第一次 | supported | violated | unknown |
| 第二次 | unknown | supported | violated |

代码把两次结果都还原到 `[supported, violated, unknown]` 后计算损失。这里的字母 **C** 只是答案标签，与条件张量 **C** 无关。

下面是 **H frozen 运行 66356、第 193 个 optimizer step** 对这个样本生成结果的真实记录。表中是两种标签排列的概率平均值，保留六位小数：

| 判据 | supported | violated | unknown |
| --- | ---: | ---: | ---: |
| requirements | 0.991613 | 0.002998 | 0.005388 |
| response | 0.996643 | 0.001010 | 0.002347 |
| persistence | 0.994844 | 0.001486 | 0.003670 |
| continuity | 0.995155 | 0.001838 | 0.003007 |

它表示 verifier 对这次生成给出了高支持概率；这些数值不是经过校准的“物理正确率”。训练取的是连续 logits，而不是先生成离散字母再对字母求导。

**3．这些输出怎样变成损失？**

令 `z[b,k,r,c]` 为映射后的 logits：`b` 是样本，`k` 是四项判据，`r` 是两种标签排列，`c` 是三种语义类别。当前温度 `T=1`，softmax 仅在这三个答案之间归一化：

$$
p_{bkrc}=\operatorname{softmax}(z_{bkr}/T)_c
$$

$$
L_{\mathrm{feedback}}
=-\frac{1}{8B}\sum_{b=1}^{B}\sum_{k=1}^{4}\sum_{r=1}^{2}
\log p_{bkr,\mathrm{supported}}
$$

直观上：希望视频得到更高的 supported 概率。若某一项的 supported 为 0.9，其 NLL 约为 0.105；降到 0.1，NLL 就变成约 2.303。unknown 占据概率质量时也会压低 supported，因此当前反馈损失不会自动忽略 unknown。

上述真实记录为：

```text
loss/verifier     = 0.005460374988615513
weighted/verifier = 0.05 × loss/verifier
                  ≈ 0.000273018755251542
```

损失先对每种标签排列分别取负对数，再平均。上一节表格已经平均了两种排列，因而不能从该表精确复算原始 NLL：`mean(-log p)` 一般不等于 `-log(mean p)`。日志中的 `verifier_score = mean(p_supported - p_violated)` 是另一项诊断指标，不是当前训练损失。

**4．如果视频出错，手算一次会是什么结果？**

以下是**教学假设**，不是上述训练视频的实际观察报告：假设生成视频里，受持续照明、没有离开画面或受到遮挡的一团明显烟雾，在烟花亮点消退时突然整体消失。这个现象会使持续性和连续性受到质疑。不能只凭画面变暗或烟雾不可见，就断言材料凭空消失。

再假设 verifier 对这段问题视频输出下表。所有三元组均已按 `[supported, violated, unknown]` 排好；数值专用于解释算法。

| 判据 | 第一次排列的概率 | 第二次排列的概率 | 两次 NLL 的均值 |
| --- | --- | --- | ---: |
| requirements | `(0.60, 0.30, 0.10)` | `(0.55, 0.35, 0.10)` | 0.554331 |
| response | `(0.25, 0.65, 0.10)` | `(0.20, 0.70, 0.10)` | 1.497866 |
| persistence | `(0.10, 0.80, 0.10)` | `(0.15, 0.75, 0.10)` | 2.099853 |
| continuity | `(0.35, 0.55, 0.10)` | `(0.30, 0.60, 0.10)` | 1.126897 |

$$
L_{\mathrm{feedback}}
=-\frac{\log 0.60+\log 0.55+\log 0.25+\log 0.20+
\log 0.10+\log 0.15+\log 0.35+\log 0.30}{8}
\approx 1.319736863
$$

加上当前反馈权重后：

$$
0.05L_{\mathrm{feedback}}\approx 0.065986843
$$

这只是为该视频构造出的损失。模型不会收到一条人工编写的“第 60 帧把烟雾补回来”的修正指令；下一步要由可微计算图求出输入怎样影响这项损失。

**5．一个标量怎样回到 C？**

先看手算例子中 persistence 的第一次判别：

```text
p = [0.10, 0.80, 0.10]
生成反馈希望的类别 y = [1, 0, 0]
```

在 `B=1, T=1` 时，这次判别对加权总反馈损失的 logit 梯度为：

$$
\frac{\partial(0.05L_{\mathrm{feedback}})}{\partial z}
=\frac{0.05}{8}(p-y)
=[-0.005625,\ 0.005000,\ 0.000625]
$$

负的 supported 导数表示：提高这个 logit 会降低损失。另两个导数为正，降低它们会降低损失。但 **logits 是中间结果**；优化器实际更新的是上游允许训练的参数。

反向路径如下：

```text
加权反馈损失
    ↓ 对三类 logits 求导
Qwen 最后读出 → 语言层 → 视觉层 → 可微像素预处理
    ↓ 对输入 RGB 求导
VAE 解码
    ↓ 对生成视频 latent 求导
Wan 第 16 步 → 第 15 步 → …… → 第 1 步
    ↓ 累积各步对正、负条件的导数
C+、C−
    ↓ 回到共享的条件构造网络
G / P / norm / alignment / Q query seeds（仅 Q 版）
```

若以 `V` 表示被观察的生成 RGB，`θ_C` 表示构造条件的参数，链式法则可写为：

$$
\nabla_{\theta_C}L
=\left(\frac{\partial C_+}{\partial\theta_C}\right)^T\nabla_{C_+}L
+\left(\frac{\partial C_-}{\partial\theta_C}\right)^T\nabla_{C_-}L,
\qquad
\nabla_{C_\pm}L
=\left(\frac{\partial V}{\partial C_\pm}\right)^T\nabla_VL
$$

`C+` 使用正常文字条件；`C−` 使用空 caption/clauses 和同一首帧。当前 CFG 两个分支都参与求导。

**冻结参数仍然可以对输入求导。** 在 feedback 这项目标中，Qwen judge、VAE、Wan 的参数不被更新，但它们参与计算 RGB 和 C 的导数。生成侧 Qwen 的 memory 已 detach，梯度到可训练的条件构造模块为止，不更新那个冻结的条件编码器。

为了节省显存，代码采用离散伴随：前向保存 16 步的边界 latent；反向从最后一步开始逐步重算，传递状态梯度，并累加每一步对 C 的贡献。它覆盖完整的 16 步反馈采样轨迹；不等于普通推理的 50 步轨迹。内部为重算而使用的 detach，配有自定义 backward 返回导数，不意味着切断整条反馈路径。

训练中的 C 是网络输出。optimizer 更新 G、P 等参数之后，下次重新编码条件会得到新的 C；当前训练并不是把每条样本的 C 当作独立参数保存和优化。梯度能指明怎样改变模型评分，但不能保证一次更新就修正烟雾过程，效果仍要重新生成视频检查。

**6．这和“训练 verifier 自己”有什么区别？**

冻结 verifier 版本直接使用上述判别器。verifier LoRA 版本另外安排 32 步校准：冻结 teacher 先描述视频早、中、晚的可见证据，然后结合要求给出三类软标签 `q`。学生读取视频与要求，用软标签交叉熵学习。

$$
L_{\mathrm{calibration}}
=\mathbb E_{(b,k)\,\mathrm{accepted}}
\left[-\frac{1}{2}\sum_{r=1}^{2}\sum_c q_{bkc}\log p_{bkrc}\right]
$$

这里 `q` 是 teacher 两种标签排列概率的均值。当前接受条件包括：观察文字可用、最大类别概率至少 0.55、两种排列的分布差异不超过 0.15。teacher 标签和视频均 detach，只训练 verifier LoRA。校准视频交替来自缓存视频与当前生成结果，不按来源自动赋予正确或错误标签。

仍用“烟雾突然消失”的教学情境，假设 teacher 对某项给出 `q=[0.10,0.80,0.10]`，学生给出 `p=[0.60,0.20,0.20]`，单次判别的交叉熵为：

$$
-0.10\log0.60-0.80\log0.20-0.10\log0.20
\approx1.499576684
$$

此时学习的是让学生 verifier 更接近 teacher 的“违背”判断。等到生成反馈阶段，verifier 参数固定，学习目标才是让生成视频得到“支持”。这两个阶段的监督对象不同。

| 阶段与目标 | 训练的参数 |
| --- | --- |
| interface：FM + 0.1 × temporal FM | 构造 C 的模块；全部 Wan 参数固定 |
| calibration：teacher 软标签交叉熵，仅 verifier LoRA 版 | verifier LoRA；生成侧固定 |
| joint：FM + 0.1 × temporal FM | 构造 C 的模块 + Wan LoRA |
| joint：额外的 0.05 × verifier feedback | 构造 C 的模块；不向 Wan 参数累积这项梯度 |

joint 每 4 个 optimizer step 追加一次完整 feedback，使用该次更新的最后一个微批，不再除以 accumulation=4。160 个 joint step 对应 40 次反馈。FM 用噪声与真实 latent 构造监督目标；temporal FM 约束其误差沿视频时间的差分，不是烟雾的物理速度标签。当前 simple 版本没有 RGB process 训练损失。

**7．真实记录能证明梯度回去了吗？**

同一样本、同为第 193 步的日志如下。H/Q 两行是各自模型生成结果的记录，并非同一段生成视频的重复打分。

| 真实日志字段 | H frozen：66356 | Q frozen：66353 |
| --- | ---: | ---: |
| `condition_tokens` | 90 | 128 |
| `loss/verifier` | 0.005460375 | 0.005674042 |
| `weighted/verifier` | 0.000273019 | 0.000283702 |
| `C_gradient_rms` | 0.004075398 | 0.00000615538 |
| `paired_C_grad_norm` | 0.056392673 | 0.080514066 |
| `feedback_C_grad_norm` | 86.767616272 | 0.094818681 |
| `feedback_to_paired_C_grad_ratio` | 1538.6328125 | 1.1776661 |
| `feedback_updates_reader` | false | false |

`C_gradient_rms` 是实际加权反馈 backward 在正条件 C 张量上触发的 hook 记录；这里的非零值证明该次梯度到达了 C。两个 `*_C_grad_norm` 则测量构造 C 的**参数**梯度：一个来自已有配对监督，一个来自新追加的 feedback，均在后续梯度裁剪之前记录。它们与 C 张量的 RMS 不是同一个量。

这也给出一个具体提醒：H 这一步的反馈 loss 只有约 0.00546，但反馈参数梯度范数约为配对监督的 1538.63 倍。loss 数值相近或权重较小，并不能推出梯度也较小，因为还要乘上生成过程和条件构造网络的导数。当前训练随后按参数组将梯度范数裁剪到 1；这个单步记录仍不足以判断长期平衡或视频质量收益。

**8．可以独立运行的手算复核代码**

下面只需要已有的 PyTorch，复现第 4、5 段的损失和 logit 梯度；它不加载 Qwen/Wan，也不代表验证了完整视频反馈效果。

```python
import torch

# 教学假设；顺序为 supported、violated、unknown。
p = torch.tensor([[
    [[0.60, 0.30, 0.10], [0.55, 0.35, 0.10]],
    [[0.25, 0.65, 0.10], [0.20, 0.70, 0.10]],
    [[0.10, 0.80, 0.10], [0.15, 0.75, 0.10]],
    [[0.35, 0.55, 0.10], [0.30, 0.60, 0.10]],
]], dtype=torch.float32)  # [1, 4, 2, 3]

# log(p) 是一组可产生这些概率的 logits，仅用于复核。
logits = p.log().detach().requires_grad_(True)
loss = -torch.log_softmax(logits, dim=-1)[..., 0].mean()
weighted = 0.05 * loss
weighted.backward()

print("feedback loss:", loss.item())
print("weighted loss:", weighted.item())
print("persistence / order 1 gradient:", logits.grad[0, 2, 0])

# 约为：
# feedback loss: 1.3197369
# weighted loss: 0.06598684
# gradient: tensor([-0.005625, 0.005000, 0.000625])
```

**代码与数据出处**

项目根目录：`/home/liuzhirui/Project/physGen/code/v5.2.2-native`。以下相对链接在仓库内有效；下载到其他位置后，可按路径回到项目核对。

| 内容 | 来源 |
| --- | --- |
| 样本原始 caption、时间戳、帧率 | [base/manifest.json](../../../data_cache/wisa_4000/base/manifest.json)，条目 `source0001856` |
| 样本实际 clauses | [requirements/source0001856.json](../../../data_cache/wisa_4000/v52/requirements/source0001856.json) |
| H 真实反馈记录 | [66356/metrics.jsonl](../checkpoints/v522-h-content-message-verifier-frozen-simple_66356/metrics.jsonl)，第 244 行 |
| Q 真实反馈记录 | [66353/metrics.jsonl](../checkpoints/v522-q-seed128-message-verifier-frozen-simple_66353/metrics.jsonl)，第 244 行 |
| 判据、标签排列、NLL、校准 CE | [criterion_verifier.py](../physgen_native/criterion_verifier.py) |
| 原生 LM head 的 A/B/C 读出 | [knowledge.py](../physgen_native/knowledge.py)，`_answer_logits` |
| 校准视频与 teacher 的隔离 | [criterion_training.py](../physgen_native/criterion_training.py) |
| 反馈生成、观察帧、C 梯度 hook | [operator_training.py](../physgen_native/operator_training.py)，`feedback_losses` |
| CFG 采样与逐步反传 | [system.py](../physgen_native/system.py)，`_sample_tokens`；[adjoint.py](../physgen_native/adjoint.py) |
| 梯度记录、反馈调度及裁剪 | [runtime.py](../physgen_native/runtime.py)；[train.py](../train/train.py) |
| 当前损失权重与训练阶段 | [H LoRA simple 配置](../configs/native_h_content_message_verifier_lora_simple_wisa3857.yaml) |
