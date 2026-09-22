# ACE-Router 5.5、质量审计 6.5/7.5 简明解释

日期：2026-09-01

本文解释：

- `wan_appearance_semantics_causal_routing_mvp_20260831.md` 的 5.5；
- `QUALITY_AUDIT_20260901.md` 的 6.5 和 7.5；
- 当前 V1 代码实际实现了什么，以及它与基础 Wan 的区别。

## 0. 先给结论

1. **存在交叉注意力处理。**基础 Wan 每个 block 用原始 prompt 做一次 cross-attention；ACE 在选定 block 中复用同一个 cross-attention 模块，额外计算正向条件和反事实条件的输出差，再把该差值作为残差加回隐藏状态。
2. **没有训练或修改基础模型权重。**Wan、T5、VAE 都保持冻结；改变的是推理时的 forward 计算路径。
3. **M3/M4 实现的是 `positive - counterfactual`；M2 是 `positive - semantic`。**M0/M1 不加 ACE 残差。
4. **审计里的“约 10%”是单个被记录 block 中，ACE 缩放后残差的 L2 范数约等于原语义 cross-attention 残差 L2 范数的 10%。**它不是准确率、像素变化率、视频质量分数，也不是整个 16 层网络的总影响。
5. **7.5 是改进建议，不是当前已实现功能。**当前代码仍使用固定 `lambda0`，没有 residual normalization、norm clipping 或按样本自适应强度。

另外，方案文档中有两个标题都叫“5.5”：

- 前一个 5.5 是“block 内的残差形式”，是本文主要解释对象；
- 后一个 5.5 是 TRACE-Lite 的三种运行模式。目前只实现了 `global-ACE`，尚未实现 `TRACE-Time` 和 `TRACE-ST`。

## 1. 基础 Wan 的 block 在做什么

把一个视频 latent 切成时空 token 后，Wan Transformer block 可以简化为：

```text
视频 token h
  │
  ├─ self-attention：视频 token 彼此交流时间/空间信息
  │
  ├─ cross-attention：视频 token 从文本 token 读取语义
  │
  └─ FFN：逐 token 非线性变换
```

基础 Wan 的文本部分是：

\[
q=\operatorname{Norm}(h),
\qquad
r_{sem}=\operatorname{CA}(q,c_{sem}),
\qquad
h'=h+r_{sem}.
\]

其中：

- `h`：当前 block 的视频隐藏状态；
- `c_sem`：原始视频 prompt 经 T5 得到的文本 context；
- `CA`：cross-attention；
- `r_sem`：原始 prompt 对视频 token 产生的语义残差。

更细一点，cross-attention 内部仍是常见的：

\[
Q=W_q q,\quad K=W_k c,\quad V=W_v c,
\]

\[
\operatorname{CA}(q,c)
=W_o\left[\operatorname{softmax}\left(\frac{QK^T}{\sqrt d}\right)V\right].
\]

也就是说，视频 token 提供 query，文本提供 key/value，模型由此判断每个视频 token 应读取哪些文本信息。

## 2. 方案文档 5.5 的含义

### 2.1 M3/M4 的公式

ACE 对同一个 `q = Norm(h)` 使用三套文本 context：

\[
r_{sem}=\operatorname{CA}(q,c_{sem}),
\]

\[
r_+=\operatorname{CA}(q,c_+),
\qquad
r_-=\operatorname{CA}(q,c_-).
\]

然后构造：

\[
\Delta r=r_+-r_-,
\]

\[
h'=h+r_{sem}+\lambda_{\ell,k}\Delta r.
\]

直观上：

- `r_sem` 负责“场景里是谁、要发生什么”；
- `r_+` 表示正确因果过程对当前视频 token 的拉动；
- `r_-` 表示邻近错误过程对当前视频 token 的拉动；
- `r_+ - r_-` 尝试保留二者的差异，例如“接触后移动”相对于“接触前移动”的方向。

这里的 `c_-` **不是 CFG negative prompt**。它是一个肯定式描述的错误事件，仅用于构造 ACE 差分。

### 2.2 M2 的公式略有不同

M2 没有使用反事实分支，而是：

\[
\Delta r=r_+-r_{sem},
\]

\[
h'=h+r_{sem}+\lambda_{\ell,k}(r_+-r_{sem}).
\]

当总 scale 为 0.5 时，文本残差部分可直观写成：

\[
r_{sem}+0.5(r_+-r_{sem})
=0.5r_{sem}+0.5r_+.
\]

所以 M2 更像把基础语义注意力向更详细的正向提示移动；它不是真正的正向—反事实对比。

## 3. 当前代码的完整流程

### 3.1 推理前

```text
原始 prompt ──────────────► T5 ─► semantic context

原始 prompt + causal_positive
                         └► T5 ─► positive context

原始 prompt + causal_counterfactual
                         └► T5 ─► counterfactual context

首帧图（仅 M1/M2/M3）────► VAE ─► known first-frame latent
```

当前 `positive_text` 和 `counterfactual_text` 都是“原始 prompt + 一句额外事件描述”。这也是审计 6.6 所说的重复和句长污染来源。

### 3.2 每个去噪 step

1. 把当前 noisy latent 输入 Wan DiT。
2. 在不启用 ACE 的 block 中，直接执行原始 Wan block。
3. 在启用 ACE 的 block 中：
   - 正常执行 self-attention；
   - 用相同 query 和相同 cross-attention 权重分别计算 `semantic`、`positive`、`counterfactual`（M2 不计算 counterfactual）；
   - 计算差分并乘以 layer gate、step gate 和 `lambda0`；
   - 加回 `h`，然后正常执行 FFN。
4. ACE 只进入 conditional 分支；unconditional CFG 分支不加 ACE。
5. 用 CFG 合并 conditional/unconditional prediction。
6. scheduler 更新 latent。
7. I2V 模式重新锚定首帧对应的 known latent。
8. 所有 step 完成后由 VAE 解码为视频。

当前缩放系数为：

\[
\lambda_{\ell,k}=\lambda_0\,g_{layer}(\ell)\,g_{step}(k).
\]

审计对应的实际配置是：

```text
lambda0 = 0.5

block 0–7   : layer gate = 0
block 8–13  : layer gate = 0.4
block 14–23 : layer gate = 1.0
block 24–29 : layer gate = 0

前 30% 去噪 step   : step gate = 1.0
30%–60%            : step gate = 0.7
60%–80%            : step gate = 0.3
最后 20%           : step gate = 0
```

因此，真正有 ACE 注入的是 16 个中层 block，而且只覆盖前 80% 去噪过程。

### 3.3 各模式差异

| 模式 | 图像首帧 | 额外交叉注意力 | 差分 |
|---|---|---|---|
| M0 | 无 | 无 | 无，基础 T2V |
| M1 | 有 | 无 | 无，基础 I2V |
| M2 | 有 | 有 | `positive - semantic` |
| M3 | 有 | 有 | `positive - counterfactual` |
| M4 | 无 | 有 | `positive - counterfactual` |

### 3.4 “处理了交叉注意力”具体是什么意思

处理方式不是修改或训练 attention 权重，也不是直接编辑 attention map，而是：

- 保留原 Wan 的 `q/k/v/o` 投影和 cross-attention 参数；
- 对同一视频 query，用不同文本 context 多调用几次同一个 cross-attention；
- 在 cross-attention **输出张量**上做差；
- 把差值作为额外 residual 加回隐藏状态。

所以这是“推理时多分支 cross-attention 输出组合”，不是新训练出的物理 attention，也没有对象 mask、轨迹或守恒约束。

当 `lambda0=0`、step gate 为 0 或所有 layer gate 为 0 时，代理直接走原始 Wan forward，严格退回基础路径。

## 4. P02 红蓝球的直观例子

P02 原始语义大意是：红球滚向静止蓝球；接触后蓝球离开，红球减速。

三条文本路径分别是：

```text
semantic:
红球撞向静止蓝球；接触后蓝球移动、红球减速。

positive:
红球明确接触静止蓝球；随后蓝球离开，同时红球减速。

counterfactual:
蓝球在红球到达之前就开始移动；之后才发生接触。
```

基础 Wan 只得到 `r_sem`。它可能知道画面应包含红蓝球和碰撞，但未必稳定遵守先后顺序。

M3 额外计算：

```text
“接触后蓝球移动”的 attention 输出
减去
“接触前蓝球移动”的 attention 输出
```

理想情况下，这个差值会把生成过程推向正确顺序。但它只是高维特征差，不是显式规则：模型没有被硬性告知只能有两个球，也没有碰撞检测器。因此它仍可能产生第三个球、球杆或身份交换。

一个简化数字例子：

```text
||r_sem|| = 100
||r_+ - r_-|| = 24
lambda0 = 0.5
layer gate = 1
step gate = 1

||scaled delta|| ≈ 12
scaled/semantic ≈ 12 / 100 = 12%
```

若当前 block 的 layer gate 为 0.4，则同一 raw delta 只得到约 4.8% 的比例；若 step gate 再变成 0.7，则约为 3.36%。实际值还取决于向量方向，不能只按标量精确推导。

## 5. 审计 6.5：固定残差强度的问题

### 5.1 指标怎么计算

代码记录：

\[
\rho_{\ell,k}
=\frac{\|\lambda_0g_{layer}(\ell)g_{step}(k)\Delta r\|_2}
{\|r_{sem}\|_2+\epsilon}.
\]

`scaled_to_semantic_max` 是一个样本在被 diagnostics 抽样记录的 layer/step 点中，最大的 `rho`。

审计表再对每个模式的 20 个样本做汇总：

- M2 的 20 个“样本最大值”落在 9.25%–15.87%，中位数 11.72%；
- M3 落在 9.40%–15.42%，中位数 11.02%；
- M4 落在 8.40%–14.18%，中位数 10.29%。

例如 M3 的 9.40%–15.42% 表示：

- 不是每个点都在这个范围；
- 是每个样本先取其已记录点中的最大值；
- 20 个最大值里，最小为 9.40%，最大为 15.42%；
- 一个典型样本的最大值约为 11.02%。

### 5.2 “最大值约 10%”说明什么

它说明在某个被记录的 block/step，额外 ACE 残差的整体 L2 能量已经达到基础语义 cross-attention 残差的约十分之一。这不算数值爆炸，但对一个未经训练验证的方向而言并不弱。

它**不说明**：

- 视频有 10% 像素发生改变；
- 生成质量提高或下降 10%；
- ACE 有 10% 的概率生效；
- 整个网络的总改动只有 10%；
- 该方向一定代表正确因果关系。

### 5.3 为什么单 block 的 10% 仍可能很强

第一，残差进入的是演化中的隐藏状态。后续 block 和后续去噪 step 会继续基于已改变的 `h` 计算，所以影响具有非线性传播。

第二，一共可在 16 个 block、前 80% 去噪 step 中反复注入。不能把它简单相加成 `16 × 10% = 160%`，因为不同向量可能相互抵消或放大；但也绝不能把一次 10% 当作全流程只有 10%。

第三，L2 比例是全张量汇总。一个总体 10% 的 delta 如果集中在很少的视频 token 上，局部改动可能接近甚至超过这些 token 原有的语义残差。直观上像整辆车只轻推了 10%，但这股力全施加在一个细小轮轴上，局部结构仍可能被扭坏。

第四，方向比大小更重要。一个幅度只有 10% 但方向错误的转向，在连续多次修正后也能让车辆明显偏离道路。

### 5.4 为什么固定 `lambda0=0.5` 不稳

不同 prompt 产生的 raw delta 范数差别很大。假设：

```text
样本 A：||raw delta|| = 20
样本 B：||raw delta|| = 200
```

两者都乘 0.5 后分别为 10 和 100。相同旋钮并不代表相同干预强度：A 可能几乎没变化，B 可能语义过冲或产生拓扑异常。

P12 蒲公英是直观案例：模型原本就容易把放射状细丝表示成线状结构；重复注入“脱落、漂移”的全局差分，可能没有正确减少既有种子，反而把细丝强化成树枝或蛛网。

### 5.5 这个诊断本身的限制

- `summary` diagnostics 只记录 schedule 边界附近的少数 layer/step，不是全部注入点；
- 表中的 max 是“已记录点最大值”，真实未记录点可能更高或更低；
- 它只比较 `scaled delta` 与 `semantic residual`，没有比较完整隐藏状态 `h`；
- 没有度量 16 层累积影响；
- `all_finite=true` 只表示无 NaN/Inf，不代表物理、对象或时序正确。

## 6. 审计 7.5：归一化、裁剪、自适应强度

三者可以用“方向盘、限位器、自动油门”来理解。

### 6.1 归一化：先统一不同样本的尺度

令：

\[
\Delta r=r_+-r_{ref}.
\]

最直接的归一化是只保留方向：

\[
\widehat{\Delta r}=\frac{\Delta r}{\|\Delta r\|_2+\epsilon}.
\]

再按 `r_sem` 的大小赋予目标尺度，例如 2%：

\[
\Delta r_{target}
=0.02\|r_{sem}\|_2\widehat{\Delta r}.
\]

这样 raw delta 为 20 或 200 的样本不会天然相差十倍。

风险是：严格归一化也会把非常小、可能只是噪声的 delta 放大。因此首版通常更适合“只压大、不抬小”的相对裁剪。

### 6.2 裁剪：设置每个 block 的安全上限

先按原配置得到：

\[
d=\lambda_0g_{layer}g_{step}\Delta r.
\]

若安全上限为 `tau = 0.02`，可使用：

\[
s=\min\left(1,
\frac{\tau\|r_{sem}\|_2}{\|d\|_2+\epsilon}\right),
\qquad
d_{safe}=s\,d.
\]

效果是：

- 原比例 1%：不变；
- 原比例 2%：基本不变；
- 原比例 10%：缩小到 2%；
- 原比例 15%：缩小到 2%。

这就是 norm clipping。它限制的是整块残差范数，不是逐元素把数值截成固定区间。

### 6.3 自适应强度：每个样本、层、step 自动调小 lambda

先计算未缩放比例：

\[
\rho_{raw}=\frac{\|\Delta r\|_2}{\|r_{sem}\|_2+\epsilon}.
\]

然后选择：

\[
\lambda_{eff}
=\min\left(
\lambda_{base},
\frac{\tau}{g_{layer}g_{step}\rho_{raw}+\epsilon}
\right).
\]

raw delta 小的样本可以保留原 `lambda_base`；raw delta 大的样本自动降低 `lambda_eff`，避免所有任务共用 0.5 导致有的无效、有的过冲。

实际实现时还应对 step 间的 `lambda_eff` 做平滑或 EMA，避免强度逐 step 剧烈跳动。

### 6.4 建议记录什么

每个样本、层、step 至少记录：

- `raw_delta_l2`；
- `semantic_l2`；
- `unclipped_scaled_to_semantic`；
- `clip_coefficient`；
- `clipped_scaled_to_semantic`；
- `lambda_eff`；
- 各层注入范数之和，以及配对基础 forward 的隐藏状态/预测差。

“累积影响”不能只把各层百分比相加；更可靠的是在相同 latent 上做 ACE/base 配对 forward，直接比较最终 prediction 或若干层后的 hidden state。

### 6.5 它能解决什么、不能解决什么

能缓解：

- 不同 prompt 的 raw delta 尺度差异；
- P12 一类过强纹理/拓扑强化；
- 单一 `lambda0=0.5` 带来的样本间不公平干预；
- 极端 block/step 对整条生成轨迹的破坏。

不能保证：

- `positive - counterfactual` 真的是纯因果方向；
- 球、液体、碎片和种子数量守恒；
- 接触检测、轨迹或终态一定正确；
- 新对象不会出现。

简言之：归一化、裁剪和自适应强度是“安全带和限速器”，不是“物理引擎或导航系统”。方向错了，即使把幅度从 10% 降到 2%，仍可能朝错误方向走，只是破坏通常会更小。

## 7. 哪些已经实现，哪些尚未实现

| 项目 | 当前状态 |
|---|---|
| 基础 Wan semantic cross-attention | 已实现，来自原模型 |
| M2 的 `positive - semantic` | 已实现 |
| M3/M4 的 `positive - counterfactual` | 已实现 |
| 固定 layer/step gate | 已实现 |
| `lambda=0` 严格回退基础 Wan | 已实现并有测试 |
| 残差 L2、比例、cosine、finite 诊断 | 已实现，但仅抽样 |
| 按视频阶段路由 TRACE-Time | 未实现 |
| 按视频阶段和空间 ROI 路由 TRACE-ST | 未实现 |
| residual normalization | 未实现 |
| relative norm clipping | 未实现 |
| 每样本/层/step 自适应 lambda | 未实现 |
| 对象 mask、轨迹、守恒约束 | 未实现 |

## 8. 与最新代码状态的关系

审计 6.5 的数字来自审计时的旧输出，不能因后来修改推理默认值而追溯改变。当前启动配置已经改成 1280×704、97 帧，并移除了通用 negative prompt；这些改动没有自动实现 7.5 的残差归一化、裁剪或自适应强度。

相关实现位置：

- block 内 ACE 差分：[model_adapter.py](../code/v1/ace_router/model_adapter.py)
- layer/step gate：[schedules.py](../code/v1/ace_router/schedules.py)
- 比例诊断：[diagnostics.py](../code/v1/ace_router/diagnostics.py)
- I2V/T2V 去噪与 CFG：[pipeline.py](../code/v1/ace_router/pipeline.py)
- 原方案文档：[wan_appearance_semantics_causal_routing_mvp_20260831.md](./wan_appearance_semantics_causal_routing_mvp_20260831.md)
- 质量审计：[QUALITY_AUDIT_20260901.md](../code/v1/outputs/ace_router/QUALITY_AUDIT_20260901.md)
