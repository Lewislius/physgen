# P02 强阶段 c+ 与结构化 JSON 注入消融

日期：2026-09-04  
范围：不增加 StateBridge、不训练新参数，只修改现有 Wan cross-attention 条件路径。

## 目标

固定 P02 的 I2V 首帧、global semantic、五条 stage c+、seed、CFG 和采样器，对比：

1. `baseline`：当前固定弱阶段残差；
2. `strong_stage`：阶段 c+ 与 global semantic 在活动 block 内按 RMS 等强；
3. `strong_stage_json`：在 2 的基础上，再注入实体数量、恒定外形、初始动量和逐阶段状态。

三组默认均使用 P02、seed=42、CFG=5、50 steps、97 帧、1280×704。

## 三组注入强度

### baseline

默认 `stage_strength=0.05`，进入 conditional 分支前除以 CFG=5，并乘去噪步门控
`0.5 / 1.0 / 0.3 / 0.1`。因此在注入 block 内，stage c+ 相对 global cross-attention
的名义系数约为：

- 前 30% 去噪步：0.5%；
- 中间 40%：1.0%；
- 后续 20%：0.3%；
- 最后 10%：0.1%。

这只是系数比，不等于实际张量能量比。

### strong_stage

每一层、每一阶段先计算：

```text
stage_delta = CrossAttn(stage_c+) - CrossAttn(empty)
stage_normalized = RMS(global) / RMS(stage_delta) * stage_delta
```

随后使用：

```text
hidden += global + stage_ratio_to_global * stage_normalized
```

默认 `stage_ratio_to_global=1.0`，所以在第 14–23 个注入 block、对应阶段的活动时间
latent 上，stage c+ 的目标 RMS 与 global semantic 相同。二者随后一起经过 CFG，因此相对比例
不再随 CFG 改变。global 仍作用于全部 30 个 block，stage c+ 只作用于 10 个中层 block；这里的
“等强”指活动 block 内的局部残差等强，不是整个网络累计贡献完全相等。

增强模式不再使用逐步衰减的 `stage_step_gate`，stage c+ 在全部去噪步持续参与。

### strong_stage_json

在 `strong_stage` 上增加两组独立、同样经过 RMS 校准的文本残差：

- JSON global：实体数量、持续身份、颜色和刚性圆形外观，目标为 global RMS 的 0.25；
- JSON stage：初始动量、当前 presence/count/state、允许变化和阶段关系，目标为 global RMS 的 0.50。

stage、JSON global、JSON stage 的合计附加残差被限制在 global RMS 的 1.75 倍以内。JSON 不会
拼接进 global semantic 或 stage c+，也不会自动推导禁用对象；它只把显式结构化字段渲染成
短的正向自然语言，再作为独立 context 注入。

## P02 结构化信息

P02 的 `structured_guidance` 显式声明：

- 始终只有一个红球和一个蓝球；
- 两球保持各自颜色、完整身份和刚性圆形；
- setup 时红球已有向右动量、蓝球静止；
- onset 只有一次边缘接触；
- evolution 后蓝球在前、红球减速；
- completion/terminal 两个原球保持可见且重新分离。

初始动量只注入 setup，不会在后续阶段重复激活。

## 运行方式

静态预检：

```bash
CHECK_ONLY=1 /home/liuzhirui/Project/physGen/code/v3/inference/run_p02_conditioning_ablation.sh
```

GPU 三组消融：

```bash
/home/liuzhirui/Project/physGen/code/v3/inference/run_p02_conditioning_ablation.sh
```

集群配置：

```text
/home/liuzhirui/Project/physGen/code/v3/inference/infer_p02_conditioning_ablation.yaml
```

Determined 实验：`59377`，trial：`59973`。baseline 与 strong_stage 已成功生成，
strong_stage_json 正在采样。strong_stage 的 500 个审计点中，stage/global RMS 比为
`0.9498–1.0073`，均值 `0.9857`，与局部等强目标一致。

每个增强运行会保存 `conditioning.audit.json`，记录各采样步和 block 的实测
stage/global、JSON/global、合计残差比例及 cap 触发率。

## 评价重点

- setup：是否只有红球运动，蓝球不提前运动；
- onset：是否出现一次清楚接触；
- evolution：蓝球是否启动、红球是否减速；
- terminal：两个原球是否都存在、未复制、未消失、未变形；
- 强化阶段信息后，桌面、摄影机和整体场景是否仍保持 global semantic 的一致性。

## 边界

这仍然是无训练的文本 cross-attention 注入。它能测试“更强阶段语义和结构化事实是否有帮助”，
但不能像未来的 StateBridge 一样提供实体级空间定位或硬约束；即使 JSON 文本权重足够大，T5/Wan
也可能不严格执行数量和状态。该实验的价值是决定是否值得继续做结构化非文本控制。
