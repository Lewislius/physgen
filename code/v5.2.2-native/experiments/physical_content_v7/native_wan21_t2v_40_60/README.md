# 原版 Wan2.1‑T2V‑1.3B 的固定 VideoPhy2 40/60 对照

读取 R20ET 使用的 [固定题目](../videophy60/protocol.json)。VP001–VP040 为固定 40 题，VP041–VP060 是补充 20 题。每题用原始提示词和官方 Wan2.1‑T2V‑1.3B 的 DiT、UMT5 和 VAE 生成一条视频。

生成按 R20ET 设置执行，832×480、81 帧、**16 fps（Wan2.1 原生帧率）**、Euler 50 步、shift 5、CFG 5，CFG 的负分支用空提示。每题使用 seed 42 在 Wan2.1 原生 latent 尺寸生成 FP32 噪声。R20ET 的 Wan2.2 latent 尺寸不同，两者共用种子和采样公式。Wan2.2 系列继续使用其 24 fps 协议，本目录不与其做跨帧率差值。

单卡 `ada-48g` 依次生成和评分。复用 R20ET 的 VideoPhy2 AutoEval，每条视频取 32 帧，SA 和 PC 都 ≥4 计为联合达标。

- `outputs/videophy60_16fps/` 保存全部 60 条 16 fps 视频和逐题分数。
- `comparison_16fps.json` 汇总前 40 条和全部 60 条；Wan2.1 的 40/60 基线只在 16 fps 内与 R20FT-Wan2.1 比较。
- 旧的 `outputs/videophy60/` 和 `comparison.json` 是 24 fps 历史结果，已被本目录的 16 fps 口径取代。
- `submission_ada48.yaml` 是 Determined 提交配置。
