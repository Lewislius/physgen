# Wan2.2-TI2V-5B 的 ProPhy VideoPhy2 对照实验

用官方 Wan2.2-TI2V-5B 权重走原生纯文生视频入口。题目直接读取 [Wan2.1 实验协议](../prophy_wan21_t2v_1_3b/protocol.json)，保持相同的 600 题、180 题 HARD 标记、题序和逐题种子 `42 + index`。使用同一条负面提示，不扩写提示词。

两版都生成 832×480、81 帧的视频，以 16 fps 写入 H.264。UniPC 50 步，shift 5，CFG 5。Wan2.2 使用自己的 DiT、文本编码器和 VAE，BF16 推理配合 CPU offload。16 fps 和 81 帧用于匹配 Wan2.1 本地实验。

生成完成后，调用 Wan2.1 实验的同一份 VideoPhy2 AutoEval 脚本和配置。每条视频抽 32 帧，SA、PC 各自达到 4 分即通过，同一条视频两项都达到 4 分计为 Joint。

本实验的对照对象是本地 Wan2.1 结果。题目取 VideoPhy2 CSV 的 `caption` 列。ProPhy 论文附录 A.5 说明原论文表格使用 `upsampled_caption`，附录 A.4 写的是 DDIM 50 步。作者后来公开的 Wan 推理脚本使用 UniPC，默认 shift 3。两处资料见 [ProPhy 附录](https://arxiv.org/html/2512.05564v1#A1) 和 [作者推理脚本](https://github.com/zijunwa/ProPhy/blob/main/inference_wan.py)。

## 启动

在这个目录提交单卡 `ada-48g` 实验。

```bash
det -m http://10.130.130.5:4096 experiment create submission_ada48.yaml
```

生成结果、逐题评分和 CSV 写入 `outputs/wan22_ti2v_5b/`。全量和 HARD 汇总写入 `summary.json`，与本地 Wan2.1 逐题对齐后的指标差值写入 `comparison.json`。通过同一配置恢复时会复用已经生成的匹配视频和评分。
