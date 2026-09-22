physgen 视频生成实验项目。各版本的源码、配置、训练/推理启动脚本及使用说明位于 `code/`；研究记录位于 `analysis/`、`log/` 和 `prompt/`。

本仓库上传源码、配置、文档、推理结果及全部视频。大型训练缓存、模型/优化器 checkpoint 和中间 tensor 留在原训练机器，不纳入 Git；本地文件未删除。`code/v4-static-lora/assets/negative.pt` 是独立推理所需的小型固定负提示词编码，随源码保留。

视频按原目录、原文件名、原字节保存，共 852 个：

| 版本 | 视频数 | 位置 |
|---|---:|---|
| v1 | 220 | [code/v1](code/v1) |
| v2 | 40 | [code/v2](code/v2) |
| v3 | 161 | [code/v3](code/v3) |
| v4 | 49 | [code/v4](code/v4) |
| v4.2 | 177 | [code/v4.2](code/v4.2) |
| v4.3 | 164 | [code/v4.3/inference_outputs](code/v4.3/inference_outputs) |
| v4-static-lora | 41 | [code/v4-static-lora/inference_outputs](code/v4-static-lora/inference_outputs) |

[VIDEO_MANIFEST.json](VIDEO_MANIFEST.json) 记录全部视频的相对路径、大小、SHA-256 和 Git blob SHA-1。视频是普通 Git 文件，克隆仓库即可取得原文件，无需 Git LFS。v4.2 的两个 quick16 工作目录视频也已显式纳入本次快照。

[v4.3 使用说明](code/v4.3/README.md)和 [v4-static-lora 使用说明](code/v4-static-lora/README.md)介绍相应实验；最新的帧内全注意力、帧间因果注意力 LoRA 对比见 [frame_causal_lora/README.md](code/v4-static-lora/frame_causal_lora/README.md)。训练和推理所需的外部模型及数据路径详见各版本配置。

本地快照日期：2026-09-22。
