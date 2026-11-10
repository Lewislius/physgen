# V5.2.2-native 推理入口

## 当前无 verifier 版本：H v29 / Q v30

使用训练实验 **66438（H）/66443（Q）** 的最终 **224 步**权重，两版都执行 **2 轮**推理。
每版已准备 demo21 和固定 VideoPhy2-40 的 SH/YAML，入口为 [infer_no_verifier_benchmarks.py](infer_no_verifier_benchmarks.py)。

此前优先执行的 **H demo21、Q demo21、H VideoPhy2-40** 三项已全部完成；用户现已授权补交 **Q VideoPhy2-40**。
全部使用 **blk-96g 单卡**，模型加载前拒绝 H100，只接受 RTX PRO 6000 / RTX 6000 Blackwell；GPU 型号不符会退出并自动重提。

```bash
bash inference/submit_no_verifier_first3_blk96g.sh
```

单独补交 Q 的 40 例：`bash inference/infer_q_videophy40_no_verifier_blk96g.sh --submit`。

完整的权重、代码、SH/YAML、当前实验 ID、重提记录、验证状态和输出路径见
[中文推理记录](../docs/无Verifier_HQ最终权重推理记录_20261027.md)，该文件由监控器更新。

输出目录中的 `66438` / `66443` 表示**训练 checkpoint 的来源实验**。例如 Q demo21 的来源是训练 66443，
本次推理实验是 66472 / Trial 67852，所以视频放在 `v522-q-seed128-message-no-verifier-simple_66443/demo21_final224_round2_seed42/`。
视频旁 JSON 分别记录 `training_experiment_id` 和 `inference_experiment_id`，重提不会改变来源权重目录。

## 历史固定 verifier 版本：H v25 / Q v27

以下为历史权重与提交记录，保留以便复现。

两份权重均已完成 224 步（interface 64 + joint 160）。推理沿用各自的训练出口：H **1 轮**，Q **2 轮**；Q 的第 0 次 grounding read 另计，之后执行两次更新。**两份训练 seed 均为 42，所有样例均固定使用 42，不按样例递增；传入其他 seed 会报错。** 当前按用户选择提交到 **amp-48g**，保持已有 LoRA 训练运行。

| 方案 | 最终权重 | 启动脚本 | 单卡任务 YAML |
| --- | --- | --- | --- |
| H-content，v25，1 轮 | `../checkpoints/v522-h-content-message-verifier-frozen-simple_66356/latest.pt` | [infer_h_content_frozen_demo21_1x48g.sh](infer_h_content_frozen_demo21_1x48g.sh) | [infer_h_content_frozen_demo21_1x48g.yaml](infer_h_content_frozen_demo21_1x48g.yaml) |
| Q-seed128，v27，2 轮 | `../checkpoints/v522-q-seed128-message-verifier-frozen-simple_66353/latest.pt` | [infer_q_seed128_frozen_demo21_1x48g.sh](infer_q_seed128_frozen_demo21_1x48g.sh) | [infer_q_seed128_frozen_demo21_1x48g.yaml](infer_q_seed128_frozen_demo21_1x48g.yaml) |

48GB 入口默认 `CPU_OFFLOAD=1`，先执行 Qwen 条件编码，再将 Qwen 移至 CPU、Wan 移到 GPU 采样，采样后释放 Wan 的显存用于 VAE 解码。此池也包含 **40GB A100**，因此按 40/48GB 运行；只改变模型驻留位置，不缩短视频或采样过程。CPU 行为测试确认 H/Q 分阶段路径和原 CFG 路径输出逐元素一致。实际 GPU 型号、总显存、峰值显存及每例的驻留方式写入结果记录。

共用入口：[infer_message_demo21.py](infer_message_demo21.py)。每组执行 `reference + P01–P20`，固定 **832×480、81 帧、24 FPS、50 步 shifted Euler、shift=5、CFG=5、condition_strength=1**。按 checkpoint 精确恢复所有 message、projection、alignment 和 Wan LoRA 权重，再应用采样设置。在线 verifier refinement 与训练配置一致，保持关闭；推理时重新执行 Qwen 物理分析和完整条件生成路径。

输入复用 `../v5.2/inference_outputs/1d-v52-lora-demo21_20261010` 中已存在的 21 组条件。预检核对原始 `Pxx-origin.txt`、基础模型路径、图片、时间轴和尺寸；运行只取 `first/text/null_text/times/valid`，不使用历史 task tokens、JEPA 数据或未来视频。既有 requirements 的来源必须为 `caption_and_first_image`。训练缓存及历史结果不修改。

从 `v5.2.2-native` 根目录执行 CPU 预检：

```bash
bash inference/infer_h_content_frozen_demo21_1x48g.sh --check
bash inference/infer_q_seed128_frozen_demo21_1x48g.sh --check
```

提交到现有 `amp-48g`，每个任务只申请一个 slot：

```bash
bash inference/infer_h_content_frozen_demo21_1x48g.sh --submit
bash inference/infer_q_seed128_frozen_demo21_1x48g.sh --submit
```

当前 amp-48g 任务：**H 66425，Q 66426**。任务及预检记录见 [jobs.json](../analysis/demo21_final_20261027/jobs.json)，实际进度以输出目录的 `summary.json` 为准。

首次 blk-96g 提交：H 为 **66421 / trial 67801**，Q 为 **66422 / trial 67802**。两次均分到 H100 `GPU-30085d8b-9408-7a7f-e544-be0280858108`，分别于 2026-10-27 02:33 / 02:35 UTC 被 SIGKILL，以 137 退出；均未完成视频。信号发送来源尚未确定。原 [H 96GB SH](infer_h_content_frozen_demo21_1x96g.sh)、[H 96GB YAML](infer_h_content_frozen_demo21_1x96g.yaml)、[Q 96GB SH](infer_q_seed128_frozen_demo21_1x96g.sh)、[Q 96GB YAML](infer_q_seed128_frozen_demo21_1x96g.yaml) 保留，其 YAML 已加入该失败 UUID 的启动检查。

YAML 指定的本次输出目录：

- H：`../inference_outputs/v522-h-content-message-verifier-frozen-simple_66356/demo21_step0224_round1_seed42/`
- Q：`../inference_outputs/v522-q-seed128-message-verifier-frozen-simple_66353/demo21_step0224_round2_seed42/`

每例视频为 `<case>/i2v/seed42.mp4`，同目录包含推理 JSON、首帧、prompt/requirements 和实际使用的五项输入张量。根目录保存 `manifest.json`、`summary.json`、原始 checkpoint 配置，以及固定最终权重的 `checkpoint.pt` 硬链接。逐例检查 MP4 的完整解码、81 帧、原生尺寸和 FPS 后才记为完成。不会给 checkpoint 增加 hash/SHA 校验。

在**已分配 GPU** 的环境中也可直接运行脚本；默认创建带时间戳的新目录。`OUTPUT` 指定目录，`SAMPLE_IDS=P01,P02` 可选样例。恢复同一输出目录时使用 `RESUME=1 OUTPUT=/原目录 bash ...sh`，会核对相同输入和采样设置、复用固定权重并验证已完成的视频；新的一批结果请使用新目录。两份 YAML 默认 `RESUME=auto`，同目录已保存 manifest 和固定权重时自动续跑。直接本地运行时，可设置 `EXCLUDE_GPU_UUID` 使用同样的失败设备检查。
