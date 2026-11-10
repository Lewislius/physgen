# 无 Verifier 的 H/Q 最终权重推理记录

更新时间（UTC）：2026-10-27T16:50:11.097277+00:00

H 来自训练实验 **66438 / v29**；Q 来自 **66443 / v30**。两者均使用最终 **224 步** checkpoint，训练预算为 64 步 interface + 160 步 joint。

**最新提交安排：Q VideoPhy2-40 已获用户授权继续提交，复用训练 66443 的最终权重和原有 40 例输入。此前优先安排的 H/Q demo21 与 H VideoPhy2-40 的状态见下表。**

推理使用训练代码中的 `PhysGenNative`、`restore`、`sample` 和 `export_video`，精确恢复 message、C 投影、alignment 与 Wan LoRA。H/Q 均执行两轮 message refinement；Q 另保留原有 R0 初始 grounding 和 128 个槽位。不构造 verifier，不启用评分、校准、反馈反传或 JEPA，不重新校准已保存的 alignment。

生成设置：832×480、81 帧、24 FPS、50 步 shifted Euler、shift=5、CFG=5、condition_strength=1。demo21 使用训练 seed=42；VideoPhy2 使用原有每例 seed 和持久化 FP32 初始噪声，H/Q 共用同一组输入与噪声。

测试清单：demo21 = reference + P01–P20；VideoPhy2 = 既有固定 VP001–VP040，共 40 例。VideoPhy2 是本项目的共享首帧 I2V 测试，不是官方完整 T2V 榜单。这里只提交视频生成；独立 SA/PC 打分未在本次自动运行。

- demo 输入：`../v5.2/inference_outputs/1d-v52-lora-demo21_20261010/`（相对于项目目录）。
- VideoPhy2 输入：`../v5.2/inference_outputs/videophy2_lora_run64543_joint400_60_seed42/`。
- 固定 40 例清单：`analysis/videophy40_20261026/case_ids.json`。
- 只读取 `first/text/null_text/times/valid`、首帧与 caption 派生的 requirements，以及保存的随机噪声；不读取历史未来视频或 JEPA/task tokens。

资源池固定 **blk-96g**，每个实验 **slots_per_trial: 1**。在读取 checkpoint、输入和模型前检查实际 GPU，只接受 RTX PRO 6000 / RTX 6000 Blackwell；H100 或其他 GPU 立即退出。监控器只对该型号拒绝重提，并在 RTX 忙碌时等待空闲，保留全部尝试 ID。未修改共享调度、GPU slot 或其他任务。

运行前检查最终权重版本/步数、输入尺寸和有限数值；每个生成结果检查实际推理轮数、视频可完整解码、尺寸/帧数/FPS。没有新增哈希或 SHA256 检查。通过这些检查表示生成链路可运行，不代表物理效果已经评测达标。

提交前已完成 4 组真实权重与输入预检（21/40/21/40 例），以及 11 项推理回归测试；覆盖 H100 在 checkpoint 打开前退出、旧固定 judge 权重拒绝、两种模式的共享噪声精确重放、输入隔离、两轮推理和只重提型号不符的失败。记录见 `analysis/no_verifier_inference_20261027/validation.json`。

## 推理代码与提交入口

- 专用入口：[infer_no_verifier_benchmarks.py](../inference/infer_no_verifier_benchmarks.py)。
- 共用生成实现：[infer_message_demo21.py](../inference/infer_message_demo21.py)。
- 权重与输出映射：[no_verifier_jobs.py](../inference/no_verifier_jobs.py)。
- 自动提交/重提/记录：[submit_no_verifier_benchmarks.py](../inference/submit_no_verifier_benchmarks.py)。
- 默认先提交 3 组：`bash inference/submit_no_verifier_first3_blk96g.sh`；单组可运行对应 SH 的 `--check` 或 `--submit`。
- 单独补交 Q 的 40 例：`bash inference/infer_q_videophy40_no_verifier_blk96g.sh --submit`。
- 相同输出目录重启会核对 manifest、复用已验证视频并继续；不覆盖已完成的历史版本。

目录名中的 `66438` / `66443` 是训练 checkpoint 的来源实验；实际推理 Experiment 和 Trial 在下表及每段视频旁 JSON 中单独记录。

## 实验与结果

| 任务 | 训练实验 | 当前推理实验 / trial | 已验证视频 | 状态 | 实际 GPU |
|---|---:|---|---:|---|---|
| H demo21 | 66438 | [66479](http://10.130.130.5:4096/det/experiments/66479) / 67859 | 21/21 | 全部完成 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| H videophy40 | 66438 | [66471](http://10.130.130.5:4096/det/experiments/66471) / 67851 | 40/40 | 全部完成 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Q demo21 | 66443 | [66472](http://10.130.130.5:4096/det/experiments/66472) / 67852 | 21/21 | 全部完成 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Q videophy40 | 66443 | [66517](http://10.130.130.5:4096/det/experiments/66517) / 67897 | 40/40 | 全部完成 | NVIDIA RTX PRO 6000 Blackwell Server Edition |

### H / demo21

- SH：[infer_h_demo21_no_verifier_blk96g.sh](../inference/infer_h_demo21_no_verifier_blk96g.sh)。
- YAML：[infer_h_demo21_no_verifier_blk96g.yaml](../inference/infer_h_demo21_no_verifier_blk96g.yaml)。
- 最终训练权重：`/home/liuzhirui/Project/physGen/code/v5.2.2-native/checkpoints/v522-h-content-message-no-verifier-simple_66438/latest.pt`。
- 输出目录：`/home/liuzhirui/Project/physGen/code/v5.2.2-native/inference_outputs/v522-h-content-message-no-verifier-simple_66438/demo21_final224_round2_seed42`。
- 视频位置：`reference/i2v/seed42.mp4`、`P01/i2v/seed42.mp4` … `P20/i2v/seed42.mp4`。
- 目录内 `checkpoint.pt` 是最终权重的硬链接，`manifest.json` 记录输入/配置，`summary.json` 记录生成进度；每段视频有同名 JSON 元数据。
- 已通过完整解码检查的首条视频：`/home/liuzhirui/Project/physGen/code/v5.2.2-native/inference_outputs/v522-h-content-message-no-verifier-simple_66438/demo21_final224_round2_seed42/reference/i2v/seed42.mp4`。
- 尝试 66470：ERROR；GPU=NVIDIA H100；GPU 放行=False。
- 尝试 66474：ERROR；GPU=NVIDIA H100；GPU 放行=False。
- 尝试 66475：ERROR；GPU=NVIDIA H100；GPU 放行=False。
- 尝试 66478：ERROR；GPU=NVIDIA H100；GPU 放行=False。
- 尝试 66479：COMPLETED；GPU=NVIDIA RTX PRO 6000 Blackwell Server Edition；GPU 放行=True。

### H / videophy40

- SH：[infer_h_videophy40_no_verifier_blk96g.sh](../inference/infer_h_videophy40_no_verifier_blk96g.sh)。
- YAML：[infer_h_videophy40_no_verifier_blk96g.yaml](../inference/infer_h_videophy40_no_verifier_blk96g.yaml)。
- 最终训练权重：`/home/liuzhirui/Project/physGen/code/v5.2.2-native/checkpoints/v522-h-content-message-no-verifier-simple_66438/latest.pt`。
- 输出目录：`/home/liuzhirui/Project/physGen/code/v5.2.2-native/inference_outputs/v522-h-content-message-no-verifier-simple_66438/videophy40_final224_round2_shared_noise`。
- 视频位置：`cases/VP001/planned.mp4` … `cases/VP040/planned.mp4`。
- 目录内 `checkpoint.pt` 是最终权重的硬链接，`manifest.json` 记录输入/配置，`summary.json` 记录生成进度；每段视频有同名 JSON 元数据。
- 已通过完整解码检查的首条视频：`/home/liuzhirui/Project/physGen/code/v5.2.2-native/inference_outputs/v522-h-content-message-no-verifier-simple_66438/videophy40_final224_round2_shared_noise/cases/VP001/planned.mp4`。
- 尝试 66471：COMPLETED；GPU=NVIDIA RTX PRO 6000 Blackwell Server Edition；GPU 放行=True。

### Q / demo21

- SH：[infer_q_demo21_no_verifier_blk96g.sh](../inference/infer_q_demo21_no_verifier_blk96g.sh)。
- YAML：[infer_q_demo21_no_verifier_blk96g.yaml](../inference/infer_q_demo21_no_verifier_blk96g.yaml)。
- 最终训练权重：`/home/liuzhirui/Project/physGen/code/v5.2.2-native/checkpoints/v522-q-seed128-message-no-verifier-simple_66443/latest.pt`。
- 输出目录：`/home/liuzhirui/Project/physGen/code/v5.2.2-native/inference_outputs/v522-q-seed128-message-no-verifier-simple_66443/demo21_final224_round2_seed42`。
- 视频位置：`reference/i2v/seed42.mp4`、`P01/i2v/seed42.mp4` … `P20/i2v/seed42.mp4`。
- 目录内 `checkpoint.pt` 是最终权重的硬链接，`manifest.json` 记录输入/配置，`summary.json` 记录生成进度；每段视频有同名 JSON 元数据。
- 已通过完整解码检查的首条视频：`/home/liuzhirui/Project/physGen/code/v5.2.2-native/inference_outputs/v522-q-seed128-message-no-verifier-simple_66443/demo21_final224_round2_seed42/reference/i2v/seed42.mp4`。
- 尝试 66472：COMPLETED；GPU=NVIDIA RTX PRO 6000 Blackwell Server Edition；GPU 放行=True。

### Q / videophy40

- SH：[infer_q_videophy40_no_verifier_blk96g.sh](../inference/infer_q_videophy40_no_verifier_blk96g.sh)。
- YAML：[infer_q_videophy40_no_verifier_blk96g.yaml](../inference/infer_q_videophy40_no_verifier_blk96g.yaml)。
- 最终训练权重：`/home/liuzhirui/Project/physGen/code/v5.2.2-native/checkpoints/v522-q-seed128-message-no-verifier-simple_66443/latest.pt`。
- 输出目录：`/home/liuzhirui/Project/physGen/code/v5.2.2-native/inference_outputs/v522-q-seed128-message-no-verifier-simple_66443/videophy40_final224_round2_shared_noise`。
- 视频位置：`cases/VP001/planned.mp4` … `cases/VP040/planned.mp4`。
- 目录内 `checkpoint.pt` 是最终权重的硬链接，`manifest.json` 记录输入/配置，`summary.json` 记录生成进度；每段视频有同名 JSON 元数据。
- 已通过完整解码检查的首条视频：`/home/liuzhirui/Project/physGen/code/v5.2.2-native/inference_outputs/v522-q-seed128-message-no-verifier-simple_66443/videophy40_final224_round2_shared_noise/cases/VP001/planned.mp4`。
- 尝试 66473：CANCELED；GPU=NVIDIA RTX PRO 6000 Blackwell Server Edition；GPU 放行=True。
- 尝试 66511：ERROR；GPU=NVIDIA H100；GPU 放行=False。
- 尝试 66513：ERROR；GPU=NVIDIA H100；GPU 放行=False。
- 尝试 66514：ERROR；GPU=NVIDIA H100；GPU 放行=False。
- 尝试 66515：ERROR；GPU=NVIDIA H100；GPU 放行=False。
- 尝试 66516：ERROR；GPU=NVIDIA H100；GPU 放行=False。
- 尝试 66517：COMPLETED；GPU=NVIDIA RTX PRO 6000 Blackwell Server Edition；GPU 放行=True。

## 机器记录与监控

- 完整尝试与进度：[jobs.json](../analysis/no_verifier_inference_20261027/jobs.json)。
- 监控日志：`analysis/no_verifier_inference_20261027/controller.log`。
- 本文由监控器每 30 秒更新；最终是否完成以本次实验状态及各目录 `summary.json` 为准。
