# step0800 EMA / demo41 单卡推理

`infer_1x96g.yaml` 和 `infer_1x96g.sh` 默认选择：

- checkpoint：`checkpoints/lora_20260917T140727Z_dae2a0/step0800`。
- 配置和权重：该目录的 `config.json`、`ema.pt`，rank=32、alpha=32。
- 测试：reference I2V 1 条，P01–P20 各 I2V/T2V，共 **41 条（21 I2V、20 T2V）**。图片、origin 提示词、模式、顺序与 v4/v4.2 一致。
- 采样：UniPC 50 步、shift=5、CFG=5、seed=42；121 帧，5 秒时间窗，24fps。reference 为 384×384，其余为 512×288。121 帧封装后的 MP4 约 5.04 秒。v4 旧版的 Euler/101 帧未用于本次 LoRA；这里对齐其测试条件。
- 资源：`blk-96g`，1 卡，原 moviestory 环境。

提交命令：

```bash
cd /home/liuzhirui/Project/physGen/code/v4-static-lora
det experiment create inference/infer_1x96g.yaml .
```

默认输出：

```text
inference_outputs/lora_20260917T140727Z_dae2a0_step0800_ema_demo_seed42/
  cases.json
  conditions_ready.json
  summary.json
  reference_i2v/video.mp4
  P01_i2v/video.mp4
  P01_t2v/video.mp4
  ...
  P20_t2v/video.mp4
  report.json
```

保留相同 OUTPUT 再启动会校验条件与已完成文件并续跑；全部完成时不再加载 Wan/VAE。`summary.json` 记录完成数和失败案例，日志每 5 个采样步骤更新。修改 checkpoint 或 suite 后默认输出目录自动变化；相同 checkpoint 的不同 single 提示词需自行指定不同 OUTPUT。

不需要 GPU 的完整适配预检：

```bash
CHECK_ONLY=1 bash inference/infer_1x96g.sh
```

预检检查保存配置、EMA SHA256、基模资产清单、原始 safetensors 中 240 个投影的形状、480 个 EMA 张量的名称/形状/FP32/有限性，以及 41 份生成条件；保存 `preflight.json`、`preflight_cases.json`。预检不会产生视频。

本次修复了两处启动问题：原入口默认最终 step1200/final81；源码校验又因共享 v4.2 的更新拒绝旧 LoRA checkpoint。现在使用本目录的 [推理兼容记录](checkpoint_compatibility.json)，仅允许已审计的精确源码摘要迁移，未知修改仍拒绝。LoRA 前向、原 Wan 加载/FP32 部分、几何处理及 T5/VAE 编码器均已核对未变。训练恢复校验独立保留。推理完整性校验只读保存配置及 EMA，不要求读取 optimizer、RNG 或 raw LoRA 文件。

CPU 审计进一步在原生 Wan 的完整尺寸模型结构中严格加载了真实 EMA（仅 LoRA 在 CPU 分配，5B 基模留在 meta），逐张量确认加载一致，并运行了一个真实训练后投影；不是仅检查文件存在。

- [适配结果](../verification/step0800_demo41_20260918/result.json)
- [完整 41 条测试计划](../verification/step0800_demo41_20260918/cases.json)
- [源码核验](../verification/step0800_demo41_20260918/source_audit.json)
- [自动化测试日志](../verification/step0800_demo41_20260918/unit_tests.log)

复现：

```bash
OMP_NUM_THREADS=2 /home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B verification/verify_step0800_demo41.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B -m unittest discover -s tests -p 'test_*.py'
PARSE_ONLY=1 bash inference/infer_1x96g.sh
```

当前验证环境没有可用 CUDA，完整 41 条 GPU 采样、96GB 实际显存峰值和生成质量尚未验证。CPU 预检通过表示权重与入口适配通过，不代表视频已经生成。
