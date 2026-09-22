# 原生 V4 三 CORRECTOR 推理

正式入口：`infer_stability_1x96g.sh` / `infer_stability_1xada48g.sh`，对应 YAML 已更新。推理加载 `corrector.pt` 和同目录 `config.yaml`，不再走旧两 CORRECTOR 的 adapter/EMA 路径，不需要在线 JEPA 或旧图像 anchor。

```bash
# 有 PRIOR 训练版本；须已经产生新版 checkpoint
CHECKPOINT=latest bash inference/infer_stability_1x96g.sh

# 无 PRIOR 训练版本
CHECKPOINT=latest_no_prior bash inference/infer_stability_1x96g.sh

# 只生成一个完整 I2V 样例
CHECKPOINT=latest SAMPLE_IDS=P01 bash inference/infer_stability_1x96g.sh
```

默认生成 P01–P20 的 I2V，121 帧/24fps/5 秒，JEPA32 坐标，Euler50、shift5、CFG5、空负提示词、seed42。正负分支都运行完整 V4 CORRECTOR，独立初始化各自 P0，每个去噪步重置状态；不额外限制写回 RMS 或输出 delta velocity。

`CHECKPOINT` 也可明确指定新版 `checkpoint-0000100` 等目录。`latest`/`latest_with_prior` 与 `latest_no_prior` 分别解析到不同训练版本；无可用 checkpoint 会明确报错，绝不回退到旧 step0400。批次输出默认带 UTC 时间戳，复跑同一输出目录需要 `OUTPUT=... RESUME_INFERENCE=1`，且采样参数和 checkpoint 必须一致。

T2V 可通过原生 Python 工具或 `DEMO_MODES=t2v` 显式运行，但这两个训练版本仅训练 I2V，T2V 不作为已经验证的训练内能力。
