# V4.3 step100 / 41 个目标 / 单卡 96GB 推理

入口为 [infer_flow_oracle_1x96g.sh](infer_flow_oracle_1x96g.sh)，集群配置为 [infer_flow_oracle_1x96g.yaml](infer_flow_oracle_1x96g.yaml)。默认固定加载：

```text
/home/liuzhirui/Project/physGen/code/v4.3/checkpoints/v43_native_joint_flow_oracle_20260918T151935Z-69fbcbf3/checkpoint-0000100
```

直接严格加载该目录的 `config.yaml` 和 `corrector.pt`，不读取优化器或 EMA，也不合并当前训练配置。三处 3072 维 P/H 交互位于 Wan 第 5、15、25 层；正负 CFG 分支独立初始化 P，每次 Wan 调用重置状态。推理不需要完整 GT 视频、训练缓存、在线 JEPA 或 Oracle 搜索。

默认生成 **41 条视频（21 I2V、20 T2V）**：

- reference I2V：`/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref5.jpg`，提示词 `The woman smiles and waves at the camera.`。
- P01–P20：依次生成每个目标的 I2V、T2V；读取 `v1/demo/Pxx/Pxx-origin.txt`。I2V 优先使用 `Pxx-i0-1280x704.png/jpg`，T2V 不读取首图。

沿用旧 V4.2 demo41 的测试图片、正提示词和主要采样参数：UniPC 50 步、shift5、CFG5、seed42、121 帧、24fps、5 秒采样时间窗。**负面提示词为空字符串**，与指定 checkpoint 的空文本 dropout 训练一致；保留 CFG，不添加旧测试中的显式负面词。reference 为 384×384，其余默认 512×288；121 帧封装后 MP4 时长约 5.04 秒。该 checkpoint 仅训练 I2V，T2V 为泛化评估。

```bash
cd /home/liuzhirui/Project/physGen/code/v4.3

# 在已分配的单卡 96GB 环境直接运行全部 41 条。
bash inference/infer_flow_oracle_1x96g.sh

# 提交到 blk-96g 资源池（1 个 slot）；源码和模型通过 bind mount 读取。
det experiment create inference/infer_flow_oracle_1x96g.yaml inference

# CPU 检查真实权重和全部 41 条条件，不生成视频。
CHECK_ONLY=1 REPORT=analysis/checks/inference_preflight.json \
  bash inference/infer_flow_oracle_1x96g.sh

# 只检查最终启动命令，不加载模型。
PARSE_ONLY=1 bash inference/infer_flow_oracle_1x96g.sh

# 只生成 P01 的 I2V。
SUITE=demo DEMO_MODES=i2v SAMPLE_IDS=P01 \
  bash inference/infer_flow_oracle_1x96g.sh

# 恢复同一批次；checkpoint、输入与采样参数必须保持一致。
OUTPUT=/absolute/path/to/existing_batch RESUME_INFERENCE=1 \
  bash inference/infer_flow_oracle_1x96g.sh
```

默认输出目录为 `inference_outputs/v43_flow_oracle_<UTC时间>_1x96g/`，包含 `cases.json`、`summary.json`；每个视频另有 `.json` 元数据与 `.steps.jsonl` 采样日志：

```text
reference/reference/i2v/seed42_full.mp4
demo20/P01/i2v/seed42_full.mp4
demo20/P01/t2v/seed42_full.mp4
...
demo20/P20/t2v/seed42_full.mp4
```

可用环境变量覆盖 `CHECKPOINT`、`OUTPUT`、`SEED`、`SOLVER`、`SAMPLING_STEPS`、`FRAMES`、`GUIDANCE` 等。`SUITE=demo` 只跑 40 条 P01–P20 双模式，`SUITE=both` 再加 reference。通用 `run_inference.sh` 的原默认值仍为 20 条 I2V / Euler；上述 96GB 专用入口才是 41 条 / UniPC。

真实权重预检检查严格加载、数值有限性、基模文件和全部 41 条条件，使用上面的 `CHECK_ONLY` 命令按需运行，报告统一写入 `analysis/checks/`。旧的一次性预检输出已清理。

已有匹配生成的质量结论与关键原始数据统一见 [分析入口](../analysis/README.md)，权重预检本身不证明视频质量。
