`4×48GB` 版本使用**单进程、四卡模型并行**，主要降低单卡显存占用。它与单卡版使用相同的标准 LoRA 和仅 FM 目标，数据、噪声配比、学习率、EMA、1200 步及每 100 步保存均保持一致；有效 batch 仍是 **1 条 × 累积 8 次 = 8**。

提交训练：

```bash
cd /home/liuzhirui/Project/physGen/code/v4-static-lora
det experiment create train/train_4x48g.yaml .
```

资源配置是同一节点的 `amp-48g`、4 个 GPU slots。脚本只启动一个 Python 进程，由它使用全部四张卡，**不要再套 torchrun**。

| 显存措施 | 实现 |
|---|---|
| 基模分布 | 30 个 transformer block 分为 8 / 8 / 7 / 7 层，每张卡只保留分配到的层及对应 LoRA |
| 加载过程 | 先在 CPU 加载基模、恢复原 FP32 部分并插入 LoRA，再逐段放到 GPU；不会先把完整模型放到 GPU0 |
| 激活重计算 | 每个 block 保留原有的非重入 checkpoint，反向时重算中间结果 |
| 激活暂存 | 使用 `save_on_cpu(pin_memory=True)` 将反向所需保存张量暂存主机内存，以增加 CPU 内存及传输开销换取较低显存 |
| 条件张量 | timestep/context/rotary 张量每次前向只向每个 stage 复制一次，前向结束后清理缓存 |
| 优化器与 EMA | LoRA 参数、梯度、AdamW 状态及 EMA 随其所在层分布在各卡；全局梯度裁剪覆盖全部 LoRA 参数 |
| 监控 | 启动日志记录实际参数分布；每步 `updates.jsonl` 记录四张卡各自的当前和峰值显存 |

原生 Wan 仍负责完整前向；设备路由使用 block/head 的 pre-hook，不添加预测分支、校正器或损失。不同设备之间的 tensor copy 保留 autograd 链，FM 仍是八条样本的平均。原有 `1×96GB` 代码及源码指纹不受这套新增 Python 文件影响。

这种执行方式按顺序经过四个 stage，没有做流水线微批次重叠，也没有四份数据并行副本，因此**不承诺四倍吞吐**。它增加了卡间及 CPU/GPU 传输，以降低每卡显存。CPU 内存占用也会增加。

输出与单卡版分开：

- checkpoint：`checkpoints/four_gpu/<run>/step0100` … `step1200`。
- 最终指针：`checkpoints/four_gpu/latest_final.json`。
- 日志：`train/train_log/four_gpu/<run>/`。
- 推理：`inference_outputs/four_gpu/`。

恢复训练时替换实际运行目录：

```bash
det experiment create train/train_4x48g.yaml . \
  --config 'environment.environment_variables=["RESUME=/home/liuzhirui/Project/physGen/code/v4-static-lora/checkpoints/four_gpu/实际run目录/step0800","PREPARE=1","OMP_NUM_THREADS=4","PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]'
```

保存的 LoRA 张量名字/形状与单卡版相同，但执行配置、源码校验和输出目录独立。使用本版入口恢复；checkpoint 中记录全部可见 CUDA 设备的 RNG 状态。

本版 checkpoint 的对应推理入口已配好：

```bash
cd /home/liuzhirui/Project/physGen/code/v4-static-lora
det experiment create inference/infer_4x48g.yaml .
```

它默认使用四卡训练的 step1200 EMA，仍为相同 81 个 final 案例及 UniPC50/shift5/CFG5/seed42。需要中间 checkpoint 或单条提示词时，沿用主 README 中的 CHECKPOINT、OUTPUT、SUITE、MODE、PROMPT、IMAGE 等环境变量，但 checkpoint/output 路径使用 `four_gpu` 下的目录。

配置和缓存检查不需要 GPU：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B four_gpu/train.py --check-config
PARSE_ONLY=1 bash train/train_4x48g.sh
PARSE_ONLY=1 bash inference/infer_4x48g.sh
```

CPU 测试：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  /home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B tests/test_four_gpu.py
```

测试覆盖 stage 路由、条件缓存清理、CPU 激活暂存后的梯度一致性、配置与断点恢复。它们不能验证真实跨 GPU 传输或完整 5B 的显存峰值。另有一个可选测试，可在已经分配四卡的容器中运行，检查小型原生 Wan 的真实跨卡前向、反向和梯度裁剪：

```bash
RUN_FOUR_GPU_TEST=1 OMP_NUM_THREADS=2 \
  /home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B tests/test_four_gpu.py
```

验证结果记录在 [../verification/results.json](../verification/results.json)。本次未启动四卡训练；完整 5B 在 4×48GB 下的峰值显存及稳定性仍需实际任务确认。
