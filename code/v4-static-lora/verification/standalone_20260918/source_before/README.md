Wan2.2-TI2V-5B 的纯 LoRA、仅 FM 对比实验。所有新增代码、checkpoint、日志和推理结果位于本目录；原 `v4.2` 及基模文件保持不变。训练及推理直接走原生 Wan 前向，只给注意力线性层增加标准 `W(x) + (alpha/r)·B(A(x))`。

新增的 **4×48GB 模型并行版**见 [four_gpu/README.md](four_gpu/README.md)，训练入口为 `train/train_4x48g.yaml`，对应推理入口为 `inference/infer_4x48g.yaml`。下文的 `1×96GB` 入口继续保留。

2026-09-17 的 checkpoint 日志参数冲突已同时修复单卡、四卡入口；旧 checkpoint 兼容和真实训练循环的 CPU 保存/续训测试见 [CHECKPOINT_RECOVERY.md](../v4.2/CHECKPOINT_RECOVERY.md)。

训练提交：

```bash
cd /home/liuzhirui/Project/physGen/code/v4-static-lora
det experiment create train/train_1x96g.yaml .
```

使用指定 step0800 EMA 提交 41 条推理测试：

```bash
cd /home/liuzhirui/Project/physGen/code/v4-static-lora
det experiment create inference/infer_1x96g.yaml .
```

两个任务均使用 `blk-96g` 单卡、原 `moviestory` 环境、相同镜像与 HOME/NAS 挂载。单卡推理 YAML 和 shell 默认读取 `checkpoints/lora_20260917T140727Z_dae2a0/step0800` 的保存配置及 EMA，`SUITE=demo`，共 41 条。详情和 CPU 核验见 [inference/README.md](inference/README.md)。

| 项目 | 本版设置及与正式 v4.2 的关系 |
|---|---|
| 基模 | 同一 Wan2.2-TI2V-5B，保持原权重精度及 FP32 数值计算部分 |
| 训练数据 | 复用 `v4.2/cache/wisa2400_nativefps_f121_a147456_fp32_jepa32`；2279 条训练、121 条验证 |
| 视频与条件 | 原 native FPS、5–121 帧、原尺寸/面积限制、相同 VAE/T5 缓存；I2V 固定已知首帧 latent，首帧 token 的 timestep=0 |
| batch | 有效 BSZ=8，microbatch=1，累积 8 次后更新一次 |
| 数据顺序 | 复用正式版的采样队列与种子；每步 6 条 I2V、2 条 T2V；1200 步共 9600 条次 |
| 噪声 | 每步 7 条 `sigma ~ U(0.02,0.999)`、1 条 `sigma ~ U(0.05,0.15)`，保留正式版的采样配比 |
| 损失 | 仅条件分支 FM：八条 `MSE(v_pred, noise−GT)` 取平均；I2V 排除已知首帧 |
| 训练步数 | 1200；从第 1 步到第 1200 步均使用相同标准 FM 目标 |
| 保存 | 每 **100** 步，step0100、0200…1200；与创建时的现行正式版配置一致 |
| 优化器 | AdamW，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01，grad_clip=1.0 |
| 学习率 | 单组 LoRA 峰值 **5e-5**，取原 core 峰值；沿用 50 步预热及 1200 步余弦日程，末值为峰值的 0.1 |
| EMA / 验证 | EMA decay=0.995，每次更新后更新 EMA；每 200 步用正式版相同 health IDs/模式/噪声强度做 FM 验证 |
| LoRA | 默认 rank=32、alpha=32、dropout=0；30 层 self/cross attention 的 q/k/v/o，共 240 个线性层、47,185,920 个参数 |
| 数值 / 显存 | 原基模冻结；LoRA 参数及 Adam 状态为 FP32，前向使用 BF16 autocast；所有 transformer block 启用激活重计算 |
| 推理 | 单卡默认 reference I2V + P01–P20 I2V/T2V，共 41 条；UniPC 50 步、shift=5、CFG=5、seed=42；正/负条件分支均使用带 LoRA 的 Wan。`SUITE=final` 可运行原 81 条（含训练/验证诊断） |

仅 LoRA 的边界：不创建原项目的 core、writer、TextInit 或状态 P；无 JEPA/anchor 条件、STRUCT、TEMP、结构梯度限幅、输出残差限幅及第 801 步 repair。训练只读取每条视频的 VAE/T5 张量，不加载 VAE/T5/JEPA 编码模型；推理准备新条件时才使用原 T5/VAE，导出视频时使用原 VAE。

保留一个低噪声队列只是为了对齐正式版的 FM 噪声配比；这条样本同样只计算 FM。基础模型文件清单及缓存 manifest 中可能保留 teacher 元数据，这些只用于确认复用了同一份数据，不参与 LoRA 的前向或损失。

LoRA 的学习率组无法与原五组自定义模块一一对应，因此显式采用原 core 的 5e-5。rank/alpha 属于新增 LoRA 超参数，见 [configs/train.yaml](configs/train.yaml)。本版对齐数据、batch 和训练日程，**没有声称训练参数量或每步计算量与 v4.2 完全相等**；标准 FM-only 与原三损失/repair 的目标差异也应在实验表中注明。相同随机种子保证各自可复现，但两种架构初始化消耗的 RNG 不同，不保证逐条噪声张量与旧训练相同。

创建时的正式版配置保存在 [configs/v42_reference.json](configs/v42_reference.json)，解析后的配置随每个 checkpoint 保存，避免后续修改正式版配置悄悄改变这个实验。现行正式版为 `save_every=100`，本版据此设置。

默认 `PREPARE=1` 检查并复用完整缓存，不重建、不复制、不覆盖原数据。已知完整时可设置 `PREPARE=0`；训练启动仍会核对数据身份。只检查配置、数据和预期 LoRA 参数数，不加载 5B 模型、不需要 GPU：

```bash
cd /home/liuzhirui/Project/physGen/code/v4-static-lora
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B train/train.py --check-config
```

输出位置：

- `checkpoints/<run>/step0100` 至 `step1200`：`lora.pt`、`ema.pt`、`training.pt`、`config.json` 及完整性标记；不保存整份 5B 权重。
- `train/train_log/<run>/`：配置、trainable 参数清单、每条数据的 ID/模式/sigma、FM 损失、梯度范数、学习率、显存、验证结果和曝光计数。
- `inference_outputs/lora_20260917T140727Z_dae2a0_step0800_ema_demo_seed42/`：单卡默认输出；每案例 MP4、latent、抽帧图片、采样轨迹和元数据，以及进度 `summary.json`、最终 `report.json`。切换 CHECKPOINT/SUITE 时默认目录自动变化，亦可指定 OUTPUT。

默认 rank32 每个完整恢复 checkpoint 约 900 MiB，12 次保存约 10.6 GiB，另加日志和视频。EMA 平均的是 LoRA 的 A/B 参数；推理加载原 Wan 后再加载 EMA LoRA。

恢复训练时，将示例路径替换为实际 checkpoint：

```bash
det experiment create train/train_1x96g.yaml . \
  --config 'environment.environment_variables=["RESUME=/home/liuzhirui/Project/physGen/code/v4-static-lora/checkpoints/实际run目录/step0800","PREPARE=1","OMP_NUM_THREADS=4"]'
```

恢复会校验 LoRA 结构、配置、源码、基模资产与数据身份，并恢复 optimizer、学习率日程、EMA、随机状态和采样队列。只能恢复本实验 checkpoint，不能把 v4.2 的 adapter checkpoint 当作 LoRA。训练恢复仍要求源码一致；本次 step0800 的兼容记录仅用于推理，不放宽恢复训练校验。

可以指定结构与源码兼容、已完整保存的百步 checkpoint 做推理；默认读取该 checkpoint 的 config.json，输出按 run/step/suite 命名。集群提交需修改 YAML 或用 `--config` 覆盖环境变量：

```bash
det experiment create inference/infer_1x96g.yaml . \
  --config 'environment.environment_variables=["CHECKPOINT=/home/liuzhirui/Project/physGen/code/v4-static-lora/checkpoints/实际run目录/step0400","OUTPUT=/home/liuzhirui/Project/physGen/code/v4-static-lora/inference_outputs/step0400_ema_seed42","SUITE=final","OMP_NUM_THREADS=4"]'
```

单个新提示词的推理同样支持 `SUITE=single`、`MODE=t2v/i2v`、`PROMPT`、`IMAGE`（I2V 必填）、`DURATION` 和 `PORTRAIT`。例如 T2V：

```bash
det experiment create inference/infer_1x96g.yaml . \
  --config 'environment.environment_variables=["SUITE=single","MODE=t2v","PROMPT=A red ball rolls across a wooden table.","OUTPUT=/home/liuzhirui/Project/physGen/code/v4-static-lora/inference_outputs/single_ball","OMP_NUM_THREADS=4"]'
```

验证命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  /home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B tests/test_static_lora.py
PARSE_ONLY=1 bash train/train_1x96g.sh
PARSE_ONLY=1 bash inference/infer_1x96g.sh
```

验证结果见 [verification/results.json](verification/results.json)。CPU 数值测试使用真实 Wan 类的缩小模型和 CPU SDPA；完整尺寸参数检查使用 meta device，不加载 5B 权重。推理流程测试使用实际 UniPC 和视频导出，预测器/VAE 为小型替身。完整 5B 的 GPU 训练、显存峰值和生成质量需要在 Determined 实际任务中验证；本次未提交训练或推理任务。
