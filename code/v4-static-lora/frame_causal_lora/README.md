本目录提供两个帧级因果注意力 LoRA 对比方法，训练入口共用 `train.py`，由各自的配置选择注意力规则。两版均在 Wan 的全部 30 层 self-attention 中使用该规则；训练、health 验证和本目录的推理入口保持一致。

| 方法 | 第 i 帧的每个 query token 可以关注的 key/value |
|---|---|
| `frame_causal_full` | 第 0…i 帧的全部空间 token |
| `frame_causal_window_k5` | 第 max(0, i−5)…i 帧的全部空间 token |

**两版的当前帧内部始终是全注意力，包括排在 query 后面的空间 token。** K=5 指前 5 个压缩后的 VAE latent 帧，不含当前帧，因此最多共看 6 帧。帧索引来自 Wan 的 `grid_sizes=(F,H,W)`，其 temporal patch size 为 1；不按原始 RGB 帧或展平后的 token 距离计算窗口。短视频或开头不足 5 个历史帧时使用已有历史。

实现先按原生绝对位置计算 3D RoPE，再逐帧取完整 Q 和允许范围内的完整 K/V，调用非因果矩形 FlashAttention；这在数学上实现了帧块级因果 mask，不会在当前帧内产生三角遮挡，也不构造整段视频的 N×N mask。文本 cross-attention 继续完整关注文本。

窗口约束作用于**每一层的直接注意力连接**。多层堆叠后，历史帧的表示可以间接携带更早的信息，因此不等价于整个网络只依赖最近 5 个原始 latent 帧。首帧同样遵守窗口，没有额外的永久首帧注意力连接。

LoRA rank32/alpha32、240 个 q/k/v/o 投影、47,185,920 个可训练参数、仅 FM 损失、数据划分、噪声采样、BS8、学习率、EMA、1200 步及每 100 步保存均沿用原实验。每帧分别调用注意力会增加 kernel 启动开销，实际训练速度和峰值显存需要在分配的 GPU 上测量。原 `static_lora/`、`four_gpu/`、训练/推理 Python 入口均未修改，原有 checkpoint 的源码指纹保持不变。

在项目根目录提交 Determined 任务，选择需要的硬件规格即可：

```bash
cd /home/liuzhirui/Project/physGen/code/v4-static-lora

# 4×48GB：单进程模型并行，30 层按 8/8/7/7 分配。
det experiment create train/train_frame_causal_full_4x48g.yaml .
det experiment create train/train_frame_causal_window_4x48g.yaml .

# 1×96GB：对应的单卡版本。
det experiment create train/train_frame_causal_full_1x96g.yaml .
det experiment create train/train_frame_causal_window_1x96g.yaml .
```

上述每个 YAML 均有同名 `.sh`；`train/` 中 YAML 是 Determined 资源/提交配置，`configs/` 中同名 YAML 是模型/训练配置。4×48GB 使用 `amp-48g`、4 slots、同一节点；1×96GB 使用 `blk-96g`、1 slot。沿用现有镜像和 HOME/NAS 挂载，四卡入口只启动一个 Python 进程。

调整 K 时，修改对应 `configs/train_frame_causal_window_1x96g.yaml` 或 `configs/train_frame_causal_window_4x48g.yaml` 中的 `attention.previous_frames: 5`。K 必须为非负整数，0 表示只看当前整帧。可复制配置后使用 `CONFIG=/绝对路径/配置.yaml`；生成的版本名、checkpoint 和日志路径会自动带实际的 K。若修改 K，同时更新提交 YAML 的 `name`/`description` 以及 `checkpoint_storage.storage_path` 中的 `k5`，让 Determined 页面与实际实验一致。已有训练的恢复必须使用原来的 K 和硬件规格。

checkpoint 路径自动包含方法名、硬件规格、UTC 时间戳和随机后缀，例如：

```text
checkpoints/frame_causal_full/4x48g/
  frame_causal_full_4x48g_20260922T120000Z_a1b2c3/step0100/

checkpoints/frame_causal_window_k5/4x48g/
  frame_causal_window_k5_4x48g_20260922T120000Z_d4e5f6/step0100/
```

每个 step 保存 `lora.pt`、`ema.pt`、`training.pt`、`config.json` 和 `complete.json`。`config.json` 保存 attention mode、K 和硬件规格，完成标记和训练状态记录相应架构版本。自定义 `RUN_NAME=trial-a` 只会追加标签，保留方法名与时间戳。日志位于 `train/train_log/<方法>/<规格>/<run>/`；各方法/规格有独立的 `latest_final.json`。

恢复示例，将路径换成实际 checkpoint：

```bash
det experiment create train/train_frame_causal_window_4x48g.yaml . \
  --config 'environment.environment_variables=["RESUME=/home/liuzhirui/Project/physGen/code/v4-static-lora/checkpoints/frame_causal_window_k5/4x48g/实际run/step0800","PREPARE=1","OMP_NUM_THREADS=4","PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]'
```

恢复保留原 run 名，并恢复 optimizer、EMA、RNG 和采样队列；不同注意力方法、K、硬件规格、源码或旧全注意力 checkpoint 会被拒绝混用。也可设置 `CONFIG=<checkpoint>/config.json` 使用 checkpoint 保存的配置。

无 GPU 的训练前置检查：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B frame_causal_lora/train.py \
  --config configs/train_frame_causal_window_4x48g.yaml --check-config
PARSE_ONLY=1 bash train/train_frame_causal_full_4x48g.sh
PARSE_ONLY=1 bash train/train_frame_causal_window_4x48g.sh
```

配套 Python 推理入口为 `frame_causal_lora/infer.py`，自动从保存的配置恢复相同的 mask、K 和硬件规格。以下命令在已分配相应 GPU 的环境中运行；`--phase check` 无需 GPU：

```bash
checkpoint=/绝对路径/实际run/step1200
output=/home/liuzhirui/Project/physGen/code/v4-static-lora/inference_outputs/实际run_demo
python_bin=/home/liuzhirui/miniconda3/envs/moviestory/bin/python
"$python_bin" -I -B frame_causal_lora/infer.py --phase check --checkpoint "$checkpoint" --output "$output" --suite demo
"$python_bin" -I -B frame_causal_lora/infer.py --phase prepare --checkpoint "$checkpoint" --output "$output" --suite demo
"$python_bin" -I -B frame_causal_lora/infer.py --phase sample --output "$output"
"$python_bin" -I -B frame_causal_lora/infer.py --phase report --output "$output"
```

CPU 数值/配置/恢复测试：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  /home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B tests/test_frame_causal_lora.py
```

测试包括独立稠密帧 mask 的前向与 Q/K/V 梯度对照、同帧最后 token 对第一个 token 的影响、未来帧/窗口外屏蔽、K=0/1/5 和短序列、原生缩小 Wan 的整网因果性、LoRA 梯度、激活重计算、四 stage 的 CPU 路由/暂存、checkpoint 版本隔离、续训及四套提交脚本。实际 CUDA FlashAttention 测试可在分配的单卡上设置 `RUN_FRAME_CAUSAL_CUDA_TEST=1` 运行；默认跳过。四卡 CPU 模拟不代表已验证真实跨卡训练显存。

本次检查记录见 [../verification/frame_causal_20260922/result.json](../verification/frame_causal_20260922/result.json)。未启动完整 5B GPU 训练，也未提交 Determined 任务。
