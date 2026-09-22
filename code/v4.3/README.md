# V4.3：原生 3072 维 P/H 交互与 Flow-Oracle

基于 V4.2 的 Wan2.2-TI2V-5B、121 帧 I2V、JEPA32 缓存，重新实现 CORRECTOR、监督与训练流程。正式训练目标只有 **FM + STRUCT + Flow-Oracle**；没有 PRIOR，也没有旧的 warp/drift WRITE。V4.2 代码和权重不覆盖。

实际维度：

```text
首图 latent + 文本 + 时空位置
             ↓
       初始化 P₀：1664
             ↓ 可学习非线性映射：1664 → 3072
     工作状态 U₀：3072
             ↓
Wan block 5  → H₅：3072 ↔ U₀：3072 → U₅：3072
                        ↓                 ↓ 独立非线性映射 3072 → 1664
                ΔH₅：3072              JEPA STRUCT
                        ↓
              H₅ + ΔH₅ → Wan block 6 … 15
                             ↕ 与 U₅ 在 3072 维交互
                        第 15、25 层重复
```

U 在插入点之间一直是 3072 维，JEPA 解码头不把状态压回 1664 后再传递。每个位置有两个完整的双流联合注意力块，24 heads × 128 channels = 3072；P/H 分别有自己的归一化、QKV、FFN，通过一次联合 attention 双向交换信息。映射中间层为 3072，初始化器宽度为 1664。**无 512 维隐藏瓶颈，也没有遇到显存不足后自动缩窄的分支。** 位置编码函数中的 512 只是时间正弦编码的一部分，不是 H/P 通道压缩。

真实参数统计：初始化/升维映射 126,337,376；每个独立插入模块 783,138,048；总可训练参数 **2,475,751,520**。完整线性层尺寸见 [native_width_manifest.json](analysis/checks/native_width_manifest.json)。参数、梯度和 Adam 两个 FP32 moment 合计约 36.89 GiB，尚不包含 Wan、激活、缓存和临时工作区。96GB 上的真实训练显存需 GPU 验证；多卡版本目前只分片 Wan，CORRECTOR 仍复制，不能把多卡直接视为 CORRECTOR 显存分片。

新增 Oracle 的具体算法、适用边界以及全部设计差异见 [设计说明](analysis/design/implementation.md)。当前研究方案与实验结论见 [分析入口](analysis/README.md)。关键实现是 [corrector.py](physgen_v43/corrector.py)、[oracle.py](physgen_v43/oracle.py)、[backbone.py](physgen_v43/backbone.py) 和 [losses.py](physgen_v43/losses.py)。

## 训练目标与默认参数

令 `a=min((step+1)/200,1)`，`q=clamp((1-sigma)/0.2,0,1)`：

```text
L = L_FM + 0.01*a*q*L_STRUCT + 60.0*a*L_Oracle
```

两个辅助项按初始运行第 27 步的原始损失校准：预热结束后，STRUCT 和 Oracle 分别约为 FM 的 5.73% 和 5.66%，以各占约 5%～10% 为起点。预热期间仍乘以 `a`；固定权重不保证所有后续样本都在该范围内。新设置只在加载新配置的训练任务中生效。

- FM：Wan 当前约定 `x_sigma=(1-sigma)*z_GT+sigma*noise`，目标 `noise-z_GT`，不计已知首 latent。
- STRUCT：三处真实工作状态 U 经各自 `3072→1664` 解码后的 JEPA 特征 MSE，使用缓存的内容权重，三处平均。
- Oracle：用停止梯度的完整修正目标监督实际 ΔH；各层误差除以停止梯度的当前 H RMS。每个被选择的位置等权；拒绝的目标贡献零，不放大其他层权重。

Oracle 默认三个位置全部参与，每处两步。初始变量是现有 `delta_pred.detach()`；每步的目标函数为 FM 加 `0.001*mean((delta/H_RMS)^2)`，提议步长 RMS 为 H RMS 的 1%，最多尝试四个逐次减半的步长。必须同时满足 FM 严格改善与正则目标的 Armijo 下降条件才接受。内部正则属于求标签过程，**不是外层第四个训练损失**；该步长也不限制实际 writer 的输出幅度。

GT 只用于标准加噪输入和训练目标。P₀ 只读取首图、完整文本和位置，没有 V4.2 的完整 GT 视频初始化路径。训练与推理都在每次 Wan 调用时重置 P；正负 CFG 分支分别使用自己的文本初始化 P。无 Oracle 的推理流程见 [inference/infer.py](inference/infer.py)。

默认 AdamW 峰值学习率 **5e-5**，100 步 warmup、800 步总训练、cosine 衰减到约 5e-6，全局 batch 8；所有可训练模块使用同一学习率。首轮以 V4.2 的 1e-4 为参照，取其一半作为起点；后续实验结论见分析入口。不是根据参数量套缩放公式，也不是已经验证的最优值。数据、基础参数及明确例外的对照见 [训练对齐与学习率依据](analysis/design/training_alignment.md)。

## 检查与运行

指定 step100 checkpoint 的单卡 96GB、41 条测试入口为 [infer_flow_oracle_1x96g.sh](inference/infer_flow_oracle_1x96g.sh) 和 [对应 YAML](inference/infer_flow_oracle_1x96g.yaml)，默认 UniPC50、空负面提示词。测试范围与运行说明见 [推理 README](inference/README.md)。

固定运行环境：`/home/liuzhirui/miniconda3/envs/moviestory/bin/python`。

```bash
cd /home/liuzhirui/Project/physGen/code/v4.3

# CPU：原生参数尺寸、缓存/配置、数值与梯度路径
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -B tools/verify_cpu.py
CHECK_CONFIG_ONLY=1 bash train/train_1x96g.sh

# 在已分配的 CUDA GPU 中：真实 Wan 的局部 oracle、suffix 重放和完整梯度
# 默认缓存第 0 条、sigma=0.6；报告写明实际帧数与尺寸；不更新参数。
bash tools/probe_1x96g.sh

# 可指定多个 sigma 和训练后的 V4.3 checkpoint
CHECKPOINT=/path/to/v43/checkpoint bash tools/probe_1x96g.sh --sigmas 0.1 0.35 0.6 0.85 0.98

# 真实优化器短检查：保留原 LR 日程，不保存训练 checkpoint
CHECK_STEPS=1 bash train/train_1x96g.sh

# 正式训练（800 步，global batch 8，峰值 lr 5e-5，checkpoint 每 100 步）
bash train/train_1x96g.sh

# 同结构对照：只关闭 Oracle，保留 FM + STRUCT
CONFIG="$PWD/configs/fm_struct_control.yaml" bash train/train_1x96g.sh

# 训练后 I2V 示例；推理没有 GT 视频、JEPA 或 Oracle 优化
SUITE=single CHECKPOINT=latest IMAGE=/path/to/first.jpg \
PROMPT='A ball rolls down the ramp.' OUTPUT="$PWD/inference_outputs/example.mp4" \
bash inference/run_inference.sh
```

Determined 配置分别为 [正式训练](train/train_1x96g.yaml) 和 [单样本验证](tools/probe_1x96g.yaml)。代码已准备，但本次没有自动提交远端任务或启动正式训练。

`RESUME` 只接受同结构、同监督定义的 V4.3；`INIT_FROM` 也只加载兼容 V4.3 权重并新建优化器。V4.2 checkpoint 不能直接当作 V4.3 断点。Oracle/对照各自保存 latest 指针，推理对照使用 `CHECKPOINT=latest_fm_struct`。

CPU 算法和尺寸检查报告见 [cpu_verification.json](analysis/checks/cpu_verification.json)。正式训练、两轮真实 Wan 修正原型及匹配生成的现有结论统一见 [分析与设计入口](analysis/README.md)；当前修正原型未达标，CPU 检查不代表生成质量通过。
