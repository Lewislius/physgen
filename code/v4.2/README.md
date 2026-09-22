# V4.2：V4 原生 CORRECTOR，121 帧，JEPA32，仅 I2V

正式实现位于 `physgen_v4/`，直接从 V4 原生实现恢复。三个独立 CORRECTOR 放在 Wan 第 5、15、25 层，注意力和状态更新宽度为 1664、26 heads，使用全时空注意力；写回作用于 Wan 的 3072 维 H。可训练参数为 **429,284,425**。原 V4 P0 初始化器保留（其内部宽度为 512，输出 1664），不存在 CORRECTOR 的 1664→512 瓶颈。未启用的 B 模块不再分配。

| 版本 | Determined 配置 | 模型/损失配置 | 有效损失 |
|---|---|---|---|
| 有 PRIOR | `train/train_stability_1x96g.yaml` | `configs/stability_v4_fullwidth3.yaml` | FM + STRUCT + PRIOR + WRITE |
| 无 PRIOR | `train/train_stability_no_prior_1x96g.yaml` | `configs/stability_v4_fullwidth3_no_prior.yaml` | FM + STRUCT + WRITE |

两个版本的结构、数据、随机种子、学习率和训练步数相同。无 PRIOR 版将 `prior_weight` 设为 0，不计算该监督项；P0 初始化器仍通过 FM/STRUCT 训练，不删除、不冻结。代码禁止两版之间误用 RESUME；明确需要从兼容权重开始一个新训练时可用 INIT_FROM。

共同设置：最多 121 帧（源视频不足时保留原连续 4k+1 窗口），JEPA 32 帧采样、16 个 tubelet 时间位置、每个位置 576×1664 特征；仅 I2V。复用 `cache/wisa2400_nativefps_f121_a147456_fp32_jepa32`，2279 条训练/121 条验证，保留源 fps 和时间戳。JEPA 相邻帧对的采样规则与该缓存一致，训练和推理共用时间坐标。第一张图使用缓存中独立编码的 first latent，不读取旧的 JEPA 图像 anchor。

训练优化参数沿用 V4 A1，总步数按用户要求设为800：全局 batch=8，单卡累积 8 次，AdamW lr=1e-4、weight_decay=0.01，100 步学习率 warmup，800 步 cosine 日程，每100步保存 checkpoint，最低 lr 比例 0.1，global grad_clip=1。10% 空文本 dropout 始终保留第一帧，因此仍是 I2V；75% 的更新允许 P0 使用训练视频 GT 条件，推理永不使用 GT。采用原始训练权重，不再使用旧版 EMA。

设 `w=min(step/200,1)`，`q=clamp((1-sigma)/0.2,0,1)`：

```text
有 PRIOR：L = L_FM + 0.1*w*q*L_STRUCT + 0.02*w*L_PRIOR + 1.0*w*q*L_WRITE
无 PRIOR：L = L_FM + 0.1*w*q*L_STRUCT                     + 1.0*w*q*L_WRITE
```

FM、STRUCT、PRIOR 定义及其原有梯度路径与 V4 一致。WRITE 比较真实 V4 写回后的 3072 维隐藏特征与配对参考特征，三个位置取平均原始 MSE。参考来自同一视频/噪声/文本/sigma 的正常 FM 前向，另一个轻微扰动 latent 的无梯度前向提供待修复 H/P；参考、H、P 均停止梯度，只训练写回路径使用的参数。WRITE 不使用预算归一化、目标裁剪或额外降维，也不应被解释为物理正确性的标签。完整121帧检查中原生 WRITE MSE 约0.0042，故将过弱的0.01系数改为1.0；取消无扰动配对，每个样本均有扰动写回监督。该系数仍不是已被视频质量验证的最优值。

四项/三项直接联合反传，不进行梯度投影、辅助梯度占比限制、写回 RMS 限幅或输出速度残差压缩。V4 原有的状态门控、写回门控和随 sigma 变化的幅度系数保留。每 25 步选择一个实际扰动的 microbatch（若整步均为干净配对则记录最后一个） 记录各加权损失的梯度范数、两两夹角及 FM 与联合梯度的内积；这些仅用于观察，不改变梯度，也不代表整个累积更新的方向。

在已分配的 96GB 单卡环境中运行：

```bash
bash train/train_stability_1x96g.sh
bash train/train_stability_no_prior_1x96g.sh
```

CPU 配置检查：

```bash
CHECK_CONFIG_ONLY=1 bash train/train_stability_1x96g.sh
CHECK_CONFIG_ONLY=1 bash train/train_stability_no_prior_1x96g.sh
```

按原使用方式提交对应的 Determined YAML 即可。训练每 100 步保存和验证；运行名称带时间戳，已有 checkpoint 不覆盖。新结构必须重新训练或显式加载兼容的原生权重，旧版 `adapter.pt` / `ema.pt` 不能直接作为新版 checkpoint。

推理默认 Euler50、shift5、CFG5、空负提示词、seed42、121 帧、24fps，正负分支都运行完整 CORRECTOR，P 在每步重新初始化。配置文件读取的 `latest` 是有 PRIOR 版；无 PRIOR 版用 `CHECKPOINT=latest_no_prior`。两者各自维护 latest 指针。详见 `inference/README.md`。

旧版活动代码已删除，源码归档位于 `analysis/v4_restoration_20260918/retired_v42_source.tar.gz`，文件清单和哈希在同目录。历史数据、日志、权重和生成视频保留用于溯源。问题分析见 `analysis/v4_restoration_20260918/diagnosis_and_changes.md`。单元检查只验证结构/数值/梯度路径，不证明新训练后的完整视频已达到 V4 画质。
