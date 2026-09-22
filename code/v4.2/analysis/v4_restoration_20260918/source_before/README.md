# physGen v4.2

2026-09-18训练默认更新为 **v4式联合 FM + 多层 STRUCT + P0 PRIOR，额外保留 WRITE**。每个 optimizer step 以75%概率让初始化器读取完整GT视频latent，同一步8条曝光共享开关；验证/推理不输入GT。取消梯度投影，四项加权后一次反传。保留v4.2现有读写架构和残差限幅；基座、teacher、已有数据缓存不变。

- [本次实现、损失公式与验证范围](analysis/20260918_v4_joint_实现说明.md)
- [新增单变量对照：辅助梯度冲突投影](analysis/20260918_梯度冲突投影对照实验.md)：使用 `train/train_stability_projected_1x96g.yaml`；正式入口仍不投影。

- [当前增量方法：ΔH监督与时序状态](analysis/20260917_v4.2_DeltaH监督与时序状态增量设计.md)
- [问题分析与修正设计](analysis/20260917_v4.2_监督信号与生成故障修正设计.md)
- [当前实现与验证范围](analysis/20260916_v4.2_代码实现与审核.md)
- [正式训练参数、选择理由与启动说明](analysis/20260917_v4.2_正式训练参数与启动说明.md)
- [推理入口说明](inference/README.md)

## 新建训练任务

```bash
cd /home/liuzhirui/Project/physGen/code/v4.2
det experiment create train/train_stability_1x96g.yaml .
```

默认 `configs/stability_v4_joint.yaml`，单卡96GB，1200次更新，每次8条曝光。每条独立以1/2概率选择I2V/T2V。数据仍是原缓存前2400条、2279 train/121 validation、原caption和fps、连续中心窗、最多121帧、原512尺寸桶。`PREPARE=0`不重建数据缓存；训练加载Wan前仅编码一次空文本，用于v4式10%文本dropout，不把原生负提示词冒充空文本。正式训练不加载VAE decoder。

STRUCT/PRIOR/WRITE权重分别为0.1/0.02/0.05，均200步预热；STRUCT和WRITE使用v4式线性噪声权重，PRIOR不乘噪声权重。STRUCT使用实际P5/P15的原始MSE均值，PRIOR为空间池化后的逐时间槽原始MSE，均不除teacher RMS。STRUCT沿实际计算图训练initializer/core及上游writer；WRITE仍仅训练writer/gate。全局grad_clip=1，学习率、1200步预算与读写限幅暂保留v4.2设置。

正式recipe为 `fm_multistruct_prior_write_gt75_joint_v1`。旧 `configs/stability.yaml` 保留step1200原配方，旧checkpoint按保存配置加载；`stability_trajectory_control.yaml`与`stability_fm_struct_control.yaml`仍为历史对照。

step0记录固定基线，每100步保存checkpoint并在固定留出面板比较raw/EMA/base。随机训练loss、固定FM改善与自由生成物理质量分别评价。

**新配方必须新建run，不设置旧step0100/step1200的RESUME。** initializer增加GT投影参数，监督与梯度路由改变；代码拒绝跨配方恢复。新配方自己的checkpoint可通过RESUME恢复。

旧 `quick16` 和 `batch1` 使用 `configs/stability_legacy_step0100.yaml`，仍是历史TEMP配方，不是本次正式训练入口或新配方质量证明。原checkpoint日志修复记录见 [历史恢复说明](CHECKPOINT_RECOVERY.md)。

## 推理和配对比较

```bash
det experiment create inference/infer_stability_1x96g.yaml .
# 或 inference/infer_stability_1xada48g.yaml
```

blk-96g、Ada 48GB YAML 和对应 shell 默认 checkpoint 均为 `v4_joint_20260918T034836Z_16183b/step0200`。加载保存配置和 EMA，新版初始化器推理不输入 GT 视频。默认 `VARIANT=full`，生成与 v4 相同的41条测试条件。`VARIANT=full/base/half/writer5_only/writer15_only` 支持完整模型、基座和写回干预。新结果后缀为 `_r3_<variant>`，自动导出ΔH曲线、空间图与摘要，不覆盖原视频。

UniPC/50步/shift5/CFG5/seed42保持。`SUITE=single MODE=t2v PROMPT='...'` 可生成单条；I2V还需IMAGE。具体提交配置见推理说明。

新版正式训练已产出上述 step0200，EMA严格加载及无GT初始化检查见[检查记录](analysis/20260918_step0200_48g_inference_check.json)。旧step1200检查保留在[历史记录](analysis/20260918_step1200_inference_audit.md)。完整视频质量收益尚待新权重生成验证。失败依据、监督公式、预期作用与实际验证范围均见设计及实现文档。
