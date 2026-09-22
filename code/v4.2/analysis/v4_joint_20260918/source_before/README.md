# physGen v4.2

2026-09-17正式默认已升级：**条件时序初始化 + 直接监督ΔH的配对修复任务**，目标为FM、STRUCT、WRITE。FM保持原高斯加噪训练；辅助支路用轻度受扰输入学习恢复真实视频参考H。保留write gate和幅度上限，取消RGB TEMP、特殊第八条、固定6/2与旧801步solver repair。基座/teacher路径、数据和预处理保持原值。

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

默认 `configs/stability.yaml`，单卡96GB，1200次更新，每次8条普通曝光。每条独立以1/2概率选择I2V/T2V。数据仍是原缓存前2400条、2279 train/121 validation、原caption和fps、连续中心窗、最多121帧、原512尺寸桶。正式提交和shell默认 `PREPARE=0`，只读取已有缓存，不运行索引重建或VAE/T5/JEPA编码；缓存缺失或不匹配会报错，不自动重算。正式训练不加载VAE decoder。

STRUCT与WRITE系数上限分别为0.1/0.05，均使用200步预热和v4式线性噪声权重；保留grad_clip=1及实际残差上限。STRUCT训练initializer/core，WRITE直接训练writer/gate，FM训练完整校正模块。初始化使用首图/均值、文本、时间位置与时长，不读取GT未来。

正式recipe为 `fm_struct_write_repair_trajectory_v1`。`configs/stability_trajectory_control.yaml` 保留时序初始化但无WRITE；`configs/stability_fm_struct_control.yaml` 是前一轮静态初始化对照。对照配置中 `loss.struct: 0` 可用于FM-only，不要将正式方案只关STRUCT误称为FM-only。

step0记录固定基线，每100步保存checkpoint并在固定留出面板比较raw/EMA/base。随机训练loss、固定FM改善与自由生成物理质量分别评价。

**新配方必须新建run，不设置旧step0100的RESUME。** initializer结构、监督任务和梯度路由均已改变，代码会明确拒绝跨配方恢复。新配方自己的checkpoint可通过RESUME正常恢复。

旧 `quick16` 和 `batch1` 使用 `configs/stability_legacy_step0100.yaml`，仍是历史TEMP配方，不是本次正式训练入口或新配方质量证明。原checkpoint日志修复记录见 [历史恢复说明](CHECKPOINT_RECOVERY.md)。

## 推理和配对比较

```bash
det experiment create inference/infer_stability_1x96g.yaml .
# 或 inference/infer_stability_1xada48g.yaml
```

96GB、Ada 48GB YAML 和 shell 默认 checkpoint 均为 `stability_20260917T181001Z_a83129/step1200`，加载其保存配置和 EMA，默认 `VARIANT=full`，生成与 v4 相同的41条测试条件。`VARIANT=full/base/half/writer5_only/writer15_only` 支持完整模型、基座和写回干预。新结果后缀为 `_r3_<variant>`，自动导出ΔH曲线、空间图与摘要，不覆盖原视频。

UniPC/50步/shift5/CFG5/seed42保持。`SUITE=single MODE=t2v PROMPT='...'` 可生成单条；I2V还需IMAGE。具体提交配置见推理说明。

正式训练已产出上述 step1200；推理适配检查见[检查记录](analysis/20260918_step1200_inference_audit.md)，完整视频质量收益尚待新权重生成验证。失败依据、监督公式、预期作用与实际验证范围均见设计及实现文档。
