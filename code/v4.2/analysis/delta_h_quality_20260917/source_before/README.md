# physGen v4.2

2026-09-17正式训练默认已修正：**FM + 有上限的 STRUCT、sigmoid 写回 gate、恢复 TextInit 的结构梯度、统一普通曝光**。TEMP、特殊第八条、固定6/2和后期repair不再启用。基座/teacher路径、数据与预处理保持原值。代码已验证训练通路，尚未通过新模型的完整视频质量验收。

- [问题分析与修正设计](analysis/20260917_v4.2_监督信号与生成故障修正设计.md)
- [当前实现与验证范围](analysis/20260916_v4.2_代码实现与审核.md)
- [推理入口说明](inference/README.md)

## 新建训练任务

```bash
cd /home/liuzhirui/Project/physGen/code/v4.2
det experiment create train/train_stability_1x96g.yaml .
```

默认 `configs/stability.yaml`，单卡96GB，1200次更新，每次8条普通曝光。每条独立以1/2概率选择I2V/T2V。数据仍是2400条、2279 train/121 validation、原caption和fps、连续中心窗、最多121帧、原512尺寸桶。PREPARE=1复用已有有效缓存，正式训练不加载VAE decoder。

STRUCT系数为 `0.1 * min(step/200,1) * clip((1-sigma)/0.2,0,1)`；保留grad_clip=1及实际残差上限。默认逐组移除与FM相反的辅助梯度分量，可用 `train.auxiliary_gradient_policy: none` 对照。`loss.struct: 0` 是同架构FM-only对照，不新增损失。

step0记录固定基线，每100步保存checkpoint并在固定留出面板比较raw/EMA/base。随机训练loss、固定FM改善与自由生成物理质量分别评价。

**新配方必须新建run，不设置旧step0100的RESUME。** gate参数、辅助梯度和队列规则都已改变，代码会明确拒绝跨配方恢复。新配方自己的checkpoint可通过RESUME正常恢复。

旧 `quick16` 和 `batch1` 使用 `configs/stability_legacy_step0100.yaml`，仍是历史TEMP配方，不是本次正式训练入口或新配方质量证明。原checkpoint日志修复记录见 [历史恢复说明](CHECKPOINT_RECOVERY.md)。

## 推理和配对比较

```bash
det experiment create inference/infer_stability_1x96g.yaml .
# 或 inference/infer_stability_1xada48g.yaml
```

默认checkpoint仍为原step0100；生成修正训练的结果时必须显式设置 `CHECKPOINT` 为新run，未擅自替换默认模型路径或权重。`VARIANT=full` 为完整模型，`VARIANT=base` 为同条件关闭全部adapter的对照。新结果后缀为 `_r2_full` / `_r2_base`，不覆盖或复用原异常视频。

UniPC/50步/shift5/CFG5/seed42保持。`SUITE=single MODE=t2v PROMPT='...'` 可生成单条；I2V还需IMAGE。具体提交配置见推理说明。

不能把内部STRUCT更小、gate更大或单步FM略低当成物理效果提升。已发现旧TextInit监督缺口、全模型梯度比例掩盖core失衡，以及基座对照本身也有色块；证据、未确定原因和后续验收条件均写入设计文档。
