# v4.2 推理与同条件基座对照

```bash
cd /home/liuzhirui/Project/physGen/code/v4.2
det experiment create inference/infer_stability_1x96g.yaml .
# 48GB: inference/infer_stability_1xada48g.yaml
```

默认仍读取 `stability_20260917T113347Z_5d0494/step0100/config.json` 与 `ema.pt`，共41条：reference I2V + P01–P20 I2V/T2V。原checkpoint和base模型路径未改。要看新训练的效果，设置CHECKPOINT为新run；更新代码不会让旧权重获得新训练结果。

## 模式与输出

- `VARIANT=full`：完整校正。P0每个solver调用重置；旧配置无gate，新配置带gate，严格按保存的模型配置加载。
- `VARIANT=base`：绕过全部adapter前向，保持相同文本、首图、seed、solver与画幅；无需JEPA首图anchor。用于实际视频配对比较。
- 默认保持UniPC 50步、shift5、CFG5、seed42、最多121帧，T2V 512×288。
- 新输出为原命名后加 `_r2_full` / `_r2_base`；不同variant不会混用旧已完成视频。OUTPUT可显式指定新目录。

例如在同一GPU环境中用新checkpoint比较单提示词：

```bash
CHECKPOINT=/home/liuzhirui/Project/physGen/code/v4.2/checkpoints/新run/step0100 \
SUITE=single MODE=t2v PROMPT='A ball rolls down a ramp.' VARIANT=full \
bash inference/infer_stability_1x96g.sh
```

保持其他变量相同，改VARIANT=base再生成对照。集群可在YAML的environment_variables中设置相同变量。

## 条件、启动与诊断

T5/首图VAE/JEPA条件缓存继续复用，只有缺失才加载对应编码器。解释器直接启动，不做Conda激活；不会为推理读取training.pt。sample独立进程明确使用保存的paths.wan_code，日志打印实际Wan源文件，模型资产路径没有替换。

启动现在分别报告Python开始、NumPy/PIL完成、PyTorch完成、本地模块完成，以及checkpoint/Wan/VAE加载。原“Loading Python/PyTorch”尚未开始加载checkpoint；十分钟停顿需要远端进程栈或分段日志才能确定，不能归因于checkpoint大小。

summary.json记录任务进度；每条video.mp4、final_latent.pt和trajectory.jsonl保留。EVALUATE默认0；设1时生成JEPA诊断和人工视频检查表。base模式没有P状态，不伪造P相似度。JEPA相似度和单步FM不等于物理正确率。

当前修正的设计与证据见 [分析文档](../analysis/20260917_v4.2_监督信号与生成故障修正设计.md)。尚未完成新配方重训后的画质验证。
