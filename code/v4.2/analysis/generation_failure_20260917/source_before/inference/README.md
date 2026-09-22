# v4.2 直接推理

```bash
cd /home/liuzhirui/Project/physGen/code/v4.2
det experiment create inference/infer_stability_1xada48g.yaml .
# 96GB入口：det experiment create inference/infer_stability_1x96g.yaml .
```

默认直接读取 `stability_20260917T113347Z_5d0494/step0100/config.json` 和 `ema.pt`，
生成reference I2V与P01–P20各I2V/T2V，共41条。保持UniPC 50步、shift5、CFG5、seed42、121帧。
I2V按首图确定尺寸，T2V默认512×288。

启动不执行预检查，不读取training.pt/adapter.pt，不校验训练日程、源码指纹、资产清单或文件SHA。
解释器直接启动，不做Conda激活。EMA使用meta构造后直接装载，避免先随机初始化6500万参数再覆盖。
保留参数名称/形状匹配，以及防止不同条件混用同一输出目录的轻量检查。

仅保留生成必需的条件编码：T5文本、VAE首图latent、JEPA首图anchor。
三者逐项保存到 `cache/inference_conditions/v1/`，同一输入跨checkpoint、跨48/96GB入口复用；
只有缺少对应缓存时才加载编码器。任务中途退出，已完成的编码也保留。
文件路径、大小和修改时间用于区分缓存，不读取大模型文件计算哈希。
I2V首图JEPA属于方法所需的P0初始化；T2V由TextInit和checkpoint中的训练均值初始化，不使用首图。

P0每个去噪步重置，独立core5/core15与有界writer照常执行。
预测仍为 `v_neg + 5*(v_base_pos-v_neg) + bounded(v_plus_pos-v_base_pos)`。
启动、模型加载、编码完成和每个采样步均立即打印进度。

Ada输出：
`inference_outputs/stability_20260917T113347Z_5d0494_step0100_ema_demo_1xada48g_seed42/`。
96GB目录不带 `_1xada48g`。`summary.json`记录进度，各案例的 `video.mp4` 为生成结果。
相同入口和输出目录会跳过已经完成的视频。

默认 `EVALUATE=0`，生成后直接结束。需要JEPA视频评估/图表时再设置 `EVALUATE=1`。
单提示词使用 `SUITE=single MODE=t2v PROMPT='...' OUTPUT=...`；I2V另设 `IMAGE`。
使用新代码须重新提交任务；已经启动的旧Python进程不会自动替换代码。
