# v4.2 推理与同条件基座对照

```bash
cd /home/liuzhirui/Project/physGen/code/v4.2
det experiment create inference/infer_stability_1x96g.yaml .
# 48GB: inference/infer_stability_1xada48g.yaml
```

blk-96g、Ada 48GB YAML 和对应 shell 的默认 checkpoint 均为 `v4_joint_20260918T034836Z_16183b/step0200`。读取所选 checkpoint 的 `config.json` 与 `ema.pt`，默认 `VARIANT=full`。新版按保存配置加载 GT-capable initializer，推理仅使用文本/首图，不输入 GT 视频；两处 Corrector 正常读写。共41条：reference I2V + P01–P20 I2V/T2V（21条I2V、20条T2V），图片和提示词与 v4 相同。切换其他权重时设置 `CHECKPOINT`；通过集群提交时须修改 YAML 中的环境变量，当前 shell 变量不会覆盖 YAML。

保持 v4.2 的采样流程：121帧、24fps、5秒时间窗、UniPC 50步、shift5、CFG5、seed42。reference 按首图生成384×384；P01–P20为512×288。v4 的旧配置为 Euler、101帧，本次沿用其41条测试条件，采样参数仍按 v4.2 设置。

## 模式与输出

- `VARIANT=full`：完整校正。P0每个solver调用重置；旧配置无gate，新配置带gate，严格按保存的模型配置加载。
- `VARIANT=base`：绕过全部adapter前向，保持相同文本、首图、seed、solver与画幅；无需JEPA首图anchor。用于实际视频配对比较。
- `VARIANT=half`：两处ΔH在限幅后乘0.5；`writer5_only`/`writer15_only`只保留对应writer，P状态仍正常计算。
- 新checkpoint按保存配置构造时序initializer，P0缓存一次，每个solver调用reset；WRITE训练参考支路不参与推理。
- 默认保持UniPC 50步、shift5、CFG5、seed42、最多121帧，T2V 512×288。
- 新输出为原命名后加 `_r3_<variant>`；不同variant不会混用旧已完成视频。OUTPUT可显式指定新目录。

例如在同一GPU环境中用新checkpoint比较单提示词：

```bash
CHECKPOINT=/home/liuzhirui/Project/physGen/code/v4.2/checkpoints/v4_joint_20260918T034836Z_16183b/step0200 \
SUITE=single MODE=t2v PROMPT='A ball rolls down a ramp.' VARIANT=full \
bash inference/infer_stability_1x96g.sh
```

保持其他变量相同，改VARIANT=base再生成对照。集群可在YAML的environment_variables中设置相同变量。

## 条件、启动与诊断

T5/首图VAE/JEPA条件缓存继续复用，只有缺失才加载对应编码器。解释器直接启动，不做Conda激活；不会为推理读取training.pt。sample独立进程明确使用保存的paths.wan_code，日志打印实际Wan源文件，模型资产路径没有替换。

启动现在分别报告Python开始、NumPy/PIL完成、PyTorch完成、本地模块完成，以及checkpoint/Wan/VAE加载。原“Loading Python/PyTorch”尚未开始加载checkpoint；十分钟停顿需要远端进程栈或分段日志才能确定，不能归因于checkpoint大小。

summary.json记录任务进度；每条video.mp4、final_latent.pt和trajectory.jsonl保留。EVALUATE默认0；设1时生成JEPA诊断和人工视频检查表。base模式没有P状态，不伪造P相似度。JEPA相似度和单步FM不等于物理正确率。

每条视频自动输出 `delta_h_summary.json`、`delta_h_curves.png`，启用adapter时还有 `delta_h_maps.png`，无需EVALUATE或额外teacher。完整设计见[ΔH监督与时序状态](../analysis/20260917_v4.2_DeltaH监督与时序状态增量设计.md)。

同条件输出可生成配对审核表：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python inference/intervention_report.py \
  --outputs /实际/full输出目录 /实际/base输出目录 \
  --comparison /home/liuzhirui/Project/physGen/code/v4.2/analysis/实际配对目录
```

脚本核对提示词、首图、seed、画幅、采样设置和基座资产，不将不同条件混在一起。step1200 的权重适配与批量条件核验见[检查记录](../analysis/20260918_step1200_inference_audit.md)；完整生成画质仍需本批视频验证。
