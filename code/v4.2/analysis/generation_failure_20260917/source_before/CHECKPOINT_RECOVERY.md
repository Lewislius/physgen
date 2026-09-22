# 2026-09-17 checkpoint 日志故障与续训

原因：`save_checkpoint(...)` 返回之后，`log(events_file, path=checkpoint_path)`
把位置参数和事件字段同时传给了旧的 `log(path, ...)`，抛出 `TypeError`。
异常发生在写事件日志时，训练循环因此退出，但此次 checkpoint 已保存完毕。

修复覆盖 `physgen_v42.runtime.log`、`static_lora.runtime.log` 和
`four_gpu.runtime.log`。日志目的文件现在使用独立的位置参数 `log_path`，
事件的 `path` 字段可以正常保存。v4.2 BS8、batch1、LoRA 单卡和四卡训练均覆盖。

原始 checkpoint 保留在：

```text
/home/liuzhirui/Project/physGen/code/v4.2/checkpoints/stability_20260917T113347Z_5d0494/step0100
```

四个文件合计 1,309,180,650 字节，逐个完整 SHA-256 校验通过。
用当前配置、数据、资产及统计信息，已在 CPU 实际加载 65,045,248 个模型参数、
300 组 AdamW 参数状态、EMA 和训练队列。已完成步数与 EMA 更新次数均为 100，
优化器各参数的更新次数也均为 100；下一步是 101，剩余 1100 步。
原文件未被修改。

随后还用此次真实权重在隔离临时目录执行了约 1.3GB 的 checkpoint 保存、
checkpoint 事件日志写入和重新加载，模型、EMA 与优化器状态逐项完全一致。
CPU 环境中的这次文件往返保留了原始 CUDA RNG 字节；未执行 CUDA 前向。
临时副本已清理。完整验证记录见 [analysis/checkpoint_fix_20260917.json](analysis/checkpoint_fix_20260917.json)。

在集群提交现成的恢复配置：

```bash
cd /home/liuzhirui/Project/physGen/code/v4.2
det experiment create train/train_stability_resume_step0100_1x96g.yaml .
```

此配置设置 `PREPARE=0` 和上述 `RESUME`，恢复模型、优化器、EMA、随机数状态、
学习率进度与数据队列，继续原 run 的第 101–1200 步，每 100 步保存。
这份配置只适合恢复此次停在 100 步的 run；一旦后续保存点已经生成，
再次恢复应使用最后一个具有有效 `complete.json` 的 checkpoint，不能反复恢复 100 步覆盖后续保存点。

由于原实现严格校验源码 SHA-256，本次增加
`physgen_v42/checkpoint_compatibility.json`，明确列举此次修复前后的代码指纹。
仅这些已审核的旧→新组合可兼容。其他源码修改、配置、数据、模型资产及统计信息
仍须匹配；checkpoint 内的训练状态身份仍必须与完成标记完全一致。
模型、损失、优化器和训练步数等数值规则未改动。

新增回归测试：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B \
  tests/test_checkpoint_training_loops.py
```

此测试使用 CPU 小模型和生产训练循环，实际执行 AdamW、EMA、队列、JSONL、
checkpoint 读写及恢复，不替换这些保存/恢复操作。四种循环分别连续训练至 1200 步，
验证 12 次保存、最终索引和保存后继续执行；再从模拟旧源码身份的 step0100
恢复剩余 1100 步，要求最终权重、优化器、EMA、随机状态、队列及学习率状态逐项完全一致。
测试替换了大型 Wan/VAE、真实视频输入与 CUDA 状态查询，不是 96GB/四卡 GPU 实训。

本次共运行 64 项测试：63 项通过，1 项实际四卡 CUDA 测试因本机没有可用 GPU
跳过。新增 6 项回归测试全部通过，v4.2 原有 24 项、batch1 原有 10 项、
LoRA 原有 23 项 CPU 测试也全部通过。

已启动的 Python 进程持有内存中的旧函数，修改磁盘代码不会自动更新它们。
新启动和恢复的任务会使用修复；旧任务若在相同位置退出，先确认最后的
`complete.json` 及文件校验，再从该步恢复。

14:08 的续训启动曾被 `Checkpoint architecture/data/assets/code differ` 拦截。
逐项检查确认模型版本、数据 manifest、统计文件和模型资产均一致，差异只有代码：
第一次修复后，`inference/infer.py`、`inference/report.py` 和 JSON 配置读取逻辑
发生了更新，整体代码 SHA-256 改变，旧兼容表尚未登记该版本。
已审核这些差异并补充精确兼容关系，同时保留原有关系和第一次修复版本的续训支持。
审核通过在内存中还原这三处改动，精确复现了此前四种版本的已测试代码指纹；
训练数值逻辑没有改变，当前 YAML 与 checkpoint JSON 解析出的配置完全相同。
本次实际 step0100 恢复检查记录见
[analysis/checkpoint_resume_identity_20260917.json](analysis/checkpoint_resume_identity_20260917.json)。
