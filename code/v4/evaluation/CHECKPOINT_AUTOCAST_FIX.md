# 实验 61001：首步 CheckpointError 修复及正式 A1 入口

当前 P₀ 训练已升级为 GT／无 GT 条件混合，详见 [专项说明](GT_CONDITION_PRIOR.md)。本文保留实验 61001 发生时的暖起与 AMP 缓存故障记录；当时的检查结果不代表新模型已通过完整 GPU 训练。

## 1. 故障与定位

用户提供的 [experiment_61001_trial_61839_logs.txt](../logs/experiment_61001_trial_61839_logs.txt) 显示：单 rank 初始化、Wan 加载、真实缓存读取、暖起、正式前向和 loss 都完成，首次 `step=1, micro=0` 的 backward 报 `torch.utils.checkpoint.CheckpointError`。运行环境为 moviestory 中的 PyTorch 2.8.0+cu128，使用 BF16。首个不一致项为 saved `[512,512]`、recomputed `[4608,512]`，对应 512 维 ConditionPrior 的激活重计算图。

原训练循环在同一个 autocast 上下文里先执行 no_grad 暖起，再执行带梯度的正式视图和独立 P₀ 监督。AMP 缓存了暖起时转换后的权重，正式前向复用该缓存；离开上下文后再做 backward，checkpoint 重算时的保存张量元数据与原前向不一致。PyTorch 项目记录过相同的 no_grad／autocast／非重入 checkpoint 组合故障：[issue 141896](https://github.com/pytorch/pytorch/issues/141896)。本次归因还通过本项目的真实模块和修改前训练入口作了直接复现。

日志前面的 determined 包版本冲突后，任务仍然进入了训练；此次退出的直接异常来自 backward。单 rank 的 FSDP `NO_SHARD` 提示符合单卡执行方式。

## 2. 修复

共用训练入口 [train_native_p.py](../train/train_native_p.py) 的训练 autocast 改为：

```python
with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
    # 合法上一视图的 no_grad 暖起、正式前向、独立 P0 监督和 loss
    ...
```

PyTorch 2.8 的 checkpoint 会捕获 `cache_enabled` 并在重算时恢复，因此正式前向和 backward 重计算都使用关闭缓存的设置。六个 block 仍各自拥有完整 corrector，同一 block 在不同去噪 step 复用自己的参数；BF16、非重入 checkpoint 的默认一致性检查和先验监督仍启用；后续 GT 初始化改造将暖起默认设为 0，非零值仅供显式对照。权重转换次数可能增加，实际吞吐需在目标节点测量。

## 3. 三份 YAML 默认运行正式 A1

三个入口均执行本目录的 `train/run_training.sh`，最终进入同一个正式 `train/train_native_p.py`。生产训练加载真实 Wan TI2V-5B、真实缓存和六套完整独立 corrector。

| 提交文件 | 当前实际分配 | PHASE | TRAIN_CHECK_STEPS | 梯度累积 |
|---|---|---|---|---|
| [4x48g YAML](../train/train_wisa_native_p_4x48g.yaml) | ada-24g，2 卡 | A1 | 0 | 4 |
| [4x80g YAML](../train/train_wisa_native_p_4x80g.yaml) | amp-80g，4 卡 | A1 | 0 | 2 |
| [4x96g YAML](../train/train_wisa_native_p_4x96g.yaml) | blk-96g，1 卡 | A1 | 0 | 8 |

当前完整 A1 配置为 500 个优化器 step、已选 1000 条训练记录、microbatch=1、global batch=8。启用 FM、L_struct、L_prior，每次 A corrector 调用进行一轮内部迭代；A1 按阶段定义冻结 B，在线 L_out 在后续 A3 阶段启用。P₀ 和 A@5／10／15／20／25／30 共 837,957,970 个参数可训练。

`TRAIN_CHECK_STEPS=0` 使训练执行完整阶段，第 250／500 步正常验证并保存中间 checkpoint，结束保存 checkpoint-final。`PREPARE=0` 复用真实预编码数据。`diagnostics.trace_steps=1` 只限制详细诊断日志。上述 1000 条是现有配置选择的训练子集，不代表使用整个 WISA 数据集。

直接提交原 YAML 即使用这些设置，例如：

```bash
cd /home/liuzhirui/Project/physGen/code/v4
det experiment create train/train_wisa_native_p_4x96g.yaml .
```

48G／80G 使用各自 YAML。此次没有替用户提交新集群任务。

## 4. 实际验证

相关 **22 项 CPU 回归最终全部通过，无跳过项**，证据分开保存：

- [修改前复现记录](results/checkpoint_autocast_before.json)：在同一个 PyTorch 2.8 环境，用修改前的正式训练循环和真实 CellBlock 构建的小型 CPU 模型，再次触发相同类型的 CheckpointError。
- [AMP 专项记录](results/checkpoint_autocast_review.json)：8 项通过，包括默认 512 维／两层先验的所有参数与输入梯度对照，两次复用 CellBlock 的梯度对照，以及原生 1664／3072 通道、六个独立 A 的 BF16 暖起＋checkpoint＋先验监督反向。
- [正式入口与训练控制流记录](results/formal_a1_launchers_review.json)：19 项最终通过，其中 5 项与 AMP 记录重复。逐份执行真实 shell 启动器并用正式 Python 参数解析器检查 A1／完整步数／卡数／累积；CPU 模型替身驱动正式训练循环完成 500 次更新，确认第 250／500 步验证和保存调用、最终保存调用及对应日志。新增控制流用例首次把内部零起始 step 当成日志的一起始 step，已修正测试预期后重跑；保留初次结果。

CPU 替身及缩小网格仅存在于 evaluation 测试中。正式 YAML 的启动链不调用这些测试模块。六位置集成使用冻结 TinyWan 替身；默认宽度先验另以真实网络测试。以上验证确认本次复现错误的修复、梯度和控制流，尚未验证修复后真实 GPU 的完整 500 步、显存峰值、吞吐或生成效果。此前独立参数的 116 项历史回归没有覆盖此次触发故障的完整 AMP 组合。

复现修复后的 CPU 检查：

```bash
cd /home/liuzhirui/Project/physGen/code/v4
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -m unittest -v \
  evaluation.test_checkpoint_autocast \
  evaluation.test_training_runtime.ShortRunTests \
  evaluation.test_launchers \
  evaluation.test_condition_prior.PriorIntegrationTests.test_bf16_warm_checkpoint_trains_prior_and_all_six_independent_a_blocks
```

修改前文件保存在 `tmp/experiment_61001_checkpoint_autocast_before`，历史实验日志及 checkpoint 未修改。
