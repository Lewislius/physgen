# v4 amp-48g：NCCL 卡死修正与检查方案

适用故障：2026-09-12 的 `20260912T080052Z-40238704`，实际两卡、global batch 8、每卡累积 4。

本页原故障对照针对历史共享 corrector。当前新配置已改为六套独立 A 和独立 B；A 阶段每卡仅 corrector 权重／梯度／Adam 状态约 12.99 GiB。原 config.json 的重放仍按旧 shared 架构运行，不能据此声称新架构的显存和吞吐已验证。独立参数专项见 [检查说明](INDEPENDENT_CORRECTORS.md)。当前 4x48g 文件实际 resource_pool=ada-24g、slots_per_trial=2；以实际分配为准。

## 1. 证据与判断边界

- 08:16:52：两个 rank 的 NCCL watchdog 心跳停滞，阈值为 480 秒。
- 08:24:52：尝试 dump 后由监控主动终止，两个进程都是 SIGABRT / exitcode=-6。ChildFailedError 是 torchrun 失败汇总；Root Cause (first observed failure) 不证明 rank 0 是最初致因。
- 退出后 multiprocessing 清理临时目录出现 `.nfs* / EBUSY`。这是清理阶段的 NFS 活跃文件问题，不能直接当作最初的卡死原因。
- 当次 micro_rank0 为空、rank 1 的 micro 日志文件缺失。rank 0 的 training_start 不代表两边都已开始训练。
- 原代码在 CUDA/NCCL/FSDP 初始化后，以 Python 3.10 默认 fork 启动 DataLoader worker；这是明确的代码风险，但没有故障线程栈证明它是唯一根因。

依据：[PyTorch NCCL/fork 提醒](https://docs.pytorch.org/docs/2.8/generated/torch.nn.parallel.DistributedDataParallel.html)、[NCCL 心跳与 trace 配置](https://docs.pytorch.org/docs/2.8/torch_nccl_environment_variables.html)、[Linux unlink 的 NFS EBUSY 说明](https://man7.org/linux/man-pages/man2/unlink.2.html)。

## 2. 修改内容

| 文件 | 修改及目的 |
|---|---|
| physgen_v4/data.py | training_loader() 为所有正数 workers 指定 spawn；worker 只读 CPU 缓存；workers=0 时 context=None、timeout=0。多 worker 取数超时默认 120 秒 |
| tools/runtime_env.sh | v4_setup_tmpdir() 默认创建 /tmp/physgen-v4.XXXXXXXX 私有目录，设置 TMPDIR/TMP/TEMP，供同节点预处理、训练和 teacher 子进程继承 |
| train/run_training.sh、tools/prepare_wisa.sh | 移除把 TMPDIR 强制指向共享项目目录的代码，调用本地临时目录设置 |
| physgen_v4/diagnostics.py | NCCL 初始化前创建每个 rank/PID 的事件与线程栈文件；阶段开始/结束/异常日志不调用 CUDA；已跟踪阶段长时间无进展时周期性输出 Python 栈 |
| train/train_native_p.py | 提前创建 Logger；每 rank 输出 rank_ready；标记初始化、取数、搬运、warm/正式前向、loss、反向、同步、梯度归约、优化器及日志；记录 GPU/版本和已完成 microbatch 的显存峰值 |
| physgen_v4/backbone.py | 可选逐 Wan block 前向标记，判断进入哪个 FSDP block 后不再返回 |
| train/train_native_p.py | --check-steps 保持原学习率调度，执行有限 optimizer step 后退出，跳过验证及 checkpoint 写入；旧 checkpoint 也能覆盖 loader/诊断参数 |
| train/train_native_p.py | 指标累积统一为 float32，修正整数帧数、前向计数在 step 汇总时调用 mean() 的错误。这是独立问题，不能解释此前的 watchdog hang |
| evaluation/check_training_runtime.py | 独立 NCCL/缓存加载检查，不被训练启动器自动调用 |

保持原 NCCL heartbeat/monitoring、P2P/IB、模型精度与 FSDP 分片策略。旧临时文件不自动删除。

## 3. 开关与日志语义

| shell 环境变量 | Python 参数/配置 | 默认 |
|---|---|---|
| TRAIN_WORKERS | --workers / train.workers | 2，固定 spawn |
| TRAIN_PIN_MEMORY | --pin-memory / --no-pin-memory | 1 |
| TRAIN_LOADER_TIMEOUT_SECONDS | --loader-timeout-seconds | 多 worker 120；0 关闭取数超时；单进程强制 0 |
| TRAIN_TRACE_STEPS | --trace-steps | 本次启动的前 1 个 optimizer step；0 仍保留启动/验证/保存标记 |
| TRAIN_STACK_TIMEOUT_SECONDS | --stack-timeout-seconds | 120；0 关闭定时 Python 栈 |
| TRAIN_TRACE_WAN_BLOCKS | --trace-wan-blocks / --no-trace-wan-blocks | 0 |
| TRAIN_CHECK_STEPS | --check-steps | 0 为正常训练；正数为短程检查 |
| TRAIN_NCCL_DEBUG | NCCL INFO、trace buffer=20000、dump、C++ trace | 0；设 1 打开，尊重显式 NCCL 环境变量 |
| V4_LOCAL_TMP_ROOT | 本任务临时目录的父目录 | /tmp；可改为节点本地 scratch，避免 NFS |
| V4_DIAGNOSTIC_DIR | 本任务各 rank 的诊断目录 | 控制台日志旁的独立目录 |

DataLoader timeout 只约束多 worker 的取数等待，不保证中断 CUDA/NCCL、worker 创建或文件系统调用。Python 栈是辅助证据，原生阻塞仍需 NCCL trace 或节点线程栈。

stage_end 表示 CPU 调用返回，CUDA 完成要看随后的 *_sync 结束。逐 block 日志仅覆盖前向，不代表反向重计算已完成。日志不会额外插入 barrier 或逐 block synchronize。默认只细查第一步；长期问题应增大 TRACE_STEPS。短程检查执行真实更新但不保存权重，不应作为正式训练结果。

## 4. 本地回归

在 v4 根目录执行：

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B -m unittest discover -s evaluation -p test_training_runtime.py -v
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B -m unittest discover -s evaluation -p test_launchers.py -v
/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B -m unittest discover -s evaluation -p test_monitoring.py -v
bash -n train/run_training.sh tools/runtime_env.sh tools/prepare_wisa.sh
```

新增测试真实启动 spawn worker，读取小型 CPU 缓存并验证顺序、可变文本长度、异常回传和 worker 未初始化 CUDA；实际触发定时线程栈；用小型 CPU 替身执行 4 次训练更新，覆盖 warm/普通前向、整数指标汇总、短程停止及不触发验证/保存。CPU 替身不验证 Wan、FSDP 或 GPU kernels。

测试需要本机 IPC socket。沙箱禁止 socket 时出现的 PermissionError 及其后续 DataLoader timeout 属于测试环境限制，应在允许本机 IPC 的环境运行，不能通过关闭测试或修改张量共享策略掩盖。

## 5. 原 amp-48g 节点验证顺序

以下是待执行的目标节点步骤，工作机的 24G RTX 3090 不能替代。各组顺序运行，保持相同节点、两张 GPU、样本选择、seed、精度和 global batch。

### 5.1 本地临时目录与纯通信

在已分配的两卡容器执行：

```bash
cd /home/liuzhirui/Project/physGen/code/v4
source tools/runtime_env.sh
v4_activate runtime
v4_setup_tmpdir
findmnt -T "$TMPDIR" -o TARGET,SOURCE,FSTYPE
df -h "$TMPDIR" /dev/shm
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi topo -m
export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_CPP_STACK=1
export TRAIN_NCCL_DEBUG=1
"$V4_PYTHON" -I -B -m torch.distributed.run --standalone --nproc_per_node=2 \
  evaluation/check_training_runtime.py --mode collectives --iterations 20
```

默认每 rank 输入 32 MiB BF16，反复 all-gather/all-reduce 并校验全部输出。两边都出现 runtime_check_complete 且正常退出才通过。失败时优先调查 GPU/驱动/互联/NCCL，保留各 rank 日志、dump、GPU Xid/ECC 信息。单次小张量成功不等于连续通信正常。

### 5.2 初始化 NCCL 后读取缓存

```bash
"$V4_PYTHON" -I -B -m torch.distributed.run --standalone --nproc_per_node=2 \
  evaluation/check_training_runtime.py --mode loader --workers 0 --iterations 8
"$V4_PYTHON" -I -B -m torch.distributed.run --standalone --nproc_per_node=2 \
  evaluation/check_training_runtime.py --mode loader --workers 2 --iterations 8
```

包含首次 NCCL、DataLoader 启动、真实缓存读取、搬到 GPU、每次读取后的 NCCL。两组保持 pin_memory=True，只改变 workers。若只有多 worker 失败，再单独重复 --workers 2 --no-pin-memory，区分 worker IPC 和 pin-memory 线程路径。该检查不加载 Wan。

### 5.3 两卡真实 Wan/FSDP 四步对照

当前共享配置另有 train.sample_limit=1000，首样本可能与最初 3788 条训练记录的运行不同。下面用原运行 config.json 重放原始数据选择；YAML 读取器也支持这个 JSON。两次使用同一配置。

```bash
export CONFIG=/home/liuzhirui/Project/physGen/code/v4/train/train_log/wisa_native_p_A1_4x48g/20260912T080052Z-40238704/config.json
PREPARE=0 PHASE=A1 WANDB_ENABLED=0 RUN_NAME=nccl_check_workers0 \
TRAIN_WORKERS=0 TRAIN_CHECK_STEPS=4 TRAIN_TRACE_STEPS=4 TRAIN_NCCL_DEBUG=1 \
bash train/train_wisa_native_p_4x48g.sh

PREPARE=0 PHASE=A1 WANDB_ENABLED=0 RUN_NAME=nccl_check_spawn2 \
TRAIN_WORKERS=2 TRAIN_CHECK_STEPS=4 TRAIN_TRACE_STEPS=4 TRAIN_NCCL_DEBUG=1 \
bash train/train_wisa_native_p_4x48g.sh
```

运行前确保未继承 RESUME、INIT_FROM、TRAIN_STEPS 等实验性覆盖。通过 Determined 新提交时，将这些变量写入独立 YAML 的 environment.environment_variables；不能假设提交端 export 自动进入容器。slots_per_trial=2，entrypoint 保持原 48G shell。

上述历史 CONFIG 的四步覆盖内部 step=0/3 的 warm 前向与 step=1/2 的普通前向。当前新配置的暖起默认关闭，GT／无 GT 条件训练见 [专项说明](GT_CONDITION_PRIOR.md)；不能把历史诊断配置当作新 GT 初始化训练配置。验收：

1. 两个 rank 都有 diagnostics_start、session_ready、rank_ready。
2. 每 rank 的 micro JSONL 为 16 条（4 steps × 4 accumulation），step 日志为 4 条。
3. loss/梯度为有限值，显存峰值在目标 GPU 容量内；两边都有 training_check_complete(completed_steps=4)，并正常退出。
4. 无 watchdog/collective timeout、SIGABRT、DataLoader timeout 或 NFS 临时文件清理错误。

workers=0 与 spawn=2 均成功只证明修改后通过短程验证；同时迁移了临时目录，不能据此声称历史故障唯一来自 fork。不为归因而在正式任务恢复 fork/NFS 组合。

若缓存检查通过但真实训练仍卡，重复短测并增加 TRAIN_TRACE_WAN_BLOCKS=1，对比两 rank 最后进入的 block、warm/普通前向和 NCCL 序列；若停在 backward，结合 Python 栈与 flight recorder 检查 FSDP 重计算和通信。

### 5.4 稳定性与正式运行

先把 CHECK_STEPS/TRACE_STEPS 提高到 20；通过后恢复正式运行参数，移除 CHECK_STEPS，按实验选择 1000 条或全量训练记录。正式运行至少验证一次 validation 和 checkpoint；短程检查跳过这两条路径，不能声称它们已通过 GPU 验证。

诊断后关闭 TRAIN_NCCL_DEBUG/逐 block trace，TRACE_STEPS 恢复 1；保持 spawn 和本地 TMPDIR。延长 heartbeat 或关闭 monitoring 不是本次修复。

## 6. 如何读日志

诊断目录含 events_rankR_pidP.jsonl、stacks_rankR_pidP.txt。启动器开启 NCCL debug 时另有 nccl.主机.PID.log、实际发生 dump 时的 nccl_trace_pidP_rankR。正式标量仍写入 run_name/session_id。

| 最后停留阶段 | 优先检查 |
|---|---|
| init_distributed | CUDA 设置、NCCL 初始化、节点 |
| session_broadcast | 初始化后的首次对象广播及 CUDA 调用 |
| wan_load_and_shard | 权重 I/O、CPU/GPU 搬运、FSDP 初始化 |
| dataloader_iter | worker 创建、序列化、pin-memory 初始化 |
| data_next | worker 栈、缓存 I/O、IPC、取数超时 |
| data_sync | H2D 搬运、驱动、pin-memory |
| warm_forward/forward/wan_block | 对比两 rank 位置，检查 FSDP all-gather、attention/CUDA |
| backward/backward_sync | 重计算、FSDP 通信、CUDA kernel/驱动 |
| backward 报 CheckpointError，saved/recomputed metadata 不同 | 检查 no_grad 暖起与带梯度前向的 AMP 权重缓存；实验 61001 已由共用训练入口的 cache_enabled=False 修复，见 [专项说明](CHECKPOINT_AUTOCAST_FIX.md) |
| gradient_all_reduce | 梯度参与集合、通信顺序/形状、另一个 rank 的进度 |
| micro_log/step_log | 标量同步、共享日志写入 |
| 退出时 .nfs*/EBUSY | 实际 TMPDIR；用 lsof/fuser 查占用，在相关进程关闭后处理残留 |

不批量删除项目 tmp，不终止无法确认归属的进程。本次修改不处理旧任务遗留的 .nfs* 文件。

## 7. 验证记录

结果见 results/nccl_hang_fix_checks.json，区分真实 CPU spawn、CPU 模型替身和未执行的目标节点 GPU 验证。

48G／80G／96G 三份训练 YAML 当前都显式设置 `TRAIN_CHECK_STEPS=0`，默认完整执行 A1。本文的正数 CHECK_STEPS 仅用于用户主动选择的短程诊断；`trace_steps=1` 仅限制详细日志。实验 61001 的首步错误、修复后的混合精度检查与正式入口检查另见 [AMP／checkpoint 修复说明](CHECKPOINT_AUTOCAST_FIX.md)。

当前直接提交三份正式 YAML 会使用 image_text_gt、25% 无 GT 条件训练及 state_warmup_every=0。无 GT step 的 GT 专用输入层不参与梯度同步，各 rank 条件选择一致；排查 gradient_all_reduce 时核对 prior/gt_conditioned 与 grad/initialization_gt_active。
