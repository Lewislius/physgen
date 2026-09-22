# physGen v4：WISA 原生 P 校正器

2026-09-12 的 NCCL watchdog hang 修正、临时目录设置、诊断开关及目标节点对照命令见 [NCCL 卡死检查方案](evaluation/NCCL_HANG_CHECKLIST.md)。

2026-09-12 已改为 **block 5／10／15／20／25／30 各自独立的完整 corrector**；同一 block 在不同去噪 step 复用自己的参数。参数归属、迁移和验证见 [独立 corrector 检查说明](evaluation/INDEPENDENT_CORRECTORS.md)。

实验 61001 首次 backward 的 `CheckpointError` 已通过关闭训练 AMP 权重缓存修复，三种硬件入口共用该修复。三份训练 YAML 均显式设置 `PHASE=A1`、`TRAIN_CHECK_STEPS=0`，直接提交执行当前配置的完整 A1 训练；故障分析、正式训练设置和回归证据见 [AMP／checkpoint 修复说明](evaluation/CHECKPOINT_AUTOCAST_FIX.md)。

方法与实现推导见 [20260909-D 实现方案](20260909-D-原生P校正器训练与推理实现方案.md)。本目录包含训练、推理、缓存、监控、checkpoint 和独立评价代码。

当前数据约定：**Wan 连续取原始帧，最多 101 帧，保留原 fps 和源内容宽高比；长边≤512、面积≤147456。** 常见 16:9 视频使用 512×288。连续窗口应与事件／描述对应；无窗口标注时使用居中截取和原视频描述，属于弱标注，不能认为完整事件一定在窗口内。

## 1. 固定路径与配置

V-JEPA 2.1 教师处理（离线 teacher pass 和在线 L_out 的教师前向／反向）固定使用 /home/liuzhirui/miniconda3/envs/vjepa2-312。索引、VAE、T5、最终 manifest、训练主进程和推理固定使用 /home/liuzhirui/miniconda3/envs/moviestory。`vjepa2-312` 是机器上的实际环境目录名。三个训练入口共用环境约束：清除继承的 Python 路径，固定 Python 与 torchrun 子进程解释器；Python 入口校验实际环境，环境缺失／不匹配直接报错，不切换到第三个环境。`VJEPA_ENV` 不再提供覆盖能力。没有依赖安装或权重下载步骤。

| 项目 | 默认位置 |
|---|---|
| 代码根目录 | /home/liuzhirui/Project/physGen/code/v4 |
| 总配置 | configs/wisa_native_p.yaml |
| WISA 视频 | /run/determined/NAS1/public/Dataset/WISA/data/videos |
| WISA 索引 | /run/determined/NAS1/public/Dataset/WISA/data/wisa-80k.json |
| Wan 代码 | /home/liuzhirui/model/Wan2.2 |
| Wan 权重 | /home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B |
| 教师代码 | /home/liuzhirui/model/vjepa2 |
| 教师权重，用户自行准备 | /home/liuzhirui/model/vjepa2/checkpoint/vjepa2_1_vitG_384.pt |
| GT 缓存 | v4/cache/wisa4000_nativefps_f101_a147456_t16 |
| 训练输出 | v4/checkpoints/run_name、v4/train/train_log/run_name/session_id |
| 推理输出 | v4/inference_outputs |

缓存选择范围为 data.limit=4000；设为 0 使用全量，也可用 selection_file 指定原索引整数列表。按视频名称分集后原有 3800 条训练、200 条验证；排除过短视频后当前缓存有 3788 条可用训练记录、200 条验证记录。16 帧教师特征缓存单独约 57.1 GiB／4000 条，另加 latent／文本。

当前小规模实验设置 `train.sample_limit: 1000`：从已有缓存按 manifest 顺序固定取前 1000 条有效训练记录，训练时仍按种子打乱；保留全部 200 条验证记录。三种硬件入口共用这一配置，并设为 `PREPARE=0` 直接复用缓存。该限制不修改缓存或分集；设为 0 可使用缓存内全部有效训练记录。样本数和优化器步数分别控制，当前 A1 为 500 步，A2／A3／B1／AB 各 250 步；目标 global batch=8 时 A1 约遍历这 1000 条训练记录 4 次。旧 checkpoint 的 RESUME 使用其保存的样本限制；此次 1000 条设置用于读取当前配置的新实验。

可用 data.clip_annotations 指定 JSON 列表，每条包含 index、start_frame、frames、caption；具体格式见实现方案。修改选片、时间窗口、caption、画幅或教师视图后，修改 cache_root 并重新编码。

当前新训练使用 `corrector.initialization: image_text_gt`。同一个 P₀ 初始化器在训练时以 75% 概率读取文字＋独立首帧＋当前 GT 片段的干净 VAE latent，以 25% 概率去掉 GT；每个样本只计算一次 P₀。GT latent 保留所有时间槽，空间池化至最多 8×8，加入真实时间和教师 letterbox 坐标后供全部 P 查询读取。两层 512 维条件网络输出完整 [1,N_P,1664] FP32 P₀；GT 的 JEPA 特征只用于监督。详见 [GT 初始化与无 GT 训练说明](evaluation/GT_CONDITION_PRIOR.md)。

训练 GT 同时提供首帧、Wan 的标准加噪输入、冻结 JEPA 目标，并在保留 GT 条件的 step 参与 P₀ 初始化。去掉 GT 条件的 step 仍计算原有训练损失。验证与推理的初始化器只读取文字、首帧和时空规格，使用已经训练过的无 GT 路径；推理未来 latent 从高斯噪声起步，CFG 正／负分支各自接续 P，换视频重新初始化。已有缓存可以复用。

`corrector.parameter_sharing: per_block` 配合 `a_blocks: [5, 10, 15, 20, 25, 30]` 注册六套完整 A corrector。各自拥有 reader／writer、归一化、文本与 sigma 投影、类型 embedding、读写注意力、两层 StateCore、状态更新头和所有门控；没有跨 block 的参数别名或共享存储。`a_every: 5` 与显式列表保持一致。P 按 A@5→A@10→A@15→A@20→A@25→A@30 顺序更新并写回 H；一次前向中不更新权重，累积结束后一个优化器统一更新六组独立参数。同一 block 跨去噪 step、同位置内部迭代及 CFG 两次调用复用自己的权重，CFG 两分支的 P 独立。B@18 也有独立核心和适配器，A1 不调用 B。

当前 `corrector.fusion: cross_attention`：先以 P 为 Query、投影后的 H 为 Key/Value 读取观察，经过读取门与残差加入 P，再由当前 block 的 StateCore 更新 P；随后以 H 为 Query、更新后 P 为 Key/Value，直接得到每个 H token 的写回信息。两路均为 1664 维、26 头全局非因果注意力，参数归当前 block 独占，A/B 也不共享；P/H token 数保持各自原值，读写均不做时间或空间插值。双方使用同一坐标系的位置编码，写注意力排除教师外层画布的纯补黑位置。A2 等内部多轮更新会重新以当前 P 查询同一 H。残差加法仍保留，避免用注意力结果整体替换状态。

FM 和更新态 L_struct 能反向训练初始化器；`loss.prior_weight: 0.02` 对 GT／无 GT 条件产生的那一份 P₀ 使用相同的保留时间次序、空间池化 JEPA 蒸馏。默认 `state_warmup_every: 0`，每个样本一次 Wan 前向；非零值仅供同视频双加噪视图暖起对照。可选暖起也只生成一次 P₀，保留其 L_prior 梯度，携带的 P 停止梯度。`early_write_width: 0.1`、`early_write_floor: 0.1` 仍减弱极高噪声写回而允许 P 更新。条件丢弃比例和暖起是否改善生成，需无 GT 完整视频验证。

初始化、fusion、parameter_sharing 和 a_blocks 随配置写入 checkpoint。旧 image_text → image_text_gt 必须在 A／AB 用 INIT_FROM，只新增 GT 输入层、保留原初始化器及 writer；RESUME 始终按保存架构恢复，旧任务不会自动升级。缺少 parameter_sharing 的历史 checkpoint 仍按 shared 架构 RESUME／推理，以保留原权重及优化器组；这不代表已转为六套独立网络。要使用独立参数，启动当前配置的新 A1，或在 A／AB 阶段用 INIT_FROM：将旧共享核心和 A 适配权重复制到各 block 的独立参数，并将旧核心复制给 B，将旧先验与文本投影复制给独立 P₀ 初始化器。初始数值可以相等，Parameter 和存储不相同，之后分别学习；重新创建优化器，禁止把旧优化器直接套到新结构。仅拆分参数时不重置 writer；同时迁移插值融合时才新建各自的 attention／读取门并重置全部 writer 与 B.b，切换旧 latent 初始化时同时重建先验。同架构阶段切换严格恢复，不重置权重。A 阶段可训练参数为 **837,957,970**，B 为 **136,224,516**，AB／模型总参数为 **974,182,486**。

## 2. 首次提交 Determined

以下命令提交集群任务；本次没有执行提交或本地 GPU 训练。先将用户准备的教师 pt 放在上述位置。

~~~bash
cd /home/liuzhirui/Project/physGen/code/v4
det experiment create train/train_wisa_native_p_4x48g.yaml .
~~~

另外两种硬件：

~~~bash
det experiment create train/train_wisa_native_p_4x80g.yaml .
det experiment create train/train_wisa_native_p_4x96g.yaml .
~~~

当前三套入口实际配置为 `4x48g → ada-24g／2 卡`、`4x80g → amp-80g／4 卡`、`4x96g → blk-96g／1 卡`；96G 入口已按单卡测试设置，实验名为 `12h-physgen-v4-wisa-A1-1x96g`。这些是用户可修改的 YAML 字段，文件名不决定资源池或容量。每次提交只需在对应 YAML 中修改 `resources.slots_per_trial`，即可改变预处理和训练使用的卡数，无需修改脚本、文件名或其他配置。启动器读取平台分配的 `DET_SLOT_IDS`，不导入 Determined SDK，不执行启动预检。`PREPARE=0` 直接进入训练，不额外扫描缓存。

例如把 `slots_per_trial: 4` 改为 `slots_per_trial: 2` 后重新提交，便使用两卡、两个训练进程。现有文件名、实验名和 checkpoint 路径中的 `4x` 是固定入口标签，不限制实际卡数，也不需要随卡数重命名。直接在已分配容器中运行 shell 时使用该容器实际分配的卡数；非 Determined 环境使用 CUDA 可见卡数。

.detignore 排除缓存、checkpoint、日志等大产物，避免后续提交把它们作为代码上下文传输。任务入口执行挂载中的 v4 文件。

当前 YAML 使用 PHASE=A1、PREPARE=0 复用已完成的缓存。需要首次构建缓存时设置 PREPARE=1：先运行索引→VAE→T5→teacher→最终 manifest，再训练 A1。三套预编码模型分进程阶段加载；teacher 在独立子 shell 中激活 vjepa2-312，结束后最终 manifest 和训练继续使用 moviestory。A3／B1／AB 每个训练 rank 启动一个 vjepa2-312 教师子进程，使用独立的 FSDP 通信组；通过私有管道传递 CPU 张量，回传教师特征和视频输入梯度，继续经 moviestory 内的 VAE／Wan 反向传播。验证和归因评价也使用该教师子进程。A1／A2 不启动在线教师。子进程退出或环境检查失败会使训练报错。预编码按 data.limit／selection_file 处理缓存选择范围，train.sample_limit 仅限制训练读取的记录数。

三份 YAML 的 `TRAIN_CHECK_STEPS=0` 关闭短程提前退出。正式 A1 按当前配置执行 **500 个优化器 step、1000 条已选训练记录、目标 global batch=8**，训练真实 Wan 上的六套独立 A corrector 和 P₀ 先验，启用 FM＋L_struct＋L_prior。第 250／500 步正常验证并保存 checkpoint，结束保存 checkpoint-final。`diagnostics.trace_steps=1` 仅限制详细诊断日志的步数，不限制训练长度；`PREPARE=0` 表示复用真实缓存。CPU 回归代码仅由 evaluation 下的检查命令调用。

在线教师跨进程传输增加 CPU 内存、数据搬运和独立 CUDA 上下文的开销；原先只统计训练进程的 PyTorch 显存指标不包含教师子进程显存。需要在目标 GPU 节点测量总显存和吞吐，不能把 CPU 梯度回归检查当作完整训练验证。

依赖缺失时只报错，不自动下载或安装；只有用户明确要求时才处理依赖。

训练 microbatch=1，目标 global batch 由 `train.global_batch_size` 设置，默认为 8。梯度累积自动取 `ceil(目标 batch / 实际卡数)`：1／2／4／8 卡分别累积 8／4／2／1 次；不能整除时向上取整，例如 3 卡累积 3 次、实际 batch=9。启动日志记录实际卡数、累积次数和 batch。三个配置均采用相同画幅／101 帧上限。Wan 的逐 block FSDP、激活重计算、分阶段驻留和稀疏 L_out 用于限制显存。corrector 自身仍在每个 rank 完整复制；计入全部 FP32 权重及活动组的梯度、两份 Adam 状态，A／B／AB 每卡静态预算约 **12.99／5.15／14.52 GiB**，还没有计入 Wan、激活、通信、VAE／教师或分配器。实际 GPU 峰值尚未测量，不能认定当前两张 24G 卡一定够用。

## 3. 阶段切换、初始化与断点恢复

| PHASE | 训练内容 | 默认初始化来源 |
|---|---|---|
| A1 | 每次调用 1 轮 cell，FM＋struct＋prior，500 步 | 新 corrector |
| A2 | 每次调用 2 轮 cell，FM＋struct＋prior，250 步 | 同硬件最近完成的 A1/checkpoint-final |
| A3 | 2 轮 cell，加真实 L_out，250 步 | 同硬件最近完成的 A2/checkpoint-final |
| B1 | 冻结 P₀ 与六套 A，仅训练完整独立 B，250 步 | 同硬件最近完成的 A3/checkpoint-final |
| AB，可选 | 全部新增权重联合，250 步 | 同硬件最近完成的 B1/checkpoint-final |

后续阶段在所选 YAML 中更改 name，并将环境变量改成例如：

~~~yaml
environment_variables:
  - PHASE=A2
  - PREPARE=0
  - INIT_FROM=/home/liuzhirui/Project/physGen/code/v4/checkpoints/wisa_native_p_A1_4x48g_20260912T150000Z-12345678/checkpoint-final
~~~

示例中的时间戳目录需换成实际保存目录，然后提交修改后的 YAML。训练不会根据 loss 自动晋级；B1 应从实际评价通过的 A3 初始化。未显式给 INIT_FROM 时，启动器从当前 CONFIG 的 checkpoint_root 中选择同硬件、前一阶段最近完成的 checkpoint-final，并打印选中的路径；兼容旧无时间戳目录，跳过未完成的运行。使用自定义 RUN_NAME 的前阶段可显式指定 INIT_FROM。

默认 run_name 前缀为 wisa_native_p_PHASE_HARDWARE，每次启动自动追加 UTC 时间戳和短唯一标识，例如 wisa_native_p_A1_4x96g_20260912T150000Z-12345678。同一秒启动也使用不同标识，全部 rank 共用同一目录名。新训练、INIT_FROM 和 RESUME 均使用新目录。

可选环境变量：

| 变量 | 用途 |
|---|---|
| CONFIG | 总配置文件路径 |
| PHASE | A1／A2／A3／B1／AB |
| PREPARE | 三套 YAML 当前设为 0，直接读取已完成的缓存；1 执行缓存准备并复用有效编码。直接运行 shell 而不指定此变量时默认为 1 |
| INIT_FROM | 新阶段只加载 corrector 权重 |
| RESUME | 同阶段恢复全部训练状态的 checkpoint 目录 |
| RUN_NAME | 输出目录名前缀，自动追加时间戳和唯一标识；也可用于续训 |
| TRAIN_STEPS | 新实验的步数覆盖值 |
| TRAIN_CHECK_STEPS | 三份训练 YAML 显式设为 0，执行完整阶段；正数仅供手动短程诊断，会提前退出并跳过验证／保存 |
| WANDB_ENABLED | 0 关闭、1 开启 W&B；三套提交 YAML 均默认 0 |

RESUME 与 INIT_FROM 互斥。RESUME 使用 checkpoint 内训练配置，保留优化器、LR 调度、步数和数据游标；输出沿用 run_name_base 前缀并追加本次时间戳，不写回源运行目录。旧 checkpoint 使用其 run_name 作为前缀，RUN_NAME 可覆盖该前缀。配置记录 resume_from 和 source_run_name，便于追溯。梯度累积随本次实际卡数重新计算。相同卡数时恢复各 rank 的 RNG；改变卡数时按恢复步数和新 rank 重新设种子，继续训练但不保证随机序列与原卡数一致。缓存与冻结模型资产需保持一致。W&B 开关可由 WANDB_ENABLED 覆盖，日志统一写入 train/train_log 下的新 session 目录；旧 checkpoint 中已移除的监控配置会在读取时丢弃。使用 RESUME 时设 PREPARE=0。

若已经处于分配好的集群 GPU 容器，也可直接使用对应 shell：

~~~bash
PREPARE=1 PHASE=A1 bash train/train_wisa_native_p_4x48g.sh
PREPARE=0 PHASE=A2 bash train/train_wisa_native_p_4x48g.sh
~~~

## 4. 训练监控与 checkpoint

每个 optimizer step 在控制台显示总 loss、FM／struct／prior／out 原始及加权分项、LR、梯度范数、sigma、输出损失开关和耗时，并把完整指标写入本地 TXT／JSONL。Determined 平台直接采集标准输出；训练进程不加载其 SDK，不上报 Core 指标曲线。TensorBoard／TensorFlow 集成及 profiling 配置已删除。训练解释器为 moviestory，不开放整个用户 site-packages，也不安装任何包。

所有完整标量指标同步写入 JSONL 和 TXT：包括 A@5 等每个调用位置的 P 误差／增量／写入、gate、梯度与权重 RMS、LR、数据几何、执行时间和各 rank 显存。每个 microbatch 还保存 latent／text／target 形状、视频几何和帧时间。保存的是标量统计与元数据；完整模型／优化器张量仍由 checkpoint 保存。out 每四步执行一次，必须结合 output/active 解读。梯度／权重组名为 initialization、A@5、A@10、A@15、A@20、A@25、A@30、B；例如 `grad/A@10_norm` 和 `weight/A@10_rms`。旧 shared checkpoint 保留旧组名。完整字段说明见实现方案第 8 节。

W&B 是可选后端，默认关闭；关闭时不导入 SDK、不读取 API key、不登录、不联网。要启用：先在 `train/wandb_credentials.py` 的 `WANDB_API_KEY = ""` 中填入自己的 key，并在 `configs/wisa_native_p.yaml` 的 `logging.wandb` 中设置 project／entity，再把所选提交 YAML 的 `WANDB_ENABLED=0` 改为 `WANDB_ENABLED=1`。直接执行 Python 时可用 `--wandb`／`--no-wandb`；没有命令行覆盖时读取 `logging.wandb.enabled`。只有 rank 0 登录并上报全局训练／验证指标，各 rank 的完整 microbatch 指标保留在本地。

API key 文件已从 `.detignore`、`.gitignore` 和 checkpoint 源码快照中排除，key 不进入训练配置和指标。W&B 关闭源码上传；只在显式开启时上传实验配置与指标，每次训练启动创建独立 run。W&B 的 `global_step` 对应 optimizer step，同一步的训练和验证指标都能保存。

~~~text
train/train_log/
  launcher_PHASE_HARDWARE_时间_PID.txt    # 启动检查、预处理、训练的完整控制台输出
  run_name/session_id/
    config.json                       # 本次有效配置
    runtime.json                      # 环境、参数量、样本数、起始 step
    steps.jsonl                       # 每个全局训练／验证 step 的全部指标
    steps.txt                         # 相同指标的可读文本
    latest.json                       # 最近训练／验证结果
    micro_rank0.jsonl ... micro_rankN.jsonl
    micro_rank0.txt ... micro_rankN.txt
    wandb/                            # 仅显式开启 W&B 时生成

checkpoints/run_name/
  latest.json
  checkpoint-0000250/
  checkpoint-final/
~~~

此处 run_name 已包含启动时间戳和唯一标识。每 250 个优化器步保存一次，目录内仍使用 checkpoint-0000250、checkpoint-0000500、checkpoint-final 等名称。若同一 checkpoint 目录已经存在，保存程序在写入任何文件前拒绝覆盖，并向所有 rank 报错；失败的部分保存也不会被静默覆盖。启动日志的 checkpoint_directory 字段给出本次绝对保存路径。代码修改只作用于之后启动的进程，已经运行的任务继续使用启动时加载的代码。

checkpoint 保存全部 corrector、优化器、LR scheduler、数据游标、各 rank RNG、数据视图清单、配置、训练／验证指标、运行信息与 v4 源码／文档快照；排除凭证文件和 train_log 内容。冻结大模型通过路径引用；用户管理外部资产。不会自动清理历史 checkpoint。

每 250 步在固定样本／固定 sigma 桶上验证；validation_loss 为平均 FM，不能解释为物理成功率。

## 5. 推理

单卡入口均固定使用 moviestory，默认读取 `wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/checkpoint-final` 内的配置与权重。该 checkpoint 是 `image_text_gt`、六套独立 A corrector、每位置一轮的 A1；`full` 在此等同于 A，后续指定 B1／AB checkpoint 才启用 B。完整用法和实际权重检查见 [单卡推理说明](inference/README.md)。

| 资源池 | 单卡启动脚本 | Determined YAML |
|---|---|---|
| ada-48g | inference/infer_wisa_native_p_1xada48g.sh | inference/infer_wisa_native_p_1xada48g.yaml |
| amp-48g | inference/infer_wisa_native_p_1x48g.sh | inference/infer_wisa_native_p_1x48g.yaml |
| blk-96g | inference/infer_wisa_native_p_1x96g.sh | inference/infer_wisa_native_p_1x96g.yaml |
| amp-80g | inference/infer_wisa_native_p_1x80g.sh | inference/infer_wisa_native_p_1x80g.yaml |

以上 YAML 的 `slots_per_trial` 都为 1。旧通用入口仍对应单卡 amp-48g。Ada 48G 集群提交入口：

~~~bash
det experiment create inference/infer_wisa_native_p_1xada48g.yaml inference
~~~

先在 YAML 的 environment.environment_variables 中填写 IMAGE、PROMPT、FPS。CHECKPOINT 可指定任一实际 checkpoint 目录；单卡配置当前显式选择 wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/checkpoint-final。默认 SUITE=both 跑原例和 v1/demo P01–P20，每个 demo 默认各跑 I2V／T2V，共 41 条视频；Ada 48G 使用 inference/infer_wisa_native_p_1xada48g.yaml。要自动查找时间戳目录，移除 CHECKPOINT 并设置 CHECKPOINT_RUN 为目标运行名前缀，脚本会选择最近完成的 checkpoint-final 并打印路径。脚本已有一个参考图和简单动作的默认值，但它不代表 WISA 物理评价条件。

已分配 GPU 的容器内直接运行示例：

~~~bash
CHECKPOINT_RUN=wisa_native_p_B1_4x48g \
IMAGE=/absolute/path/to/reference.png \
PROMPT='The red ball rolls toward the blue ball and hits it.' \
FPS=25 FRAMES=101 SEED=42 \
bash inference/infer_wisa_native_p.sh
~~~

IMAGE 示例需换成实际存在的首帧。输出包括 MP4、.steps.jsonl 和 .json（完整参数、训练配置、实际 F/H/W、时间轴、画布变换、fps、前向次数）。

| 变量 | 默认值／作用 |
|---|---|
| CHECKPOINT | 显式 checkpoint 目录，优先于自动查找；当前默认时间戳 A1 运行的 checkpoint-final |
| CHECKPOINT_RUN、CHECKPOINT_ROOT | CHECKPOINT 未设置时，可用运行名前缀开启自动查找；根目录默认 v4/checkpoints，仅选最近完成的 checkpoint-final |
| IMAGE、PROMPT、NEGATIVE_PROMPT | 首帧与正／负文本，负文本默认空 |
| FPS、FRAMES | 24、101；请求 F 向下对齐 4n+1 并受训练上限限制 |
| DURATION | 可选，指定首尾秒长后由它计算输出 fps |
| SAMPLING_STEPS、SOLVER | 50、euler；可选 unipc／dpm++ |
| SHIFT、GUIDANCE、SEED | 5、5、42 |
| VARIANT | full；可选 wan（全部旁路）、A（关闭 B） |
| RESET_STATE | auto 按保存的训练配置选择；此次最终 checkpoint 默认每步重置，0/1 显式选择接续/重置 |
| DEVICE、OUTPUT | GPU 编号与输出 MP4 路径 |

单张图没有原 fps；WISA 条件应填写源视频 fps。25 fps／101 帧的首尾相隔 4 秒。cond／uncond 独立持久 P，每步两次 Wan 前向、一次 scheduler 更新，推理不加载教师。

## 6. 评价入口

这些入口用于用户后续在集群开展独立效果评价，本次未运行 GPU 评价。

导出固定留出条件及空评分表（CPU 视频读取）：

~~~bash
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory
python evaluation/export_cases.py \
  --config configs/wisa_native_p.yaml --count 8 \
  --output evaluation/cases
~~~

已分配 GPU 的容器中生成全部指定对照：

~~~bash
python evaluation/generate_cases.py \
  --checkpoint checkpoints/wisa_native_p_B1_4x48g_20260912T150000Z-12345678/checkpoint-final \
  --cases evaluation/cases/cases.json \
  --output inference_outputs/paired_B1 \
  --variants wan A full --seeds 42 43
~~~

checkpoint 示例中的时间戳需替换为实际目录。默认生成每个 case 的全部 seed／variant；每条重新加载模型，成本应计入正式评价时间。默认少量 case 用于观察，不能支撑强成功率结论。按预定规则填写 ratings_template.csv：process_correct 为 0／1，quality 为 1…5，记录失败原因；不凭空填入评分。

~~~bash
python evaluation/summarize_ratings.py \
  --ratings evaluation/cases/ratings_template.csv \
  --baseline A --candidate full \
  --output evaluation/cases/B1_vs_A.json
~~~

同状态、同后缀的逐位置归因（已分配的集群 GPU）：

~~~bash
torchrun --standalone --nproc_per_node=4 evaluation/attribute_correctors.py \
  --checkpoint checkpoints/wisa_native_p_B1_4x48g_20260912T150000Z-12345678/checkpoint-final \
  --samples-per-rank 1 --sigmas 0.1 0.35 0.6 \
  --with-output-loss --downstream on \
  --output evaluation/outputs/B1_attribution
~~~

该入口有额外后缀／教师计算，衡量合法噪声端点的写入收益；不进入正式推理，也不替代完整采样的物理评价。B1 的 A-only 保留其冻结来源；AB 的 A-only 已共同训练，评价 AB 还需使用原 A3 checkpoint 作独立基线。

## 7. 本次实际完成的检查

本次 GT／无 GT 改造的代码、测试和限制见 [专项检查](evaluation/GT_CONDITION_PRIOR.md)，结果单独记录，不沿用历史通过数证明新架构。

**独立参数改造时的 116 项 CPU 回归最终全部通过，其中独立参数／六位置集成 27 项。** 两项 DataLoader 检查在支持本机 IPC 的环境重跑通过；当时静态检查通过 34 个 Python 文件、7 个 shell 和5份 YAML。

随后实验 61001 暴露了此前未覆盖的 BF16＋no_grad 暖起＋非重入 checkpoint 组合。本次补充及相关回归共 **22 项最终通过**，包含默认 512 维先验的全参数／输入梯度对照、六套独立 A 的混合精度反向、三份 YAML 的实际 shell 参数解析，以及 CPU 替身驱动正式训练循环完成 500 次更新并触发验证／保存。记录见 [修复与验证说明](evaluation/CHECKPOINT_AUTOCAST_FIX.md)；该 CPU 循环检查不代表真实 GPU 已完成 500 步。

本次独立参数的验证入口与结果见 [独立 corrector 检查说明](evaluation/INDEPENDENT_CORRECTORS.md)，覆盖完整模块隔离、单位置 AdamW 更新、六位置前后向与后缀重放、保存／恢复、旧结构迁移。历史共享架构的 CPU 数学／梯度记录见 [cpu_contracts.json](evaluation/results/cpu_contracts.json)。复现命令：

~~~bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
/home/liuzhirui/miniconda3/envs/moviestory/bin/python evaluation/contract_checks.py
~~~

检查使用完整通道数、小空间网格，未进行 GPU 训练、正式教师前向或生成效果检验。另已读取实际 WISA 元数据并检查 Python／shell／YAML，结果见 [静态检查记录](evaluation/results/static_review.json)。这些不能保证训练后必然提升效果。数据窗口是否覆盖事件、教师是否看清目标、FSDP 实际峰值、训练收敛和完整视频收益仍需各自的真实证据。
