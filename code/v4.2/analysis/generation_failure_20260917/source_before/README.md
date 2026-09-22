# physGen v4.2

当前版本使用原缓存前2400条有效样本，保留原训练/验证划分；第5/15层各有独立的reader、core、writer。原caption、原fps、连续中心窗与图像尺寸处理沿用v4，视频帧数上限121。所有新代码与输出位于本目录。

实现说明：[analysis/20260916_v4.2_代码实现与审核.md](analysis/20260916_v4.2_代码实现与审核.md)。

2026-09-17 checkpoint 日志故障已修复；本次 step0100 的完整性及恢复已验证，现成续训配置和测试说明见 [CHECKPOINT_RECOVERY.md](CHECKPOINT_RECOVERY.md)。

## 先运行16条快速版

```bash
cd /home/liuzhirui/Project/physGen/code/v4.2
det experiment create train/train_quick16_2epochs_1x96g.yaml .
```

一次提交自动完成：16条原视频索引 → VAE/T5/JEPA编码 → 12条训练数据遍历两轮 → 4条留出验证 → 保存checkpoint → 新进程加载EMA → I2V/T2V各生成一条4步视频。共24次训练曝光、6次优化器更新，覆盖第1步开始的FM联合训练、可微VAE解码与低噪声修复。详见[quick16/README.md](quick16/README.md)。

日志出现`quick_pipeline_passed`、生成`quick16/work/SUCCESS.json`才表示这一GPU链路真正完成；文件中有checkpoint和两个视频的路径。它不能判断正式训练的生成质量。

## 正式训练

```bash
det experiment create train/train_stability_1x96g.yaml .
```

默认`PREPARE=1`，自动运行`index → vae → text → teacher → finalize → train`。逐样本、逐组件复用有效缓存，缺失或损坏时补算；某阶段全部命中时，不加载该编码器。teacher与首帧anchor独立检查，只缺anchor时不会重算teacher。`ready.json`存在也会检查完整性，缺文件会重新进入准备流程。只有实际需要生成某个组件时才要求对应原视频和GPU。

正式训练入口会先用训练环境执行CUDA初始化和一个小tensor操作，通过后输出`gpu_ready`，再准备数据。进入训练时会再次检查GPU。失败时`gpu_unavailable`保留原始CUDA异常、Python/PyTorch版本、GPU可见性环境变量及`nvidia-smi`结果，便于区分环境与节点驱动问题。在Determined分配的GPU容器内可单独运行`source tools/runtime_env.sh`后执行`v42_python runtime "${V42_ROOT}/train/train.py" --check-gpu`。

如果`cache_finalized`之后、训练启动之前因CUDA失败退出，重新提交上述任务即可复用有效缓存，不需要`RESUME`。已确认缓存完整时，也可提交时追加`--config 'environment.environment_variables=["PREPARE=0","OMP_NUM_THREADS=4"]'`，直接跳过准备阶段。新容器仍出现CUDA错误时，根据`gpu_unavailable`排查该容器的GPU设备、驱动和动态库；修改训练超参数不能修复CUDA初始化失败。

数据逻辑与v4对齐：

- 从`cache/wisa4000_nativefps_f121_a147456_fp32_jepa32/manifest.json`取前2400条有效记录，不是简单截取编号0–2399。
- 保留这些记录原有split，当前为**2279 train / 121 validation**。原划分按前4000个候选文件名、seed=20260909、validation_stride=20生成。若原manifest不存在，则按同一候选集合划分并顺序索引到2400条有效记录。
- 默认保留原始`captions`，不调用语言模型改写。
- 原fps，连续中心窗，最多121帧；短片按`1+4*floor((min(T,121)-1)/4)`帧保留，少于5帧排除；不补帧、不稀疏抽取源视频。
- 原v4尺寸桶：最长边512、最大面积147456、边长32的倍数；等比例缩放和居中padding，不放大小图。

v4.2的FP32 VAE、32帧JEPA输入和16时间槽状态保留，它们属于新模型的编码/监督定义。短片仍按实际长度编码latent，teacher按该片段取16个相邻帧对（共32帧），极短片段重复教师采样位置；不会把重复帧写入源视频或VAE训练窗。

正式缓存目录为`cache/wisa2400_nativefps_f121_a147456_fp32_jepa32/`，原4000缓存保留。复用来源包括原4000目录和`quick16/work/cache_v4_f121/`；同一文件系统用硬链接，跨文件系统复制，修复采用原子替换，避免改写来源文件。首次检查tensor形状、dtype和有限性，`validated_cache.json`记录输入契约和文件元数据，后续启动复用校验结果。预处理兼容性独立于模型结构版本，旧编码实现只接受明确审核过的指纹。统计量重新按这2279条训练数据计算。

截至此次修改，2400条VAE均已校验并接入新目录，文本/teacher/anchor各复用16条及negative；剩余组件由启动流程生成。可用`STAGE=reuse bash tools/prepare.sh`只做CPU缓存复用和缺失统计。资产默认采用`metadata_only`检查；VAE/T5/JEPA仍分进程顺序加载。训练时TEMP需要VAE解码器；跳过的是准备阶段已完成的VAE编码。

## 正式训练行为

| 项目 | 实现 |
|---|---|
| GPU与环境 | `blk-96g`单slot；训练/推理`moviestory`，teacher使用`vjepa2-312` |
| 基座 | 冻结Wan2.2 TI2V 5B；原始FP32 VAE，关闭decoder autocast/TF32 |
| 结构 | 第5/15层各自reader + core + writer；两套corrector参数与Adam状态完全独立，无gate |
| 初始化 | I2V真实首图JEPA；T2V文本加固定训练集均值，无真实未来条件 |
| 目标 | FM + `.10*q*struct` + `.05*temp`，再按本步辅助梯度预算缩放STRUCT |
| 累积 | microbatch=1，accumulation=8，每批6 I2V/2 T2V；FM/struct除8，唯一temp不除8 |
| 梯度 | STRUCT detach输入重算，仅更新两个core状态分支（包括各自reader）；有效辅助梯度最多占两路范数之和的20%，生成梯度按全体可训练参数计算 |
| 1–800步 | 同一套联合训练：8个样本全部计算FM/struct，其中1个额外计算temp；所有外挂模块从第1步可训练 |
| 学习率/预算 | 全部参数组前50步只预热学习率，随后余弦衰减；writer/输出残差预算从第1步启用，不再在第101步切换或重置Adam |
| 801–1200步 | 每批唯一temp样本使用真实CFG的一步修复 |
| EMA | 第1步开始累计，最终1200次；第801步延续同一个优化器与EMA |
| 验证/保存 | 每200步固定留出健康检查；每100步保存可恢复checkpoint，step0100/step0200/…/step1200 |

状态仍按`P0 → core5 → P5 → core15 → P15`传递，最终P15的STRUCT沿状态链更新两个独立core。reader是各自StateCore内的读入注意力与投影；`correctors["5"]`和`correctors["15"]`没有共享的可训练模块。FM/TEMP经writer训练生成路径，STRUCT不回传writer或TextInit。参数组为`core5/core15/text_init/writer5/writer15`，共65,045,248个参数。

令`G=||g_(FM+TEMP,全部可训练参数)||`，`A=||g_(0.10×mean(q×STRUCT),两个core)||`，合并前取`alpha=min(1,0.25G/(A+eps))`。因此`alpha*A/(G+alpha*A)≤20%`；这是上限，不会放大弱梯度来凑满20%。该口径也不等于合成梯度向量或Adam实际参数更新的20%。`q`控制不同噪声样本的监督强度，0.10控制STRUCT基础尺度，alpha控制实际梯度预算。0.10是更强的起点，是否足够影响生成需要查看残差日志和同条件消融。

## 训练进度与损失日志

正式训练的控制台只显示每步的进度条和最终损失汇总。每处理完一个样本推进一次 `micro 0/8 → 8/8`；终端原地刷新，Determined 日志中按样本输出简短进度行。每次参数更新后显示 `FM`、`STRUCT`、`TEMP` 的有效权重 `w`、加权贡献 `add` 和 `TOTAL`。FM从第1步就计算并反向传播，不再有仅STRUCT的预热阶段。

- `FM.add = mean(L_fm)`，权重为 1。
- `STRUCT.qL = mean(q_i × L_struct_i)`，其中 `q_i = clamp((1-sigma_i)/0.5,0,1)^2`；控制台显示 `w=0.10*aux_alpha=有效权重`，`STRUCT.add = w × qL`。每个样本相对于原始损失的准确系数为 `0.10 × q_i × aux_alpha / 8`。
- `TEMP.add = 0.05 × L_temp`，每个联合/修复训练步只有一个时序样本，不再除以 8。
- `TOTAL = FM.add + STRUCT.add + TEMP.add`，包含本步辅助梯度缩放。它是固定本步 `aux_alpha`、保留结构梯度仅流向 core 时的等效目标值；实际训练仍采用独立反向传播与梯度合并，随后执行全局梯度裁剪和 AdamW。

所有详细记录位于 `train/train_log/<run_name>/`：`micro.jsonl` 保留逐样本原始损失、噪声权重和完整诊断；`updates.jsonl` 保留每步 `loss_total`、有效 `loss_struct`、原 `loss_struct_nominal`、`loss_terms` 中的逐样本系数与贡献，以及梯度、学习率和显存指标；`health.jsonl` 保存验证明细；`events.jsonl` 保存启动与 checkpoint 事件；`run.json` 保存配置和各损失的统计定义。验证与 checkpoint JSON 不再打印到控制台。

控制台`STRUCT.grad_share`显示实际梯度占比和20%上限。JSONL同时保存`generation_grad_all`、`core_generation_grad`、`core_auxiliary_grad`、`effective_aux_grad`、`aux_grad_budget`、`aux_alpha`与每组参数更新范数；两套core分别记录梯度和更新。

训练日程为 `fm_joint_independent_cores_struct_global_share20_v2`。前800步的样本选择、损失与预测分支规则一致；普通样本sigma仍在0.02–0.999采样，时序样本仍在0.05–0.15采样，FM使用条件预测，TEMP使用CFG预测。第801步起仅时序样本先做一次不反传的CFG预测，将sigma降到原来的0.8，再对同一更新后输入计算FM/STRUCT/TEMP，FM改用CFG预测与更新后的速度目标。

writer零初始化使首步生成输出等于基座，但writer的FM梯度通常非零，因此全模型G非零，两个core从首步即可获得STRUCT监督；由于core输出头也为零初始化，最初先更新其输出头，再逐渐传入内部层。缓存可复用，旧共享core checkpoint不可作为本结构的断点恢复。已运行Python进程保留内存中的实现，共享目录下随后启动的新进程会读取当前配置。

恢复训练时指定实际checkpoint目录：

```bash
det experiment create train/train_stability_1x96g.yaml . \
  --config 'environment.environment_variables=["RESUME=/home/liuzhirui/Project/physGen/code/v4.2/checkpoints/实际run目录/step0800","OMP_NUM_THREADS=4"]'
```

## 正式推理

```bash
det experiment create inference/infer_stability_1xada48g.yaml .
# 或单卡96GB
det experiment create inference/infer_stability_1x96g.yaml .
```

默认直接加载指定step0100 EMA，生成reference I2V与P01–P20各I2V/T2V，共41条。
启动已删除预检查、整包SHA、源码/资产/训练日程校验和Conda激活。
只对缺失缓存的文本/首图进行必要编码，跨checkpoint和硬件入口共享缓存；生成后默认结束，评估按需开启。
双独立校正器、P0逐步重置、原生CFG加一次有界残差及50步采样逻辑保留。
说明见[inference/README.md](inference/README.md)。

## 验证范围

本次收尾结果在[analysis/data_alignment_completion.json](analysis/data_alignment_completion.json)：两版启动命令、配置和语法检查通过；45/57/121帧真实视频与v4设置121帧上限后的输出逐像素一致，最大差值为0；短片时间插值前向/反向、12条训练数据各曝光两次及现有索引缓存复用检查通过。v4的79个受保护文件均未修改。

历史CPU与小规格真实Wan/VAE GPU数值证据在`analysis/contract_checks*.json`。本次没有执行quick16的GPU训练或正式1200步训练，完整GPU链路与生成质量以实际任务结果为准。独立检测脚本已删除；用户明确要求的quick16运行入口保留。
