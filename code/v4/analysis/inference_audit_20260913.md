# v4 推理审查补充：20260912T151800Z-e189c27c

审查时间：2026-09-13。只读检查代码、checkpoint配置、实际输出metadata和step日志；没有运行GPU生成、没有新增测试、没有修改实现。视频画面由主审查报告单独描述。本报告不把2步质量当作50步质量，也不把相关性当成已证明因果。

## 1. 实際执行路径与可确认范围

- 50步目录 `inference_outputs/wisa_native_p_1xada48g/final_20260913-070408-373326450` 的已完成metadata均指向用户指定run的 `checkpoint-final`；均为50 scheduler updates、100 Wan forwards、600 corrector calls。seed42、Euler、shift5、CFG5、101帧、24fps；demo 512×288、reference 384×384。
- 此次产物仍在增加。审查中summary从10条完成增加到11条，total=41、status=running；不能说20个demo的40个组合都已看完。应以最终主审查观测时点为准。
- 正式50步均记录 `state_policy.mode=reset`，cond/uncond的P独立，P为FP32。2步的 `smoke_before` 是旧的persistent；`batch_smoke_21` 已改为reset。
- 代码从checkpoint本身的config构造架构，791个tensor严格load，不进行迁移，不使用当前训练配置覆盖：`inference/infer_native_p.py:43-68`。全参数包含冻结未启用分支；该run实际为A1/stage=A，6个独立A@5/10/15/20/25/30、iterations=1、B不启用。`full` 是“启用本stage训练过的模块”，不是“所有规划阶段均已训练完成”。
- 当前 `physgen_v4/encoders.py` 与checkpoint的source完全一致；backbone差异仅 `first is None` 时timestep不再固定第一帧；corrector差异仅允许无首帧prior/T2V。I2V路径语义未改变。不能凭加载成功宣称推理效果正确，但未找到“错checkpoint、漏权重、wrong Wan版本”的证据。
- 运行metadata记录路径、config、tensor计数；batch signature仅包含config hash、weights大小和mtime，不含执行代码hash/weights内容hash（`inference/cases.py:131-152`）。因此能核对执行声明与现代码/存档，但不是执行文件字节完整性的证明。
- 所谓原“测试集”在本次batch里是1条reference：`overfit_ref5.jpg` + `The woman smiles and waves at the camera.`；不等于WISA held-out集合。demo直接读Pxx-origin.txt（`inference/cases.py:53-71`），未做提示词扩写。

## 2. 不是已确认bug的基础环节

- FM训练目标 `(noise-clean)` 与 `z += (sigma_next-sigma)*prediction` 的符号一致（`physgen_v4/losses.py:5-12,68`；`inference/infer_native_p.py:200-207`）。未发现把noise预测误当velocity预测。
- I2V干净首帧latent的初始替换、每步重置，以及首个latent时间片对应token的t=0，与本地Wan TI2V代码一致。对比 `backbone.py:103-118`、`infer_native_p.py:173-207` 和 `model/Wan2.2/wan/textimage2video.py:548-598`。
- CFG公式正确，cond/uncond没有共享P链（`infer_native_p.py:178-201`）。当前reset符合训练 `state_warmup_every=0`；不能拿旧smoke的persistent解释正式50步。
- Decoder的inverse scale、conv2、每latent片因果解码、cache递推、first_chunk和unpatchify顺序与Wan源实现一致：`physgen_v4/encoders.py:76-93` 对比 `wan/modules/vae2_2.py:812-838`。自建cache每视频新建，模型encode自身前后clear_cache（原实现783-809）。没有发现跨视频cache污染或色彩通道维度交换的代码证据。
- 输出decoder先clamp[-1,1]再转uint8，缺少clamp导致wrap不是本次实现问题（`encoders.py:93`、`infer_native_p.py:390-393`）。

## 3. 明确存在、但影响程度需要对照的推理差异

### 3.1 Euler并非Wan原生默认solver

脚本默认Euler（`inference/run_inference.sh:27-28`），Wan原生仅提供UniPC/DPM++，默认UniPC（`wan/textimage2video.py:527-546`）。v4已有UniPC/DPM++分支，因此可以原地切换，不需要再写一套生成器。50步shift5的实际sigma前高后低：step0=1，step20≈0.88235，step40≈0.55556，step45≈0.35714，step49≈0.09259，最后单步跳到0。2步只有1→0.83333→0，最后一步跨度0.83333，不能代表正常收敛质量。

Euler可合法解同一flow ODE，不能说“用Euler就是推理写错了”；但当corrector扰动向量场时，一阶误差与终端大步可能放大细节问题。需同seed/model的UniPC对照才可定量。

### 3.2 VAE BF16不是原生FP32

`physgen_v4/encoders.py:15-18` 将VAE权重和scale构成BF16；编码/解码也在BF16 autocast中（`infer_native_p.py:363-364,388-389`）。Wan TI2V原类构造VAE不传dtype（`wan/textimage2video.py:98-100`），VAE默认 `dtype=torch.float`（`wan/modules/vae2_2.py:894-899`）。因此，尽管cache实现相同，数值精度不是完全原生一致。

这不是训练/推理之间新产生的差异（训练cache也是这套load_vae），也不足以凭代码断言造成疯狂闪色。推荐保留同一final latent，用原生FP32 VAE仅解码一次；若画面一致，迅速排除decoder主因。若差异显著，再统一VAE dtype和缓存版本。不要通过改dtype后重新生成整条轨迹混淆encoder、sampler、decoder三个变量。

### 3.3 低分辨率与负提示词

原生TI2V典型canvas是1280×704（原类 `textimage2video.py:165-166`），目前因训练约束只512×288/384×384，总面积约原来的16.4%；demo Wan latent 32×18，hidden每时间片16×9=144个token。球、球拍、手、脸、接触点丢失细节，物体边界稳定性更难。但不意味着合法小canvas一定会疯狂变色。

本次negative_prompt为空，而原生Wan对空串自动替换其质量负提示词（`textimage2video.py:496-497`；`wan/configs/shared_config.py:19`）。这会改变CFG无条件分支。v4训练本就有empty-text dropout，故空串不是非法条件；仍是原生对照时必须对齐的参数。不可承诺补负提示词即可修好物理。

### 3.4 T2V是明确的未训练使用方式

T2V实现做了必要的无首帧timestep调整、latent不clamp、prior移除image tokens（`backbone.py:109-111`、`corrector.py:94-114`）。但该run训练每个样本都有first、first latent总被固定，FM loss总排除首片；没有image dropout或T2V混合训练（`train/train_native_p.py:288-320`、`losses.py:5-7,68`）。T2V既改变prior模态，又要求corrector处理此前从未训练的“第一片也在去噪”的hidden。基础Wan支持TI2V并不代表外挂corrector已支持T2V。T2V失败不能和I2V同等解释成“训练任务内性能”。

## 4. 最具体的模型行为证据：状态更新退化，写回仍强

来源：reference 50步 `.steps.jsonl`，合计cond/uncond 100个forward均值。RMS是feature范数统计，不是像素误差。

| 层 | state gate平均 | effective P update RMS平均 | write gate平均 | 写回/hidden RMS平均 |
|---|---:|---:|---:|---:|
| A@5 | 0.995 | 0.229 | 1.000 | 14.6% |
| A@10 | 8.62e-5 | 1.42e-5 | 1.000 | 14.1% |
| A@15 | 0.00262 | 0.000433 | 0.687 | 7.74% |
| A@20 | 4.22e-5 | 6.88e-6 | 0.596 | 3.82% |
| A@25 | 4.62e-6 | 7.04e-7 | 0.874 | 7.65% |
| A@30 | 1.50e-7 | 1.93e-8 | 0.117 | 0.251% |

A@5所有token的state gate>0.95，后5层所有token<0.05。A@5单次写回峰值19.37%；P02 T2V达到21.86%。这个模式跨所有已检查case重复。P RMS仍稳定，reference约1.288–1.323，无NaN/Inf，**不应叫作P数值爆炸**。

训练日志证明该现象不是推理才发生：step25/50六层gate≈0.98–0.997；step100以后后层迅速闭合；step500 gates为 `[0.99494,0.0001496,0.002267,3.878e-5,4.609e-6,1.542e-7]`。后5层struct_gain约 `[-1.24e-6,-1.61e-5,-3.28e-7,-1.49e-8,0]`，实际已经没有逐层接近teacher的改善。

机制：所有层都对同一个teacher状态做MSE（`losses.py:50-62`），关闭状态更新有可能成为简单的损失最小化路径。与此同时写回有独立gate、独立projection，依然读取前层留下的P并注入hidden（`corrector.py:437-457`）；所以“P没有再变化”不等于“这些外挂层没有作用”，它们可退化成用近似静态P的多层adapter。关闭状态更新还会压低相应delta/core的有效学习信号。

需要区分：门控退化和强写回是直接证据；其导致多少闪色、畸变是待ablation验证的因果假说。不能据此强行把gate全部打开，那会启用长期未学习好的增量。

## 5. 从训练到推理的核心分布差异

- 训练75%概率用完整clean GT视频给P0，推理100%没有GT。训练仅25%步骤学习真正部署条件，而且模态选择整optimizer step共用（`train/train_native_p.py:106-116,271-320`）。P0通过GT时序token可直接读取未来，生成时需要凭首帧/文本预测未来；没有no-GT分支单独足够训练和生成验证，就容易学到依赖不可用信息的捷径。代码支持dropout只能缓解，不能证明已经适配。
- 训练每次输入由GT直接加噪，而推理输入来自自己前几十步的错误积累；当前训练不做sampling rollout，state reset解决P分布不一致，不解决latent exposure bias。
- A1的loss/out=0，VAE/teacher对生成端点的路径从未用于这个run的优化。训练struct只是内部P对齐teacher，无法保证它写回后图像的脸、颜色、轮廓、碰撞都合理。
- CFG5会把两个分支误差组合成 `5*delta_cond - 4*delta_uncond`；两个分支都启用corrector。空文本、无GT的部署分支训练更少，可能把小的条件不一致放大。现有step日志只有每分支各层RMS，没有cond-uncond prediction差、latent统计，不能凭日志确定CFG已失控。
- 50步与2步都只证明合法完成，不代表输出良好。当前检查仅isfinite/shape，无物理事件验证（`infer_native_p.py:209-214,390-391`）。

## 6. 最小且有辨别力的解决顺序（没有在本次执行）

1. 先固定reference和1个事件demo、seed、首帧、prompt、分辨率，在现有接口跑 `--variant wan` 对比 `--variant full`。若base健康/full坏，首先处理corrector；若base也坏，再排查公共sampling/VAE/低分辨率。两者都差仍可继续检查full是否额外恶化，不能简单宣布模型全坏。
2. 公共路径对照：使用已有 `--solver unipc`，对齐原生negative_prompt。只换一个变量。若仍显著闪色，保留同final latent做原生FP32 decode；不先写海量单元测试。
3. corrector对照：现有 `--writer-off A` 可保留P更新而关闭全部写回；再只关 `A@5 A@10` 或逐个关可定位高写回层。实现已有hooks（`backbone.py:142-145`），无需新推理框架。
4. 训练方向：减少或移除GT输入，或只给独立teacher后验、把no-GT student蒸馏到该后验；I2V/T2V采用明确image dropout+全部latent的模式条件loss。针对当前gate崩塌，优先改善目标/优化/分层权重，监测实际struct_gain，不要盲目增加iterations或强行开门。
5. 写回保守化：更小初始/分层写回系数、较低writer LR、对基础Wan的预测一致性或残差幅度约束；用matched baseline的画质和事件完成率选checkpoint。高噪声阶段强写回及多层重复条件注入值得单独调整，但没有消融前不指定唯一最优系数。
6. 用少量真正完整50步生成作为训练验收，至少含训练分布无GT I2V、独立物理demo；不把单噪声点GT加噪的FM loss下降当作生成质量证据。

