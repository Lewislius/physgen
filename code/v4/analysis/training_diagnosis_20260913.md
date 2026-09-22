# 2026-09-13：61018 / 61858 训练诊断（准确 run 的独立核对）

本文件只诊断训练侧；实际视频、50 步 solver 和推理版本由综合报告覆盖。没有修改训练/推理实现，没有新增测试或启动 GPU。所有统计来自本 run 完整 500 条 train step、2 条 validation step、4000 条 micro 记录。

## 1. 证据身份与边界

- 主日志：[experiment_61018_trial_61858_logs.txt](/home/liuzhirui/Project/physGen/code/v4/logs/experiment_61018_trial_61858_logs.txt:36) 的 session_ready 明确指向 `20260912T151800Z-e189c27c`；[68 行](/home/liuzhirui/Project/physGen/code/v4/logs/experiment_61018_trial_61858_logs.txt:68) 是实际 training_start；[69 行](/home/liuzhirui/Project/physGen/code/v4/logs/experiment_61018_trial_61858_logs.txt:69) 确认 GT 初始化策略。
- 完整逐步记录：[steps.jsonl](/home/liuzhirui/Project/physGen/code/v4/train/train_log/wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/20260912T151800Z-e189c27c/steps.jsonl)，逐样本：[micro_rank0.jsonl](/home/liuzhirui/Project/physGen/code/v4/train/train_log/wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/20260912T151800Z-e189c27c/micro_rank0.jsonl)。第 250 步验证为 steps.jsonl 第 251 行，500 步训练为第 501 行、验证为第 502 行。
- 权重自带配置：[checkpoint-final/config.yaml](/home/liuzhirui/Project/physGen/code/v4/checkpoints/wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/checkpoint-final/config.yaml)。模型完成于主日志 [779 行](/home/liuzhirui/Project/physGen/code/v4/logs/experiment_61018_trial_61858_logs.txt:779)。
- `analysis/20260912_实验61002_训练损失平台与P门控退化分析_v1.md` 分析的是旧 run `125641Z-a822428d`；它是机制与历史背景，不能把其中六门全关闭、无 GT、warm every=3 等数字当本次事实。
- 保存的训练源码与目前 train_native_p.py / losses.py / data.py / cache.py 相同。当前 corrector.py / backbone.py 比保存源码多了 `first is None` 的 T2V 支持；I2V 有首帧路径对应计算没有发生这类改动。使用源码快照作为训练侧行号基准最稳妥。

## 2. 实际训练任务，比目录名和 final 名称更关键

| 项目 | 本 run 事实 |
|---|---|
| 阶段 | 仅 A1，stage=A，六个 A 各内部迭代一次；B 不参与训练/生成 |
| 数据池 / 实际训练 | 缓存清单 4000 条，3800 train / 200 validation；其中12个train因不足5帧排除，有效train3788；sample_limit=1000，仅前 1000 个有效 train 记录参与 |
| 训练量 | 500 optimizer steps × 8 accumulation = 4000 样本；恰好每条 4 次，共 1000 唯一视频 |
| 并行 | world_size=1；硬件标签 4x96g 不代表这次用了 4 卡；主日志平台资源也 slots_per_trial=1 |
| 参数 | A1 实际可训练 837,957,970 参数，其中初始化器 20,610,880，每个 A 136,224,515；checkpoint 包含总 974,182,486 corrector 参数，含未参与 A1 的 B |
| 主干 | 预训练 Wan2.2-TI2V-5B 冻结，主要训练新增 corrector |
| P₀ 条件 | image_text_gt；374/500 步给完整干净视频 latent，126/500 步不给；每 optimizer step 所有 8 micro 共用这个选择 |
| 图像条件 | 训练所有样本均固定干净首帧；无 image dropout，未训练真正无首帧 T2V |
| 文本 dropout | 407/4000 = 10.175%；其中无 GT 且无文本只有 93 样本，2.325% |
| 无 GT 覆盖 | 1008 micro，685 个唯一视频；315/1000 视频四次训练都没有无 GT 条件；正文本且无 GT 915 micro / 643 唯一视频 |
| 状态分布 | state_warmup_every=0；每个训练 forward 从 P₀ 起步，不训练 sampler 产生的连续去噪状态轨迹 |
| 学习率 | 1e-4，100 步 warmup，后续 cosine 至约 1e-5；500 是阶段预算，final 不代表最佳视频质量 |
| 输出约束 | out_weight=0；没有在线 VAE 解码+teacher 的 out loss，更没有完整 50 步生成评价 |

关键来源：[配置 26 行](/home/liuzhirui/Project/physGen/code/v4/checkpoints/wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/checkpoint-final/config.yaml:26)、[配置 54 行](/home/liuzhirui/Project/physGen/code/v4/checkpoints/wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/checkpoint-final/config.yaml:54)、[配置 64 行](/home/liuzhirui/Project/physGen/code/v4/checkpoints/wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/checkpoint-final/config.yaml:64)、[训练实际载入数据](/home/liuzhirui/Project/physGen/code/v4/train/train_native_p.py:229)、[GT 模式选择](/home/liuzhirui/Project/physGen/code/v4/train/train_native_p.py:106)、[GT 输入与 forward](/home/liuzhirui/Project/physGen/code/v4/train/train_native_p.py:303)。

解释：1000 样本配约 8.38 亿新增可训练参数、只有 4000 训练视图，明显缺少充分覆盖复杂物理现象、人物身份和各种场景的依据；这造成优化/泛化风险。但“样本少”不自动证明已发生过拟合，不能拿它代替配对证据。A1 没启用 A2/A3/B1/AB，也不能称为整个设计都完成训练。

## 3. Loss 到底改善了什么

当前目标是 `L_FM + 0.1*a(step)*w(sigma)*L_struct + 0.02*a(step)*L_prior`。`a=min(1,step/200)`，`w=clip((1-sigma)/0.2,0,1)`。struct 对六个 A 的提交态 MSE 求平均，不是六份直接求和；prior 是每个时间槽的空间池化 MSE。FM 比较未来 latent 的 velocity 与 noise-clean；首帧不参与 FM。详见 [losses.py 28 行](/home/liuzhirui/Project/physGen/code/v4/physgen_v4/losses.py:28)、[40 行](/home/liuzhirui/Project/physGen/code/v4/physgen_v4/losses.py:40)、[65 行](/home/liuzhirui/Project/physGen/code/v4/physgen_v4/losses.py:65)。

| optimizer 步区间 | total | FM | struct 原值 | struct 加权 | prior 原值 | prior 加权 |
|---|---:|---:|---:|---:|---:|---:|
| 1-50 | 0.310593 | 0.280201 | 2.581223 | 0.026269 | 1.747308 | 0.004124 |
| 51-100 | 0.317364 | 0.251803 | 1.771672 | 0.060158 | 0.749870 | 0.005402 |
| 101-200 | 0.403644 | 0.283084 | 1.649579 | 0.113042 | 0.502833 | 0.007518 |
| 201-250 | 0.400483 | 0.251066 | 1.580116 | 0.140023 | 0.469670 | 0.009393 |
| 251-300 | 0.397914 | 0.250505 | 1.548528 | 0.138508 | 0.445036 | 0.008901 |
| 301-400 | 0.400782 | 0.257301 | 1.514356 | 0.134956 | 0.426259 | 0.008525 |
| 401-500 | 0.388455 | 0.248058 | 1.474141 | 0.132348 | 0.402399 | 0.008048 |

末 100 步按均值统计，FM / struct / prior 对总 loss 的数值贡献分别 63.86% / 34.07% / 2.07%。前 200 步 total 受辅助 warmup 影响，不能以它上升证明变差。FM 确有一些学习：逐样本 sigma∈[0.8,1) 的前 50 步与末 100 步均值为 0.22623 → 0.16050。但 sigma∈[0.2,0.4) 为 0.27023 → 0.27178，且不同窗口样本和 sigma 分布不同，不能当成配对因果收益。

全部500步标量有限，全局梯度最大 0.191437，grad_clip=1 从未触发；没有数值爆炸、NaN、过强裁剪导致训练失效的证据。末100步初始化器占总梯度平方范数的平均比例 84.27%，提示优化集中在初始化器，但此统计按参数组，不能解释为 84.27% 梯度来自 JEPA，或直接证明 FM 被辅助梯度压制。

## 4. 最强直接证据：后五个状态校正器退化，写回仍在作用

末100步均值：

| 位置 | state gate | 实际 ΔP RMS | write gate | write RMS / H RMS | 单次 struct gain |
|---|---:|---:|---:|---:|---:|
| A@5 | 0.993061 | 0.207134 | 0.999933 | 0.149777 | 0.0546703 |
| A@10 | 0.000157668 | 2.50743e-05 | 0.999987 | 0.13557 | 2.96533e-08 |
| A@15 | 0.00188356 | 0.00029954 | 0.485297 | 0.0460187 | 4.17233e-09 |
| A@20 | 4.08896e-05 | 6.41496e-06 | 0.495376 | 0.0389596 | -1.14739e-08 |
| A@25 | 4.76779e-06 | 7.01036e-07 | 0.832542 | 0.0831295 | 1.63913e-09 |
| A@30 | 1.60571e-07 | 1.99227e-08 | 0.0867237 | 0.00291003 | 1.63913e-09 |

A@5 仍有有意义的 P 更新；后五层虽然 delta 的候选向量没有消失，却被接近零的 state gate 压到很小。后五次 dense MSE 增益平均约 1e-8 量级，实际工作接近“P₀ → A@5 一次更新 → 后五层对几乎同一 P 用 H 查询并写回”。不能用旧报告的“六个状态门全关”描述本 run。

这尤其能解释为何“P 学得更好”没有自动带来稳定生成：关 state gate 并不关 writer；A@5 和 A@10 write gate 几乎饱和到 1，写回相对当前 H 的 RMS 仍约 15.0% 和 13.6%，A@25 约 8.3%。后续去噪每步都会再应用。这些不是无作用微扰，也不能把各层 RMS 比例直接相加当总破坏幅度。是否是脸部形变、色彩变化的直接原因仍需原 Wan/关闭 writer 的相同条件生成对照，单靠日志不能断言。

实现路径：[corrector.py 411 行](/home/liuzhirui/Project/physGen/code/v4/physgen_v4/corrector.py:411)：`P += eta * sigmoid(state_logit) * delta`，写回为独立 `gamma * sigmoid(write_logit) * writer(...)`；state gate 没有正下限，也没有约束 write 的相对能量；sigmoid 饱和会同时削弱 delta 路径梯度和门自身导数。

## 5. 机制解释：初始化器、GT 与逐层目标容易走向一个局部解

以下有计算图依据，但不能把每一项都说成已完成因果证明：

1. P₀ 初始化器本身能看文本、首帧、时空位置，75%训练步还能看所有干净未来 latent（空间压到最多8×8，所有时间槽仍参与）。这是很强的未来条件，推理不存在它。[ConditionPrior](/home/liuzhirui/Project/physGen/code/v4/checkpoints/wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/checkpoint-final/source/physgen_v4/corrector.py:94)
2. 不只0.02权重的 pooled prior 在训练 P₀；六处 dense struct 都沿 `P_after=P_before+increment` 恒等路径反传到 P₀。`initial.detach()`只用于诊断，不切断states上的梯度。降低 prior_weight 一项不能切断 dense监督初始化器的捷径。[状态和 prior 返回](/home/liuzhirui/Project/physGen/code/v4/checkpoints/wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/checkpoint-final/source/physgen_v4/backbone.py:120)
3. 每层都逼近同一个完整未来 JEPA目标，初始化器或第一层已降低误差后，后层最容易通过关门避免增量造成误差；FM又可通过仍开放的writer继续学习，因此FM不强制动态状态更新必须有价值。
4. JEPA密集 MSE + pooled prior 都是表征代理；代码没有显式检查接触发生、碰撞前后速度、刚体形状、质量守恒、人物身份或色彩时间稳定。它们可以帮助表征学习，但不等价于这些物理/视觉事件的正确性。
5. 本run末100步 P 初始MSE=1.52881、最终平均struct=1.47414；该批GT自己的全视频常量均值 oracle MSE=1.23419，说明还没达到充分学习时空细节的证据。该均值使用未来真值，不能当可部署基线；也不能仅凭这个比较证明P严格塌成常量。需要实际P的时空方差/条件敏感性证据。

GT条件差异是**已存在**，GT主导或造成最终失败则**未证明**：末100步无GT组24步的struct=1.46109、prior=0.38619，给GT组76步分别1.47826、0.40752，不能从随机分组均值证明给GT更优或更差。这些样本、sigma不同；需要同一样本配对或训练配方对照。更稳妥的路线是先训部署条件一致的 image_text / image dropout，再让全视频GT作为监督教师，避免直接作为大多数学生前向的可见条件。若保留GT，应设计显式配对的有/无GT一致性或逐渐提高无GT比例，而不是凭25%这个数认为已解决分布差异。

## 6. 监督数据与想学的现象可能不对齐

这是本 run 数据清单的直接统计：实际1000训练记录全部 `caption_scope=video`；974条不是完整源视频；527条只使用原视频帧数的一半以下；中心窗口时长中位3.3333秒，200条短于2秒，最短0.1333秒。典型宽高288×512，teacher对完整窗口均匀抽至最多16帧。source fps含30/25/24/50/60等，原生连续采样，窗口真实时长并不固定。

实现：[中心截窗](/home/liuzhirui/Project/physGen/code/v4/physgen_v4/data.py:20)、[source timestamps 与 teacher indices](/home/liuzhirui/Project/physGen/code/v4/physgen_v4/data.py:48)、[整段 caption 绑定窗口](/home/liuzhirui/Project/physGen/code/v4/physgen_v4/data.py:75)、[CachedWISA 限前1000](/home/liuzhirui/Project/physGen/code/v4/physgen_v4/data.py:104)。

含义：caption描述了整段视频的事件起因/结果，而GT训练片段可能仅含中间局部动作，甚至没有那个事件；无法指望FM或JEPA监督从错误配对中学出“提示词中的整个现象一定发生”。这里只能证明存在系统风险，尚未逐条人工确认1000窗口中有多少caption事实不匹配。teacher最多16帧也可能跳过很短接触/碰撞瞬间，不能把语义特征拟合直接当逐帧动力学约束。原始帧率多样且无显式统一时间速度条件时，还需审查骨干的速度分布适配，但不应无证据说坐标代码完全错误。

优先修复是以事件为单位采样完整“前态→交互→后态”窗口，并使用window caption，过滤超短/无关片段、剪辑跳变、描述中未出现的物体/作用；验证首帧与文本一致。比直接把limit从1000改大，更关键的是事件、时窗、文字和teacher视图覆盖一致。

## 7. 验证盲区，为什么训练日志没挡住差视频

实际只有一条验证视频（index12，航拍山谷/河流风景）的五个固定sigma，不是200条；两次验证在250/500步，训练之前没固定baseline，亦没实际多步生成。[validate 126 行](/home/liuzhirui/Project/physGen/code/v4/train/train_native_p.py:126)

| sigma | step250 FM | step500 FM |
|---|---:|---:|
|0.10|0.4124985|0.4113215|
|0.35|0.1572262|0.1560319|
|0.60|0.0958804|0.0952464|
|0.85|0.0694645|0.0696142|
|0.98|0.1690287|0.1549626|
|五者平均|0.1808197|0.1774353|

500比250仅下降1.87%；没有证据说500的验证回归损失比250更差，故不能直接诊断后半程过拟合导致生成崩溃。反过来，一条风景的标准加噪GT速度回归变好，完全没有验证人物脸、物体形状、复杂相互作用、无首帧T2V或完整采样轨迹的稳定性。teacher-forced的一步端点训练与sampler闭环生成之间存在任务分布差异，reset P仅修复P状态初始分布，不等于修复latent闭环误差。

## 8. 按证据优先级处理，不盲目追加训练阶段

- **先确定生成路径是否可靠**：用少量代表性已失败案例，固定输入/seed/solver，比较原生Wan、相同封装禁用corrector、当前final、当前250。这是归因必须的最小配对，不要求新增大量单元测试。若禁用corrector仍失败，先修推理/基础模型条件；若只有corrector失败，进入以下训练修复。仅因50步比2步正常并不能确认增强有效。
- **降低强写回的风险并定位位置**：将现有write strength按原值、半值、零做少量paired generation；必要时只保留A@5。不要在推理中强行打开已饱和关闭的state gate。若退火写回改善脸/色彩但事件仍缺失，说明视觉扰动和事件学习至少部分是不同问题。
- **简化可训练容量和部署条件**：先用少数较早A/参数共享或低秩适配，image_text条件；同时训有首帧/无首帧时要明确混合概率、全latent时间条件和loss mask。B未训练前勿启用；A2仅把失效更新重复两遍不会自动救回；A3增加同一JEPA代理也不保证物理改善。
- **修复门控优化**：新训练或更早初始化使用小/零delta输出与温和门，不让乘法门过早饱和；可降低门学习率，或给状态门小正下限，配合delta范数/增益监控。门非零本身不是目的，要有正的生成或状态改进。只从已关闭final继续训，不能当干净消融。
- **查清目标竞争再调权**：固定小批分别取加权FM/struct/prior对初始化、state core、writer的梯度范数和余弦。观察辅助与FM冲突，而不是比较原始loss数值。最低成本训练对照可先比较FM-only与struct/prior分别降10倍的一组；这些是诊断起点，非已证明最优系数。
- **划清P₀和动态更新职责**：对初始化器先学粗prior，再冻结或控制dense辅助梯度路由，让动态corrector对H的新观察真正负责；注意整体detach(P₀)会同时切掉FM路径，不能当按loss来源路由的替代。将深层重复完整GT对齐减少到少数有意义层/目标，避免六层追同一个易解均值。
- **把事件数据放在容量前面**：事件完整窗口与window caption；teacher采样覆盖瞬间；保留人物/刚体/流体等代表性留出集；同一视频近重复不能跨split。训练时预定少量完整采样周期与基线差值，按生成结果选checkpoint，而不是按“final已保存”。

明确未被证明的解释：完全由struct_weight过大导致、GT条件一定泄漏并主导、teacher输出全部错误、500比250过拟合、所有P token严格常量、Ada48硬件直接改变物理质量。当前硬证据支持的是后五层状态更新退化、写回仍显著、有限数据/事件监督风险、训练/部署条件不一致和验证覆盖不足；各项对脸形变、色彩变化和事件缺失的因果占比仍需最小配对拆分。

机器可读统计：[training_diagnosis_20260913.json](/home/liuzhirui/Project/physGen/code/v4/analysis/training_diagnosis_20260913.json)。
