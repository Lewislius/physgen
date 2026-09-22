# 训练使用 GT 初始化 P₀，推理使用已训练的无 GT 路径

更新：2026-09-12。配置入口为 [wisa_native_p.yaml](../configs/wisa_native_p.yaml)。本次保留六套独立 corrector，修改 P₀ 的训练条件；默认完整 A1 为 500 个优化器 step，暖起默认关闭。

## 1. 同一个初始化器，每个样本只算一次 P₀

| 情况 | P₀ 的输入 | P₀ 监督 | Wan 前向次数／样本 |
|---|---|---|---:|
| 训练，提供 GT，概率 75% | 文字、独立首帧、当前 GT 训练片段的干净 VAE latent、时空坐标 | GT 的冻结 JEPA 特征 | 1 |
| 训练，去掉 GT，概率 25% | 文字、独立首帧、时空坐标 | 同一 GT 的冻结 JEPA 特征 | 1 |
| 验证 | 文字、独立首帧、时空坐标 | GT 仅用于计算评价损失 | 1／sigma |
| 正式推理 | 文字、用户首帧、请求的时空坐标 | 不计算损失，不运行 JEPA | 每个去噪步 2 次，两路 CFG 各一次 |

`initialization: image_text_gt` 注册一个支持可选视频条件的初始化器。两种训练情况共用这个初始化器的文字层、首帧层、attention 和输出头；没有另外创建一套无 GT 初始化器，没有每个样本算两份 P₀，没有新增两份 P₀ 之间的蒸馏损失。

`gt_condition_dropout: 0.25` 控制去掉 GT 的概率。选择按优化器 step 确定，同一步所有 rank 和累积 microbatch 使用同一选择；每个 microbatch 仍使用自己的视频并重新生成 P₀。随机选择由 seed 和 step 单独决定，不受各 rank 的随机数状态影响，RESUME 可复现。概率不表示每四步严格出现一次；0／1 分别供“总给 GT”／“总不给 GT”的显式消融使用。

去掉 GT 条件，仍保留训练所需的 GT 标签：Wan 依旧输入标准加噪 GT、计算 FM；JEPA 依旧监督 P₀ 和各次更新的 P。这与推理时凭噪声生成视频的区别是正常的去噪训练关系。推理初始化器不会读取未来 GT；其 eval 模式拒绝显式传入 `gt_video`。

## 2. GT 到底怎样进入 P₀

读取已有缓存的 `sample['latent']`，即当前训练片段的干净 VAE 编码 `[B,48,T,h,w]`。不是另读一个无关视频，不是 Wan 的加噪输入，也不是把 JEPA 答案 T 直接复制给 P₀。对于原视频较长的情况，这里指选定训练片段；没有声称读取了片段之外的完整原视频。

GT latent 停止输入梯度，空间自适应平均池化至最多 8×8；所有 latent 时间槽保留，全部空间像素参与池化。101 帧最多产生 26×8×8=1664 个 GT 条件 token。每个 token 经独立 LN(48) 和 48→512 投影，加入时间、教师 letterbox 坐标及视频类型编码，再与文字和首帧序列拼接。全部 P 查询通过同一两层条件网络读取这些输入，输出原生 `[B,N_P,1664]` 的 FP32 P₀。池化位置使用相同平均区间的坐标中心，避免内容和位置错位。

GT latent 提供视频内容，映射到 JEPA 高层语义仍然需要学习，不能把 VAE latent 本身叫作 JEPA 语义。固定 JEPA 只产生监督目标，其参数不训练。已有缓存包含所需 latent，不增加预编码阶段或在线视频编码器。

随后仍是 `P₀ → A@5 → A@10 → A@15 → A@20 → A@25 → A@30`。每次提交的 P 都计算原生稠密 L_struct；writer 消费同一提交 P，并通过真实 Wan 后缀接受 FM 监督。不同 A 的完整网络、GT 初始化器、可选 B 的参数彼此独立。同一 A 在不同去噪步复用自己的参数。

## 3. 为什么训练无 GT 路径

训练总有 GT、推理突然去掉 GT，会产生模型未训练过的输入组合，也允许模型过度依赖真实未来。混合训练让实际推理使用的文字／首帧路径直接接受 L_prior、L_struct 和 FM 的训练。初始 writer 为零时，FM 对初始化器的梯度可以为零；L_prior／L_struct 仍训练初始化器，writer 学开后 FM 也能更新它。

保留 A1 原来的三类损失及权重，没有为了新条件额外叠加损失：FM；各 corrector 提交 P 的稠密结构 MSE；P₀ 与 JEPA 按时间槽分别进行空间池化后的先验 MSE。L_prior 保留粗时序信息，权重仍为 `0.02 × warmup`，不强迫无 GT 的 P₀ 精确复原每个未来空间 token。L_struct 权重仍随噪声减弱。这里的损失／学习率 warmup 是权重逐渐增加，与额外 Wan 暖起前向是两件事。

研究上的参考是训练时额外信息与部署输入需要匹配，以及共同训练条件存在／缺失的情况：[Unifying distillation and privileged information](https://arxiv.org/abs/1511.03643)、[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)。本实现采用单初始化器的 GT 条件丢弃，不是这两篇论文的方法复现；论文不能证明本项目的 25% 比例最优或画质必然提升。

## 4. 跨去噪步连续与可选暖起

跨步接续只发生在同一视频内部。例：视频甲第一次前向得到 P₃₀，下一去噪步 A@5 从这个 P₃₀ 接着更新；训练换成视频乙时，重新从乙的输入产生 P₀。模型参数经过训练保留学习结果，P 激活不跨视频传递。

原来的每 3 步暖起在同一视频、同一噪声下，构造较高 sigma 和较低 sigma 两个标准加噪视图。先无梯度运行高噪声视图，把输出 P detach 后交给低噪声正式前向，后者计算损失和反向。它训练当前步接收更新过的 P，但没有让第一个预测通过 scheduler 产生第二个输入，没有跨两次前向反向传播，也没有显式相邻 P 相似度损失。因此它不是完整采样展开，不能保证 50 步不会漂移。

GT 初始化提供真实片段内容；暖起训练接收已更新状态，两者作用不完全相同。因尚无暖起生成收益的对照证据，当前正式默认 `state_warmup_every: 0`；保留非零配置作为明确选择的对照，避免将未经验证的额外 Wan 计算固定加入 A1。启用时，P₀ 先以所选 GT／无 GT 条件生成一次；暖起从其 detach 副本出发，正式视图接收暖起 P，原 P₀ 保留 L_prior 梯度。不再为了 L_prior 重新生成一次初始化结果。关闭 L_prior 的暖起步不更新初始化器。

推理继续为 cond／uncond 分别保存 P，默认不重置；`--reset-state` 可用于相同条件的状态接续对照。单步训练和长程推理的分布差异仍需完整采样评估，不能用 CPU 的有限值检查替代。

## 5. 多卡、checkpoint 与正式入口

无 GT step 仅排除 GT 专用 `video_norm`、`video`、`video_modality` 的梯度同步和裁剪；它们的 grad 为 None，AdamW 不更新它们。共用初始化层和所有活动 A 正常同步。所有 rank 使用相同选择，避免一侧同步不存在梯度或出现不同通信序列。原 BF16 非重入 checkpoint 保留 `cache_enabled=False`，不关闭一致性检查。

新增 GT 输入层合计 **25,696** 个参数；默认初始化组 **20,610,880**，A 阶段合计 **837,957,970**，B **136,224,516**，模型总计 **974,182,486**。六套 A 的数量、宽度和参数归属不变。空间池化限制新上下文大小，实际 attention 开销与 GPU 峰值还需实测。

三份 `train_wisa_native_p_4x48g.yaml`／`4x80g.yaml`／`4x96g.yaml` 共用新配置，均为 `PHASE=A1`、`TRAIN_CHECK_STEPS=0`、1000 条训练记录、500 个优化器 step、目标 global batch=8，250／500 步验证保存及最终保存。96G 仍为 1 卡；另外两份实际资源分别为 ada-24g／2 卡、amp-80g／4 卡。默认每个样本一次正式 Wan 前向，使用真实缓存和真实冻结 Wan；evaluation 下的 CPU 替身不会由 YAML 调用。

当前架构的 checkpoint 严格恢复全部 GT 层和独立 block。旧 `image_text` → `image_text_gt` 必须用 A／AB 的 `INIT_FROM`：保留全部兼容初始化器和 corrector 权重，只新建 GT 专用输入层，并新建优化器；不会清零 writer。B1 不允许冻结刚加入的随机初始化层。`RESUME` 使用保存的配置和优化器，旧 checkpoint 的 RESUME 不会自动升级成 GT 初始化。已运行进程也不会因为修改磁盘代码自动换架构。启动器已修正非 A1 的 RESUME 被错误附加上一阶段 INIT_FROM 的冲突。

日志直接显示 `prior/gt_conditioned`（本步是否输入 GT）、`state/warm_active` 和 `compute/wan_forwards`（每样本 Wan 次数）。

## 6. 检验与生成收益验收

本次 **112 项相关 CPU 检查最终通过**。记录见 [gt_condition_prior_review.json](results/gt_condition_prior_review.json)，静态检查见 [gt_condition_prior_static_review.json](results/gt_condition_prior_static_review.json)。测试涵盖 GT 未来内容确实改变 P₀、GT／教师输入梯度边界、两种条件的初始化与六位置 FM 梯度、BF16 checkpoint 梯度一致性、按时间保留的池化坐标、验证不传 GT、跨 rank／恢复的条件选择、初始化只调用一次、参数独立和 checkpoint 迁移、三份正式 YAML 启动链。小网格真实 corrector＋冻结 TinyWan 的三步 Euler／CFG 检查只确认状态传递和有限值，不代表正式 Wan 的长期稳定性。

另用现有真实 WISA 缓存、默认 512 维两层初始化器及完整 `[1,4608,1664]` P₀，在 CPU BF16＋checkpoint 下分别完成 GT／无 GT 前向与反向；76／71 个活动参数张量梯度全部有限，无 GT 时仅 5 个 GT 专用参数张量无梯度。结果见 [正式尺寸检查](results/gt_prior_full_shape.json)，复现入口为 [check_gt_prior_full_shape.py](check_gt_prior_full_shape.py)。这仍未运行真实 Wan 后缀或 GPU。

效果应以固定留出首帧／文字、多随机种子的无 GT 完整视频生成判断，记录画质、运动、物理过程和失败样本；不能以“给 GT 的训练 loss 更低”替代。使用 [generate_cases.py](generate_cases.py) 和已有评分入口，分别比较当前混合训练、全程无 GT 训练，以及显式开启暖起的训练 checkpoint，统一条件、采样器、步数和基础数据，并报告额外计算量。`--reset-state` 只改变推理状态策略，不能单独证明暖起训练是否有效。

尚未执行新架构的真实 GPU 训练、完整生成或评分，故没有生成质量提升结论。当前代码提供可运行的机制、明确的部署输入训练和对照入口；25% 条件丢弃、空间池化大小、暖起开关的最终选择应服从无 GT 生成结果。
