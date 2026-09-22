# V4.3 设计与实现边界

> 本文记录现有实现及历史配置依据；当前设计与效果结论见 [分析入口](../README.md)。稳定基线为 STRUCT=.1、Oracle=1，不能将默认 YAML 的 .01/60 当成质量推荐。

本版本落实用户确认的三个外层损失：FM、STRUCT、Flow-Oracle。用户明确要求保留 Wan H 的原生宽度，让 P 升维适应 H，并禁止隐式压到 512 维。方案保留第 5、15、25 层插入、原始 Wan2.2-TI2V-5B 与现有数据缓存。没有假定文本里举例的 Wan2.1 scheduler 就是本机实现。

## 1. 为什么用“已有残差 + 新增修正”

当前层 Wan 输出记为 h，实际 writer 输出 d₀，后续网络接收 h+d₀。Oracle 优化变量 d 表示**相对当前层写前 h 的完整残差**，初始化为 `stopgrad(d₀)`：

`J(d) = FM(F_suffix(stopgrad(h)+d), noise-z_GT) + lambda_r * MSE_valid(d/s)`。

`s=stopgrad(max(RMS_valid(h), 1e-4))`。默认 `lambda_r=0.001`。

梯度 `g=∂J/∂d` 在当前 d 上求，方向为 `-g/RMS_valid(g)`。第 k 步提议：

`d_candidate = mask * (d_k - rho*s*g/RMS_valid(g))`。

默认 rho 为 0.01，回溯依次尝试 0.01、0.005、0.0025、0.00125。对每个提议都真实执行当前后续网络，只有 FM 至少改善 `max(1e-8, 1e-5*FM_current)` 且 `J_candidate <= J_current + 1e-4 * <g, candidate-current>` 才接受。点积使用求和；不能把训练 MSE 的 mean 再错误应用到 Taylor 点积上。下一步从已接受的**完整 d**继续。最多两步，找不到可接受步长就停止。

最后返回 `d_star.detach()`。不是把额外的 `-eta*g` 单独拿去替换已有残差。标量测试中 h=10、d₀=-2、目标输出=6，归一化步长为 1 时，完整目标依次是 -3、-4，最终输出是 7、6；不存在错误地退回到 9 的路径。

`autograd.grad(J, temporary_delta, create_graph=False)` 只求临时变量梯度，不填充模型参数 `.grad`；所有教师输入和最终标签 detach。模型参数在求标签时不发生更新，但后续 Wan 运算仍保留对临时 H 的导数。把整个 suffix 放在 no_grad 中会破坏 oracle，因此只有候选评分和参考前向使用 no_grad。

## 2. 后续网络到底是什么

快照在**当前 Wan block 执行后、当前 CORRECTOR 已算出 P′ 和 ΔH 时**采集。固定该处写前 H 和当前 P′；把 H+d 交给下一个 Wan block，重算后面所有 Wan block 和 CORRECTOR。

例如第 5 层：

```text
已有前缀 → Block5 → h5 → 当前交互模块 → U5, d5
                                     ↓ 固定 U5
                            h5 + 临时变量 d
                                     ↓
                  Block6…15 → 重新计算 U15/d15
                                     ↓
                  Block16…25 → 重新计算 U25/d25
                                     ↓
                          Block26…30 → 原生 head → velocity
```

不能从写前 H 再执行一次第 5 层 CORRECTOR，把当前修正重复计入；也不能跳过下游 CORRECTOR，却把得到的标签当作部署路径中的标签。CPU 重放测试覆盖三个位置，要求原始快照残差送入 suffix 后复现完整前向输出。

多处 Oracle 是在同一次参考前向上分别构造的。单处可验证改善不意味着三处标签同时替换、一次参数更新或整段去噪轨迹都必然改善。最终正常路径的 FM 仍然参与联合训练。

## 3. 唯一的新增外层损失

`L_Oracle = (1/|S|) * sum_l a_l * MSE_valid((delta_pred_l - stopgrad(d_star_l))/s_l)`。

S 为本次选中的位置集合，默认三个；a_l 表示是否至少接受过一步。失败位置贡献零，不用零 ΔH 伪装成成功标签，也不按成功个数缩小分母而放大其他位置。已知首 latent 对应的 H token 被写回 mask 排除；它们可提供上下文，但不能被 Oracle 修改来走捷径。

当前默认训练使用 `FM + 0.01*a*q*STRUCT + 60.0*a*Oracle`。`a=min((step+1)/200,1)`，`q=clip((1-sigma)/0.2,0,1)`；q 只沿用于 STRUCT。2026-09-18 初始运行的 `struct_weight=0.1`、`oracle_weight=1.0`。现按用户要求将两个辅助项校准到预热结束后各占 FM 约 5%～10%：以初始运行第 27 步数据静态重算，分别为 5.73% 和 5.66%；保留原搜索参数和 200 步预热。固定权重不能保证后续实际占比始终处于该范围，生成质量收益仍待验证。无 Oracle 对照同步采用 `struct_weight=0.01`。Oracle 根据实际改善筛选，不额外使用旧 WRITE 的 sigma 权重或扰动分支。Oracle 的尺度归一化使用 H RMS，不除以很小的“写回预算”，没有目标限幅或推理残差限幅。

求标签的正则是内层优化的一部分，不额外进入联合训练的标量和。三个损失一次正常 backward，没有梯度投影或隐藏的梯度比例限制。

从计算图看，FM 与 Oracle 可以训练初始化器、升维映射、各交互模块和 writer；STRUCT 训练 JEPA 解码头及产生对应 U 的真实前向路径，也可经后层 U→H 依赖训练前面的 writer。最后一个 writer 位于最后一个 STRUCT 状态之后，直接监督来自 FM 和 Oracle，这是预期的因果关系。

## 4. P 适应 H 的具体结构

1. 首图 latent 48 维和完整 UMT5 文本构造 1664 维 P₀；初始化器本身宽度也是 1664，两个 attention 块。所有时间槽使用真实位置，允许有不同输出。
2. 使用独立的 `Linear skip + LayerNorm→Linear→SiLU→Linear` 非线性映射，把 P₀ 升到 3072。隐藏层为 3072，残差分支最后一层零初始化，线性 skip 保留初始梯度路径。
3. 三个位置分别拥有两个双流 joint-attention 块。P/H 都是 3072，各自的归一化、QKV、输出投影和 FFN 不共享；Q/K 拼接后进行联合 attention，P 可以读 H，H 也可以读 P。坐标用于 Q/K 寻址，values 保留内容。每个 head 为 128 维，24 个 head 拼接仍是 3072。
4. 每个位置还让 P 读取完整文本 tokens；噪声条件和文本条件驱动 AdaLN 调制。内部注意力/FFN 的可学习残差门初始化为 0.1，避免内部全部关闭造成训练初期状态难以学习。
5. 交互后的 U 保持 3072 维传给下一位置。各位置有独立 `3072→1664` 非线性解码头做 STRUCT；不把该解码结果作为下一位置的窄状态，也不对 H 做 3072→1664 的读入压缩。
6. 交互后的 H 分支经 `LayerNorm→Linear(3072,3072)` 输出 ΔH，最后线性层零初始化。写回为 FP32 的 `h + delta`，防止小残差在 BF16 加法中被舍掉。零初始化时生成路径严格等于关闭 CORRECTOR 的基座路径。

Teacher letterbox 中完全无效的 P token 不进入联合 attention；有效 P/H token 保留各自时空坐标，不强行让两者 token 数相等。STRUCT 使用缓存的内容权重。

双流分别参数化、允许双向信息交换的架构思想可参考 [Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206)。这里把它用于 P/H，是本项目的设计选择，不是原论文对当前视频任务的效果证明。

## 5. 与 V4.2 的差异，明确列出

| 项目 | V4.3 |
|---|---|
| Wan、VAE、JEPA 与缓存 | 保留原资产及121帧/JEPA32数据定义，只读复用缓存 |
| 插入点 | 保留5/15/25，三个独立模块 |
| 交互宽度 | 从1664改为3072，让P适应H；不存在512瓶颈 |
| 初始化器 | 从原内部512改为1664；只读首图/文本，不再额外输入完整GT视频 |
| 跨位置状态 | 持续保持3072；每处独立解码到1664接受STRUCT |
| writer | 改为联合交互后H分支的3072→3072零初始化输出头，已知首片不写回 |
| 写回幅度机制 | 未沿用旧1664维writer的sigmoid门及固定sigma taper；保留内部可学习调制门、输出零初始化、全局grad clip；不添加幅度上限 |
| 损失 | 删除PRIOR及扰动WRITE，新增经验证的Flow-Oracle |
| 优化与数据曝光 | 800步/global batch8/峰值lr5e-5/100步LR warmup/200步辅助warmup/10%空文本；不把GT视频送入学生initializer |
| 推理 | P每次调用重置，两个CFG分支均运行CORRECTOR，不运行Oracle或JEPA教师 |

这些是明确记录的结构和训练变化。不要把与旧 V4.2 的结果差异全部归因于 Oracle：必须先比较**同结构**的 `flow_oracle.yaml` 与 `fm_struct_control.yaml`。两个配置只在 run 名称和 `oracle_weight` 上不同。

学习率由 V4.2 的 1e-4 调为首轮 5e-5，依据与基本训练参数逐项对照见 [training_alignment.md](training_alignment.md)。Oracle 的 relative_step=0.01 是 H 空间的内层搜索尺度，不是参数优化器学习率。

## 6. 可行性、成本与必须验证的内容

- 目标在同一层、同一 H 坐标下定义，匹配 actual writer 的职责，且用真实 GT-FM 评分；这比相邻层 hidden 差分更契合任务。
- 一步、很小步长、detach 标签的 MSE 回归，可能等价于对 FM 梯度重新缩放。归一化、mask 或“Oracle”命名都不会让它成为独立物理真值。两步优化及回溯只是提供经过有限步验证的目标，仍须证明比直接 FM 更有效。
- 目标随当前模型、噪声、sigma、上下文变化，是模型相关伪标签。teacher 冻结是指**单次求标签期间参数固定**，不是另有永久冻结的高质量模型。
- GT 优化出的 3072 维残差可能不容易由条件有限的学生预测，高噪声下尤其存在不可辨识性。要观察 target change、接受率、验证 FM、STRUCT 和完整视频，不能只看 Oracle loss 下降。
- 默认每个微批执行一次无梯度参考前向；依次求三处标签并释放内层图；随后执行一次带梯度正常训练前向。每处默认2次 suffix backward，以及每步1次求梯度前向和最多4次候选前向。相较 V4.2 可能显著更慢，推理仅去掉 Oracle 额外计算，较大的交互模块本身仍有成本。
- 多GPU使用FSDP Wan时，各rank必须有相同的suffix调用次数。代码同步接受/终止决定，只有所有rank的局部样本都通过才接受同一次步长；不允许不同rank各自提前break造成集合通信挂起。
- 本版本默认2.476B可训练参数，未缩窄模型来规避显存。单卡96GB的实际峰值必须运行真实检查；当前没有“已验证可装入96GB”的声明。

建议验收顺序：运行CPU检查；在96GB GPU运行指定完整121帧样本的真实probe；做一次保留完整优化器状态的训练短检查；相同初始化/数据计划比较有无Oracle；最后对固定案例生成完整121帧视频。Oracle改善只能先称为局部FM改善，不能换算物理正确率或保证自由生成质量。

目前CPU检查覆盖目标累计、有限步回溯、梯度隔离、全部插入点重放、结构/生成梯度路径、BF16/FP32接口、CFG采样首帧固定与原生尺寸。后续真实GPU、训练和视频效果结论见分析入口。原V4.2源文件哈希保存在 `../checks/v42_source_sha256.json`，并由验证脚本检查是否发生变化。
