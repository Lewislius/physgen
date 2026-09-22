**P0、状态门与 WRITE：旧版实现与诊断记录**

依据：无 PRIOR 的 checkpoint-0000300、experiment_62293_trial_63458_logs.txt、对应结构化日志、当前 physgen_v4 实现。本文保留旧版实现与日志事实；当前改造采用[原生 Wan 物理监督方案](native_wan_physics_implementation.md)，尚未接入正式训练。

**一、完整视频应该怎样教会 P0**

完整视频作为训练期教师信息是合理的。目标是让只看首帧和文本的 P0 学会预测合理的过程初态；不能把“训练中出现 GT 视频”本身认定为错误。相关方法背景是 privileged information / generalized distillation：https://arxiv.org/abs/1511.03643 。

当前实现却是同一个初始化器随机有/无 GT：前 300 步中 227 步有 GT、73 步无 GT，没有在同一样本上显式匹配两种 P0。GT 分支学好，不等于无 GT 分支自动学会同等能力。

**二、状态门不是写回门**

当前每个 CORRECTOR 的内部 iterations=1。A@5、A@15、A@25 是不同网络深度的三处修正，不是三个采样时间步。

P_next = P + eta(sigma) * state_gate * delta_P，其中 eta <= 0.2。

state_gate=1 表示允许该候选增量通过；不表示覆盖 100% 的 P，更不表示把 100% 的 H 替换。

| 统计窗口 | A@5 状态门 | A@15 状态门 | A@25 状态门 |
|---|---:|---:|---:|
| step1–25 | 0.576260 | 0.578401 | 0.793680 |
| step76–100 | 0.995849 | 0.350750 | 0.002290 |
| step176–200 | 0.996719 | 0.000326 | 0.000000843 |
| step276–300 | 0.994225 | 0.000721 | 0.000001279 |

最后一个窗口三个 write_gate 分别为 1.000000、0.999735、0.995152；三个实际 P 增量 RMS 分别约 0.23556、0.00011471、0.000000199。后两层几乎不更新 P，却仍向 H 写入。P03 推理三处 ΔH/H 的 RMS 比约为 19.28%、15.97%、10.70%。这些比值不是内容错误率。

代码支持的机制解释：三个状态都拟合同一个 JEPA 目标；前层更新影响所有后续状态损失，后层发现自己的候选增量不能继续降低损失时，可以通过关门退出。sigmoid 接近 0 后梯度也很小，退出状态容易持续。记录证明门是训练中饱和的，不是初始就被指定为 1/0。上述是合理的优化机制解释，尚不是对具体训练路径的完整因果证明；V4 原版也出现后层状态门接近零，因此不能将它单独认定为新版闪烁根因。

**三、读取和写回投影的训练与参数量**

H 的读取路径是 LayerNorm(3072) → Site.read = Linear(3072, 1664, bias=True)。输出路径是 Site.writer = Linear(1664, 3072, bias=False)，后者零初始化。set_stage('A') 开启全部 A 参数，parameter_groups 将它们交给 AdamW。读取投影参与 FM/STRUCT，现有 WRITE 重新计算检索 query 时也会训练它；正常 FM 可穿过冻结 Wan 对 writer 反传，额外 WRITE 分支也训练 writer。冻结 Wan 权重不等于禁止对 Wan 输入求梯度。

- 每处输出投影：1664 × 3072 = 5,111,808 参数。
- 三处独立输出投影：15,335,424 参数。
- 每处读取投影：3072 × 1664 + 1664 = 5,113,472 参数；三处共 15,340,416。
- 两个方向的六个线性层合计 30,675,840 参数，尚未包含 LayerNorm、门和注意力内部投影。
- 若还计入该处 write_attention、write_norm、write_gate：每处 16,205,697 参数；此口径不包括共享用于读取的 site.read、sigma 等部分。

读取投影服务于旁路交互，原始 H 在 Wan 主干中仍保留 3072 维。read 与 writer 也不是必须互逆的编码/解码矩阵：writer 接收从 P 检索出的内容，而非简单还原 read(H)。注意力内部 Q/K/V/O 投影同样可训练。

step300 三处 writer 权重 L2 范数约为 2.426、2.643、2.239，已从零初始化发生更新。维度正确和参与训练，不代表已经学得正确的语义变换。

**四、当前 WRITE 监督的确切含义**

正常加噪训练输入产生 reference；额外施加小幅空间 warp 和平滑通道漂移的输入产生 disturbed 分支。两支保持同样的 P0、文本、sigma、坐标。

L_old = mean_sites MSE(H_disturbed + W(H_disturbed, P_disturbed), stopgrad(H_reference + W(H_reference, P_reference)))。

目标、扰动分支的 H/P 均 detach。只重新计算实际写回路径得到 WRITE 梯度；正常 FM/STRUCT 的完整图仍在。首帧对应 token 不计入该损失，三个位置的原生 3072 维 H-MSE 取平均。

当前总目标为 FM + 0.1*w*q*STRUCT + lambda_prior*w*PRIOR + 1.0*w*q*WRITE；lambda_prior 为 0.02 或 0，w=min(1,(step+1)/200)，q=clamp((1-sigma)/0.2,0,1)。

它监督的是扰动前后一致性。reference 自身包含当前模型写回，不能提供独立的“正确写入内容”标签。两输入相同时，任意 W 都能使损失为零；现有 mse_gain 甚至可以等于写回能量 ||W||²，不能当作重建改善证据。

step276–300 平均加权 FM/STRUCT/WRITE 约 0.28262/0.14179/0.002727。step100–300 九次单 micro 的 WRITE/FM 梯度范数比，在三个完整 CORRECTOR 参数组上平均约 2.8%/3.0%/3.1%。这不支持“WRITE=1 已压倒 FM”的说法，也不能用 loss 数值大小直接判断贡献。首要问题是正确性依据不足。
