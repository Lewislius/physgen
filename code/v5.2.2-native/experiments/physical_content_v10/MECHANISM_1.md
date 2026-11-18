# 机制1：原生语义关联定位的双路径物理写入

## 核心计算

沿用N1，不增加分段标签、global scene pool或video-to-C反复回读。
一条路径原样读取全宽物理记忆，另一条路径让原生text attention的同一组关联分配物理value。两条路径都只有物理value，没有额外纯文本writer。

直接读取：

    zd = softmax(Qp [Kc;k0]^T / sqrt(dh) + [b_c;0]) [Vc;0]

请求相关的物理value：

    Binding = softmax(Kt_valid Kc^T / sqrt(dh) + b_c)
    Vp_request = Binding @ Vc

原生语义位置关联：

    Atext = softmax(Qt Kt_full^T / sqrt(dh) + native_attention_mask)
    zs = Atext @ PadZero(Vp_request)

这里Atext随当前hidden_state、位置和去噪步骤改变；没有“前段只读A类、后段只读B类”的硬规则。它不是准确分割图，只是主干实际采用的语义关联。即使权重同源，也不能从公式推断对象对应一定准确。

    state = Wstate LN0(x)
    delta_d = Wd [zd + state*zd]
    delta_s = Ws [zs + state*zs]
    Hmix = Hsa + text_update + task_write_mask(delta_d+delta_s)
    Hout = native_FFN_residual(Hmix)

Wd/Ws是独立全宽3072x3072矩阵，无bias、zero-init；同层共享Wstate。与N1一致，不使用固定0.1缩放或条件可信度门。

## 主干信息参与在哪里

第一，Qt/Qp来自主干当前视觉状态，决定本位置的物理读取。
第二，Atext是原生文本attention的实际请求—视频关联，用来定位请求相关的物理内容。
第三，state由当前视觉状态决定，和物理消息形成已有的逐位置交互。
第四，融合后的视觉状态由主干原有FFN与后续self-attention执行；主干没有被另一个抽象物理video block替代。

这不是宣称主干信息已被写回Cr/Ce；本版本明确不增加这个回路。历史R12A已尝试回读，不能仅为模仿ProPhy图示而重复添加。

## 如何理解可能的收益与不足

PC：保留未先经请求平均的direct物理value；SA：物理写入同时利用主干正在使用的请求关联；Joint：希望同一个对象上请求与物理响应更协调。
不足：Atext可能定位不准，两个writer可能相互干扰，模型可能只使用公共物理偏置。正确physics与真实生成收益必须实测；不把“使用了文本attention”当成已解决SA。

## 成本和实现

相比单writer多一个3072x3072 writer，每处9,437,184参数，10处约94.37M。没有额外N×N视频attention，但增加N×L的文本关联加权和L×256绑定。
不要永久物化[B,24,8190,512] attention。可按query分块重算完整keys上的softmax并立即乘物理value；key分块不能各自softmax再相加。原生text forward不改，物理投送允许单独计算相同logits，梯度必须到H与Vp_request。
推理kernel若尝试拼接native/physical values同时计算，须先证明原生text输出等价；默认不要求实现这种优化。
