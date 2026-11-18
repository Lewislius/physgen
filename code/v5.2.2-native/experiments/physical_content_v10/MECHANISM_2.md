# 机制2：联合物理读取＋原生视觉特征条件适配

## 核心计算

保留N2计算，不附加之前准备中的三段描述、局部监督head或L_local。整个视频共用同一组全局Cr/Ce；不同位置通过当前视觉query和已有真实时空坐标读取不同内容。

先构造两类物理记忆：

    Vbound = softmax(Kt_valid Kc^T/sqrt(dh)+b_c) @ Vc
    Kj = concat(Kc, Kt_valid, k0)
    Vj = concat(Vc, Vbound, 0)
    z = softmax(Qp Kj^T/sqrt(dh)+b_joint) @ Vj

b_joint沿用B400+I：direct各组槽-log(n_g)-log2；bound各请求槽log(R/Lvalid)-log2；reference=0。这是关系先验质量守恒，不是把最终输出硬减半。直接+请求两份记忆不是两段时间内容。
Qp继续使用native Q，不新增R15I的独立巨大可训练Q。

全宽物理写入：

    direct = Wo [z + Wstate(LN0(x))*z]
    Hmix = Hsa + text_update + task_write_mask(direct)

原生FFN真实视觉激活：

    u = native_norm_and_diffusion_time(Hmix)
    f = GELU((W1_base + LoRA1)u + b1)
    y = (W2_base + LoRA2)f + b2

唯一新增条件执行：

    delta_ff = Bp [ (Ap f) * (Rp LN0(z_full)) ]
    Hout = Hmix + native_ffn_gate * (y + task_write_mask(delta_ff))

Ap:14336->64，Rp:3072->64，Bp:64->3072，均无bias。Ap/Rp随机非零，Bp=0。z_full是把同一次物理读取结果填回视频grid，不是新teacher状态。

## 主干信息参与在哪里

不是让抽象physics P独自经过复制的Wan FFN。f是真实主干在同一次forward里已有的14336维post-GELU视频特征，已包含当前视觉状态、请求、物理直写和原生diffusion-time。
物理z与这个实际视觉激活共同决定额外输出组合。后续原生self-attention负责时空交流。无需把Cr/Ce分成多个时间段，也不需要新的逐帧物理脚本。

此处rank64只限制额外adapter内部维度；3072维物理读写通路完整存在。与普通静态LoRA不同，adapter输出依赖本视频位置的物理z；与R13A不同，不替代direct，也不调制整个FFN输入；与R18不同，原生视觉FFN始终处理视觉状态，不处理孤立物理语言向量。

## 初始化和梯度

Bp=0时新增adapter为0；第一个反向Bp可更新，之后Ap/Rp/z可接收生成梯度。全部参数从step1在optimizer里，不人为添加冻结阶段。direct的Wo=0独立存在，而raw Ce含anchor，不形成整个物理网络永久零梯度。
不增加sigmoid物理门、范数配额或把严重伪影靠后处理平滑掉。仍有学习到错误更新或条件被忽略的风险，必须用真实生成判断。

## 成本

每处r*(14336+2*3072)=1,310,720；10处13,107,200。没有额外主干前向，没有额外N×N attention，没有复制FFN。f原本就需要计算，新增激活主要是rank64条件特征。实际显存和速度需GPU测量。

该版本优先研究完整物理内容怎样改变材质/边缘/几何/持续性相关视觉特征。不能仅因使用f就称其为准确的材质编码或保证物理定律被学会。
