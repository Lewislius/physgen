# 共同实施契约：官方初始化、1000步、有效batch=4

## 0. 本版的优先级与状态

本契约与各方案正文、对应YAML共同构成本次实现的唯一规格。它覆盖前面对话中关于R10A检查点续训、原生全参数解冻、数千步能力训练、偏好后训练的建议。

本次只交付设计与实施计划，不代表已完成Wan GPU训练或生成达标。目标是VideoPhy2固定40例同时PC≥3.85、进一步争取3.95，SA≥3.25，Joint≥14/40，并改善完整视频质量及WMB、Physics-IQ、VBench表现。目标不是已测结果。

四个候选独立实现、独立训练、独立报告，每个候选最多1000次optimizer.step，有效全局batch固定4。若全部运行，合计是四次独立实验，不是把四个模型串联训练成4000步的一个模型。

### 0.1 “从头”准确含义

- Wan从官方Wan2.2-TI2V-5B预训练权重初始化，不是把Wan随机初始化。
- Qwen/T5/VAE使用官方冻结权重或由这些权重生成的输入缓存。
- 所有项目新增可训练权重在本次run新建；允许从官方Wan复制/引用冻结算子作为新模块的基础，再新建LoRA。
- 禁止加载R1/R10A/R15/R17或任何其他项目生成训练checkpoint的model、LoRA、optimizer、EMA、scheduler或RNG。
- 禁止先合并某个R版本LoRA再称为“官方底座”。
- 本run中断恢复允许，但必须恢复相同run、相同配置、相同初始化清单和已完成步数，1000步上限不重置。
- 训练输入的闭式统计校准不是项目模型checkpoint：应使用当前train split重新计算并记录来源，不能不加说明地读取旧实验interface.pt。

### 0.2 “Wan全训1000”准确含义

Wan视频self-attention和FFN上的新LoRA在第1—1000次更新全部处于可训练状态。不是全参数训练，也没有前128/256步冻结Wan LoRA的阶段。

原生文本cross-attention Q/K/V/O、text_embedding、T5、Qwen、VAE、patch/head/time原生参数始终冻结。主干视频Q/K/V/O和FFN的base矩阵也冻结，仅更新LoRA。新模块中明示为dense的参数正常全矩阵训练。

“可训练1000步”不等于每个参数在第1步必须获得非零梯度：标准LoRA B=0时A在首个反向可为零梯度，新输出头为零时内部路径首步也可能暂时为零。需要验证输出打开后梯度可达，不能据此错误追加随机扰动或解冻base。

### 0.3 本次明确不做

不做DPO/Flow-DPO/RL/奖励模型优化、偏好后训练、训练外额外预训练、额外自蒸馏rollout；不使用V-JEPA等新视觉教师；不使用动作标签/动作约束；不使用评测集目标视频训练；不使用推理best-of-N挑选作为主结果；不开放原生文本Q/O LoRA。

默认不加入时间差分loss、RGB重建loss、C多样性、正交、attention熵、残差范数、专家负载均衡等附加目标。不要把“预留配置”变成默认启用项。

## 1. 本轮相对于历史设计的实质修订

R17已经使用了主干视频LoRA；所以“加LoRA”本身不是未经尝试的突破。这里保留LoRA，是为了让官方生成运算能与新Expert共同适配。改变的是Expert的计算结构、semantic作用位置，以及短预算下如何利用官方全宽算子。

以前A/C/D提出的多个完整新执行块若全部作为新dense参数训练，在4000次样本曝光下风险很大。本版保留完整3072宽度和14336宽度FFN，但其中复制自官方的视频运算采用冻结全宽base＋本run独立LoRA。它仍执行全宽attention/FFN，不是3072→512→3072的唯一信息瓶颈。物理读取K/V、最终写入、Q/C增强以及D的条件映射继续拥有全宽dense可训练能力。

B不再开放六个原生block的完整参数，而改为主干内LoRA协同执行；这是对旧B的明确替换，不同时保留全参数版本。

## 2. 主干与LoRA的唯一规格

官方结构以本地实际config核实：d=3072，ffn_dim=14336，30层，24头，每头128，patch_size=(1,2,2)。不从README中的“5B”反推模块形状。

主干LoRA目标严格只有：

```python
blocks[i].self_attn.q
blocks[i].self_attn.k
blocks[i].self_attn.v
blocks[i].self_attn.o
blocks[i].ffn[0]
blocks[i].ffn[2]
```

必须检查实际ffn结构并按Linear层定位，不能用字符串“所有q/o”之类宽泛匹配。

A/C/D：所有30层rank=32、alpha=32。B：注入的10层rank=64、alpha=64；其余20层rank=32、alpha=32。LoRA dropout=0。

每个线性层：

\[
W_{eff}x=W_0x+\frac{\alpha}{r}B(Ax).
\]

W0来自官方且冻结；A使用Kaiming初始化；B=0。基础bias冻结。禁止LoRA A与B同时清零。

主干LoRA主要参数：rank32全30层=57,016,320；B混合rank=76,021,760。这些是矩阵形状计算，不是生成收益证明。rank64不是默认越大越好，B只是把更多适配能力放在负责执行条件的原生位置。

### 2.1 新Expert对官方算子的使用

A/C/D的每个物理时空执行器均有独立的expert LoRA，rank32、alpha32，作用于自己的视频Q/K/V/O和FFN两层。新语义读取器A/B拥有独立cross-attention Q/K/V/O LoRA，rank32。这里的语义读取参数属于新模块，不属于wan.blocks.*.cross_attn。

允许共享官方冻结base存储，但不共享主干与Expert的LoRA。推荐在安装主干LoRA前建立官方权重注册表。Expert按官方参数键引用base，显式使用自己的adapter。严禁Expert调用已经安装主干LoRA的整个Linear，从而隐式混入主干增量。

可以采用`FrozenBaseRefLinear.forward(x, base_bank)`：注册表只含requires_grad=False的官方W/b，模块自身只注册本模块A/B。保存时保存官方模型revision/hash与新参数；恢复时先装载同一官方base，再重新绑定引用。

若实现选择物理复制官方矩阵，必须仍冻结复制base，且如实报告额外内存、独立参数存储和推理计算；不能宣称与共享实现相同的唯一参数量。Expert增加额外前向运算，即使共享权重也不是零计算开销。

## 3. 数据、条件和张量契约

### 3.1 默认I2V输入

- RGB片段：历史协议81帧、480×832；不得擅自改成官方默认121帧。
- latent：通常[B,48,21,30,52]，必须以本地VAE/cache元数据验证，不硬编码替代验证。
- patch后grid：(21,15,26)，N=8190，已知首帧token数K=390；其他尺寸按grid动态计算。
- text为冻结T5输出，实际有效长度包含EOS；通过原`native_text_context`形成[B,512,3072]上下文。
- 原生文本前向保留官方padding处理；新语义attention可以mask人工padding，但不能mask掉EOS/null合法token。
- Qwen条件缓存仅来自首帧＋原请求；读取scope并要求`first_image_and_request_only`。不能用未来视频编码的“第一个token”假冒首帧条件。
- 文本、帧率、时长、首帧、Qwen正负条件必须彼此匹配。

### 3.2 两种时间不能混淆

`sigma`是去噪噪声时间，未来视频token使用sigma×1000，已知首帧使用0。

`video_times`是片段呈现时间/PTS。它用于数据、报告与实际视频时长，不能被当成sigma。RoPE使用真实时空grid。物理槽编号不具有视频空间坐标，不给256个槽随意添加视频RoPE。

### 3.3 数据分割与曝光

沿用当前已审计的训练cache，不要求本轮额外获取20K数据。1000×4恰好4000次视频样本曝光。优先按seed42洗牌、整轮无放回遍历train，轮末再洗牌；若N=3857，前4000次覆盖一轮再加143次，不是“每次随机抽四个”导致大量样本未被看到。

禁止把VideoPhy固定40/60、WMB、Physics-IQ的测试目标视频或评分写入训练cache、loss、采样权重、reasoning。数据清单保存source_id并排查重复泄漏。已有train/validation split不能按40例评分临时重切。

本轮不自动改写caption。发现训练片段/描述不一致可在训练前统一修正并冻结新的data_revision，四版一致使用；禁止每个候选单独人工优化测试提示词。

## 4. 物理内容生成：复用结构，不复用R10A权重

使用历史R10A增强结构作为可读源码起点：Qwen宽度4096、Q数256、32头（16关系内＋16全局）、两轮共享增强、分组交互和route/execute分工。所有learnable Q、ContentBlock、投影和offset在当前run新建，不加载旧model tensors。

\[
C_r=A+b+P_r Z,\qquad C_e=A+P_e U.
\]

两者均[B,256,3072]。U/Z是已有增强模块计算得到的张量，不额外命名为真实物理状态。Pr/Pe可沿用历史零初始化；A的存在使初始物理记忆并非全零。输出头打开后检查Pe及Q增强能获得生成梯度。

### 4.1 训练输入闭式接口重新生成

采用已核实的`physical_content_v7/fit_interface.py`数学形式，但更改输出目录为本run准备目录，不覆盖旧文件：

\[
\bar X=(X-\mu_X)/s_X,\quad Y_c=Y-\mu_Y,
\]
\[
G=\bar X^T\bar X/n,\quad H=\bar X^TY_c/n,
\]
\[
\lambda=0.01\,mean(diag(G)),\quad W=(G+\lambda I)^{-1}H.
\]

X来自train侧官方Qwen逐词特征，Y来自对应官方原生文本特征；保存mean、全局scalar scale、weight、bias、target_scale，形状完全沿用历史接口。统计不使用test数据。它是闭式数据校准、没有optimizer.step；需单独报告使用的train IDs/hash与prep成本。不要声称“完全没使用数据做初始化”。

若用户另行禁止一切训练输入校准，必须单独修订输入投影方案；不得隐瞒读取旧interface.pt。当前默认是重新做上述明确、无梯度的校准。

## 5. 公共全宽物理读取器

新读取器在完整视频token上读取物理C，包括作为条件的首帧位置；最终新增写入仅作用于未来token。允许首帧信息参与内部关系计算，不等于修改已知输入图像。

1. Q使用当前对应block官方冻结cross-attention Q及norm_q；输入先经独立非仿射LN。它不调用原生文本LoRA，因为原生文本LoRA根本不存在。
2. K/V为两个全宽dense可训练矩阵，初值拷贝官方cross-attention的Wk/Wv，bias=False；K-normalization复制官方norm_k，可作为新模块参数训练。
3. 由Cr生成K、Ce生成V。
4. 保留等关系数量归一先验`prior[j]=-log(number_of_slots_in_group[j])`。只为消除同关系重复槽数量偏差，不设物理强度门。
5. 本新reader不添加零value reference槽；所有memory均为实际物理内容。
6. attention输出通过官方冻结cross-attention Wo的**无bias权重**转到视频residual坐标。该变换不是最终写入头；它不能零初始化。
7. 不额外安装旧`state_value`乘法分支。A/C/D已有完整物理执行器；B由原生FFN与视频LoRA执行。

\[
P=O_{official}\left[softmax\left(QK^T/\sqrt{128}+b_{group}\right)V\right].
\]

此处P是“物理消息特征”，不是可测力场、运动或真实物理量。新增最终writer是3072×3072全宽dense矩阵，bias=False，独立零初始化，所有候选正常推理倍率1。

### 5.1 两种attention的实现区别

物理读取是N×256，可以chunk查询并保持相同softmax维度；显式float32 logits是允许的。空间时间self-attention是N×N，N=8190、24头时不得物化完整概率矩阵；必须复用可微FlashAttention/高效SDPA。所有heads沿同一memory集合归一，不能按chunk对key分块各自softmax再相加。

## 6. 训练步骤与损失：只有一个1000步训练过程

四版均从第1步开始共同训练所有允许参数。没有先单独训练Expert几十步再打开Wan的阶段，没有在1000步之后精修。

| 更新号（1-based） | 参数归属 | 学习率／目标变化 |
|---|---|---|
| 1—64 | Wan视频LoRA＋新物理模块全部active | 线性warmup；route辅助升起后保持 |
| 65—200 | 同上，无解冻变更 | peak LR；route辅助逐步退出 |
| 201—320 | 同上 | cosine LR开始；route继续衰减 |
| 321—1000 | 同上 | 只有真实视频生成目标；LR继续cosine |

这是一套优化日程，不是不同训练任务。所有步骤只有真实视频监督，不引入自动评价器梯度。

### 6.1 Flow Matching数学约定

未来latent：

\[
z_\sigma=(1-\sigma)z_0+\sigma\epsilon,\quad
\sigma\sim U(0.001,0.999),\quad \epsilon\sim\mathcal N(0,I).
\]

I2V前向实际使用`cat(first_clean, noisy_future)`，已知首帧不加噪、不计算生成loss。

\[
v^*=\epsilon-z_0.
\]

对未来有效区域先做channel mean，得到每个时空位置误差e。FM是有效区域的加权平均，不包括首帧和无效padding。

### 6.2 主生成损失

\[
L_{gen}=0.8L_{FM}+0.2L_{tail}.
\]

沿用已审查实现的四个pool窗口：

```python
windows = [
 (1, H, W),
 (min(4, F_future), H, W),
 (min(4, F_future), max(1,H//5), max(1,W//4)),
 (F_future, max(1,H//5), max(1,W//4)),
]
```

每个窗口用误差×valid与valid同时avg_pool3d（stride=window，ceil_mode=True），相除得到有效平均；在支持非零的窗口中取top ceil(25%×窗口数)，求均值，再对四种尺度求均值。空有效区域必须报错，不能伪造零loss。

高误差聚合不是“物理loss”，也不能直接监督SA或Joint。它只是现有真实视频误差的聚合方式，本轮不把它改名包装成额外能力。

### 6.3 route辅助只作用于原始Cr

\[
L_{route}=mean[(C_r-C_{native})^2]/max(s_{target}^2,10^{-8}).
\]

使用以下唯一系数（s为1-based optimizer update）：

\[
\lambda_r(s)=\begin{cases}
0.03\min(s/32,1),&s\le64,\\
0.03(320-s)/(320-64),&64<s<320,\\
0,&s\ge320.
\end{cases}
\]

总loss：`Lgen + lambda_route * Lroute`。C方案只约束语义增强前Cr，不约束C_tilde_execute或C_tilde_route。其他方案也不约束Ce。

target `native_words`只能进入loss侧，不得混入生成条件。lambda为0时可以不加载target，但正负condition读取仍一致。不得使用测试集target_scale。

### 6.4 实际batch=4

默认单GPU microbatch1、grad_accum4，loss先除4，四次backward后clip一次、optimizer.step一次、scheduler推进一次。

允许2GPU×micro1×accum2或4GPU×micro1×accum1，但`world_size*microbatch*accumulation==4`。DDP需正确处理no_sync和梯度平均，不能又除world_size造成额外缩放。

1000更新恰好4000个训练视频条件；不得用两个view、额外正负训练前向、rollout或偏好对偷偷扩大每更新数据预算。CFG训练通过10%条件dropout单前向完成，不每条训练样本都计算CFG双臂。

### 6.5 学习率与优化器

AdamW，betas=(0.9,0.999)，eps=1e-8，矩阵weight_decay=0.01；bias/norm参数decay0。trainable参数用FP32存储/优化器状态，前向BF16 autocast；norm、softmax统计、loss与residual关键加法使用FP32。

| 参数组 | peak LR |
|---|---:|
| Wan视频LoRA | 2e-5 |
| Q/C增强 | 3e-5 |
| 物理reader dense K/V及新norm | 2e-5 |
| 全宽最终writer | 3e-5 |
| 新Expert内部LoRA | 2e-5 |
| A/B独立语义读取LoRA | 2e-5 |
| C记忆增强LoRA与route映射 | 2e-5 |
| D语义执行映射 | 2e-5 |

统一multiplier：s≤64时s/64；65≤s≤200时1；s>200时`0.2+0.8*0.5*(1+cos(pi*(s-200)/800))`。第1000步仍为0.2×peak，结束后不多执行一步。

按上述角色分别clip_grad_norm_(1.0)，先完成全部4个micro，再clip。记录preclip norm与缩放比。clip属于优化器数值处理，不限制推理物理幅度。参数必须恰好进入一个优化器组；共享冻结base不能进入任何组。

## 7. 条件dropout、CFG与采样

训练10%概率同时使用null文本与对应null Qwen物理条件。不能只drop T5而仍用含正请求的physics缓存。C的语义增强器也必须使用同一null上下文；D直接复用该臂的原生文本更新。

推理两臂分别计算所有条件：`v = v_neg + 5*(v_pos-v_neg)`。这次不增加独立physics CFG三臂或第二个可调gamma。若旧runner具有text_guidance与physical_guidance双字段，要验证该设置下与上述公式严格一致，而不是默认它们可互换。

固定主协议：50步，shift5，shifted Euler，seed42；noise tensor持久化hash。每步重新覆盖已知首帧latent。C可在推理每个正/负条件开始时缓存增强后的记忆；训练时不要detach记忆，也不要跨optimizer更新复用计算图。D的当前文本消息必须逐层逐噪声步重新获得，不能缓存整条采样轨迹。

## 8. 初始化与梯度审计

只有最终write矩阵为零；新expert的官方base非零，内部LoRA A随机/B零。C的记忆增强没有额外零输出门，Wr为identity初始化；D的语义q/k/g矩阵可零初始化，因为它们是加性条件项/1+调制，不是串联全零value路径。

初始化应满足：新writer=0且主干LoRA B=0时，所有候选与官方同条件I2V前向一致。不能拿R10A输出作初始化等价目标。

首步合理梯度：最终writer和主干LoRA B可有梯度；内部Expert LoRA、Q/C执行投影可能需writer首次更新后才获得生成梯度。应运行至少3个toy优化步验证可达性，而不是强求第1步所有参数都有梯度。

官方主干base冻结不意味着可用`torch.no_grad()`包住Wan前向。那会截断新条件到最终预测的梯度。只对冻结输入编码/缓存读取使用no_grad；整个生成计算保留autograd。

## 9. 推荐代码模块和调用契约

新增独立目录建议：`experiments/physics_expert_abcd_fresh1000/`。这是待创建路径，不是声称仓库已有实现。不要覆盖旧R版本。

```text
model.py                 # 构造官方Wan、各方案、共同velocity
base_bank.py             # 官方冻结参数引用与可核验加载
lora.py                  # 视频LoRA、Expert独立LoRA、拒绝原生文本适配
content.py               # fresh Q/C增强、C专属记忆增强
readers.py               # 全宽物理reader、新语义reader
experts.py               # A/C/D执行器、B in-block融合
training.py              # 唯一1000-step循环
objectives.py            # FM、tail、raw-route辅助
schedule.py              # 本契约的解析日程
checkpointing.py         # fresh与same-run-resume分开
inference.py             # 同一forward + CFG2臂
reports.py               # PC/SA/Joint、逐例结果、质量审查
configs/{a,b,c,d}.yaml
```

核心接口：

```python
content, groups = model.enhancement(evidence)  # raw route, execute
prediction = model.velocity(
    noisy_latent, first_latent, text_tokens, sigma,
    content=content, groups=groups,
    video_times=video_times, valid=valid,
)
loss = generation_loss(prediction, noise, clean, valid)
if lambda_route(step) > 0:
    loss += lambda_route(step) * raw_route_loss(content['routing'], route_target)
```

主干只遍历30个block一次。B/D需要当前层text_update，重构一个显式`native_block_parts`方法返回它；不得先调用完整block再重复运行self-attention或text-attention来“取中间量”。

activation checkpoint使用non-reentrant；闭包绑定layer/index，所有有梯度条件张量显式传入。不能在forward中修改全局adapter角色或跨层可变缓存。为LoRA dropout0保持推理与checkpoint重算一致。

## 10. 保存、恢复和可追溯性

每100更新保存latest；保留step0100/0250/0500/0750/1000，以及异常前最后完整step。保存：方案ID、schema版本、全配置、trainable参数、必要新buffer、optimizer、scheduler step、sample cursor、Python/NumPy/Torch CPU/CUDA RNG、官方revision/hash、接口拟合hash、train manifest hash、LoRA target map与rank、参数计数。

fresh和resume必须是不同入口。fresh禁止`project_checkpoint`非null；resume要求run_id、config hash、param key/shape和completed step一致。resume到1000时立即结束，不再更新。

不能用strict=False吞掉初始化或恢复的缺失键。新增run不载入旧优化器。不要将旧R10A的400步计入本run后再只训练600步；本run从0到1000，与旧run无继承关系。

## 11. 最终评测和目标

固定40为开发集目标，不冒充独立泛化证据：PC≥3.85、挑战3.95；SA≥3.25；Joint≥14/40（同一视频SA≥4且PC≥4）。SA/PC五分均值、通过率、Joint计数分开报告。

最终step1000固定生成40和不重叠20，60=40+20，不能重新生成40挑更好的一次。记录各例原seed/noise/首帧/caption一致性。看原强样本和新增成功样本的完整视频，不仅看新Joint数量。

外部benchmark：先用当前固定WMB70、Physics-IQ20场景×3视角作为回归；完整竞争扩展官方套件，注明版本、输入模态、prompt、judge版本。Physics-IQ三个视角按场景聚合，不作为60个独立场景。VBench与实际视频质量一起报告。不能把I2V40均值直接当成超过ProPhy的T2V600通过率。

大色块/雪花/横线等明显崩坏要求所报告集合零观测。失败样本不删除、不换种子补成绩。全40自动分数不能保证画质，需要完整视频审查。用户目标是研发验收线，不是本计划已保证的分数。

## 12. 编码AI必须通过的最小检查

1. fresh加载来源仅官方及本次train输入统计；检测禁止项目checkpoint路径/权重hash。
2. 所有原生cross_attn参数冻结且没有LoRA；官方base参数更新前后逐一hash相同。
3. 主干视频LoRA从step1到1000参数归属不变，step1000后不能再optimizer.step。
4. 主干LoRA与Expert LoRA没有对象别名；共享仅发生在冻结base。
5. zero-writer初始化与官方输出一致，恢复后同输入可重复。
6. 三步toy反向证明生成梯度到reader、内容增强、Expert与主干LoRA；首步zero梯度按上述预期判断。
7. 4个micro累积等价于显式batch4均值，scheduler只更新一次。
8. FM符号正确；first frame损失为0；valid padding无贡献；tail与历史公式一致。
9. 正/负CFG条件完全分开；C cache不跨更新；D text_update来自正确层正确臂。
10. C执行value和route都受语义影响；D在输入P全0时新增输出为0。
11. 未重复执行主干block；大分辨率用高效attention，不能CPU toy通过后直接N²物化上GPU。
12. 恢复step/cursor/RNG后与连续训练一致；报告不能从FM低值推导视频达标。

这些是正确性检查，不应扩展成取代训练的大型验证项目；先确保不会跑错模型，再完成本候选1000更新与实际视频评测。
