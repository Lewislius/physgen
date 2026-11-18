# I1｜图生视频：首帧条件下的语义定位物理写入

版本：NoChunks-v2-20261111。状态：候选实施规格，尚无真实训练结果。

## 方案边界

本版保留原项目明确的I2V输入：固定首图＋原请求。读取、增强与执行可以利用由首图参与形成的主干视觉状态，但不会把首图纹理直接复制到所有未来帧。
首帧latent和token timestep按官方I2V边界处理；新增物理残差只写未来位置，原生主干仍完整处理整个视频。
这是原VideoPhy2固定40图生视频目标的直接竞争版本。不能用另一个纯T2V模型的成绩替代它的验收。

本版继承N1的双writer机制，新增物理内容来源仍为Ce，无纯文本残差，也无新的局部标签监督。


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


---

# 共同实施契约：不分段、不改造噪声策略

## 0. 状态、版本及优先级

这是可供编码的候选规格，不是已通过性能验收的最终模型。四个实验是两个机制各自对应T2V/I2V任务，不是四套不断叠加的新网络。
本版覆盖此前准备中的`PhysGen_*_T2V_I2V_Design.md`同名方案的局部分段监督内容；这些旧文档中的三时间段、L_local、局部target缓存、辅助head均不再使用。
它保留已交付N1/N2的核心计算，不把每次讨论都改成全新架构。该版本只固定两类机制、模态边界、评测标准。

共同条件：官方Wan2.2-TI2V-5B；官方冻结Qwen/T5/VAE；本run新建项目参数和视频LoRA；不加载任何R系列训练checkpoint；每版最多1000 optimizer updates、effective batch4；主干视频LoRA从第1步训练；原生文本Q/K/V/O与norm冻结；无ST、无后训练、无V-JEPA等新教师。

删除项：固定时间段物理描述、分段memory、分段attention硬路由、局部描述生成器、额外局部对齐loss、全视频“复制Expert”、额外视觉teacher、动态物理强度预算、按评价结果修改提示、推理best-of-N。

保留的是完整时空视频主干。物理消息依赖当前视频token，token含原生空间/时间信息；不分段不意味着每帧读到同样的内容。

## 1. T2V/I2V只按各自标准任务入口实现

同一TI2V-5B基础模型支持两条官方路径。此处不发明新的加噪分布、sigma重权、分段noise schedule、桥接扩散或噪声curriculum。
共同目标仍为z_sigma=(1-sigma)z0+sigma*epsilon，v*=epsilon-z0。已有sigma采样/主比较采样器参数继续保持。

T2V：输入仅原文本；没有外部已知首帧。所有视频位置都是待生成位置，也包括第一个latent时间位置。训练时它们正常参与同一FM目标，推理从完整视频噪声开始。不允许通过默认first参数偷偷拼入真实首图。
I2V：输入首图I0+原请求。首帧latent是硬条件，采用官方I2V token timestep与采样覆盖逻辑；FM只监督未知未来区域。所有新增物理残差只写未来位置。首帧可正常通过原生self-attention参与后续生成。

以上差别定义任务，并不是预期提高PC的新方法。T2V去掉首图但仍保留I2V首帧硬拼接/首帧零timestep/未来-only loss，是错误实现。

用一个通用unknown_mask表示未知位置：T2V所有有效位置为1；I2V首帧为0、未来有效位置为1。不引入学习到的mask，不从测试视频提取物理区域mask。
原generation_loss硬编码prediction[:,:,1:]。实现时添加显式task/known_frames参数并与旧I2V结果数值对齐；T2V包含第一帧的同一聚合，不能仅在YAML加标志而不改函数。

## 2. 数据与条件来源

真实训练视频/latent仍取现有约3857条，实际train数量以manifest为准。没有正负视频对、物理正确性标签、动作标注、区域框、光流或分段描述。

I2V可复用当前合规的input-only Qwen缓存：scope='first_image_and_request_only'。正常与null缓存都保留同一首图；null丢弃文本与其派生physics，不能偷用positive memory。
T2V必须单独建立scope='request_only'的正常与null Qwen缓存。Qwen只处理文本，不输入训练首图/未来帧。不是在已有看图缓存上删除图片字段：这种缓存已经含图像信息，不能复用。
T2V允许使用目标完整视频作为FM训练target，这是正常监督；它不能通过Qwen条件、首图分支或planner提前泄漏给生成器。

每种模态的训练输入坐标拟合接口单独由train split拟合，记录数据hash。T2V/I2V共用真实视频不等于可共用模态不同的物理特征缓存或interface统计。
图文不匹配、时间戳、裁剪、首帧/视频错位的检查在训练集进行，不依据测试40分数修改内容。冻结固定40首图与原caption；任务不可能/首图冲突样本不偷偷剔除，而是另外报告。

Qwen物理文本覆盖请求相关的物体/材料、关系、可见条件与预期物理现象，不强制一定出现动作，不把任意四个关系组解释为四个时间阶段。对未知质量、温度、速度不编造数值。

## 3. 物理增强和内存

保留R10A形状契约：Q为256x4096，32头，其中关系内/全局各16头；两次共享Q更新，关系内交互和跨关系摘要；输出Cr/Ce均为256x3072。

    Cr = A + b + Pr Z
    Ce = A + Pe U

A是本run训练输入拟合的原生语义anchor；U是两轮Q更新；Z是精炼Q从当前可用request/reasoning词读取的内容。符号只表示已有计算，不增加物理置信度、能量预算或动作状态。

Pr、Pe和shared b本run新建，沿已有零初始化约定；Ce有非零anchor。route MSE只直接约束raw Cr，Ce由生成目标学习。不新增Ce词向量对齐，不对所有槽强制不同/正交，不把Q槽编号当视频坐标。

## 4. 原生block及坐标

所有block的原生self-attention/text attention/FFN按原计算执行。额外注入每三层一次，0-based索引[2,5,8,11,14,17,20,23,26,29]。不复制额外完整video self-attention/FFN。

设Hsa为原生self-attention后的residual，x=block.norm3(Hsa)。

    Qt = norm_q(Wq_native x)
    Kt = norm_k(Wk_native T_native)
    Vt = Wv_native T_native
    text_update = Wo_native Attention(Qt,Kt,Vt)

这里T_native是官方T5经Wan文本投影后的context，不能把原始4096维T5和3072维native context混用。

物理reader：

    Qp = norm_q(Wq_native x + E_video_time(times))
    Kc = norm_k_phys(Wk_phys Cr)
    Vc = Wv_phys Ce

物理K/V独立全宽dense、从官方对应weight复制初始化；物理key norm独立可训练；Q仍为冻结native Q。E_video_time编码presentation time与相邻帧间隔，不是diffusion sigma。sigma继续走官方block时间调制。

z=head-concat后的物理attention value消息；写入前保留原状态交互：

    Fstate(x,z) = z + Wstate(LN0(x))*z

Wstate无bias、zero-init；writer无bias、zero-init，不加sigmoid强度门或固定上限。原生FFN bias/time scale/shift/gate全部保留。

T2V物理query覆盖所有视频位置。I2V物理query和write只覆盖未来位置；把消息按正确grid填回时首帧为0。不得在T2V分支继续使用known=Hgrid*Wgrid隐式去掉第一帧。

## 5. 主干与参数归属

30层self_attn.q/k/v/o及FFN两个Linear安装新rank32、alpha32、dropout0 LoRA。A随机、B=0；官方base冻结。text cross-attention及text_embedding不加LoRA、不解冻。
原生文本参数冻结不意味着把其forward放在no_grad：梯度必须能经视频query到达主干video LoRA。只有真正的输入缓存生成器可no_grad。
物理reader的K/V、state/time、writers、Q增强为可训练；所有新增adapter按机制清单明确列出。不得广泛字符串匹配解冻模块。

I2V首帧硬覆盖只是条件边界；不能推断其余画面一定稳定。zero-init只证明开始时新增残差为0；之后仍会学出错误更新，必须检查实际视频。

## 6. 损失与更新计划不再因任务名改变

    L = 0.8 L_FM + 0.2 L_tail + lambda_route(s) L_route

L_FM目标是noise-clean。T2V对全部有效未知位置求均值；I2V对未来有效位置求均值。L_tail沿用原四尺度误差聚合与top25%，同样按task mask处理。它不是物理定律监督，也不对视频变化本身施加平滑惩罚。

    L_route = mean((Cr-Cnative)^2) / max(target_scale^2,1e-8)
    lambda_route = .03*min(s/32,1)           (1<=s<=600)
    lambda_route = .03-.015*(s-600)/400     (601<=s<=1000)

不存在L_local、ST、动作约束、DPO、守恒伪标签、评测器梯度、输出norm处罚。没有新target preprocessing。

每版1000 optimizer updates、effective batch4，4000次样本曝光。每版独立运行；四版合计是四次预算，不谎称四版共计1000步。单卡micro1 accum4，多卡使world_size*micro*accum=4。
视频LoRA从step1可训练；其LR预热128更新，物理参数LR预热32更新，这不是前128冻结。峰值保持至600，601–1000 cosine降至0.2倍。400/600/800/1000属于同一训练计划的checkpoint，不为400单独压缩日程。
学习率工程起点：Enhancement5e-5，physics_read2e-5，physics_write3e-5，videoLoRA2e-5，机制2的adapter2e-5。AdamW(.9,.999),eps1e-8,weight_decay.01(矩阵)，bias/norm无decay。trainable master FP32，BF16前向，softmax关键统计/残差与loss按官方及参考代码用FP32。

sigma仍为原0.001–0.999均匀采样，每条视频一个sigma；不引入分层sigma或额外sigma loss权重。采用共享数据顺序/noise seed做机制配对。

## 7. 推理

主比较仍为81帧、832x480、50步shifted Euler、shift5、CFG5、seed42，不通过更改采样器/CFG弥补结构问题。同任务baseline必须使用同一配置。
TI2V官方默认示例可使用其他solver/分辨率；与官方等价检查时必须建立相同solver和相同输入的本地reference，不能把Euler与UniPC差异算为结构故障。
I2V正/负CFG两支始终保留同一首图，每一步覆盖其latent；T2V两支都无首图，不进行覆盖。两支的physics由各自合法文本条件生成，动态tensor/缓存不相互覆盖。
原生T5 padding规则完整保留。物理请求绑定仅使用合法T5长度含EOS。物理消息的value padding=0，不把删除native padding后重归一当“原生attention权重”。

## 8. 学到了物理的验收证据

直接指标：同一个checkpoint、同一输入协议，VideoPhy2 I2V固定40目标PC>=3.85(冲刺3.95)、SA>=3.25、Joint>=14；画质与请求完成同时合格。目标不是预测。
强样本继续比较物理细节/对象持续性/材料和光学正确性；新联合成功案例要有正确现象，不能靠静止逃避任务。

条件检验：同一输入/种子比较正确physics、同首图/原请求但无关physics、训练均值physics、关闭物理。不能只让错误条件更差；正确条件要对匹配baseline产生实际收益。推理开关只说明依赖，最终结构价值以匹配训练的base/videoLoRA对照检验。
机制1另检查去掉语义投送writer；机制2另检查去掉条件FFN增量，但保留完整直接物理。不要把两版所有开关组合成大矩阵。
监督与容量控制：匹配主干videoLoRA baseline，报告训练token/updates、新增参数/FLOPs和数据处理成本。条件使用检验不要求构造训练负样本。

画质检查：先验证真实Wan起点等价、patch顺序、时间/latent缩放、CFG与checkpoint一致，再在开发集做完整采样。目标是0个严重棋盘格/雪花/横线/大色块/整体融化样本，但有限样本零故障不等于分布上零故障保证。
R18原始故障未凭README定位；不把“来自I2V”当作网格崩塌的直接原因。

## 9. 跨基准与模式差距

I2V固定40与同文本T2V固定40分别报告，不能用T2V达到35% Joint替代原I2V目标。论文的PC/SA通过率与项目均分分开。
同参数量比较至少区分5B主干、实际生成参数、Qwen等辅助模型与额外推理计算；不只写5B就忽略外部编码器。
WorldModelBench：完整350为正式目标，固定70为历史子集。T2V用官方text_first_frame+text_instruction；I2V用first_frame+text_instruction；分项报告instruction/physics/quality。
Physics-IQ：固定Original或Verified；I2V与multiframe/V2V、single-sample与BoN分别报告。纯T2V不具有与指定真实首图对齐的任务输入，不能直接称I2V排行榜SOTA；需要图像条件时使用I2V版并明确模式。
VBench：固定版本、prompt集合、帧率/时长和原始视频，检查动态程度、主体/背景一致性、闪烁与质量，不把只有静止的高PC当成功。
测试集已经被反复使用，fixed40应定位为历史回归集；独立holdout及完整benchmark才能支持泛化主张。测试judge不能进入训练/挑样本。

## 10. 实现前后应保存的证据

保存config、base revision、code commit、data/condition/interface hash、trainable allowlist与优化器组、采样顺序、首帧mask、真实FPS/PTS、模型checkpoint与RNG。T2V单独断言：任何输入tensor不来自首图或未来标签缓存。
基本检查：T2V第一帧有FM梯度、I2V首帧无FM梯度；I2V未来梯度正常；writer为0时新增路径零输出；改physics可影响输出且梯度可达；冻结文本矩阵训练前后不变；native FFN只执行一次；save/reload输出一致。
这些是实现正确性检查，不是physics有效性证据，不以通过数量替代视频。


# 默认配置（需要真正接入解析器，不是已有代码即插即用承诺）

```yaml
revision: NoChunks-v2-20261111
plan: I1
task: i2v
mechanism: 1
status: unvalidated_candidate
base:
  model: Wan2.2-TI2V-5B
  project_checkpoint: null
  freeze_text_qkvo: true
  video_lora_rank: 32
  video_lora_alpha: 32
  video_lora_from_update: 1
model:
  physics_slots: 256
  physics_width: 3072
  q_width: 4096
  q_heads: 32
  q_rounds: 2
  injection_layers:
  - 2
  - 5
  - 8
  - 11
  - 14
  - 17
  - 20
  - 23
  - 26
  - 29
  extra_video_blocks: 0
  split_temporal_physics_memory: false
  conditioned_ffn_rank: null
input:
  external_image: true
  evidence_scope: first_image_and_request_only
  known_latent_frames: 1
  future_targets_in_condition: false
  chunk_descriptions: false
  local_annotation_preprocessing: false
train:
  max_updates: 1000
  effective_global_batch: 4
  microbatch_per_device: 1
  gradient_accumulation: 4
  world_size: 1
  seed: 42
  physics_warmup: 32
  video_lora_warmup: 128
  peak_end_update: 600
  minimum_lr_ratio: 0.2
  post_training: false
  noise_strategy: unchanged_linear_FM
  sigma_min: 0.001
  sigma_max: 0.999
loss:
  fm: 0.8
  tail: 0.2
  tail_fraction: 0.25
  route_peak: 0.03
  route_warmup: 32
  route_decay_start: 600
  route_final: 0.015
  local_semantic_enabled: false
  st_enabled: false
  supervise_known_latent: false
  unknown_scope: valid_future_frames
inference:
  num_frames: 81
  width: 832
  height: 480
  steps: 50
  sampler: shifted_euler
  shift: 5.0
  cfg: 5.0
  seed: 42
  best_of_n: 1
  clamp_first: true
```

---

# 一手资料与证据边界

以下资料用于核对机制，不表示论文验证了本候选。引用日为2026-11-11；实际复现需另外冻结各仓库commit、基础模型revision及缓存hash。

1. ProPhy论文：https://arxiv.org/abs/2512.05564 ，核对v2第4–6、8页：视觉latent与物理先验结合、类别/局部对齐、20K训练视频、VideoPhy2通过率口径。
2. ProPhy公开Wan实现：https://github.com/zijunwa/ProPhy/blob/main/prophy/models/transformer_wan.py 。首个physical block把hidden_states加到control state；随后物理分支推进，预先形成hints后注入主干。不是每一层都回读该层最新主干状态。
3. PhysVid论文：https://arxiv.org/abs/2603.26285 ，核对v2第6、8页。约53K样本、有效batch64、1000+2000更新；VideoPhy2消融中baseline SA0.2845/PC0.6144，PhysVid SA0.2775/PC0.6411。这是论文自己的口径，不是我们40例均分。
4. 官方TI2V入口：https://github.com/Wan-Video/Wan2.2/blob/main/wan/textimage2video.py 。同一TI2V-5B支持标准T2V与I2V。I2V首帧条件覆盖与token timestep处理，和T2V无已知帧是任务边界差别，不是新噪声策略。
5. 项目现有I2V输入：https://github.com/Lewislius/physgen/blob/main/code/v5.2.2-native/experiments/physical_content_v7/physical_content/teacher.py 。缓存scope=first_image_and_request_only。
6. 项目现有FM聚合：https://github.com/Lewislius/physgen/blob/main/code/v5.2.2-native/experiments/physical_content_v7/physical_content/supervision.py 。当前函数硬编码[:, :, 1:]；增加T2V时必须将监督范围参数化，I2V保持数值等价。
7. 历史视频回读：https://github.com/Lewislius/physgen/blob/main/code/v5.2.2-native/experiments/physical_content_v7/reciprocal_fusion_r12a/DESIGN.zh.txt 。已有state读回C再写回的尝试；不是本次新发现。
8. 历史局部视觉算子：https://github.com/Lewislius/physgen/blob/main/code/v5.2.2-native/experiments/physical_content_v7/local_state_operator_r15c/DESIGN.zh.txt 。已有基于视觉K/V的执行尝试，不能仅凭视觉value存在推断有效。
9. R10A增强：https://github.com/Lewislius/physgen/blob/main/code/v5.2.2-native/experiments/physical_content_v7/execution_value_r10/physical_content/model.py 。采用其全宽计算机制，不采用项目训练checkpoint。
10. B400+I联合读取：https://github.com/Lewislius/physgen/blob/main/code/v5.2.2-native/experiments/physical_content_v7/b400_joint_read_eval/joint_read.py 。推理增益不能视为从头训练的收益保证。
11. R18评测：https://github.com/Lewislius/physgen/tree/cde6a88e/code/v5.2.2-native/experiments/physics_expert_abcd_fresh1000/eval_artifacts/r18c_r18d_20261111 。已公开结果不等于已公开完整训练源码/日志/权重。本文件不声称定位了R18的具体实现故障。
12. VideoPhy2官方：https://github.com/Hritikbansal/videophy/blob/main/VIDEOPHY2/README.md 。Joint为PC>=4且SA>=4；均分与通过率分别报告。
13. WorldModelBench：https://github.com/WorldModelBench-Team/WorldModelBench/blob/main/README.md 。T2V用text_first_frame+text_instruction；I2V用first_frame+text_instruction。
14. Physics-IQ：https://github.com/google-deepmind/physics-IQ-benchmark 。Original/Verified、I2V/V2V、单样本/BoN需要区分；纯T2V不能冒充I2V成绩。

尚未完成：真实Wan集成、GPU优化、任何本候选视频生成、PC/SA/Joint/WMB/Physics-IQ/VBench评测。配置检查不能替代这些验证。
