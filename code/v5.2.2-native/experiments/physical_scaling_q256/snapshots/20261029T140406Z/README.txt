Q256 / 1000 步独立训练版本，2026-10-29

包含 v6_write（原残差回归定义）和 v6_assoc（关联 KL 定义）。
新代码、配置、运行目录、缓存写入均位于本目录；不修改 physical_variants。
新模块和 LoRA 从头训练，仅加载官方 Wan/Qwen/T5/VAE 基础权重。
不使用旧项目 CKPT 初始化；--resume 只接受本版本、相同配置的训练中断存档。
用户已授权按修订学习率提交；实际任务和GPU记录见 reports/latest_submission.json。

规模与结构
  Q：[1,256,4096]；两次共享 enhancement 更新；投影后 C：[1,256,3072]。
  Wan 仍为 3072 维，30 层，每三层注入；原生文本路径完整保留。
  残差版读取 256 个 C；KL 版读取 256 个 C 加一个零值 reference key。
  KL 版保留分组槽位数量的 -log(group_size) 修正。token 数翻倍不会
  自动给某关系增加先验权重；复制相同 token 不会使写入幅度翻倍。
  仅 Q seed 增加 128*4096=524288 个参数，不是整个条件模块参数翻倍。
  Qwen 冻结，无 verifier、CE、额外 temporal FM 或新增物理损失。
  knowledge.reasoning_mode 的 q_seed128_message 是固定 Qwen 加载器的
  兼容选择名称，不决定 Q 大小；实际 Q 由 model.q_tokens=256 创建。
  knowledge.reasoning_tokens 同步为256；Qwen 文本分析上限仍为128 tokens，
  与可学习 Q 槽位数量是两件事。原输入/教师生成方式完全保留。

训练与学习率
  128 步接口训练：训练 enhancement、融合读取、融合输出；冻结 Wan。
  872 步联合训练：继续前者，并训练 Wan self-attention/FFN rank32 LoRA。
  总计1000个优化器更新步；每步累积4个样本；seed42；request dropout=0.1。
  C 提取/Q/projector 峰值学习率       5e-5
  融合 K/V LoRA、key norm、time、ref  2e-5
  融合输出 O（直接写入 Wan）         1e-5
  Wan self-attention/FFN LoRA        2e-5
  前三组用32步线性预热，保持到第600步，之后余弦衰减到峰值20%。
  Wan 从第129步解冻，128步线性预热：第192步到1e-5，第256步到2e-5；
  保持到第600步，之后余弦衰减，第1000步到4e-6。
  阶段切换不重置 optimizer，不将物理模块学习率突然提高。
  AdamW betas=(0.9,0.999)，weight_decay=0.01，各优化组 grad_clip=1。
  日志记录各组裁剪前范数、裁剪比例、实际学习率及辅助损失系数。

两种目标（仍各两项）
  L = L_FM + lambda(step)*(1-sigma)*L_aux，FM系数始终为1。
  sigma 是扩散噪声强度，不是视频时间或物体速度。
  v6_write：lambda 前32步线性升至0.005，保持到600步，余弦降至0.002。
  v6_assoc：lambda 前32步线性升至0.030，保持到600步，余弦降至0.015。
  原两版都是固定0.02*(1-sigma)。本轮保留损失的数学定义和教师目标含义。
  无正支持的残差样本跳过辅助项；文本 dropout 同步作用于 C 和原生文本。

调整依据
  原残差版前32步31次触发裁剪，最大裁剪前物理模块范数75.7965；
  同窗口加权辅助损失总量约为FM的34.72%。例如第18步的一个样本，
  FM=0.6903、加权残差项=1.0925。不是仅凭模型规模或数据比例解释不稳定。
  这支持降低早期直接写入的更新步幅和残差系数；日志没有分离两种损失
  的梯度贡献，因此不能把所有大梯度都断言成辅助损失单独造成。
  KL版联合160步没有触发裁剪，最后32步平均KL仍为0.4384；旧学习率已
  降到峰值20%。延长保持期用于让C读取与Wan执行继续共同适配。
  Wan LoRA 进一步修订（2026-10-29）：原224步方案的预热是16步，
  不是新版的64步。原两版Wan联合160步均未触发裁剪，裁剪前最大范数
  分别为0.05419/0.06956；这些记录未显示梯度爆炸，也不证明学习充分。
  新版Wan峰值由1e-5调为2e-5，预热由64步延长为128步。峰值/预热步数
  的比值不变，故联合训练前64步的学习率与本目录初稿相同，随后继续
  缓升至更高峰值；升温斜率是原224步方案的1/4。目的是加强Wan执行
  适配，同时避免一解冻就加大步幅。保持期与更长训练已增加累计更新
  机会，不能把本次翻倍说成已经证实必要；AdamW下小梯度也不等于小更新。
  不改变其他三组学习率、损失定义或系数。无新增GPU试验或训练提交。
  固定四条验证FM：残差版0.19630->0.18573，KL版0.19048->0.18675。
  这些不是物理一致性或视频质量分数，不能据此保证扩展后指标提升。
  上述系数是结合记录选择的初始训练方案，尚非已验证最优参数。
  残差版的合成响应目标仍存在目标有效性局限；降低系数不能消除它。
  证据明细：reports/prior_training_analysis.json；完整调度：reports/schedule.csv。

数据与缓存
  保留已有3666训练/191验证划分，按原SampleOrder逐epoch重排。
  1000*4=4000次样本抽取，覆盖一个完整训练epoch并进入下一轮。
  只读复用原 physical_variants/supervision_cache/r1 中定义相同的输入和
  教师目标；未缓存样本只写本目录 supervision_cache/r1。Q数量不改变
  教师关系目标维度，因此无需重新生成已有弱目标或伪造新目标。
  输入只读取首帧与请求，真实未来视频只用于训练辅助监督。

保存与验证
  每50步更新 latest.pt；第128/250/500/750/1000步额外保留独立存档。
  存档包括全部新参数、Wan LoRA、optimizer、RNG和样本cursor。
  第128/500/1000步沿用原来的4条固定验证FM和1条固定seed生成视频。
  Q256 attention计算量增加；使用原bf16和梯度checkpointing配置。
  已检查CPU前后向、分组归一化、调度和数据接口；尚未做96GB GPU实跑。

使用（在本目录）
  bash train/train_v6_write.sh --check
  bash train/train_v6_assoc.sh --check
  bash ../../tools/runtime_python.sh tests/test_scaling.py
  det -m http://10.130.130.5:4096 experiment create train/v6_write_q256_s1000_blk96g.yaml submission_context
  det -m http://10.130.130.5:4096 experiment create train/v6_assoc_q256_s1000_blk96g.yaml submission_context
  单卡blk-96g，仍在加载模型前检查RTX PRO6000/RTX6000 Blackwell。
  H100/其他设备退出86；实际提交时须终止该任务并重提直到RTX确认。
  不更改软件环境、旧训练、旧模型或正在运行的推理任务。
