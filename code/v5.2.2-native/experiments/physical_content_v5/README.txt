V5 C-language CE：Q256 / 1000步独立扩展，2026-10-29

目的与实现
  保留已完成224步V5的设计，扩展Q容量与执行适配训练时长。
  从官方预训练Wan/Qwen/T5/VAE开始；新模块和LoRA重新初始化，不续训旧V5。
  原模型与语义目标函数引用固定快照 physical_variants/snapshots/20261029T064043Z。
  本目录实现独立分组、调度、训练入口和缓存写入，不影响V6运行任务。

结构与可训练部分
  Q种子[1,256,4096]；两轮共享增强；4096->3072后C为[1,256,3072]。
  每三层Wan注入一次，共10处。原生文本注入及原生时空注意力保留。
  保留V5全参数C专用K/V、key norm、time和3072->3072输出O。
  不把V5的K/V替换成V6的LoRA。语义桥接层仍为3072->4096。
  Wan全部30层self-attention Q/K/V/O及FFN使用rank32 LoRA。
  Qwen本体、原生Wan基础权重、T5、VAE冻结；无verifier。
  相比旧V5只因Q新增128*4096=524288参数；总可训练575611904。
  五组预计参数：enhancement215002112、fusion_read196638720、
  fusion_write94371840、semantic_bridge12582912、wan_lora57016320。
  reasoning_mode=q_seed128_message仅是原冻结Qwen加载器的兼容选择名，
  不决定Q数量；实际model.q_tokens和knowledge.reasoning_tokens均为256。

训练和学习率（峰值）
  128步接口：训练C、融合读取/输出、语义桥；Wan LoRA冻结。
  872步联合：继续前者，并解冻Wan LoRA；合计1000步，accumulation4，seed42。
  C提取/Q/projector                    5e-5
  全参数C K/V、key norm、time           2e-5
  融合输出O                            1e-5
  语义解码桥3072->4096                 5e-5
  Wan self-attention/FFN LoRA           2e-5
  前四组32步预热，保持到600，余弦衰减到20%。
  Wan第129步解冻，128步预热至第256步2e-5，保持到600，最终4e-6。
  AdamW(.9,.999)，weight_decay=.01；各组裁剪阈值1，记录裁剪前梯度。

仅两项损失
  L = L_FM + lambda(step)*L_CE。
  FM沿用原视频去噪目标；CE沿用原C-only冻结Qwen解码器的答案token平均交叉熵。
  CE不乘(1-sigma)，不随Q token数量乘倍数。标签来自既有WISA物理说明、
  首帧和请求形成的弱语义目标，定义与旧V5完全相同。
  lambda由原固定0.05改为：前64步线性升到0.03，保持到600，余弦降到0.015。
  按原V5顺序先FM反传并释放Wan图，再重算C做CE反传，最后统一优化器更新。
  Qwen decoder块使用checkpointing，参数冻结但保留到C/bridge的输入梯度。
  request dropout=0.1时C/T5同步用空请求，并跳过该样本CE。

依据与边界
  旧V5前64步63次裁剪，最大物理模块梯度56.5363。末32步平均CE1.9613，
  加权0.09807，FM0.27656，比值35.46%。固定4例验证FM0.20586->0.18615。
  新CE峰值降低40%，并逐渐接入；同样损失值下后期加权CE/FM约21.28%。
  分组学习率使语义桥和C不再与直接写入共用较高学习率及一次总裁剪。
  旧日志未分离CE/FM梯度，裁剪不能单独归因于CE；CE下降不是执行正确的证明。
  原V5在固定VideoPhy2-40得SA3.15/PC3.40/平均3.275/联合25%。新设置尚无评分，
  不能将更长训练、Q翻倍或调度修改表述成已经提升物理一致性。

缓存、存档与提交
  数据仍3666训练/191验证，1000*4=4000次抽样。
  只读复用旧V5语义缓存和旧/V6输入缓存；新增目标与锁仅写本目录。
  每50步更新latest.pt，另保留128/250/500/750/1000；验证128/500/1000。
  独立运行目录runs；保存配置、全部新参数/LoRA、optimizer、RNG和样本cursor。
  --resume只接受本版本相同配置。RTX6000检查在基础模型加载之前；H100退出86。
  实际提交快照、任务、GPU和首步记录见reports/latest_submission.json。

使用
  bash train/train_v5_ce.sh --check
  bash ../../tools/runtime_python.sh tests/test_v5_scaling.py
  det -m http://10.130.130.5:4096 experiment create train/v5_ce_q256_s1000_blk96g.yaml submission_context
