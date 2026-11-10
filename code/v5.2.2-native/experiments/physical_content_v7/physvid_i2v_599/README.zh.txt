本轮评测：匹配本机PhysVid已完成的599提示词、1组生成结果。成功任务种子9898539；另外4个日志种子为失败重试，不计为完整种子集。
五版使用各自400步checkpoint。Wan+LoRA、R1：amp-80g单卡A10080GB；R9、R10-A/B：blk-96g单卡，启动时拒绝H100，仅接受RTX PRO6000/RTX6000 Blackwell。
重新独立生成599张动作开始前的公共首帧，使用现有官方Wan2.2-TI2V-5B，无项目LoRA；五个版本和所有视频种子共用，无PhysVid视频帧、参考后续视频或评测标签输入。各任务准备不重叠的120/119例输入后复用。
视频81帧、832x480、16fps；50步Euler、shift8、文本CFG6，物理分支遵照用户既有要求gamma5。原生负文本沿用本机PhysVid配置；负C保持本图无请求。Qwen根据5秒呈现区间重新提取条件，读取规则与冻结训练模型一致。
第一条正式样本生成和评分后继续全量，结果复用，无额外诊断视频。单个任务输出outputs/<model>/seed9898539；评分后summary.json同时保存1-5均值及SA>=4、PC>=4和联合通过率。保留PhysVid原始解析文本和异常0分，不能当作有效1-5分隐藏。
所有模型使用CPU逐案例种子9898539+index生成相同初始噪声。这不是PhysVid分布式CUDA随机流的逐像素重现；后者基座/潜空间也不同。I2V与PhysVid原T2V条件不同，原PhysVid成绩作为参考，五版之间为共享输入配对比较。
不下载模型，不改变旧源码、训练或结果。测试5项通过：全集与分片覆盖、评分口径、作者解析器一致、作者sigma网格/步进一致、各400步权重和资源/随机种子映射。
后续汇总：bash tools/runtime_python.sh experiments/physical_content_v7/physvid_i2v_599/summarize.py

代码与版本对应：
本轮新增公共首帧/条件准备prepare_inputs.py、推理generate.py、评分evaluate.py、编排run.py以及汇总/协议测试。五个YAML入口均指向snapshots/20261101T141534Z/run.py，以--model选择版本。
generate.py根据protocol.json中的models[版本].source加载该版本physical_content/model.py，再加载该版本自己的400步checkpoint；模型结构和权重来自各自冻结快照，共享的只是评测编排与输入。Wan+LoRA使用native_lora分支，不启用物理增强模块。
Wan+LoRA: source=snapshots/20261030T141432Z; checkpoint=runs/native_lora_66781/step0400.pt; Search67037/Run68424。
R1: source=snapshots/20261030T141432Z; checkpoint=runs/physical_content_66778/step0400.pt; Search67038/Run68425。
R9: source=r1_content_reader_r9/snapshots/20261031T150803Z; checkpoint=r1_content_reader_r9/runs/physical_content_66968/step0400.pt; Search67043/Run68430。
R10-A: source=execution_value_r10/snapshots/20261101T034557Z; checkpoint=execution_value_r10/runs/physical_content_67003/step0400.pt; Search67040/Run68427。
R10-B: source=grounded_execution_r10b/snapshots/20261101T054248Z; checkpoint=grounded_execution_r10b/runs/physical_content_67008/step0400.pt; Search67041/Run68428。
以上source/checkpoint相对experiments/physical_content_v7；绝对路径和提交YAML见protocol.json与prepared.json。R9早先两个H100分配均在加载模型前退出，当前任务已确认RTX。

2026-11-02服务器中断恢复：Wan+LoRA67037/68424和R1 67038/68425仍在实际生成，未重复提交。R9原67043、R10-A原67040、R10-B原67041为agent故障后挂载目录缺失而终止。公共输入599全部保留；原完成视频28/69/64条均保留。新R9 67115/68502、R10-A 67110/68497、R10-B 67111/68498均已确认RTX PRO6000。恢复YAML/入口和记录见recovery/20261102T052708Z，直接调用原冻结推理/评分脚本，跳过已完成视频与评分。四次H100尝试均在执行模型前退出或终止，不再活动。
