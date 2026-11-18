# 来源、事实边界与设计取舍记录

审阅日期：2026-11-10。  
项目内容固定到commit `1defd6f4b41e42302bc3e4b9b2bf938abbbc686c`。  
本轮只使用R16之前的设计与结果；没有读取或借用R16候选。

## 1. 项目事实来源

### E01 — 已完成40例汇总

[completed_full40_summary.csv](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/reports/completed_full40_summary.csv)

支持Native400、R1、R10A等结果；不同训练步数不可混成同预算结果。R1续训到1000的PC回落，不支持“只要更久就更好”。

### E02 — R10A设计与代码

[DESIGN.zh.txt](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/execution_value_r10/DESIGN.zh.txt)

[physical_content/model.py](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/execution_value_r10/physical_content/model.py)

支持两轮全宽增强、route/execute分工、原生Q复用、独立物理K/V/O、state_value乘法、时间与reference实现。代码事实不等于每一个子模块都已有独立有效性证据。

### E03 — R13B

[设计](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/interaction_query_r13b/DESIGN.zh.txt)

[正式比较结果](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/interaction_query_r13b/inference_outputs/physical_content_67610/step0400_original_fixed40_controls16_seed42/comparison_report.json)

支持“物理query调制+直接写入”的设计及40/60结果。不能把它当作P2所有Q/K/V共同重算的现成成功证据。

### E04 — R11A原生语义适配及历史成绩

[request_adapt_r11a/DESIGN.zh.txt](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/request_adapt_r11a/DESIGN.zh.txt)

[R11A_to_R13B_scores_20261105.csv](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/r11_r12/reports/R11A_to_R13B_scores_20261105.csv)

已尝试文本Q/K/V/O rank32、文本先物理后的顺序。新方案不应把文本LoRA本身包装成从未验证的突破。

### E05 — R15B与机制复盘

[设计](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/request_bound_fusion_r15b/DESIGN.zh.txt)

[实现](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/request_bound_fusion_r15b/physical_content/model.py)

[完整机制复盘](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/local_state_operator_r15c/review/r15b_mechanism_completed.json)

支持请求value聚合、C公共成分变化、state_value的混合效应及完整40成绩。不能从token cosine推出生成效用。

### E06 — 训练参数与400步审计

[completed400_training_audit.json](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/request_bound_fusion_r15b/reports/completed400_training_audit.json)

支持670,010,368参数分组、400更新/1600次样本使用。新方案参数为从这些模块与矩阵形状推导的估计，尚非真实部署模型统计。

### E07 — R11B反例与R12A记录

[reciprocal_fusion_r12a/DESIGN.zh.txt](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/reciprocal_fusion_r12a/DESIGN.zh.txt)

支持独立请求memory副作用、移除后局部收益，以及“文本先更新+C读取视频状态”已经尝试过的事实。本包没有把这些旧组合改名当成新方案。

### E08 — B400联合读取与跨基准

[b400_joint_read_eval/reports/results.zh.md](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/b400_joint_read_eval/reports/results.zh.md)

支持40/60、WMB70、Physics-IQ60视角分数以及仓库记录的视频复盘。本次没有重新播放全部原始视频，因此本包关于这些旧视频的具体描述来自该复盘，不冒充新的独立视觉判断。

### E08b — ProPhy提示词重放的输入任务限制

[prophy_videophy2_600/README.md](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/prophy_videophy2_600/README.md)

明确I2V首帧适配与ProPhy T2V主报告不同，也不保证judge完全一致。

### E09 — 缓存训练视频元数据

[cached_temporal_metadata_audit.json](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/request_bound_fusion_r15b/reports/cached_temporal_metadata_audit.json)

支持3857条记录与时间戳审计；它不是完整语义—片段一致性审计。

### E10 — 生成损失

[physical_content/supervision.py](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_content_v7/physical_content/supervision.py)

高误差聚合是通用生成误差归约，不等价于物理标签或物理规律损失。

### E11 — 原生结构和LoRA接口

[Wan官方model.py](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py)

[Wan官方TI2V-5B配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)

[项目physical_scaling/model.py](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/experiments/physical_scaling_q256/physical_scaling/model.py)

[项目LoRALinear](https://github.com/Lewislius/physgen/blob/1defd6f4b41e42302bc3e4b9b2bf938abbbc686c/code/v5.2.2-native/physgen_native/lora.py)

原生block的self→text→FFN顺序、norm/time语义与LoRA `.base/.lora_a/.lora_b`命名已经核对。官方main会变化，正式实验必须再锁定实际安装commit；本次读到的官方model.py blob SHA为`6982fa15db1156dfcfc50d480ae9a42d8525f83d`，这是文件blob，不是repo commit。

## 2. 外部一手资料

### E12 — ProPhy

[arXiv v2](https://arxiv.org/html/2512.05564v2)

[PDF](https://arxiv.org/pdf/2512.05564v2)

本次已通过PDF截图核对第6页Table1和第7页Table2；数值也与v2 HTML表格一致。引用其协议与报告数字，不认为其机制已经验证本包三个新设计。ProPhy使用额外VLM标注；本包选择不引入此监督，因此不能声称拥有等价训练信号。

### E13 — VideoPhy2官方定义

[官方README](https://github.com/Hritikbansal/videophy/blob/main/VIDEOPHY2/README.md)

支持SA/PC 1—5与Joint双阈值定义。数据集自身名称含action-centric，不意味着本项目必须采用动作约束。

### E14 — Physics-IQ官方

[Physics-IQ / Verified](https://github.com/google-deepmind/physics-IQ-benchmark)

支持Original/Verified、输入模式、场景视角、同规模记录及SOTA四次运行要求。网页/榜单会更新，本包不声称锁定了所有未来或全部同规模SOTA。

### E15 — WMB官方

[项目页](https://worldmodelbench-team.github.io/)

[官方实现](https://github.com/WorldModelBench-Team/WorldModelBench)

支持350样本、7领域及输入任务对应规则。

### E16 — VBench官方

[官方实现](https://github.com/Vchitect/VBench)

[VBench论文](https://arxiv.org/abs/2311.17982)

支持多维质量评测、I2V/T2V区分。不同prompt集合和版本的总分不可直接等同。

### E17 — PyTorch SDPA

[官方API说明](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)

支持bool mask True为允许、attention默认缩放与dropout需要显式设置等实现注意事项。实际依赖版本应锁定，不默认使用main文档的所有新API。

## 3. 三类陈述必须分开

| 类别 | 本包内容 | 可据此得出的结论 |
|---|---|---|
| 仓库事实 | 已完成结果、真实源码、参数审计 | 描述历史能力与实现 |
| 推导/实现性质 | 参数恒等式、零初始化等价、source=0输出0 | 证明数学/工程性质 |
| 新研究假设 | 请求key改善SA、预写入改善执行、协同专家共同提高质量 | 必须靠训练和生成验证 |

不要把第三类变成“已证明”；也不要因为第三类待验证就把设计写成无法执行的空泛建议。

## 4. 本次结构收敛记录

1. 删除方案三及所有新增视觉教师、热图监督。
2. P1从多项score融合收敛为一个标准请求→物理key cross-attention；执行value不经二次请求聚合。
3. P2删除复制大块与双前向相减，改为每3/5层的原生block输入预写入；同时把state_value预算转为rank64原生生成适配。
4. P4删除干预学习与“因果专家”未经验证的命名，用两个来源明确的条件专家替换state_value；其参数量可精确预算匹配。
5. 三版分别训练，不第一轮全堆一起；所有中间量来自已有视频/物理/文本运算。
6. 共同保持高目标，不使用降低CFG、后处理平滑、挑seed或自动关闭物理作为达标方式。

## 5. 没有做过的事

本次未访问项目训练服务器、未提交GPU任务、未运行完整Wan推理、未重新评分40例、未获得任何新候选的PC/SA/Joint。附带CPU测试已执行23项并通过，是合成数学/梯度检查，不是项目GPU训练或视频效果实测。Markdown、配置、预算与检查结果均在本次对话环境中创建。

## 6. 附带检查记录

[reference_math_checks_results.json](checks/reference_math_checks_results.json)记录本次23项CPU检查及torch/Python版本；[检查脚本](checks/reference_math_checks.py)可独立重跑，不下载任何模型。它验证小尺寸算子的数学性质，没有加载Wan checkpoint，因此不能作为真实Wan集成或质量报告。

方案二额外补充了旧残差也会进入后续原生层的事实，避免把“注入位置变更”夸大成首次原生执行。三版的新生成收益仍属于待验证假设。
