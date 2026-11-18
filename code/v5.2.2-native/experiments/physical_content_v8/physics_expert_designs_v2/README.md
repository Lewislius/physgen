# Physics Expert：方案一、二、四完整实现设计包

版本：design-v2 / 2026-11-10。方案三不在本包中。

**建议先实现P2-S3-R64与P4-S3-H512作为两个独立竞争方案；P1-S3用于验证读取/寻址改进。** 不把三版第一轮全部叠加。

目标：VideoPhy2-40同时PC≥3.85、争取3.90，SA≥3.25，Joint≥15/40；强样本继续提升；实际画质不崩坏；进一步在大规模、同任务、同预算评测中竞争。包内没有任何新模型已达标的声明。

## 阅读顺序

| 文件 | 作用 |
|---|---|
| [00_IMPLEMENTATION_CONTRACT.md](00_IMPLEMENTATION_CONTRACT.md) | 公共张量、原生计算、训练、数据、精度、接口、梯度及恢复契约 |
| [01_P1_REQUEST_KEY_ENHANCEMENT.md](01_P1_REQUEST_KEY_ENHANCEMENT.md) | 方案一：请求条件化物理key，原始全宽value保留 |
| [02_P2_DISTRIBUTED_NATIVE_EXECUTION.md](02_P2_DISTRIBUTED_NATIVE_EXECUTION.md) | 方案二：每3/5层预写入已有原生block，不复制、不双前向差分 |
| [03_P4_COOPERATIVE_EXPERTS.md](03_P4_COOPERATIVE_EXPERTS.md) | 方案四：物理—语义上下文条件专家，无新教师、无干预学习 |
| [04_EVALUATION_AND_EXECUTION.md](04_EVALUATION_AND_EXECUTION.md) | 对照顺序、40例及大规模目标、画质与公平比较 |
| [05_EVIDENCE_AND_DECISIONS.md](05_EVIDENCE_AND_DECISIONS.md) | 原始来源、历史反例、推导与新假设区分 |
| [06_HANDOFF_TO_CODING_AI.md](06_HANDOFF_TO_CODING_AI.md) | 可直接复制给后续AI的开发任务说明 |
| [configs/README.md](configs/README.md) | 独立候选与对照YAML的使用说明 |
| [checks/reference_math_checks.py](checks/reference_math_checks.py) | CPU小张量代数/梯度检查；不是Wan项目实现 |
| [checks/reference_math_checks_results.json](checks/reference_math_checks_results.json) | 本次实际CPU检查结果，不包含生成评分 |

## 三版的边界

| 方案 | 独立变化 | 每3层参数估算 |
|---|---|---:|
| P1 | key读取原请求，value不经过请求预平均 | 713,657,344 |
| P2 | 物理写入提前到原生self-attention前，state_value预算转给rank64原生适配 | 638,553,088 |
| P4 | 两个来源条件专家替换state_value，保留全宽直接write | 675,908,608 |

以上是训练参数估算，不是新增推理总参数，也不是实际部署统计。原始R10A/R15B审计为670,010,368，详见E06。

已有Qwen只继续作为输入条件模型，T5作为原生文本编码器；不引入V-JEPA或其他新教师，不做热图监督/干预训练/动作约束。设计来源限定R1—R15与公共底座，不读取或继承R16及之后候选。

## 当前交付状态

完成：Markdown设计、YAML配置契约、预算推导和可运行的小型CPU参考检查。
未完成：与真实Wan的集成、项目GPU训练、视频生成、VideoPhy2/WMB/Physics-IQ/VBench新评分。

CPU检查验证的是函数关系、shape和梯度性质，不能推出PC、SA或Joint的数值。正式测试结果使用`results_template.json`从真实输出填写。
