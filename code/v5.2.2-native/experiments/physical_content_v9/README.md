# PhysGen ABCD：官方初始化／LoRA／1000步／batch4

版本：2026-11-11，v4。四份Markdown均是可独立交给编码AI的完整计划，包含共同契约、该方案精确结构、默认YAML与核验来源。

## 本版唯一训练定义

**官方Wan/Qwen/T5/VAE基础初始化；不载入任何R系列项目checkpoint；原生视频LoRA从第1步训练到第1000步；原生文本Q/K/V/O完全冻结；每次更新有效batch4；没有后训练。**

“从头”不是把Wan随机初始化。“全训1000”不是原生全参数解冻。没有400＋600的R10A续训，也没有1000后附加精修。

## 四个方案

| 文件 | 主线 |
|---|---|
| [A计划](PLAN_A_FRESH1000.md) | 独立联合时空专家：semantic与physics在完整attention/FFN中共同处理 |
| [B计划](PLAN_B_FRESH1000.md) | 主干内LoRA协同执行：不开放原生完整参数，新增融合由原生视频LoRA执行 |
| [C计划](PLAN_C_FRESH1000.md) | semantic增强physics记忆与执行value，再由强物理时空专家执行 |
| [D计划](PLAN_D_FRESH1000.md) | semantic指导物理传播/变换，没有额外语义value写入 |

本轮优先研究C、D；A保留较自由的联合内容路线，B保留较低额外时空计算路线。它们都是待测方案，不代表已达到目标或已有排序证据。

## 使用方法

给编码AI一份对应`PLAN_*.md`即可；它已经内嵌必要共同规格与配置。需要一次了解四版时使用包内`ABCD_COMPLETE_PLAN.md`。独立YAML在configs目录。

只需让AI实现该方案及共用基础，不要求它自动执行其他三版、混合四版或提交训练任务。路径中的环境变量和官方checkpoint hash由实际运行环境填写，不能猜测不存在的文件。

## 代码与测试边界

`tools/validate_specs.py`检查交付文件/配置的训练边界；`tools/toy_contract_checks.py`只检查小张量LoRA、gradient accumulation、归一化损失和zero-preserving物理执行等代数。它们不是完整生成模型，不应被当成已经实现了Wan Expert。

目标：VideoPhy2-40 PC≥3.85、挑战3.95，SA≥3.25，Joint≥14/40，画面无明显崩坏，并在完整跨基准协议下检验泛化。本包未运行真实GPU训练，没有为这些方案生成分数。

## 重要变化

旧版本关于R10A warm-start、B的原生全参数训练、前期冻结Wan LoRA和后训练的描述全部由本版替代。训练输入闭式校准重新计算并记录；数据缓存可复用，但已训练项目权重不能复用。
