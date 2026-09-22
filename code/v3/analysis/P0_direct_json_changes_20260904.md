# P0 人话分阶段直接 JSON 基线：本轮修改说明

## 先回答：现在就一定能生成高质量视频吗？

不能保证。P0 做的是把明显错误的条件链路清干净，让 Wan 收到短、明确、像人说话的
动作描述，并让 I2V 从物理上可行的首帧开始。这应当明显减少提示词被标签淹没、负面名词
反向增强、初始朝向错误和阶段重复，但 Wan 本身仍可能发生物体变形、身份丢失或动作没有
完整结束。P0 是可信的干净基线，不是已经训练好的物理控制器。

后续只有在固定 seed 对比确认 P0 仍稳定失败后，才值得训练一个共享 StateBridge。本轮没有
恢复旧 Writer、Corrector、Reader 子系统或增加新损失，只在统一推理文件中恢复必要的
五阶段时间路由。

## 这次实际改了什么

### 1. 重新制作了 P01–P20 的首帧

20 个首帧都按“事件发生前一刻”重新安排，必要的原因直接出现在画面中：

- P01 的滑板车已经正对垃圾桶，不再先背对目标再被迫转弯。
- P02 明确是红球已有向右速度，红球和蓝球在碰撞前分开，不虚构看不见的持续外力。
- P06 只保留一本书，并同时看见推书的手、书架边缘、完整下落空间和地板。
- P10 显示气球连接泵嘴；P19 显示两只手拉纸；P20 显示戴手套的手控制水壶。
- 熔化、风吹和扩散任务明确显示温暖环境、风向线索或正在下落的液滴。

demo 目录中每个样本现在都有以下三个固定尺寸文件，三者全部经过检查，尺寸均为
`1280×704`：

- `PXX-i0.png`
- `PXX-i0-1280x704.png`
- `PXX-i0-1280x704.jpg`（Wan I2V 实际读取这个文件）

20 张新首帧总览：
[P0_I0_contact_sheet_20260904.jpg](/home/liuzhirui/Project/physGen/code/v3/analysis/P0_I0_contact_sheet_20260904.jpg)

旧首帧已备份到：
`/home/liuzhirui/Project/physGen/code/v1/demo/_archive_i0_pre_p0_20260904/`

### 2. 重写了首帧和视频提示词

提示词规范现在只允许普通英文句子，不允许 `[STAGE ENTITY INVENTORY]`、
`[VALID RELATION]`、`gap=contact`、c1–c5 或“声明的原因”这类抽象代词。

P01–P20 的权威提示词集中保存在：
[p0_direct_prompts_p01_p20.json](/home/liuzhirui/Project/physGen/prompt/p0_direct_prompts_p01_p20.json)

每个任务包含：

- 一段只描述首帧可见内容的 `pre_event_image_prompt`；
- 一段按动作顺序描述完整视频的 `global_semantic`；
- setup、onset、evolution、completion、terminal 五句互不重复的人话阶段 c+。

五句英文原文现在直接分别进入 T5，并在各自的 DiT 时间 token 上生效；它们不会先经过
Python 改写，也不会被拼成一条长文本。例如 P02 依次使用“红球接近”“两球接触”
“蓝球启动且红球减速”“间距拉开”“两球都仍可见”，不再用两组重复关系冒充五个阶段。

同时更新了提示词说明：

- [generate_high_quality_i0.txt](/home/liuzhirui/Project/physGen/prompt/generate_high_quality_i0.txt)
- [generate_trace_writer_v1_plan.txt](/home/liuzhirui/Project/physGen/prompt/generate_trace_writer_v1_plan.txt)
- [generate_cplus_cminus.txt](/home/liuzhirui/Project/physGen/prompt/generate_cplus_cminus.txt)

### 3. 更新了全部 demo JSON

P01–P20 的 `i0.json`、`V2-plan.json`、`V2-planimg.json` 和 `cplus-cminus.json` 均已更新。

- `plan.json` 是 T2V 数据；`planimg.json` 是 I2V 数据，并保存人工确认的首帧实体框。
- 模型的全局正向提示是 `global_semantic`；五条 `stages[].positive` 是逐字使用的阶段 c+。
- 不设置阶段 c-，也没有 predicates、states、constraints、violations 或自动推导的禁用对象表。
- `cplus-cminus.json` 只保留停用旧格式的说明；真正的 c+ 直接来自 plan/planimg 的
  `stages[].positive`。
- `cfg_negative` 在 JSON 中直接保存为一个空格。原因是 Wan 会把真正的空字符串替换成
  内置负面提示词；这个空格没有视觉语义，又能阻止该回退。运行时不再替换它。
- 每个 plan/planimg 还直接保存 5×25 的 `temporal_route.weights`：97 帧对应 25 个
  DiT 时间 token，相邻阶段各用两个 smoothstep token 过渡。

### 4. 推理改成一个文件直接读取

当前活动推理只有：
[infer_trace_writer.py](/home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer.py)

它的条件传递关系非常直接：

| JSON/文件 | 直接交给 Wan 的位置 |
|---|---|
| `plan["global_semantic"]` | 全程使用的全局语义条件 |
| `plan["stages"][i]["positive"]` | 第 i 阶段的 c+，逐字编码 |
| `plan["cfg_negative"]` | CFG 负向条件，同时作为各阶段的空 c- |
| `plan["temporal_route"]["weights"]` | 五阶段在 25 个时间 token 上的直接权重 |
| `planimg["source_files"]["reference_input"]` | I2V 的 `img` |

没有关键词搜索、子串匹配、同义词猜测、模板拼接、负面词扩写或 Python 文本改写。
相邻阶段只按 JSON 已写好的数值权重平滑混合，不对句子做任何合并或加工。
代码只检查固定 schema、模式、文件是否存在和图片是否正好为 `1280×704`，不会处理提示词
内容。每次生成只把实际 prompt 原文、CFG、steps、solver、shift、seed、帧数和输出尺寸
写入结果目录，不再计算无必要的输入/输出哈希。

旧实体账本、语义扩写、minimal pair、Writer/Reader 子系统、动态 support、SafeCap 和两个
旧工具仍然不在活动路径中，备份在：
`/home/liuzhirui/Project/physGen/code/v3/archive/pre_p0_20260904/`

### 5. 验证结果

- 4 项必要检查通过：20 组文本逐字一致、五阶段不重复、5×25 路由合法、I2V/T2V 读取正确、
  60 张图片均为 `1280×704`。
- M3/I2V 全 20 样本无模型预检通过。
- M4/T2V 全 20 样本无模型预检通过。
- 数值检查不再叫“物理通过”，生成后只会写 `numeric_pass`。

本轮没有批量运行 20 个正式 Wan 视频，因此不能把“代码和数据预检通过”说成“视频质量已
验证”。下一步应先用 P01、P02、P06、P10、P13、P19、P20 各 3 个固定 seed 跑 P0，再与旧
结果并排比较；只有仍然稳定出现身份丢失、动作重放或阶段突变时，才进入一个 StateBridge、
一个状态损失的 P1 训练。
