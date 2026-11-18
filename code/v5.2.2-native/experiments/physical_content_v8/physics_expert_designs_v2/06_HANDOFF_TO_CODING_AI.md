# 可直接交给编码AI的实施任务

## 任务正文

请基于本包选定的**一个**候选实现Physics Expert；不要把三个候选自动合并。先读取`00_IMPLEMENTATION_CONTRACT.md`、对应方案文件、`04_EVALUATION_AND_EXECUTION.md`与来源文件。默认优先`P2-S3-R64`，或按用户明确选择实现P1/P4。

历史起点固定为Lewislius/physgen的commit `1defd6f4b41e42302bc3e4b9b2bf938abbbc686c`中R10A相关代码及公共底座。禁止读取或使用R16及以后候选，禁止修改旧实验结果/权重。新工作区独立。

第一步交付代码映射：列出将复用的R10A增强、loss、native文本、LoRA与采样接口；列出实际Wan版本和旧接口形状。若路径/签名不一致，先定位真实源码，不要凭相似名字导入后续版本。

第二步实现模型：明确`ConditionBundle`、`PhysicsReadOutput`、原生base访问、site选择、native运算拆分。P1只增强route；P2直接在residual预写入后调用一次native block；P4将两个来源条件专家放到同一native FFN中。所有native模块每个block只执行设计规定次数。

第三步安装训练参数并审计：每个参数恰好一个owner，LoRA不得重复包装。新增模块所有output初始化零，上游source/value保持非零。冻结权重不能使用`no_grad()`切断输入梯度。落实`joint400`而非静默继承旧128+272冻结规则。

第四步做测试：公共测试、方案专项测试、参数预算、CFG正负缓存、首帧mask、真实Wan单batch、gradient checkpointing开关、保存恢复。附带CPU检查仅作参考，不替代实际Wan测试。

第五步交付训练/推理/评测入口，但**除非用户明确授权，不自动提交GPU任务、下载新模型或使用外部API**。YAML中null路径需先依据本地实际文件填写并验证，不编造服务器路径。记录未完成项。

禁止新增：V-JEPA等教师、热图监督、动作约束、干预/反事实训练数据、因果置信度、语义保护门、动态注入预算、未知用途中间变量、自动调小guidance、best-of-N结果挑选。已有Qwen/T5只是已有输入编码器，不承担新教师损失。

## 代码交付目录建议

```text
physics_expert_redesign_v2/
  common/
    enhancement.py
    physics_read.py
    native_ops.py
    lora_install.py
    ownership.py
    condition_cache.py
  variants/
    p1_request_key.py       # 仅选定方案必需
    p2_pre_self.py
    p4_cooperative.py
  configs/
  tests/
  train.py
  infer.py
  evaluate.py
  README.md
```

只实现被选择的方案也可以，但公共代码不得假装已经支持另外两版。没有实际评分时`quality_goal_achieved=false`、分数字段null。

## 验收输出

请报告：代码差异、模块调用顺序、所有训练参数owner与数量、CPU/真实Wan测试分别通过哪些、训练入口命令、恢复规则、推理协议、尚未执行的步骤。参数统计分别列base/新增常驻模块/LoRA/总训练/系统外部条件模型。

数学性质通过不等于模型效果通过。PC≥3.85/3.90、SA≥3.25、Joint≥15/40以及高画质与跨基准竞争，是实际训练后的验收目标。既追求新增Joint成功，也保留并提高原本强样本的PC/SA和细节质量。

## 三个最容易写错的重点

P1：`C_execute`必须保持原张量，辅助loss对raw route而非effective route；请求attention只计算一次/条件，不写成十个复制模块。

P2：`hidden = hidden + physical_write`发生在真实residual上，然后调用一次当前native block；不是`norm3(hidden)+write`，不是复制大block，也不是两条forward相减。state_value删除，self/FFN LoRA rank64真实安装。

P4：条件专家读raw物理message和native text output；不用已经零初始化后的physical_write作唯一source；两个专家512宽加起来等于原state_value的参数预算。原生text与FFN不得重复计算。
