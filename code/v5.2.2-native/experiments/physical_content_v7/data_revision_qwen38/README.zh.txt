本轮目标：改进真实训练视频和文本的匹配，同时保持 WISA 原有语言风格、篇幅、细节密度。

API：使用用户明确提供的百炼北京工作空间，模型 qwen3.8-flash。
调用配置在 /home/liuzhirui/.config/physgen/qwen_api.json；不写进代码、训练 YAML 或结果日志。
没有下载、更换本地模型，也没有改写已有训练数据和运行中的 R11A。

标注方法
1. 严格解码原来 81 帧训练窗口及原来的裁剪几何，提供 41 张有时间标记的有序画面。
2. 明确这些是长视频的时间截取片段，首尾是任意边界，不要求完整动作、首尾匹配或闭环。
   不因首尾不同判断物理错误，不为了顺畅或故事完整补造触发、碰撞、停止、落地或完成。
   看见内部剪切时也不把两侧画面强行解释成连续物理变化。
3. 先在不读旧标签的情况下观察视频；再依据视频和 WISA 的文体统计写描述。
   文体统计保留通用开头、原文词数、段落和语气，不把原来可能错误的事件塞回提示。
   篇幅目标为 WISA 原文的 85%-115%，事实准确优先；例外必须记录，不能靠编造凑词数。
4. 使用 Flash 思考模式。此前非思考视频列表试验出现“移动的球相对静止”和复制旧因果等错误，
   相关结果保存在 runs 中且不采用。新版使用逐帧时间标签并显式处理时间截断。
5. 增加一次独立的视频事实核查，直接纠正描述中的错误，再生成两侧输入 reasoning。
6. 已完成先行 16 例：4 个已有具体问题样本 + 12 个实际400步训练样本中固定随机抽取的例子。
   按真实画面复核后确定批量版本，不能将模型自己的确认当作人工正确率。

兼容现有架构
完整训练片段只用于离线修正训练 caption，这是监督标签。
输入侧 reasoning 重新请求时，只给首帧 + 请求文字，正负 CFG 两个分支分别处理。
API 生成的是 1-4 条简短关系文字，不提供模型隐藏特征、不替换 Qwen3-VL embedding。
encode.py 使用本地冻结 Qwen3-VL-8B-Instruct 和完全相同的输入模板重新编码：
memory / reasoning words / request words 都保留4096维；Q256、Q→C 和 Wan 物理融合不改。
新 caption 同步使用已有 UMT5 编码；native word target 也用已有冻结 T5 + Wan text_embedding 重建。
禁止只改 caption 字符串却沿用旧文本、reasoning、words 或 native_words 张量。
cache_reader.py 为下一数据版本显式读取新缓存，并核对 caption、relation、sample、revision 对应。
不把 API 通用 embedding 当作 Qwen3-VL 隐藏状态，不能把两者当作同一特征空间。

执行入口
使用既有 tools/runtime_python.sh，不修改 Python/conda/Determined 环境。
run_dataset.py --scope train400 --output runs/20261103_dataset --workers 16
encode.py --output runs/20261103_dataset --stage all
finish_preparation.py --output runs/20261103_dataset
prepare_inference.py 分 reasoning 和 encode 两阶段；首帧/原始请求不变，只生成新 reasoning 候选。
所有 GPU 编码仍要求单卡 RTX6000 Blackwell；H100 会在加载模型之前退出。

完成范围与采用条件
文本 API 已调用成功；小批候选标注和接口检查不等于新模型生成质量已经提升。
pilot 编码目录是明确标记的子集，不是可直接代替完整训练集的 manifest。
未来完整数据版本须保留原始训练/验证划分和采样顺序，完成实际训练与接口拟合所需覆盖，
并从新训练输入重新拟合接口/统计量；不得把试标16例当成新的训练数据集直接训练。
当前 R11A67214 继续原架构对照；新数据候选接后续已授权的版本，不在训练中途换标签。
采用后仍用相同400步、原40+20/60及固定16配对评测，分别报告 PC、SA、Joint 和实际视频质量。
最大版本仍是 R12B。

最新用户范围修正：只处理400步实际需要的1600条训练样本，暂不处理剩余2066条训练数据，
不为验证集调用API。原始3857条manifest及采样顺序保留，只覆盖所选1600条的文本和配套特征。
每条均生成正/负两份输入缓存以供配对使用；接口拟合按真实训练使用的1446正/154负取样，
不会因为缓存同时存在两份就错误地把无文本训练权重提高到50%。
接口使用其中固定64条验证可预测性后，在全部1600条实际训练输入上重新拟合。
这是训练输入间的接口拟合，不是生成质量评测，也不会改变未来视频为生成条件。
缓存仅声明支持seed42、accumulation4、最多400步；其他采样设置须另行准备，不能静默回退旧特征。

自动衔接：advance_preparation.py每30分钟更新本地API进度，完整后才提交冻结编码YAML。
只排一个blk-96g RTX6000编码作业；GPU排队每30分钟检查，H100加载模型前拒绝。
编码完成后在CPU拟合接口并生成training_data_overlay.yaml、encoded/readiness.json。
本流程到数据准备完成即停，不提交新模型训练、不修改R11A67214。
1600条完整API流程按21条实际token测量预计76.81元，约80元；不含前期试标/重试/GPU。

后续数据对比的推理与消融接入（已实现，尚未启动新训练/评测）
- condition_controls.py 的 InferenceCache 仅读取明确指定的新输入目录；校验原始请求、首帧路径、时长和本地4096维特征，缺失时不回退旧 reasoning。
- prepare_inference.py --stage encode 可附 --controls empty_reasoning reasoning_swapped --reasoning-donor DONOR_JSON；使用已有 API 文本重新本地编码自己的首帧/请求，不额外调用 API。无关 donor 必须具有同样关系条数；空文本保留同样数量的结构分隔符，仍含首帧/请求上下文。
- 新 prepare_mean_condition.py 从新版本训练输入计算64例双分支完整 C 均值，兼容不同关系条数。推理需用 mean_with_own_prior 保留各例各分支的原关系先验；固定的是 C 张量，关系数量元数据仍保留。均值还会改变 C 幅度，不能声称纯内容消融或完全不看 Qwen。
- 原 group_log_prior 在原始注意力 logits 相等时给予每个关系和零值参考各一份先验；1/2/3/4关系的物理总质量分别为1/2、2/3、3/4、4/5。API正负分支数量分布不同，故不能悄悄统一分组再将变化全部归为文本内容。此公式不是实际训练后注意力/质量测量。
- 四项新控制边界测试通过，仅证明输入和先验接入正确。旧 R11A 源码、训练与评测、冻结的065716编码任务不变；待本轮证据与1600条数据准备齐全后，再接入独立的新数据版 R10A/R11A runner。
