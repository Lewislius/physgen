R1-400 / PC 3.625 参考版本备份

这是用户要求保留、供后续消融的原始参考设计。训练实验66778，第400步；完整40例由66855恢复完成。固定gamma=5，seed=42，50步shifted Euler，shift=5。它与R3短训修订是不同版本。

完整VideoPhy2-40：SA=2.95，PC=3.625；SA>=4为11/40，PC>=4为18/40，Joint为10/40。配对native LoRA基线实验66781第400步：SA=3.075，PC=3.45。

独立保存了两方400步checkpoint（包含原有训练状态）、训练配置/日志、实际训练与推理源码快照及共享代码、语义接口与词特征缓存、固定40例首图/条件/噪声/Qwen证据、两方生成视频和逐例评分、已经完成的原400步模块对照及对照实现。manifest.json逐文件列出来源与副本路径。所有文件是独立副本。

workspace/保持各文件相对于/home/liuzhirui的原路径。原始配置与checkpoint内部的绝对路径保留，用于明确复现来源；后续消融应从这个版本的代码与配置另建实验目录，保持本备份不变。原训练架构位于workspace/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/snapshots/20261030T141432Z；原推理入口在inference_snapshots/20261030T151158Z。新增固定gamma5的16例模块对照代码单独存于module_controls16/snapshots/20261031T053400Z，不替代原实现。

完整官方Wan/Qwen/评估模型、已有完整训练视频缓存和全部训练Qwen证据仍复用本机原目录，详见manifest.json的external_dependencies；本备份不是携带全部基础模型和训练数据的可迁移环境。已单独保存固定40例的生成输入与Qwen证据。没有启动训练、下载或重评估。
