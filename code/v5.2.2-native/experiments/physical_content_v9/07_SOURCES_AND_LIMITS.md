# 核验来源与事实边界

本包核对的用户仓库基准commit：`24b957dd4c962db24621cc94286ad6a2ce8a7c23`。实际执行时锁定本地源码与模型revision，不把日后变化的main当作同一实验。

## 已读取的一手源码

- [R17模型：实际安装self/FFN LoRA与文本Q/O LoRA](https://github.com/Lewislius/physgen/blob/24b957dd4c962db24621cc94286ad6a2ce8a7c23/code/v5.2.2-native/experiments/physics_expert_redesign_v2/physical_content/model.py)。本包取消原生文本LoRA，不声称R17没有视频LoRA。
- [R10A物理内容结构](https://github.com/Lewislius/physgen/blob/24b957dd4c962db24621cc94286ad6a2ce8a7c23/code/v5.2.2-native/experiments/physical_content_v7/execution_value_r10/physical_content/model.py)。仅参考结构，不载入该版本模型权重。
- [真实生成损失](https://github.com/Lewislius/physgen/blob/24b957dd4c962db24621cc94286ad6a2ce8a7c23/code/v5.2.2-native/experiments/physical_content_v7/physical_content/supervision.py)。本包固定其FM/tail数学约定。
- [训练侧闭式接口拟合](https://github.com/Lewislius/physgen/blob/24b957dd4c962db24621cc94286ad6a2ce8a7c23/code/v5.2.2-native/experiments/physical_content_v7/fit_interface.py)。本轮重新拟合且只用train输入，不把旧拟合文件当生成checkpoint。
- [官方Wan2.2 TI2V-5B配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)。本次读取blob SHA为`d5d5aed0d5a6c88b58b5661538d575036b22caa2`，宽度3072、FFN14336、30层、24头。
- [官方Wan block](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py)。本次读取blob SHA为`6982fa15db1156dfcfc50d480ae9a42d8525f83d`，包含bias、QK norm、RoPE、时间调制及SA→text→FFN顺序。C/D刻意使用无bias的物理value执行，不得宣称与完整官方block等价。
- [R17结果与归因边界](https://github.com/Lewislius/physgen/blob/24b957dd4c962db24621cc94286ad6a2ce8a7c23/code/v5.2.2-native/experiments/physics_expert_redesign_v2/R17_RESULTS.zh.md)。R17未达标，不能从这组同时变化多个因素的结果断言文本LoRA是唯一根因。

## 方法依据

- Hu等，LoRA，[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)：冻结官方base与低秩可学习增量。论文不证明本包视频方案或rank32/64必然达标。
- Lipman等，Flow Matching，[arXiv:2210.02747](https://arxiv.org/abs/2210.02747)：生成速度场回归基础。本包时间方向以本项目实际实现为准。
- ProPhy，[项目页](https://zijunwa.github.io/prophy/)、[arXiv:2512.05564](https://arxiv.org/abs/2512.05564)：作为物理分支/生成能力研究参照，不声称本包复现了其方法或已经超过其成绩。

## 不能从本包得到的结论

本包没有新GPU训练、没有新VideoPhy2/WMB/Physics-IQ/VBench评分、没有新视频画质验证。代码中的CPU数值检查只能说明被检查的代数/日程/冻结契约，没有执行真实Wan大模型。

新版A/B/C/D不是R17-A/B/C的重命名。新增方案的具体效果未知。四个候选使用LoRA的选择、注入位置、初始化及学习率都是有依据的设计处方，不是已验证最优超参数。
