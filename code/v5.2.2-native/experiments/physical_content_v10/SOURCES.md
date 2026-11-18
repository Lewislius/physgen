# 一手资料与证据边界

以下资料用于核对机制，不表示论文验证了本候选。引用日为2026-11-11；实际复现需另外冻结各仓库commit、基础模型revision及缓存hash。

1. ProPhy论文：https://arxiv.org/abs/2512.05564 ，核对v2第4–6、8页：视觉latent与物理先验结合、类别/局部对齐、20K训练视频、VideoPhy2通过率口径。
2. ProPhy公开Wan实现：https://github.com/zijunwa/ProPhy/blob/main/prophy/models/transformer_wan.py 。首个physical block把hidden_states加到control state；随后物理分支推进，预先形成hints后注入主干。不是每一层都回读该层最新主干状态。
3. PhysVid论文：https://arxiv.org/abs/2603.26285 ，核对v2第6、8页。约53K样本、有效batch64、1000+2000更新；VideoPhy2消融中baseline SA0.2845/PC0.6144，PhysVid SA0.2775/PC0.6411。这是论文自己的口径，不是我们40例均分。
4. 官方TI2V入口：https://github.com/Wan-Video/Wan2.2/blob/main/wan/textimage2video.py 。同一TI2V-5B支持标准T2V与I2V。I2V首帧条件覆盖与token timestep处理，和T2V无已知帧是任务边界差别，不是新噪声策略。
5. 项目现有I2V输入：https://github.com/Lewislius/physgen/blob/main/code/v5.2.2-native/experiments/physical_content_v7/physical_content/teacher.py 。缓存scope=first_image_and_request_only。
6. 项目现有FM聚合：https://github.com/Lewislius/physgen/blob/main/code/v5.2.2-native/experiments/physical_content_v7/physical_content/supervision.py 。当前函数硬编码[:, :, 1:]；增加T2V时必须将监督范围参数化，I2V保持数值等价。
7. 历史视频回读：https://github.com/Lewislius/physgen/blob/main/code/v5.2.2-native/experiments/physical_content_v7/reciprocal_fusion_r12a/DESIGN.zh.txt 。已有state读回C再写回的尝试；不是本次新发现。
8. 历史局部视觉算子：https://github.com/Lewislius/physgen/blob/main/code/v5.2.2-native/experiments/physical_content_v7/local_state_operator_r15c/DESIGN.zh.txt 。已有基于视觉K/V的执行尝试，不能仅凭视觉value存在推断有效。
9. R10A增强：https://github.com/Lewislius/physgen/blob/main/code/v5.2.2-native/experiments/physical_content_v7/execution_value_r10/physical_content/model.py 。采用其全宽计算机制，不采用项目训练checkpoint。
10. B400+I联合读取：https://github.com/Lewislius/physgen/blob/main/code/v5.2.2-native/experiments/physical_content_v7/b400_joint_read_eval/joint_read.py 。推理增益不能视为从头训练的收益保证。
11. R18评测：https://github.com/Lewislius/physgen/tree/cde6a88e/code/v5.2.2-native/experiments/physics_expert_abcd_fresh1000/eval_artifacts/r18c_r18d_20261111 。已公开结果不等于已公开完整训练源码/日志/权重。本文件不声称定位了R18的具体实现故障。
12. VideoPhy2官方：https://github.com/Hritikbansal/videophy/blob/main/VIDEOPHY2/README.md 。Joint为PC>=4且SA>=4；均分与通过率分别报告。
13. WorldModelBench：https://github.com/WorldModelBench-Team/WorldModelBench/blob/main/README.md 。T2V用text_first_frame+text_instruction；I2V用first_frame+text_instruction。
14. Physics-IQ：https://github.com/google-deepmind/physics-IQ-benchmark 。Original/Verified、I2V/V2V、单样本/BoN需要区分；纯T2V不能冒充I2V成绩。

尚未完成：真实Wan集成、GPU优化、任何本候选视频生成、PC/SA/Joint/WMB/Physics-IQ/VBench评测。配置检查不能替代这些验证。
