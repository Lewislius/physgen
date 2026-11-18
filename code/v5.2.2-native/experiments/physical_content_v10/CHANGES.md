# 本轮修订清单

1. 删除T2/I2中按三个时间段生成局部物理描述的预处理。
2. 删除L_local、J:64->3072辅助head、局部标签缓存和其有效性mask。
3. 全部四版只有FM、现有tail聚合、raw route MSE；ST仍关闭。
4. 不创建新noise schedule。显式区分标准T2V无已知帧和标准I2V有已知首帧；原训练公式/采样器不作为创新项。
5. T1/I1保留N1的核心双路径物理写入；T2/I2保留N2的native FFN条件adapter。
6. 不因ProPhy图示增加video-to-C全局反馈环，不复制R18视频Expert。
7. 引入明确task-dependent监督范围：旧I2V loss的[:, :, 1:]不能直接用于T2V。
8. 冻结物理机制与训练计划；不把推测性分数写进实施文档。

此前带L_local或三时间段的T1/T2/I1/I2草稿不适用于本轮。历史N1/N2文档保留，不能把其I2V-specific first-frame处理直接当作T2V实现。
