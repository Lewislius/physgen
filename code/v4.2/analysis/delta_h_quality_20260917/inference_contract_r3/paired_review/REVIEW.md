# 同条件完整视频对照

逐条播放完整视频，按 case 对照不同 variant/checkpoint。CSV 中 failure 列填 0=未发现、1=失败、na=不适用；未填项不计成功。reviewer 与 reviewed=1 标明已审核。正常遮挡、出入画、液体形变不能算物体失败。静止视频即使不闪烁也不能视为动作成功；同时检查提示词动作和事件是否完成。

half 在限幅后缩放两个 writer 的 ΔH；writer5_only/15_only 只关闭另一个 writer，仍计算并传递两个 P 状态。不同 variant 的完整轨迹随后会分叉；它们不是同一个 x 上的单步干预。训练 health 中的 writer_ablation 才是同一个 x、noise、sigma 的单步干预。对照脚本不把 ΔH 大小、特征相似度或 FM 自动换算成画质/物理分数。
