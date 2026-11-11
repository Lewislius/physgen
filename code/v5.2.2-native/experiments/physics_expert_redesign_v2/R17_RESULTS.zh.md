# R17-A/B/C 结果摘要

固定 VideoPhy2 40/60、seed42、gamma5、shift5、50 steps；同一输入协议。
分数是 1–5 均值；Joint 为 SA>=4 且 PC>=4 的视频数。

| 版本 | 40 SA | 40 PC | 40 Joint | 60 SA | 60 PC | 60 Joint |
|---|---:|---:|---:|---:|---:|---:|
| R17-A / P1 | 3.050 | 3.450 | 8 | 2.967 | 3.533 | 11 |
| R17-B / P2 | 2.950 | 3.450 | 8 | 2.900 | 3.550 | 11 |
| R17-C / P4 full-width | 3.000 | 3.475 | 10 | 2.900 | 3.583 | 12 |

## 重要说明

- P1 的 `p1_s3_v5` 是修正 text-attention 顺序之后的版本；旧 `p1_s3_v4`/68998 已取消。
- P4 的 `p4_s3_v5` 使用两个全宽 source-conditioned experts；
  旧的 3072→512→3072 P4 `p4_s3_v4`/69049 已取消。
- 三个实现都没有上传视频、checkpoint 或媒体文件；本目录只保留源码、配置、
  scores 和逐例 SA/PC JSON。
- 这些结果没有达到目标 PC3.85 / SA3.25 / Joint15。
- v8 要求的 C0（R10A + joint400）和 C1（C0 + text Q/O LoRA16）
  尚未训练，因此不能把当前 PC 下降单独归因于 Q/O LoRA、joint400 或某个结构。
