# v4 单卡推理

默认 checkpoint 已固定为训练 500 步的最终 A1 权重：

```text
/home/liuzhirui/Project/physGen/code/v4/checkpoints/wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/checkpoint-final
```

推理严格读取此目录的 `config.yaml` 和 `corrector.pt`。791 个 FP32 张量、974,182,486 个参数严格匹配且全部有限；六套 A corrector 分别位于第 5、10、15、20、25、30 层，每个位置执行一轮。`stage=A`，因此 `VARIANT=full` 等于 A；B writer 和 B 状态参数为零，不启用 B。

此 checkpoint 的初始化为 `image_text_gt`，训练使用 `gt_condition_dropout=0.25`，验证也使用无 GT 条件。推理采用训练过的首帧＋文本路径，不需要目标视频、JEPA 教师或 WISA 缓存。

**状态适配：** 此次训练 `state_warmup_every=0`，每个去噪训练样本重新初始化 P。因此默认 `RESET_STATE=auto` 按保存的训练配置选择每步重置 P，两路 CFG 保持独立。旧 checkpoint 若训练启用了 warmup，则 auto 保留跨步接续。`RESET_STATE=0` 显式接续、`RESET_STATE=1` 显式重置；每个结果 JSON 记录实际选择。每步内部仍按六个 A 位置依次传递并更新 P。

## 提交单卡 Ada 48G

```bash
cd /home/liuzhirui/Project/physGen/code/v4/inference
det experiment create infer_wisa_native_p_1xada48g.yaml .
```

此配置使用 `resource_pool: ada-48g`、`slots_per_trial: 1`、`DEVICE=0`，固定 moviestory 环境，默认 **41 条视频（原例 I2V 1 条＋20 demo 各 I2V/T2V 2 条）、101 帧、24 fps、50 步 Euler、CFG=5、shift=5、seed=42**。当前已分配到 GPU 的容器中也可直接运行：

```bash
bash /home/liuzhirui/Project/physGen/code/v4/inference/infer_wisa_native_p_1xada48g.sh
```

其他入口仍可用：`1x48g` 对应 `amp-48g`，`1x80g` 对应 `amp-80g`，`1x96g` 对应 `blk-96g`。同一批条件和模型结构适用于这些单卡入口。提交时任务环境来自 YAML；当前 shell 的变量不会替换 YAML 内的变量。

## 两组测试条件

默认 `SUITE=both`，依次生成：

1. 原有示例：`overfit_ref5.jpg`，提示词 `The woman smiles and waves at the camera.`，与原 `inference_outputs/wisa_native_p_1x96g` 中的条件一致。
2. 自建 P01–P20 的 I2V：读取 `v1/demo/Pxx/Pxx-i0-1280x704.png` 和 `Pxx-origin.txt`。文本只去除文件首尾空白，不读取 V2 plan、cplus/cminus 或旧 archive。所有 20 组文件已核验配对及解码。
3. 自建 P01–P20 的 T2V：使用相同 origin.txt、seed、50 步和输出时序，完全不读取首帧图片；默认输出 512×288。

无首帧模式的全部 latent 帧都从噪声开始，不固定第一帧；Wan 的全部 token 使用当前噪声时间步；P₀ 初始化器只接收文本和时空位置，不传入图片 token、零图或 GT。**该 checkpoint 训练时始终提供首帧，T2V 是在完整已保存权重上的扩展测试模式，其效果不能等同于经过无首帧训练的模型。** 输出 JSON 会记录 generation_mode、image_conditioned、first_frame_clamped 和该训练差异。

首帧按训练相同的 letterbox 与画幅规则处理：最大长边 512、面积 147456、32 倍数。原示例输出 384×384，20 个自建例输出 512×288；1280×704 是输入图片尺寸。101 帧、24 fps 的首尾时间差为 100/24 秒，MP4 播放长度为 101/24 秒。

| 变量 | 用法 |
|---|---|
| SUITE | `both` 原例＋20 自建的双模式，共 41 条；`demo` 只跑 40 条自建；`single` 单条 |
| MODE | 原例／单条模式，默认 `i2v`；`t2v` 不使用 IMAGE |
| DEMO_MODES | 默认 `both`；也可 `i2v` 或 `t2v` 只跑 20 条 |
| WIDTH、HEIGHT | T2V 默认 512、288；不从图片读取尺寸，须满足训练画幅限制 |
| SAMPLE_IDS | 默认 `all`，严格对应 P01–P20；也可 `P01,P02` |
| DEMO_ROOT | 默认 `code/v1/demo` |
| IMAGE、PROMPT、NEGATIVE_PROMPT | 原例或单例的条件，负向默认空 |
| CASES | 可选 JSON 条件清单，替换 `both` 中的原例；兼容 `evaluation/export_cases.py` 的输出，每项有 case_id/image/prompt，可附 mode/frames/fps/duration，T2V 可附 width/height |
| CHECKPOINT | 显式目录，默认上面的最终权重 |
| CHECKPOINT_RUN、CHECKPOINT_ROOT | 未设置 CHECKPOINT 时，显式按训练运行名前缀查找最近完成的 checkpoint-final |
| FRAMES、FPS、DURATION | 默认 101、24；帧数向下对齐 4n+1，且不超过训练 max_frames；duration 可显式指定首尾时间差 |
| SAMPLING_STEPS、SOLVER | 默认 50、euler；也支持 unipc 和 dpm++ |
| SHIFT、GUIDANCE、SEED | 默认 5、5、42 |
| VARIANT、WRITER_OFF | 默认 full；wan 为校正旁路；WRITER_OFF 只关闭指定位置写回，例如 `A@5 A@10` |
| RESET_STATE | 默认 auto；按 checkpoint 训练设置选择，0/1 可显式覆盖 |
| OUTPUT_DIR | 批量输出目录；默认硬件目录下创建带时间戳的新目录 |
| OUTPUT | `SUITE=single` 的 MP4 文件路径 |
| RESUME | 1 配合原 OUTPUT_DIR，跳过已完成且条件一致的样例 |
| CHECK_ONLY、REPORT | 1 在 CPU 严格检查权重、模型资产、全部输入、画布、时间和状态策略；REPORT 保存 JSON |
| PARSE_ONLY | 1 只打印最终命令 |

单例与自建子集示例：

```bash
SUITE=single IMAGE=/absolute/first_frame.png PROMPT='Your prompt.' \
  bash inference/infer_wisa_native_p_1xada48g.sh

SUITE=demo SAMPLE_IDS=P01,P02 \
  bash inference/infer_wisa_native_p_1xada48g.sh
```

脚本末尾可附加 Python 参数，重复标量以最后一个为准。正式配置的默认 50 步不会被本地冒烟验证的 2 步覆盖。

## 输出、模型复用和续跑

批量输出默认位于 `inference_outputs/wisa_native_p_1xada48g/final_<时间戳>/`：

```text
cases.json
summary.json
reference/reference/i2v/seed42_full.mp4
reference/reference/i2v/seed42_full.json
reference/reference/i2v/seed42_full.steps.jsonl
demo20/P01/i2v/seed42_full.mp4
demo20/P01/t2v/seed42_full.mp4
...
demo20/P20/i2v/seed42_full.mp4
demo20/P20/t2v/seed42_full.mp4
```

每个样例都有 MP4、逐步诊断和结果 JSON，记录实际权重、阶段、画布、时序、状态策略、输入图片哈希、origin.txt 路径与哈希、显存峰值。`cases.json` 固定条件和参数；`summary.json` 持续记录完成情况。50 步的每个样例执行 100 次 Wan 前向、50 次采样更新、600 次 A 校正；两路 CFG 和不同视频之间不会混用 P。

整批只加载一次 T5、Wan 和校正器；先编码全部不同提示词并保存 CPU 特征，再释放 T5。每条视频采样期间 VAE 留在 CPU；解码前把 Wan/校正器移回 CPU，解码后复用模型处理下一条。Wan 主要权重为 BF16，关键时间/输出/norm、校正器权重和 P 保持 FP32。

续跑时使用原有目录及完全相同的输入与采样参数：

```bash
RESUME=1 OUTPUT_DIR=/absolute/path/to/the/original/final_run \
  bash inference/infer_wisa_native_p_1xada48g.sh
```

参数、权重标识或输入内容改变时拒绝复用旧任务目录；应创建新的输出目录。中断样例会重跑，完成样例保留。

## 本次检查

此前 I2V 权重与 21 个条件的报告：[validation/checkpoint_final_20260913.json](validation/checkpoint_final_20260913.json)。相关 CPU 回归覆盖实际权重、无 GT 分支、三种采样器、CFG、首帧固定、状态选择、输入配对、批量模型复用与断点续跑。

```bash
CHECK_ONLY=1 REPORT=/tmp/physgen-v4-final-check.json \
  bash /home/liuzhirui/Project/physGen/code/v4/inference/infer_wisa_native_p_1xada48g.sh
```

快速 GPU 检查使用本机 RTX 3090、101 帧、2 步，已实际生成原例及 P01–P05；按快速验证要求停止了余下冒烟用例。通过样例的采样显存峰值约 14.41 GiB，完整启动还需容纳 T5 加载峰值。这些已有视频仅覆盖 I2V。本次新增 T2V 只执行小型 CPU 逻辑检查，未启动新视频生成。已有 I2V 验证支持单卡 48G 配置的可运行性判断，但没有在目标 Ada 上执行正式 50 步批量，也不评价物理质量。详见 [validation/checkpoint_final_20260913_gpu_batch.log](validation/checkpoint_final_20260913_gpu_batch.log)。
