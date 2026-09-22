# PhysGen v3：P0 人话分阶段条件基线

当前活动实现都从对应计划 JSON 中读取
`global_semantic`、五个 `stages[].positive`、`cfg_negative` 和 `temporal_route.weights`，
原样用于 Wan。代码不会扩写提示词、搜索关键词、拼接模板或根据实体名称猜测约束。

可切换的三组条件路径是：

- `baseline`：保留原来的弱阶段 c+ 残差；
- `strong_stage`：在活动 block/time token 内，把阶段 c+ 残差校准到 global semantic
  cross-attention 的 `1.0×` RMS；
- `strong_stage_json`：在 `strong_stage` 上，另行注入显式作者化的 JSON 全局事实
  (`0.25×`) 与阶段事实 (`0.50×`)。四路 context 独立编码，不拼成一个长 prompt。

- I2V 读取 `PXX-V2-planimg.json`，并使用文件内明确指定的
  `PXX-i0-1280x704.jpg`。
- T2V 读取 `PXX-V2-plan.json`，不读取首帧。
- 所有首帧 JPG 和 PNG 均为固定的 `1280×704`。
- 五条 `stages[].positive` 就是 setup、onset、evolution、completion、terminal 各自的
  c+，逐句编码，不与 `global_semantic` 拼成一段长提示词。
- 阶段 c- 不单独设置；五个阶段都用 `cfg_negative` 的一个空格作为空比较条件。
- 97 帧视频对应 25 个 DiT 时间 token。每个阶段占一段核心 token，相邻阶段之间各有
  2 个 smoothstep 过渡 token；分配权重直接读取 JSON，不在 Python 中重新推断。
- `cfg_negative` 是 JSON 中直接保存的一个空格。Wan 会把真正的空字符串替换为内置
  负面提示词；一个空格可以关闭该回退，同时不引入任何物体名称。

活动推理只有一个 Python 文件：

`/home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer.py`

旧 Writer、Reader、实体账本、模板编译和 SafeCap 等实现已移到：

`/home/liuzhirui/Project/physGen/code/v3/archive/pre_p0_20260904/`

先做不加载模型的完整检查：

```bash
CHECK_ONLY=1 /home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer_m3.sh
CHECK_ONLY=1 /home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer_m4.sh
```

正式运行：

```bash
/home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer_m3.sh
/home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer_m4.sh
```

P01–P20 的完整 `strong_stage_json` I2V Determined 配置：

```text
/home/liuzhirui/Project/physGen/code/v3/inference/infer_p01_p20_strong_stage_json.yaml
```

对应入口可先本地预检：

```bash
CHECK_ONLY=1 /home/liuzhirui/Project/physGen/code/v3/inference/run_p01_p20_strong_stage_json.sh
```

20 条结构化信息的作者化与同步工具：

```bash
python3 /home/liuzhirui/Project/physGen/code/v3/tools/sync_p0_structured_guidance.py --check
```

P0 的作用是得到一个干净、可解释、可分阶段的 Wan 基线。全局语义始终提供完整事件，
当前阶段的人话 c+ 只在对应时间 token 上补充局部进度。这能减少旧提示词压过主语义、
负面名词反向强化和阶段重复，但不能保证每个随机种子都产生高质量且严格守恒的视频。
本版没有恢复旧 Writer/Corrector/Reader 子系统，只在统一推理文件内保留一个无额外训练参数
的轻量阶段残差；正式质量仍需用固定 seed 视频对比确认。
