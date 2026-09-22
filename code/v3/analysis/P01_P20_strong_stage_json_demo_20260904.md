# P01–P20 strong_stage_json 完整 Demo

日期：2026-09-04  
范围：I2V、seed 42、97 帧、1280×704、50 steps；不增加 StageBridge，不训练新参数。

## 为什么此前 19 条会被拒绝

`strong_stage_json` 不允许从自由提示词自动猜实体数、状态、不变量或允许变化，因此加载器要求
每份 plan 显式包含 `structured_guidance`。此前只有 P02 完成了该字段，P01 和 P03–P20 会在
模型加载前主动失败。这是数据完整性保护，不是运行时或模型故障。

现在 P01–P20 均已使用 `structured-text-guidance-v1` 补齐，并同步到：

- 主源：`prompt/p0_direct_prompts_p01_p20.json`；
- I2V：20 份 `PXX-V2-planimg.json`；
- T2V：20 份 `PXX-V2-plan.json`。

每条显式声明全部计划实体的 count、身份/外观/材料不变量、初始条件及其活动阶段，以及五个
阶段的 presence、count、state、mutable 和关系事实。刚体只开放位置、速度或姿态；融化、
爆裂、碎裂、撕裂、扩散和海绵弹性压缩仅在对应阶段开放形状、相态或拓扑变化。

## 四路条件及默认强度

四路文本 context 独立编码，不互相拼接：

1. global semantic：全部 30 个 Transformer block 的主条件；
2. stage c+：仅 block 14–23，并按 25 个 latent 时间 token 路由；
3. JSON global：仅增强注入 block，目标 RMS 为 global 的 `0.25×`；
4. JSON stage：仅增强注入 block，按阶段路由，目标 RMS 为 global 的 `0.50×`。

stage c+ 在活动 block/time token 的目标 RMS 为 global 的 `1.0×`。三路附加残差相加后，
统一限制在 global RMS 的 `1.75×`。这表示阶段 c+ 在其局部位置与 global 等强，但 global
仍覆盖全部 30 层，所以不能理解为整网累计贡献完全相等。

配置中保留的 `STAGE_STRENGTH=0.05` 只属于 baseline 兼容参数；在 `strong_stage` 和
`strong_stage_json` 中不参与计算，增强模式以以上 RMS 比例为准。

## 已完成检查

- 41 份 JSON 同步检查通过；
- P01–P20 的 structured entity ID 与原 plan entity ID 完全一致；
- 每条严格包含 setup/onset/evolution/completion/terminal；
- 初始条件事实只进入声明的活动阶段；
- 全部结构化 stage 文本均小于 350 个英文词；
- 7 个 v3 单元测试通过；
- `CHECK_ONLY=1 + SAMPLE_IDS=all + strong_stage_json` 完整预检通过；
- 新 Determined YAML 解析通过。

## 运行入口

配置：

```text
/home/liuzhirui/Project/physGen/code/v3/inference/infer_p01_p20_strong_stage_json.yaml
```

提交命令：

```bash
DET_MASTER=10.130.130.5:4096 det experiment create \
  /home/liuzhirui/Project/physGen/code/v3/inference/infer_p01_p20_strong_stage_json.yaml
```

运行时每条视频保存 `structured_prompts.used.json`、`conditioning.audit.json`、
`manifest.json` 和 `video.mp4`；整批最终以 `results.json` 汇总成功/失败数。

## 运行状态

P02 三臂预跑：experiment `59377`，trial `59973`；JSON 分支已通过首个扩散步。  
P01–P20 完整任务：experiment `59387`，trial `59984`，已于 2026-09-04 14:49 UTC 启动。
运行目录为 `outputs/trace_writer/trace-v3-p01-p20-strong-stage-json-20260904-144957`；
其运行时 preflight 再次确认 20/20 样本均启用 structured guidance。

## 解释边界

这里的 JSON 会先渲染成显式、阶段局部的自然语言，再通过现有 T5/cross-attention 注入。它比
单纯把 JSON 拼进 global prompt 更可控，也能做强度审计；但仍属于文本软约束，不能保证像未来
实体级空间控制或硬物理约束一样逐帧严格执行。20 条视频用于判断该软约束是否带来总体收益。
