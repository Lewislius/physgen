# physical_content_v7 目录地图

更新时间：2026-11-10 05:50 UTC（基于当前目录、`prepared.json`、`DESIGN.zh.txt`、`reports/` 和 `runs/` 实际内容核对）

## 0. 先记住三条规则

1. **一个 `*_rN` 目录 = 某一轮 R 版本的一个候选设计**，但目录名是“设计主题”，不是版本号。例如 `execution_value_r10` 是 R10A，不是“R10 全部”。
2. **目录里的 `runs/<experiment_id>/` 才是训练权重和日志**；`reports/` 是复盘/中间报告；`inference_outputs/` 是生成视频与评分；`snapshots/` 是冻结源码。找版本先看 `runs/` 的 experiment id，再看 `reports/`。
3. **R1 没有单独的 `*_r1` 训练目录**。R1 主体在顶层 `runs/physical_content_66778`、`snapshots/20261030T141432Z`、`inference_outputs/physical_content_66778`。`request_anchor_r1` 只是 R1-400 的推理期变体。

## 1. 版本主线（R1 → R16K）

| 版本 | 对应目录 | 设计一句话 | 当前状态 |
|---|---|---|---|
| **R1** | 顶层 `runs/physical_content_66778`、`snapshots/`、`inference_outputs/` | 物理内容接口 + 独立物理 K/V/O + C 调制；质量基准 | 400步完成，保留 |
| R1-1000 | 顶层 `runs/physical_content_66856` | R1 续训到 1000 步 | 完成，PC 下降，未采用 |
| R1-request-anchor | `request_anchor_r1` | R1-400 推理期请求锚定变体 | 仅推理，无新训练 |
| **R2** | `process_r2`、`inference_guidance_r2` | R2 过程监督 / 推理侧 guidance 试验 | 均未形成正式40例；guidance 在 heldout8 为负向 |
| **R3** | `early_execution_r3` | 早期执行修订（64步短预算） | 仅固定16例；权重已删 |
| **R4** | `grounded_read_r4` | 关系物理词读取 + Q→C 初始梯度（64步） | 40例完成；权重已删 |
| **R5** | `fixed_interface_r5` | 固定语义坐标，检验生成梯度能否学到 Q 修正 | 仅实现检查，未提交训练；无权重 |
| **R6** | `content_modulation_r6` | 让 Q 更新作用于输入物理内容（128步） | 40例完成；权重已删 |
| **R7** | `native_coordinate_r7` | 原生坐标写入（128步） | 40例完成；权重已删 |
| **R8** | `content_reader_r8` | 内容读取器（128步） | 40例完成；权重已删 |
| **R9** | `r1_content_reader_r9` | R1 复现 + 内容读取（400步） | 40/60完成，保留 |
| **R9.1** | `content_value_r9_1` | Q 选择真实内容值，并分离通用修正 | 40/60完成，保留 |
| **R10A** | `execution_value_r10` | 语义寻址 C_route 与物理执行 C_execute 分工 | 40/60完成，保留 |
| R10A-API | `r10a_api400` | R10A 在 API 修订数据上重训 | 40/60完成，保留 |
| **R10B** | `grounded_execution_r10b` | 原生执行语义核 + 视觉依据 + 内容路由修正 | 40/60完成，保留 |
| R11A-old | `request_adapt_r11a` | 原生语义适配 + 顺序物理执行（旧版，67214） | 40/60完成；权重已删 |
| **R11A-reworked** | `aligned_request_r11a` | 保留 R10A 物理执行 + 按原请求寻址的执行通路（67265） | 40/60完成；权重保留 |
| R11A-rework-design | `native_request_r11a` | R11A 分阶段 + API 数据的设计稿 | 未提交、未训练 |
| **R11B** | `request_memory_r11b` | 独立、全宽的请求语义记忆 | 40/60完成，保留 |
| **R12A** | `reciprocal_fusion_r12a` | 原生文本先更新，物理 C 双向读取/修正当前视频状态 | 40例完成；权重已删 |
| **R12B** | `semantic_execution_r12b` | 以原生文本更新约束物理执行 | 40例完成；权重已删 |
| **R13A** | `native_operator_r13a` | 物理条件通过原生视觉运算执行（FFN调制） | 40/60完成；权重已删 |
| **R13B** | `interaction_query_r13b` | 物理条件控制原生时空读取，保留直接物理写入 | 40/60完成；权重已删 |
| **R13C** | `coupled_attention_r13c` | 物理条件参与原生视频注意力 Q/K 调制（32+368阶段） | 40例完成；60例被用户取消；权重仍在 |
| **R14** | 无目录 | R14A/R14B | 用户明确跳过，未运行 |
| **R15A** | `grounded_fusion_r15a` | 有语义依据的物理内容 + 独立物理读写 | 40例完成；权重已删 |
| **R15B** | `request_bound_fusion_r15b` | 物理内容通过原请求参与生成 | 40例完成，保留 |
| **R15C** | `local_state_operator_r15c` | 物理条件驱动的局部视频状态执行 | 40例完成，保留 |
| **R15D** | `denoising_matched_fusion_r15d` | 与文本状态、去噪阶段配合的直接物理融合 | 40例完成；权重保留 |
| **R15E** | `semantic_addressing_r15e` | 语义寻址（主版本，67955） | 40例完成；权重已删 |
| R15E-redesign | `semantic_response_fusion_r15e` | 物理写入后的原生文本响应 + 真实视频时间关系监督 | 重设计稿，无训练权重 |
| R15E-probe | `state_response_fusion_r15e` | 只读探针/诊断 | 无训练权重 |
| R15E-local-probe | `native_request_local_r15e` | local-K 条件预检 | 预检失败（“Local K conditioning does not affect the prediction”），无训练 |
| **R15F** | `grounded_semantic_fusion_r15f` | 有依据的语义融合 | 40例完成；权重已删 |
| **R15G** | `source_residual_fusion_r15g` | 源残差融合 | 40/60 + WMB/PIQ 完成；权重已删 |
| **R15H** | `semantic_feedback_fusion_r15h` | 直接物理执行 + 原请求语义响应 | 40例完成，保留 |
| **R15I** | `joint_memory_fusion_r15i` | 物理分支内联合读取直接物理信息与请求关联信息 | 40/60完成，保留 |
| R15I-probe | `r15i_design` | 为 R15I 做的只读联合读取探针（68309） | 探针完成，不是正式 R15I |
| **R16A** | `native_joint_read_r16a` | native-request 条件先验 + 直接读取原始 C value | 400步完成；40/60已评分，controls收尾 |
| R16A-NOZERO | `native_joint_read_r16a_nozero` | R16A 去掉物理零值 reference key/attention | 训练400步完成；评测在22/40时被取消，无评分 |
| R16B-draft | `grounded_joint_read_r16b` | R16B 早期草案 | 34步后取消，禁止恢复权重 |
| **R16B-final** | `native_budget_joint_read_r16b` | 原生 query 分配 direct/request/reference mass 的联合读取 | 400步完成；40/60已出分，controls收尾 |
| **R16C** | `native_request_local_r16c` | request-bound local video-state execution：两路 C 读条件化 local video Q，K/V 来自视频状态 | 训练400步；40例生成36/40，未评分 |
| **R16D** | `native_request_local_r16d` | request-bound local QK modulation：条件同时偏移 local Q/K，V 为视频状态 | 400步；40/60已出分，controls收尾 |
| **R16E** | `native_request_local_r16e` | local QKV modulation + request-bound full-width V correction | 400步；40已出分，extra20进行中 |
| **R16G** | `native_response_fusion_r16`（runs 68625/68627） | pure R15C local + 三路全局 execution 读；68627 是 G-SAFE 安全变体 | 400步；40/60已出分，部分controls进行中 |
| **R16H** | `native_response_fusion_r16`（run 68621） | joint-conditioned local + 三路全局读 | 400步；40/60已出分，controls收尾 |
| **R16I** | `native_response_fusion_r16`（run 68622） | full local conditioning + R16E V correction + 三路全局读 | 400步；40已出分，extra20进行中 |
| R16J/K | `native_response_fusion_r16`（runs 68654/68699/68698 等） | 更新的候选标签，配置在 snapshots 中 | 仅训练到19–50步，未完成；当前有活动训练 |
| R16F | 无目录 | — | 不存在 |

## 2. 目录分类总表

### A. R 版本候选目录（每个目录是一个候选设计）

| 目录 | 版本/内容 | 关键实验ID | 权重状态 |
|---|---|---|---|
| `request_anchor_r1` | R1-400 推理期请求锚定变体 | 66778 + 推理 | 复用 R1 权重 |
| `process_r2` | R2 过程监督训练设计（未提交） | — | 无 |
| `inference_guidance_r2` | R2 推理侧 guidance（引导拆分） | 66778 + 推理 | 复用 R1 权重 |
| `early_execution_r3` | R3 早期执行修订（64步） | 66870（候选）、66866/66913（native基线） | 候选已删；native基线保留 |
| `grounded_read_r4` | R4 grounded read（64步） | 66871 | 已删 |
| `fixed_interface_r5` | R5 固定语义坐标（未提交） | — | 无 |
| `content_modulation_r6` | R6 content modulation（128步） | 66910 | 已删 |
| `native_coordinate_r7` | R7 native coordinate（128步） | 66920 | 已删 |
| `content_reader_r8` | R8 content reader（128步） | 66926 | 已删 |
| `r1_content_reader_r9` | R9 content reader（400步） | 66968 | 保留 |
| `content_value_r9_1` | R9.1 content value（400步） | 66970 | 保留 |
| `execution_value_r10` | R10A execution value | 67003 | 保留 |
| `grounded_execution_r10b` | R10B grounded execution | 67008 | 保留 |
| `r10a_api400` | R10A + API 修订数据 | 67271 | 保留 |
| `request_adapt_r11a` | R11A-old（67214） | 67214 | 已删 |
| `aligned_request_r11a` | R11A-reworked（67265） | 67265 | 保留 |
| `native_request_r11a` | R11A-rework 设计稿（未提交） | — | 无 |
| `request_memory_r11b` | R11B request memory | 67308 | 保留 |
| `reciprocal_fusion_r12a` | R12A reciprocal fusion | 67346 | 已删 |
| `semantic_execution_r12b` | R12B semantic execution | 67456 | 已删 |
| `native_operator_r13a` | R13A native operator | 67598 | 已删 |
| `interaction_query_r13b` | R13B interaction query | 67610 | 已删 |
| `coupled_attention_r13c` | R13C coupled attention | 67682 | 保留（用户停止，未采用） |
| `grounded_fusion_r15a` | R15A grounded fusion | 67762 | 已删 |
| `request_bound_fusion_r15b` | R15B request bound fusion | 67829 | 保留 |
| `local_state_operator_r15c` | R15C local state operator | 67904 | 保留 |
| `denoising_matched_fusion_r15d` | R15D denoising matched fusion | 67920 | 保留 |
| `semantic_addressing_r15e` | R15E 主版本 semantic addressing | 67955 | 已删 |
| `semantic_response_fusion_r15e` | R15E 重设计稿（无训练） | — | 无 |
| `state_response_fusion_r15e` | R15E 只读探针（无训练） | — | 无 |
| `native_request_local_r15e` | local-K 预检失败（无训练） | — | 无 |
| `grounded_semantic_fusion_r15f` | R15F grounded semantic fusion | 68038 | 已删 |
| `source_residual_fusion_r15g` | R15G source residual fusion | 68103 | 已删 |
| `semantic_feedback_fusion_r15h` | R15H semantic feedback fusion | 68243 | 保留 |
| `joint_memory_fusion_r15i` | R15I joint memory fusion | 68312 | 保留 |
| `r15i_design` | R15I 前置只读探针，不是正式 R15I | 68309 | 无正式权重 |
| `native_joint_read_r16a` | R16A native joint read | 68429 | 保留 |
| `native_joint_read_r16a_nozero` | R16A-NOZERO | 68458 | 保留（训练完，评测取消） |
| `grounded_joint_read_r16b` | R16B 草案（取消） | 68490 | 无可用权重 |
| `native_budget_joint_read_r16b` | R16B 最终版 | 68514 | 保留 |
| `native_request_local_r16c` | R16C local video-state execution | 68550 | 保留 |
| `native_request_local_r16d` | R16D local QK modulation | 68561 | 保留 |
| `native_request_local_r16e` | R16E local QKV + V correction | 68562 | 保留 |
| `native_response_fusion_r16` | R16G/H/I/J/K 共用工作区 | 68621/68625/68627/68622 等 | 多个权重，部分仍在训练 |

### B. 跨版本/评测工作区

| 目录 | 内容 |
|---|---|
| `r11_r12` | R11–R13 的共享目标、监控、R11A/R11B 复盘、R10A 数据交叉20例、R13 支持实验 |
| `r16_evaluation_runtime` | R16A 的评测-only 恢复运行时（68459）；含 R16A-NOZERO 评测待提交清单 |
| `b400_joint_read_eval` | 冻结 R15B-400 权重做“直接C + 请求关联C”联合读取的推理评测；不是训练版 R15I |
| `cross_benchmark_70_60` | WMB70 + Physics-IQ 20场景×3视角；模型为 Wan+LoRA/R1/R10A |
| `physvid_i2v_599` | PhysVid I2V 599 提示词评测；模型为 Wan+LoRA/R1/R9/R10A/R10B |
| `data_revision_qwen38` | 用 qwen3.8-flash API 修订训练 caption/reasoning 的数据工作区；R10A-API、R11A-reworked、R11B 使用 |
| `r10_preparation` | R10 早期设计原型（历史未选中；readout 思路后来成为 R9.1） |
| `recovery_20261031` | 第一次服务器中断后的恢复脚本与提交 YAML |

### C. 评测、数据与基础设施

| 目录 | 内容 |
|---|---|
| `videophy60` | full60（原40+额外20）评测工作区；含 R1/R9/R9.1/R10A/R10B/R11A-old 的60例结果 |
| `training_budget_reference` | R1-400 的250步固定16参考（模块归因用） |
| `module_controls16` | R1-400 固定16模块对照（no_c/anchor_only/reasoning_swapped/empty_reasoning/mean/global_train_mean） |
| `heldout8` | 第一轮8例 heldout 验证集 |
| `reports` | 第一轮汇总报告、探针、已完成40例总表（`completed_full40_summary.csv/json`）等 |
| `evaluation` | 评测脚本与16例过程准则 |
| `quality_eval` | VBench 资产；用户已说“先不用vbench”，状态 deferred |
| `inference_outputs` | 第一轮顶层推理产物（R1/native 400/1000） |
| `runs` | 第一轮顶层训练 run（R1 66778、续训 66856、native 66781 等）；**受保护** |
| `snapshots` | 第一轮 R1/native 源码冻结快照 |
| `inference_snapshots` / `input_snapshots` / `review_snapshots` | 第一轮推理/输入/复盘代码冻结 |
| `cache` | 预计算条件、词向量、reasoning controls 缓存 |
| `configs` | 基础训练配置 `native_lora.yaml` / `physical_content.yaml` |
| `physical_content` | 模型架构代码（model.py、supervision.py 等） |
| `train` | 早期训练提交 YAML |
| `inference` | 早期推理提交 YAML |
| `preparation/66775` | 第一轮输入准备 |
| `backups` | R1-400 PC3.625 gamma5 参考备份（不是训练目录） |
| `determined_checkpoints` | 空目录 |
| `tmp` | 空临时目录 |
| `__pycache__` | Python 缓存，可忽略 |

### D. 顶层脚本/日志（不是某个版本）

- **汇总/建表**：`build_results_tables.py`、`summarize_first_wave.py`、`summarize_physvid_style.py`、`summarize_r10_final.py`、`merge_completed40.py`、`compare_iteration.py`、`build_video_review.py`
- **图表**：`plot_manuscript_evidence.py`、`plot_r6_content_evidence.py`
- **探针**：`probe_*.py` + 对应 `.log`（content、execution gradient、process features、reasoning prompt、bounded reasoning 等）
- **准备/拟合**：`prepare_*.py`、`prepare_configs.py`、`fit_interface.py`、`prepare_words.py`
- **审查/审计**：`review_*.py`、`audit_training_budgets.py`、`watch_declared_stages.py`、`dispatch_primary_tail.py`、`repair_heldout_evaluation.py`、`resume_baseline_and_review.py`
- **文档**：`AGENTS.md`（长期运行日志）、`MODULE_EFFECTIVENESS.zh.txt`（模块有效性摘要）

## 3. 最容易混淆的名字

| 看起来像 | 实际是什么 |
|---|---|
| `request_anchor_r1` | R1 的推理期变体，不是 R1 训练目录 |
| `r1_content_reader_r9` | R9；不叫 `r9_*` 是为了强调它从 R1 复现出发 |
| `content_value_r9_1` | R9.1，不是“R9 的一部分” |
| `execution_value_r10` | R10A |
| `grounded_execution_r10b` | R10B |
| `r10a_api400` | R10A 的 API 数据重训版，不是 R11 |
| `request_adapt_r11a` / `aligned_request_r11a` / `native_request_r11a` | R11A-old / R11A-reworked / R11A 未提交设计稿 |
| `reciprocal_fusion_r12a` / `semantic_execution_r12b` | R12A / R12B |
| `native_operator_r13a` / `interaction_query_r13b` / `coupled_attention_r13c` | R13A / R13B / R13C |
| `grounded_fusion_r15a` … `joint_memory_fusion_r15i` | R15A–R15I 的主候选目录 |
| `semantic_addressing_r15e` / `semantic_response_fusion_r15e` / `state_response_fusion_r15e` / `native_request_local_r15e` | 四个 R15E 相关目录：主版本 / 重设计稿 / 只读探针 / 预检失败 |
| `r15i_design` | R15I 前的只读探针，不是正式 R15I |
| `grounded_joint_read_r16b` / `native_budget_joint_read_r16b` | R16B 草案（取消）/ R16B 最终版 |
| `native_joint_read_r16a` / `native_joint_read_r16a_nozero` | R16A 主版 / 去掉零值 reference 的对照 |
| `native_request_local_r16c/d/e` | R16C/R16D/R16E 三个局部执行算子候选 |
| `native_response_fusion_r16` | R16G/H/I（以及未完成的 J/K）共用一个工作区，不是单一版本 |
| `b400_joint_read_eval` | 冻结 R15B 权重的推理评测，不是训练版 R15I |
| `cross_benchmark_70_60` | WMB70+PhysicsIQ60，只测 Wan/R1/R10A；R15G 的跨基准在它自己目录里 |

## 4. snapshots 是干什么的，CKPT 到底在哪

### 4.1 每个版本目录的标准结构

```
<版本目录>/
├── prepared.json            # 提交清单：实验ID、YAML、快照路径、状态
├── DESIGN.zh.txt            # 设计说明（部分版本有）
├── snapshots/<UTC时间>/      # ★ 提交时冻结的源码+配置，不是权重
│   ├── physical_content/    # 当时的模型代码（model.py 等）
│   ├── configs/*.yaml       # 当时的训练/推理配置
│   ├── guidance.py / infer.py / train.py ...
│   └── review/              # 设计证据、预检结果（部分版本有）
├── submissions/             # Determined 提交 YAML
├── runs/<experiment_id>/    # ★ 真正的训练目录，CKPT 在这里
│   ├── config.json / gpu.json / status.json / train.jsonl
│   ├── latest.pt
│   └── stepNNNN.pt
├── inference_outputs/<experiment_id>/<阶段>/
│   ├── videophy40/ videophy60/ full60/ controls/
│   └── cases/、scores.json、manifest.json、summary.json
├── reports/                 # 复盘、观测、closure、jobs 日志
├── controls/                # 均值控制小张量（.pt，几MB）
└── helper_outputs/          # 辅助作业（部分版本有）
```

**`snapshots/` 的作用**：在提交训练/推理的那一刻，把当时的完整代码和配置冻结一份，保证以后即使工作区代码被改，结果仍可复现/审计。`prepared.json` 的 `snapshot` 字段指向它。它通常**不存模型权重**；里面的 `.pt` 只有 `controls/*_mean_content.pt` 这类几MB的控制张量，或输入条件快照，不是模型 CKPT。

**顶层几个 snapshot 目录**：
- `snapshots/`：第一轮 R1/native 的源码冻结。
- `inference_snapshots/`：第一轮推理代码冻结。
- `input_snapshots/`：第一轮输入条件冻结。
- `review_snapshots/`：第一轮复盘脚本冻结。
- `snapshots/.../review/`：设计证据、预检、对照选择，不是权重。

### 4.2 CKPT 的规范位置

**唯一规范位置：`<版本目录>/runs/<experiment_id>/`**，文件是 `latest.pt` 和 `stepNNNN.pt`。`latest.pt` 通常与最后一个 `stepNNNN.pt` 是**同一个 inode 的硬链接**（`nlink=2`），只删其中一个不会释放空间，必须把该 run 下的权重文件都删掉。

当前保留权重的版本与路径：

| 版本 | CKPT 目录（相对 `physical_content_v7/`） | 现有权重文件 |
|---|---|---|
| R1 | `runs/physical_content_66778/` | latest, step0128/0250/0400 |
| R1 续训 500/750 | `runs/physical_content_66833/` | latest, step0500, step0750 |
| R1-1000 | `runs/physical_content_66856/` | latest, step1000 |
| R9 | `r1_content_reader_r9/runs/physical_content_66968/` | latest, step0128/0250/0400 |
| R9.1 | `content_value_r9_1/runs/physical_content_66970/` | latest, step0128/0250/0400 |
| R10A | `execution_value_r10/runs/physical_content_67003/` | latest, step0128/0250/0400 |
| R10B | `grounded_execution_r10b/runs/physical_content_67008/` | latest, step0128/0250/0400 |
| R10A-API | `r10a_api400/runs/physical_content_67271/` | latest, step0128/0250/0400 |
| R11A-reworked | `aligned_request_r11a/runs/physical_content_67265/` | latest, step0128/0250/0400 |
| R11B | `request_memory_r11b/runs/physical_content_67308/` | latest, step0128/0250/0400 |
| R13C | `coupled_attention_r13c/runs/physical_content_67682/` | latest, step0032/0128/0250/0400 |
| R15B | `request_bound_fusion_r15b/runs/physical_content_67829/` | latest, step0128/0250/0400 |
| R15C | `local_state_operator_r15c/runs/physical_content_67904/` | latest, step0128/0250/0400 |
| R15D | `denoising_matched_fusion_r15d/runs/physical_content_67920/` | latest, step0128/0250/0400 |
| R15H | `semantic_feedback_fusion_r15h/runs/physical_content_68243/` | latest, step0128/0250/0400 |
| R15I | `joint_memory_fusion_r15i/runs/physical_content_68312/` | latest, step0128/0250/0400 |
| R16A | `native_joint_read_r16a/runs/physical_content_68429/` | latest, step0128/0250/0400 |
| R16A-NOZERO | `native_joint_read_r16a_nozero/runs/physical_content_68458/` | latest, step0128/0250/0400 |
| R16B-final | `native_budget_joint_read_r16b/runs/physical_content_68514/` | latest, step0128/0250/0400 |
| R16C | `native_request_local_r16c/runs/physical_content_68550/` | latest, step0128/0250/0400 |
| R16D | `native_request_local_r16d/runs/physical_content_68561/` | latest, step0128/0250/0400 |
| R16E | `native_request_local_r16e/runs/physical_content_68562/` | latest, step0128/0250/0400 |
| R16G | `native_response_fusion_r16/runs/physical_content_68625/` | latest, step0128/0400 |
| R16G-SAFE | `native_response_fusion_r16/runs/physical_content_68627/` | latest, step0128/0400 |
| R16H | `native_response_fusion_r16/runs/physical_content_68621/` | latest, step0128/0400 |
| R16I | `native_response_fusion_r16/runs/physical_content_68622/` | latest, step0128/0400 |
| R16I 重跑（未完成） | `native_response_fusion_r16/runs/physical_content_68652/` | latest（约85步） |
| R16I 失败重试 | `native_response_fusion_r16/runs/physical_content_68689/` | latest |
| R16H 重跑 | `native_response_fusion_r16/runs/physical_content_68690/` | latest |
| R16J/K 重跑（未完成） | `native_response_fusion_r16/runs/physical_content_68698/`、`68699/` | latest |
| Native LoRA 基线 | `runs/native_lora_66781/` | latest, step0128/0250/0400 |
| R3/R6 native 基线 | `early_execution_r3/runs/native_lora_66866/`、`native_lora128_66913/` | 各自的 step 文件 |
| R1 参考备份 | `backups/r1_step0400_pc3.625_gamma5_20261031T055039Z/.../step0400.pt` | 独立副本 |

**已删除模型权重的版本**（只删权重，快照/日志/视频/评分都还在）：
R3、R4、R6、R7、R8、R11A-old、R12A、R12B、R13A、R13B、R15A、R15E、R15F、R15G。

清理记录：
- `analysis/storage_cleanup_20261110T052857Z/`
- `analysis/storage_cleanup_20261110T053608Z/`
- `analysis/storage_cleanup_20261110T053911Z/`
- 各 `runs/<id>/CHECKPOINTS_REMOVED_*.json`

### 4.3 那些不是模型 CKPT 的 .pt

| 路径 | 是什么 | 大小 |
|---|---|---|
| `*/controls/*_mean_content.pt` | 固定训练均值 C 的控制张量 | 6–13 MB |
| `*/controls/*_mean_raw_query_update.pt` | 均值 Q 更新的控制张量 | 8 MB |
| `*/prepared_inputs/*/positive.pt`、`negative.pt` | 每例输入条件张量 | 几百 KB |
| `inference_outputs/**/planned.latent.pt` | 每条视频的初始 latent | 约6 MB |
| `cache/*.pt`、`cache/**/*.pt` | 条件/词向量缓存 | 不定 |
| `analysis/physical_scaling_audit_*/c_samples_*.pt` | 诊断采样张量 | 18–150 MB |
| `checkpoints/`、`analysis/recovery_lora_20261027/*.pt` | 旧 v5.2.2 代际权重，不属于 R 系列 | 0.6–2.8 GB |

`determined_checkpoints/` 和 `tmp/` 是空目录；`__pycache__/` 可直接忽略。

## 5. 当前活跃工作（2026-11-10 05:50 UTC）

- `native_response_fusion_r16` 仍有训练任务在写入（68690/68698/68699 等，状态为 training；最新出现 `latest.pt.*.tmp`）。**不要清理该目录。**
- R16A：40/60 已评分，controls 大多完成；R16A-NOZERO 评测已取消。
- R16B-final：40/60 已评分，`direct_read_only` 等 controls 仍在跑。
- R16C：40例生成停在36/40，未评分。
- R16D：40/60 已评分，`no_c` control 在跑。
- R16E：40例已评分，extra20 在跑。
- R16G/G-SAFE/H：40/60 已评分，部分 controls 在跑；R16I：40已评分，extra20 在跑。

## 6. 如果想让后续不再乱（建议，不自动执行）

1. 新版本目录统一用 `r<N><letter>_<主题>/` 命名，例如 `r16c_native_request_local/`，把版本号放在最前。
2. 每个版本目录放一个 `STATUS.md`，只写：版本号、设计一句话、实验ID、当前状态、权重是否保留、主结果文件路径。
3. 把纯评测套件移到 `_bench/`（`videophy60`、`cross_benchmark_70_60`、`physvid_i2v_599`、`b400_joint_read_eval`）。
4. 把纯基础设施移到 `_shared/`（`cache`、`configs`、`physical_content`、`train`、`inference`、`runs`、`snapshots`、`reports` 等）。
5. 把未训练/失败/被取代的设计移到 `_archive/`，例如 `fixed_interface_r5`、`process_r2`、`native_request_r11a`、`r10_preparation`、`semantic_response_fusion_r15e`、`state_response_fusion_r15e`、`native_request_local_r15e`。

如果希望，我可以按这个建议实际整理目录，但会先给出移动清单供你确认，不会直接改动正在训练的目录。
