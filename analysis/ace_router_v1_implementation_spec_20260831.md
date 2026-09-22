# ACE-Router V1 外置实现文档（Wan2.2-TI2V-5B）

> 日期：2026-08-31  
> 状态：待实现的工程规范；本文档本身不包含已完成代码  
> 对应设计：[wan_appearance_semantics_causal_routing_mvp_20260831.md](/home/liuzhirui/Project/physGen/analysis/wan_appearance_semantics_causal_routing_mvp_20260831.md) 第一部分第 5–8 节  
> 新代码唯一根目录：`/home/liuzhirui/Project/physGen/code/v1`  
> Wan 源码只读根目录：`/home/liuzhirui/model/Wan2.2`  
> 首选基模：`/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B`

## 0. 实施结论

ACE-Router V1 应实现为 **Wan2.2 之外的无训练推理适配层**。它读取 Wan2.2 源码与 TI2V-5B checkpoint，在内存中复用基模已有模块，但不编辑、不复制回、不缓存到 `/home/liuzhirui/model`。新增 Python、配置、测试、脚本、运行日志和输出均放在 `/home/liuzhirui/Project/physGen/code/v1`。

推荐的最小结构是：

1. 用外部 `AceWanTI2V` 继承官方 `WanTI2V`，复用模型、T5、VAE、scheduler 和首帧重锚逻辑。
2. 模型加载完成后只包一层 `AceModelProxy`；proxy 内仍持有同一份已加载的 `WanModel`，不再加载第二份 5B 权重。
3. `lambda0 == 0`、step gate 为零或 layer gates 全零时，proxy 直接调用原 `WanModel.forward`，形成严格的基线快路径；有效 scale 大于零却缺少 ACE 上下文属于配置错误。
4. ACE 生效时，由 `code/v1` 中的外部 helper 按官方 forward 顺序执行同一组 block；只在启用的 block 内把原语义 cross-attention 改为

   \[
   r=r_{sem}+\lambda_0g_{layer}(\ell)g_{step}(k)(r_+-r_-).
   \]

5. 正确因果与反事实条件只进入 CFG 的 conditional forward；unconditional/negative-prompt forward 保持官方行为。
6. I2V 继续采用 TI2V-5B 原生 VAE 首帧 latent 重锚，不新增视觉编码器，不把初态图像转成额外 cross-attention token。

本文档定义的是 **ACE-Router 消融基线/MVP**。它只有 DiT block 深度 \(\ell\) 和 denoising iteration \(k\) 两个路由轴，不包含 TRACE-Lite 的输出视频时间 \(f\) 与空间 \((x,y)\) gate。后续即使继续做 TRACE，也应先保留本实现作为可复现的 global-causal baseline。

---

## 1. 目标、非目标与不可破坏约束

### 1.1 目标

- 在不训练新参数的前提下，把原始语义、正确因果和最小反事实拆成三条文本路径。
- 在指定 Wan DiT block 和指定采样阶段注入正反事实 cross-attention 差分。
- 同时支持 TI2V-5B 的原生 T2V 与首帧 I2V，以便完成 M0–M4 对照。
- `lambda0=0` 时严格退回官方 Wan 路径。
- 每个结果具备完整、机器可读、可追溯的输入、基模、schedule、性能与诊断记录。
- 单卡实现完成后，通过多 GPU 的“每卡独立样本 worker”并行批量推理；首版不依赖模型内并行。

### 1.2 非目标

- 不修改 `/home/liuzhirui/model/Wan2.2/wan/modules/model.py`、`textimage2video.py` 或其他 Wan 文件。
- 不在 `/home/liuzhirui/model` 下新建脚本、配置、输出、日志、缓存或 `__pycache__`。
- 不训练 adapter、LoRA、router 或新 attention 权重。
- 不把 `c_-` 当作 CFG negative prompt。
- 不实现逐帧 phase、ROI/tube、token mask、Reader–Writer/Corrector 或 TRACE-Field。
- 不宣称 ACE 已解决“逐帧演化正确性”；它只验证全局因果差分在层/去噪步路由下是否有用。

### 1.3 写入边界

| 类型 | 允许路径 | 行为 |
|---|---|---|
| 新代码、测试、脚本 | `/home/liuzhirui/Project/physGen/code/v1/**` | 可写 |
| 新配置、运行 manifest | `/home/liuzhirui/Project/physGen/code/v1/**` | 可写 |
| 视频、日志、缓存、临时文件 | `/home/liuzhirui/Project/physGen/code/v1/**` | 可写 |
| 本实现文档 | `/home/liuzhirui/Project/physGen/analysis/**` | 可写 |
| Wan 源码 | `/home/liuzhirui/model/Wan2.2/**` | 只读导入 |
| TI2V-5B 权重 | `/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B/**` | 只读加载 |

启动器必须设置：

```bash
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=/home/liuzhirui/model/Wan2.2:/home/liuzhirui/Project/physGen/code/v1
HF_HOME=/home/liuzhirui/Project/physGen/code/v1/.cache/huggingface
TORCH_HOME=/home/liuzhirui/Project/physGen/code/v1/.cache/torch
XDG_CACHE_HOME=/home/liuzhirui/Project/physGen/code/v1/.cache
TMPDIR=/home/liuzhirui/Project/physGen/code/v1/tmp
```

`PYTHONDONTWRITEBYTECODE=1` 是硬要求，否则仅仅 import `wan` 就可能更新 Wan 源码树中的 `__pycache__`。所有输出参数在解析后都必须做 `resolve()`，如果目标位于 `/home/liuzhirui/model` 内，程序应立即拒绝运行；检查时要使用解析后的真实路径，防止符号链接绕过。

---

## 2. 已核对的本地 Wan 接口

本设计不是基于抽象的 Diffusers pipeline，而是针对当前本地 Wan2.2 实现编写。

| 项目 | 本地事实 | 对 ACE 的影响 |
|---|---|---|
| 基模配置 | `Wan2.2-TI2V-5B/config.json`：`num_layers=30`、`dim=3072`、`num_heads=24`、`text_len=512`、`in_dim=48` | layer gate 必须恰好有 30 项；模型不匹配时拒绝加载预设 |
| 文本编码器 | UMT5-XXL 输出每条 `[L,4096]`，Wan 内部 `text_embedding` 投影到 3072 | 三条条件都先经同一个冻结 T5，再分别经同一个冻结投影层 |
| block 文本路径 | `WanAttentionBlock.forward` 在 self-attention 后执行一次 `cross_attn(norm3(x), context)`，然后执行 FFN | ACE 差分必须加在 cross-attention residual 合并处、FFN 之前 |
| 模型 forward | `WanModel.forward(x,t,context,seq_len,y=None)`；内部负责 patchify、time embedding、context padding/投影、30 个 block 和 unpatchify | 外部 ACE forward 需保持这些操作及顺序；不能只在 block 输出后补一个 residual |
| TI2V pipeline | `WanTI2V.generate` 依据 `img is None` 分流 T2V/I2V | 同一外部 pipeline 可完成 M0–M4 |
| I2V 初态 | VAE 编码输入图像；初始采样前以及每次 scheduler 更新后都用 mask 重锚已知首帧 latent | ACE 不得替换或跳过该重锚步骤 |
| CFG | conditional 与 negative-prompt/unconditional 各做一次完整模型 forward，再按 guide scale 合并 | ACE 只改 conditional forward；unconditional forward 不增加正反 context |
| checkpoint 加载 | `WanTI2V.__init__` 调用 `WanModel.from_pretrained(checkpoint_dir)` | 首版先让官方 pipeline 加载一次，再用 proxy 包装；禁止再次加载一份模型 |

对应本地源码锚点：

- [WanCrossAttention 与 WanAttentionBlock](/home/liuzhirui/model/Wan2.2/wan/modules/model.py:158)
- [WanModel.forward](/home/liuzhirui/model/Wan2.2/wan/modules/model.py:410)
- [WanTI2V](/home/liuzhirui/model/Wan2.2/wan/textimage2video.py:34)
- [WanTI2V.generate](/home/liuzhirui/model/Wan2.2/wan/textimage2video.py:162)
- [WanTI2V.i2v](/home/liuzhirui/model/Wan2.2/wan/textimage2video.py:410)
- [I2V 首帧重锚与模型调用](/home/liuzhirui/model/Wan2.2/wan/textimage2video.py:550)

### 2.1 当前上游快照

截至 2026-08-31，本地核对值如下：

```text
Wan git HEAD:
08e890ff01ff8daf9161db472f362efe5d6ba656

sha256 wan/modules/model.py:
8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432

sha256 wan/modules/attention.py:
c79158e1cc9b7a17bc934e6acb34f42e588a41adb89801146957cda2d6d4379d

sha256 wan/modules/t5.py:
8b0cebf3192c542f92a344255a06c203df3ba24160715899a055cc8de0cd930f

sha256 wan/textimage2video.py:
228f2fabf23014ed41ec6b8cd713d5be7371bb558501c71a62a0258b491a877b

sha256 wan/configs/wan_ti2v_5B.py:
26d9e7b9c555eb0900b13751267a596556f869e7aee2d1572afc2a0a4a76f4c7

sha256 checkpoint config.json:
d1fea36899d00c2501b836c13ad65af56e2f9529ba622e50886d3f5c3e6c02bc

sha256 checkpoint weight index:
bfa2337f1163e195d24151a72298daf34a620543898109be47e414c8daa5b3fe
```

本地 Wan 工作树不是 clean 状态，因此只记录 git commit 不足以复现。每次运行应把上述源文件的实际 hash、checkpoint index hash、权重分片文件名与大小写入 manifest。hash 不匹配时：smoke test 可在显式 `--allow-upstream-drift` 下继续，但正式 pilot 必须先复核外部适配层与新源码签名。

当前 `wan` 环境核对为 Python 3.10.19、PyTorch 2.6.0+cu124、Diffusers 0.36.0；运行 manifest 仍须实时记录实际版本，不能把本文数值当作永久环境锁。

---

## 3. ACE-Router V1 的冻结方法定义

### 3.1 四个输入

对每个样本定义：

- `I0`：可选的 pre-event 初态图像；I2V 方法必需，T2V 方法不使用。
- `c_sem`：原始短视频 prompt，原样使用。
- `c_plus_clause`：一条紧凑、肯定式、可见、按时间顺序描述的正确因果句。
- `c_minus_clause`：保持实体、场景和大部分动作词不变，只改变一个因果/演化关系的最小反事实句。

实际送入 T5 的三条字符串固定为：

```text
semantic_text = original_prompt
positive_text = original_prompt + " " + causal_positive
negative_text = original_prompt + " " + causal_counterfactual
```

这里的 `negative_text` 是 ACE 的反事实条件，不是 CFG negative prompt。为避免歧义，代码和日志中不得把它简称为 `negative_prompt`，统一使用 `counterfactual_text` 或 `causal_neg_context`。

拼接只做首尾空白归一化和单空格连接，不自动扩写、不翻译、不改标点、不去重。三个最终字符串、UTF-8 字节 hash、T5 token 数都必须落盘。任意文本超过 `text_len=512` 时直接报错，不能静默截断。

### 3.2 block 内计算

令 \(q=\operatorname{Norm}_3(h)\)，同一个冻结 cross-attention 模块计算：

\[
r_{sem}=\operatorname{CA}(q,c_{sem}),\qquad
r_+=\operatorname{CA}(q,c_+),\qquad
r_-=\operatorname{CA}(q,c_-).
\]

ACE block 更新为：

\[
\tilde h=h+r_{sem}+s_{\ell,k}(r_+-r_-),
\]

\[
s_{\ell,k}=\lambda_0g_{layer}(\ell)g_{step}(k),
\]

然后只对 \(\tilde h\) 执行一次原 Wan FFN。self-attention、time modulation、FFN、head、patchify/unpatchify 均保持官方顺序和参数。

三个 cross-attention 必须共享该 block 原有的 `norm3` 与 `cross_attn` 权重。不能分别创建三个 attention 模块，否则既会引入未加载权重，也会破坏“差分主要来自文本”的实验含义。

数值策略固定为：三次 attention 在官方 autocast 环境中计算；`r_pos-r_neg` 先转 FP32 做差和缩放，再转回 `r_sem.dtype` 后与语义 residual 相加。V1 不做 norm matching、归一化、裁剪或可学习缩放；这些若加入，必须成为独立消融而不能静默改变基线。

### 3.3 CFG 作用域

每个 denoising iteration 的语义如下：

```text
conditional prediction:
  semantic context + ACE positive/counterfactual residual

unconditional prediction:
  官方 sample_neg_prompt（或显式 n_prompt），ACE scale 强制为 0

final prediction:
  pred_uncond + cfg_scale * (pred_cond_ace - pred_uncond)
```

禁止以下做法：

- 把 `causal_counterfactual` 塞入 CFG 的 `n_prompt`；
- 在 unconditional forward 中也注入 `r_plus-r_minus`；
- 用 `positive_text` 直接替换全层的 `semantic_text`；
- 对 c+、c- 各跑一遍完整 DiT 后在模型输出端相减。

### 3.4 layer gate：消除原方案中的实现歧义

原设计同时出现了“只有中间约 16 层多算两次 attention”和“边缘层 scale=0.1”的描述。两者在计算上不等价：只要 gate 非零，该层就必须计算 c+ 与 c- 两次额外 cross-attention。因此 V1 明确定义两个不同 preset，日志必须保存完整 30 项数组，不能只保存名称。

**默认 `mvp_mid16`：**

```text
blocks  0–7  : 0.0
blocks  8–13 : 0.4
blocks 14–23 : 1.0
blocks 24–29 : 0.0
```

它只在 8–23 共 16 个 block 增加计算，与“中间约 16 层”的工程目标一致。

**消融 `paper_soft30`：**

```text
blocks  0–7  : 0.1
blocks  8–13 : 0.4
blocks 14–23 : 1.0
blocks 24–29 : 0.1
```

它会让全部 30 个 block 产生额外计算，只用于复现原始分段权重，不作为首轮默认。

层窗口扫描使用半开区间，避免 `14–23` 到底包不包含 23 的歧义：

| 名称 | 配置区间 | 实际 block |
|---|---:|---|
| `early` | `[0,10)` | 0–9 |
| `mid_a` | `[10,20)` | 10–19 |
| `mid_b` | `[14,24)` | 14–23 |
| `late` | `[20,30)` | 20–29 |
| `all` | `[0,30)` | 0–29 |

窗口扫描时区间内 gate 为 1、区间外为 0，只比较位置，不混入分段强弱。加载后必须断言 `model.num_layers == len(layer_gates) == 30`。

### 3.5 denoising-step gate

schedule 依据采样循环的 `step_index`，而不是 scheduler 的原始 timestep 数值：

```text
ratio = step_index / total_sampling_steps

0.00 <= ratio < 0.30 : 1.0
0.30 <= ratio < 0.60 : 0.7
0.60 <= ratio < 0.80 : 0.3
0.80 <= ratio < 1.00 : 0.0
```

因此 50 steps 时恰为：

```text
0–14: 1.0, 15–29: 0.7, 30–39: 0.3, 40–49: 0.0
```

40 steps 时自动变为：

```text
0–11: 1.0, 12–23: 0.7, 24–31: 0.3, 32–39: 0.0
```

V1 的 `lambda0` 候选是 `0.25, 0.5, 1.0`，默认从 `0.5` smoke test。最终 block scale 只在一个地方计算，禁止调用方与模型内部各乘一次 `lambda0`。

---

## 4. 建议目录与文件职责

以下是后续实现时应建立的结构；本文档不会创建这些代码文件：

```text
/home/liuzhirui/Project/physGen/code/v1/
├── ace_router/
│   ├── __init__.py
│   ├── paths.py                 # 只读/可写路径解析与写入边界检查
│   ├── schema.py                # sample/config JSON 读取和严格校验
│   ├── schedules.py             # 30 层 gate、step gate、preset 解析
│   ├── upstream_guard.py        # Wan 源文件与 checkpoint 身份核对
│   ├── model_adapter.py         # AceModelProxy 与外部 ACE forward/block helper
│   ├── pipeline.py              # AceWanTI2V、T2V/I2V ACE 采样循环
│   ├── diagnostics.py           # residual norm、耗时、显存、非有限值检查
│   └── artifacts.py             # manifest、原子写入、输出命名和 hash
├── configs/
│   ├── ace_router_smoke.json
│   └── ace_router_pilot.json
├── scripts/
│   ├── run_ace.py               # 单样本/单方法入口
│   ├── run_matrix.py            # M0–M4 × prompt × seed 批量入口
│   ├── run_smoke.sh
│   └── run_pilot.sh
├── tests/
│   ├── test_paths.py
│   ├── test_schema.py
│   ├── test_schedules.py
│   ├── test_model_adapter.py
│   └── test_upstream_parity.py
├── demo/                        # 已有 P01–P20 条件文件；补入选定 I0 图像/选择记录
├── outputs/ace_router/          # 视频与逐运行 artifact
├── logs/ace_router/             # batch 索引和失败记录
├── .cache/                      # HF/Torch/XDG 缓存
└── tmp/                         # 临时文件
```

目录原则：

- 不把整份 `model.py` vendoring 到 `code/v1`；只实现最少的外部 forward helper，继续复用基模对象里的冻结子模块。
- 不在运行时改写 `wan` 包源码，也不通过生成 patch 文件后应用到 Wan 目录。
- 不保存修改后的 5B checkpoint；ACE V1 没有新参数，运行只需保存配置与基模身份。
- `outputs`、`logs`、`.cache` 和 `tmp` 都必须显式落在 `code/v1`，不能依赖当前工作目录的隐式相对路径。

---

## 5. 输入数据契约

### 5.1 单样本目录

现有 `demo/P01` 等目录已经包含：

```text
P01-origin.txt
P01-cplus-cminus.json
P01-i0.json
```

其中 `P01-i0.json` 只是首帧 prompt 与审计规则，不是输入图像。截至本文最终核对时，P01–P20 均已有实际 `Pxx-i0.png`，但当前未发现独立的候选选择 manifest，而且这些图片尚未统一成 TI2V-5B 的 20:11/11:20 宽高比。I2V 正式运行前，每个样本必须最终冻结：

```text
P01-i0.png                       # 或 jpg/jpeg，选定的原始首帧资产
P01-i0-wan.png                   # 不覆盖原图；裁剪到冻结视频宽高比后的 Wan 输入
P01-i0-selection.json            # 候选生成与只看图筛选记录
```

如果实际图像或冻结后的 Wan 输入缺失：

- `M0/M1/M4` 的 T2V 可以运行；
- `M2/M3` 必须 fail fast；
- runner 不得悄悄在线生成图像、改用 Wan 自己的首帧或从其他样本寻找图片。

### 5.2 `origin.txt` 规则

- UTF-8；去掉文件末尾换行后必须是非空单条 prompt。
- 不自动做 prompt extension。
- `cplus-cminus.json.original_prompt` 与它按首尾空白归一化后必须完全相同。

### 5.3 `cplus-cminus.json` 必需字段

```json
{
  "original_prompt": "...",
  "event_type": "...",
  "changed_state_variables": ["..."],
  "preserved_context": ["..."],
  "causal_positive": "...",
  "causal_counterfactual": "...",
  "counterfactual_type": "cause_effect_reversal",
  "single_changed_relation": "...",
  "warnings": [],
  "confidence": 0.98
}
```

强校验：

- `causal_positive` 与 `causal_counterfactual` 均非空且不相同；
- 每句最多 32 个按空格切分的英文词；
- 不包含规则禁用的显式否定词；
- `counterfactual_type` 属于预定义枚举；
- `confidence` 在 `[0,1]`；建议正式 pilot 不低于 0.7；
- 拼接后的三条 T5 文本均不超过 512 tokens。

“实体集合相同”和“只改变一个因果关系”至少保留人工审核结果；V1 可用字段和词面做报警，但不能把脆弱的字符串规则冒充可靠语义判定。

### 5.4 `i0.json` 与实际图像校验

强校验：

- `original_prompt` 与 `origin.txt` 一致；
- `pre_event_image_prompt`、`required_visible_entities`、`initial_states`、`reject_if` 非空；
- 实际图片可解码、尺寸满足预处理要求、通道为 RGB 可转换格式；
- `i0-selection.json` 记录生成模型/revision、候选 seeds、选择规则、选中编号和人工复核；
- 同一 sample 的 M2/M3、全部视频 seeds 使用同一个图像内容 hash。

现有 PNG 并不自动等于可直接进入正式矩阵。例如核对时 P01 为 `1536×1024`（3:2），而 TI2V-5B 官方横版尺寸为 `1280×704`（20:11）；若直接传入，官方 `best_output_size` 会沿用 3:2，而不会变成 20:11。正式运行前应从原图生成一个不覆盖原图的冻结派生图 `Pxx-i0-wan.png`：横版统一到 20:11，竖版统一到 11:20，并做只看首帧的实体/布局复核。官方 pipeline 内部仍沿用按 `max_area` 等比缩放和中心裁剪的策略，不拉伸。需要记录原图大小、派生裁剪框、派生图大小/hash、pipeline 最终缩放/裁剪信息和实际输出宽高。

---

## 6. 运行配置契约

建议使用 JSON，避免为 MVP 新增 YAML 解析依赖。一个 resolved config 至少包含：

```json
{
  "paths": {
    "wan_repo": "/home/liuzhirui/model/Wan2.2",
    "checkpoint": "/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B",
    "sample_dir": "/home/liuzhirui/Project/physGen/code/v1/demo/P01",
    "output_root": "/home/liuzhirui/Project/physGen/code/v1/outputs/ace_router"
  },
  "method": "M3_I2V_ACE",
  "generation": {
    "frame_num": 49,
    "width": 640,
    "height": 352,
    "max_area": 225280,
    "sampling_steps": 50,
    "sample_solver": "unipc",
    "shift": 5.0,
    "guide_scale": 5.0,
    "seed": 42,
    "offload_model": true,
    "t5_cpu": true,
    "convert_model_dtype": true
  },
  "ace": {
    "enabled": true,
    "lambda0": 0.5,
    "layer_preset": "mvp_mid16",
    "layer_gates": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "step_schedule": [[0.0, 1.0], [0.3, 0.7], [0.6, 0.3], [0.8, 0.0]],
    "cfg_scope": "conditional_only",
    "fp32_delta": true
  },
  "runtime": {
    "device_id": 0,
    "use_sp": false,
    "dit_fsdp": false,
    "t5_fsdp": false,
    "allow_upstream_drift": false
  }
}
```

校验规则：

- `frame_num` 必须满足 `4n+1`；smoke 使用 49。
- 宽、高必须同时可被 `vae_stride(16) × patch_size(2) = 32` 整除；横版 smoke 使用 `640×352`，竖版使用 `352×640`，分别保持与官方 `1280×704`、`704×1280` 相同的 20:11 或 11:20 宽高比。`max_area` 必须等于宽×高。
- I2V 使用 `max_area` 且实际宽高由冻结 I0 的宽高比经 `best_output_size` 决定；runner 必须先预计算并断言结果等于配置宽高。T2V 使用同一组已解析宽高。
- `sampling_steps > 0`，step schedule 区间连续、单调且覆盖 `[0,1)`。
- `lambda0 >= 0` 且为有限数；首轮不允许负数。
- `layer_gates` 长度等于模型层数，所有值非负且有限。
- `M0/M1/M2` 强制 `ace.enabled=false`；`M3/M4` 强制为 true。
- `M2/M3` 要求 I0；`M0/M1/M4` 不向 pipeline 传图片。
- V1 强制 `use_sp=false`、`dit_fsdp=false`、`t5_fsdp=false`。

最后一条是明确的支持边界：官方 sequence-parallel 会在初始化时替换 `model.forward`，FSDP 会包装/shard 模型；外部 helper 若直接访问 block 内部结构，必须另行适配。首版不要在未验证时把这两条路径标记为可用。多卡吞吐通过多进程、每卡一个独立单卡 worker 实现。

---

## 7. 模块接口与执行逻辑

### 7.1 `AceModelProxy`

建议接口：

```python
forward(
    x,
    t,
    context,
    seq_len,
    y=None,
    *,
    causal_pos_context=None,
    causal_neg_context=None,
    layer_gates=None,
    step_gate=0.0,
    lambda0=0.0,
    diagnostic_sink=None,
)
```

proxy 只持有 `base_model`，不复制参数。初始化后应断言：

- `base_model.model_type == "ti2v"`；
- `base_model.num_layers == 30`；
- `len(base_model.blocks) == 30`；
- `base_model.text_len == 512`；
- 所有参数仍是 `requires_grad=False`。

快路径条件：

```text
lambda0 == 0
OR step_gate == 0
OR all(layer_gates == 0)
```

命中快路径时，直接：

```python
return base_model(x=x, t=t, context=context, seq_len=seq_len, y=y)
```

当任一有效 scale 大于零时，c+、c- 任一缺失都应报错，不能自动退化。是否启用 ACE 由 pipeline/config 层先行校验，proxy 不再维护第二份容易失配的 `enabled` 状态。

### 7.2 外部 `ace_model_forward`

ACE 路径需严格镜像当前 `WanModel.forward`：

1. 检查/拼接可选 `y`；TI2V 当前不传 `y`。
2. 用基模 `patch_embedding` 处理 latent。
3. 计算 `grid_sizes`、`seq_lens` 并 padding 到 `seq_len`。
4. 使用基模 `time_embedding` 与 `time_projection` 产生 modulation。
5. 分别把 semantic、positive、counterfactual T5 context pad 到 512，再通过同一个 `base_model.text_embedding`。
6. 遍历原 `base_model.blocks`：gate 为零的 block 直接调用官方 `block.forward`；gate 非零的 block 调用外部 `ace_block_forward`。
7. 使用原 `head`。
8. 使用原 `unpatchify`，返回与官方相同的 `List[float Tensor]`。

不得复制、重建或重新初始化 block。上下文 batch 数必须和 latent batch 数一致；V1 正式运行限定 batch size 1，但 helper 不应通过错误 reshape 默默接受不一致 batch。

### 7.3 外部 `ace_block_forward`

该 helper 接收“已有 block 实例”，按当前官方实现执行：

```text
1. 官方 modulation 分块
2. 官方 self-attention residual
3. q = block.norm3(x)
4. r_sem = block.cross_attn(q, semantic_context, context_lens=None)
5. r_pos = block.cross_attn(q, positive_context, context_lens=None)
6. r_neg = block.cross_attn(q, counterfactual_context, context_lens=None)
7. scaled_delta = cast(lambda0 * layer_gate * step_gate
                       * (float(r_pos) - float(r_neg)), r_sem.dtype)
8. x = x + r_sem + scaled_delta
9. 官方 FFN residual，仅执行一次
```

需要保留官方 self-attention 和 FFN 中的 FP32 autocast/modulation 行为。不要调用完整 `block.forward` 后再额外注入，因为那会把差分放到 FFN 之后，与方法公式不一致。

### 7.4 `AceWanTI2V`

初始化流程：

1. 调用官方 `WanTI2V.__init__`，只加载一次 T5、VAE 和 `WanModel`。
2. 验证未启用 SP/FSDP。
3. 用 `AceModelProxy(base_model)` 替换 pipeline 内的运行时引用。
4. 保持 base model 的 eval、dtype、device 与 offload 行为。

应保留两个入口：

- 官方 `generate(...)`：用于 M0/M1/M2 和 `lambda0=0` parity；不编码 ACE contexts。
- 新增 `generate_ace(...)`：用于 M3/M4；额外接受 `causal_positive`、`causal_counterfactual`、layer gates、step schedule 和 diagnostic sink。

这样 `lambda0=0` 验证可以直接走官方 pipeline 逻辑，不因无意义的额外 T5 编码或自定义循环影响基线。

### 7.5 T5 编码与生命周期

在采样循环外一次性编码：

```text
[semantic_text, positive_text, counterfactual_text]
```

可在一次 batch 调用中编码三条文本；CFG `n_prompt` 可一并编码或沿用官方单独调用。无论采用哪种方式，T5 只在每个视频开始时运行一次，不得每个 denoising step 重编码。

当 `t5_cpu=true` 时，T5 输出应按官方方式转到 DiT device。offload 时要确保 contexts 的生命周期覆盖完整采样循环，结束后显式释放，不能把持久 tensor 缓存在全局对象导致连续样本显存增长。

### 7.6 ACE 采样循环

T2V 与 I2V 分别镜像官方 `t2v`/`i2v`。每一步的核心调用为：

```python
step_gate = schedule(step_index, len(timesteps))

pred_cond = model_proxy(
    latent_model_input,
    t=timestep,
    context=semantic_context,
    seq_len=seq_len,
    causal_pos_context=positive_context,
    causal_neg_context=counterfactual_context,
    layer_gates=resolved_layer_gates,
    step_gate=step_gate,
    lambda0=lambda0,
)

pred_uncond = model_proxy(
    latent_model_input,
    t=timestep,
    context=cfg_negative_context,
    seq_len=seq_len,
    step_gate=0.0,
    lambda0=0.0,
)

pred = pred_uncond + guide_scale * (pred_cond - pred_uncond)
latent = scheduler.step(pred, ...)
```

I2V 还必须在以下两个位置保留官方 mask 混合：

```text
进入 denoising loop 前：latent = known_first_frame + unknown_noise
每次 scheduler.step 后：再次覆盖已知首帧 latent
```

这两个步骤不能因为 ACE 重构采样循环而遗漏。初态图像只进入这一原生路径；c+ 与 c- 不改变 mask。

---

## 8. 方法矩阵与公平对照

### 8.1 冻结的 M0–M4 定义

| 方法 | 模式 | I0 | conditional 文本 | ACE | 用途 |
|---|---|---:|---|---:|---|
| M0 | T2V | 否 | 原始短 prompt | 否 | 原始语义基线 |
| M1 | T2V | 否 | 预先冻结的长物理扩写 | 否 | 复现长 prompt 失败；若样本无该输入则标记缺失，不临时生成 |
| M2 | I2V | 是 | 原始短 prompt | 否 | 初态图像收益 |
| M3 | I2V | 是 | 原始短 prompt | 是 | ACE 完整 MVP |
| M4 | T2V | 否 | 原始短 prompt | 是 | 去掉 I0 后验证 causal routing 本身 |

公平性要求：

- 同一样本的 M2/M3 使用同一 I0 文件 hash。
- 同一方法间的对照使用相同 seed、尺寸、帧数、scheduler、steps、shift、CFG 与 negative prompt。
- M0/M1/M4 的输出尺寸应与 M2/M3 实际裁剪后的输出尺寸一致。
- 所有方法共用同一个基模 checkpoint 和源码快照。
- layer window、`lambda0` 等只能用开发 prompt 选择；冻结后才能扩展到完整 20 prompts。
- 不根据视频结果回头重选 I0、c+ 或 c-。

### 8.2 计算量口径

对每个 active denoising step，conditional forward 的每个 active block 比官方多两次 cross-attention，但仍只有一次 self-attention 和一次 FFN。unconditional forward 没有额外 attention。

`mvp_mid16` 每个 active step 增加 `16 × 2 = 32` 次 block-level cross-attention；`paper_soft30` 增加 60 次。不能仅凭调用次数宣称固定百分比开销，必须实测：

- wall-clock 总时长与每 step 中位数；
- `torch.cuda.max_memory_allocated`；
- 首个 warm-up step 与稳定 step 分开；
- M2 与 M3 在同机同配置比较。

---

## 9. P01 滑板车完整落地示例

### 9.1 输入来源

```text
/home/liuzhirui/Project/physGen/code/v1/demo/P01/P01-origin.txt
/home/liuzhirui/Project/physGen/code/v1/demo/P01/P01-cplus-cminus.json
/home/liuzhirui/Project/physGen/code/v1/demo/P01/P01-i0.json
```

当前原始语义：

```text
A black scooter rolls into a metal trash can, then tilts to one side and stops beside it.
```

正确因果 clause：

```text
The scooter visibly contacts the trash can before tilting, slows during the tilt, and remains stopped beside the can.
```

最小反事实 clause：

```text
The scooter tilts and stops while still separated from the trash can; visible contact occurs afterward.
```

### 9.2 三条最终 T5 文本

```text
c_sem:
A black scooter rolls into a metal trash can, then tilts to one side and stops beside it.

c_plus:
A black scooter rolls into a metal trash can, then tilts to one side and stops beside it. The scooter visibly contacts the trash can before tilting, slows during the tilt, and remains stopped beside the can.

c_minus:
A black scooter rolls into a metal trash can, then tilts to one side and stops beside it. The scooter tilts and stops while still separated from the trash can; visible contact occurs afterward.
```

`c_plus` 与 `c_minus` 共享 scooter、trash can、tilt、stop 等主体和动作词，主要差异集中在“接触先于响应”还是“响应先于接触”。这正是 ACE 差分要放大的方向。

### 9.3 I0 前置条件

`P01-i0.json` 已定义侧视、直立滑板车、与垃圾桶分离、留有运动空间等条件，当前也已有 `1536×1024` 的 `P01-i0.png`。但该图是 3:2，且尚需选择记录和事件前状态人工复核；P01 的正式 M2/M3 应使用经复核、派生且冻结的 20:11 `P01-i0-wan.png`。原图不得被覆盖，M2/M3 必须引用同一个派生图 hash。

### 9.4 推荐 smoke 配置

```text
method             = M3_I2V_ACE
frame_num          = 49
size               = 640 x 352
max_area           = 225280
sampling_steps     = 50
solver             = unipc
shift              = 5.0
guide_scale        = 5.0
seed               = 42
layer_preset       = mvp_mid16
lambda0            = 0.5
step_schedule      = 1.0 / 0.7 / 0.3 / 0.0
offload_model      = true
t5_cpu             = true
convert_model_dtype= true
```

`640×352` 是诊断分辨率，不进入最终质量主表。正式冻结设置后切到 `1280×704`，其余条件不变并重新记录性能。

---

## 10. Artifact 与日志规范

### 10.1 每个运行的目录

```text
outputs/ace_router/<run_id>/<sample_id>/<method>/seed_000042/
├── video.mp4
├── manifest.json
├── config.resolved.json
├── contexts.json
├── diagnostics.jsonl
├── reference_image.json          # I2V 时保存路径、hash、尺寸与裁剪信息
└── failure.json                  # 失败时存在；与成功 video 互斥
```

`run_id` 建议使用 UTC 时间 + 短配置 hash。写 manifest 时先写同目录临时文件，再原子 rename，避免中断后留下看似完整的 JSON。

### 10.2 `manifest.json` 最低字段

```json
{
  "status": "success",
  "run_id": "...",
  "sample_id": "P01",
  "method": "M3_I2V_ACE",
  "seed": 42,
  "output_video": "...",
  "input_hashes": {
    "origin_txt": "sha256:...",
    "causal_json": "sha256:...",
    "i0_image": "sha256:..."
  },
  "contexts": {
    "semantic_text": "...",
    "positive_text": "...",
    "counterfactual_text": "...",
    "cfg_negative_prompt": "...",
    "token_lengths": {"semantic": 0, "positive": 0, "counterfactual": 0}
  },
  "ace": {
    "lambda0": 0.5,
    "layer_gates": [0.0],
    "step_gates": [1.0],
    "cfg_scope": "conditional_only",
    "fp32_delta": true
  },
  "generation": {
    "frame_num": 49,
    "width": 640,
    "height": 352,
    "max_area": 225280,
    "sampling_steps": 50,
    "solver": "unipc",
    "shift": 5.0,
    "guide_scale": 5.0
  },
  "upstream": {
    "wan_repo": "/home/liuzhirui/model/Wan2.2",
    "wan_git_head": "...",
    "source_hashes": {},
    "checkpoint": "/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B",
    "checkpoint_config_hash": "...",
    "weight_index_hash": "..."
  },
  "runtime": {
    "python": "...",
    "torch": "...",
    "diffusers": "...",
    "cuda": "...",
    "gpu": "...",
    "peak_cuda_bytes": 0,
    "elapsed_seconds": 0.0
  }
}
```

示例中的 `[0.0]` 只是字段占位；实际必须保存完整 30 项 layer gates 和完整 `sampling_steps` 项 step gates。

### 10.3 residual 诊断

smoke test 对每个 active step、active layer 保存标量，不保存完整 activation：

```text
||r_sem||_2
||r_pos||_2
||r_neg||_2
||r_pos-r_neg||_2
||scaled_delta||_2 / (||r_sem||_2 + eps)
cos(r_pos, r_neg)
isfinite
```

批量 pilot 可只保存预选 steps 或在线聚合的 min/median/max，避免频繁 GPU 同步拖慢生成。判断建议：

- 所有 active 层的差分比都接近 0：先查文本是否相同、T5 是否截断、context 是否传错，再提高 lambda；
- 差分长期远大于语义 residual 且画质崩：降低 lambda 或缩小窗口；
- 出现 NaN/Inf：立即终止该视频并写 `failure.json`，不能继续保存为有效结果。

这些 norm 只用于机制诊断，不作为论文效果指标。

---

## 11. 测试与验收门槛

### 11.1 无 GPU 单元测试

1. **路径边界**：所有输出/caches/tmp 解析后均在 `code/v1`；任何 `/model/**` 写目标都被拒绝，包括符号链接逃逸。
2. **schema**：P01–P20 的三个现有文本/JSON 文件可解析；字段缺失、原 prompt 不一致、非法 counterfactual 类型会失败。
3. **schedule 边界**：50 steps 与 40 steps 的分段索引完全符合第 3.5 节。
4. **layer preset**：`mvp_mid16` 只有 16 个非零 gate；五个扫描窗口使用半开区间且索引正确。
5. **block 公式**：用微型 mock block 验证输出等于 `semantic + scale*(positive-counterfactual)`，FFN 仅调用一次。
6. **调用计数**：active block 有 3 次 cross-attention，inactive block 只有 1 次；unconditional forward 全部只有 1 次。
7. **配置防错**：M3 无图、ACE enabled 无 c+/c-、layer 数不匹配、非法帧数、非法尺寸均 fail fast。

### 11.2 GPU 集成测试

1. **基模加载**：只出现一份 DiT 参数；proxy 不增加 trainable parameter，不产生 missing/unexpected checkpoint keys。
2. **lambda-zero parity**：同一输入/seed 下，外部 pipeline 的官方快路径与直接官方 `WanTI2V.generate` 输出 latent 在允许误差内一致；优先要求 `max_abs <= 1e-6`，如底层 kernel 非确定则同时记录环境并设经论证的 dtype 相对误差。
3. **late-step bypass**：后 20% step 不再投影 c+/c-，也不计算额外 block attention，model proxy 命中原 forward 快路径；采样前已经得到的 T5 context 不需要重复编码。
4. **CFG 隔离**：日志证实每 step 仅 conditional call 带 ACE，unconditional call 的 `lambda0=0`。
5. **I2V 重锚**：首帧 latent 在初始化和每次 scheduler 更新后按官方 mask 重置；测试计数为 `sampling_steps + 1`。
6. **确定性**：同机同环境、相同 seed/config 重跑结果一致；不同 seed 的 manifest 不覆盖。
7. **资源释放**：连续运行至少 3 个 49 帧样本，第二、第三个样本开始前的保留显存不持续增长。

### 11.3 smoke 通过条件

- M2 `lambda0=0` parity 通过；这是继续 ACE 实验的前置条件。
- M3 生成完成，无 NaN/Inf，输出 49 帧且尺寸正确。
- c+ 与 c- 的 residual norm 非零，且启用/禁用层调用计数正确。
- `/home/liuzhirui/model/Wan2.2` 的受监控源文件 hash 在运行前后不变。
- 视频、manifest、contexts、resolved config 和性能数据齐全。
- M2/M3 使用同一 I0 hash，M0/M4 不使用 I0。

任何一项失败都不能进入 64 视频 pilot。

---

## 12. 两天实施顺序

### 12.1 Day 1 上午：接口、边界和 parity

1. 建立 `ace_router` 包、配置 schema、路径守卫与 upstream hash guard。
2. 实现并测试 layer/step schedule。
3. 实现 `AceModelProxy` 的官方快路径，先不加 ACE 数学。
4. 用 P01、1 seed、49 帧、640×352 验证 M0/M2 官方生成与 artifact 落盘。
5. 完成 `lambda0=0` latent parity；未通过前不实现批量脚本。

### 12.2 Day 1 下午：ACE block 与单例诊断

1. 实现外部 `ace_model_forward` 与 `ace_block_forward`。
2. 接入三条 T5 context，确认 T5 只编码一次。
3. 强制 ACE 只进入 conditional CFG 分支。
4. P01 运行 `mvp_mid16 × lambda0=0.25/0.5/1.0`，记录 residual、显存和耗时。
5. 检查实体、事件、可见接触、先因后果、末态保持与画质。

若所有 lambda 都无差异，不应立刻批量跑视频；先检查：

```text
三条最终字符串与 token hash
T5 context 是否真的不同
positive/counterfactual text_embedding 后的差异 norm
active layer 是否命中
每层 r_pos-r_neg norm
lambda 是否被重复乘或始终为零
```

### 12.3 Day 2 上午：层窗口扫描和冻结

在 6 个机制 prompt、每个 1 seed 上比较：

```text
early [0,10)
mid_a [10,20)
mid_b [14,24)
late  [20,30)
all   [0,30)
```

扫描阶段窗口内 gate 都为 1，只研究位置。根据预先定义的七项人工问题选择一个窗口，再冻结 `lambda0`；不能一边看 20 条测试样本一边调参数。

### 12.4 Day 2 下午：最小矩阵

第一批：`8 prompts × 2 seeds × 4 methods = 64 videos`，四方法固定为 M0、M1、M2、M3。若长物理扩写没有冻结输入，则用 M4 替代 M1，同时在报告中明确矩阵变化。

只有 smoke、窗口扫描和 64 视频矩阵均稳定后，才扩展到：

```text
20 prompts × 3 seeds × 冻结方法集合
```

多 GPU 运行采用静态 job 列表分片：每个进程只绑定一张 GPU，每个输出目录唯一。不要让多个 worker 写同一个 batch JSON；各 worker 写独立结果，最后由单独聚合步骤生成只读索引。

---

## 13. 盲审问题、诊断解释与止损

每个视频至少回答：

1. 关键实体是否在应该存在的阶段出现，合理相变/破裂后是否仍保持材料与事件连续性？
2. 初始状态是否正确？
3. 事件是否真正发生？
4. 接触或作用是否可见？
5. 原因是否先于响应？
6. 指定末态是否出现并保持？
7. 实体连续性、初态一致性和画质是否不低于原始短 prompt？

结果解释：

| 现象 | 优先判断 | 下一步 |
|---|---|---|
| M2 优于 M0，M3≈M2 | 主要收益来自 I0，ACE 未提供额外因果信号 | 查 delta norm、层位置和 c+/c- 质量；看 M4 |
| M4 优于 M0，M3 优于 M2 | causal routing 本身有效，且能与 I0 组合 | 进入 TRACE-Time 对照 |
| 接触顺序改善但背景/外观崩 | residual 过强或层窗口过宽 | 降 lambda、比较 mid 窗口，不改 prompt |
| 开场已是终态 | I0 或全局文本发生 outcome preset | 先审计 I0；ACE 本身没有 output-time gate |
| 中间态仍缺失/瞬间替换 | ACE 的全局条件能力边界 | 不继续堆 lambda，转 TRACE 的 output-time phase |
| 无关区域也变化 | ACE 的全局空间广播边界 | 转 TRACE 的 ROI/token gate，不在 ACE 中暗加 mask |
| c+、c- delta 极小 | 条件过近、T5 截断或传参错误 | 先做 embedding/attention 诊断 |
| delta 大但行为无变化 | 层/step 位置不对，或生成器未把方向映射成运动 | 做窗口扫描；不要只继续增大强度 |

止损条件：

- `lambda0=0` 无法复现官方路径；
- 三条 context 不能稳定区分或有静默截断；
- 所有窗口在 `lambda0<=1.0` 下都没有可见/可测差异，且 attention residual 确认已注入；
- 一旦有差异就普遍造成实体、画面或数值崩坏；
- I0 质量不足导致 M2 本身不成立。

触发止损时保留日志和失败样本，不用“继续扩写 prompt”掩盖实现或方法边界。

---

## 14. 与 TRACE-Lite 的接口边界

ACE V1 输出应为后续 TRACE 保留以下可复用模块：

- 输入 schema 与 c+/c- 审计；
- upstream 只读加载与 hash guard；
- 模型 proxy、block helper 和 CFG conditional-only 约束；
- layer/step schedule；
- artifact、residual diagnostics、batch runner 与 parity tests。

TRACE-Lite 后续新增的是：

```text
单对全局 c+/c-
  -> 多 phase c_m+/c_m-

标量 layer-step gate
  -> G_m(f,x,y) × layer-step gate

全 token 广播
  -> 输出视频时间 × 局部空间 support
```

因此 ACE 的 context 接口命名和 artifact schema 不应假设永远只有一个 phase；但 V1 实际执行仍严格限定 `M=1`，不得提前塞入未经验证的 phase/ROI 逻辑。

---

## 15. 实现完成清单

### 工程边界

- [ ] 所有新 Python/shell/config/test 均在 `code/v1`。
- [ ] 输出、日志、cache、tmp 均在 `code/v1`。
- [ ] 使用 `PYTHONDONTWRITEBYTECODE=1`，未在 Wan 目录生成 pyc。
- [ ] 运行前后 Wan 受监控文件 hash 相同。
- [ ] checkpoint 只加载一次，proxy 不复制 5B 参数。

### 方法正确性

- [ ] 三条 T5 文本按固定规则组装并记录。
- [ ] c+、c- 使用同一个 block 的同一个 cross-attention 权重。
- [ ] 差分在 FFN 前注入，FFN 只执行一次。
- [ ] ACE 只进入 CFG conditional forward。
- [ ] `lambda0=0`、late step 和全零 layer gate 命中官方快路径。
- [ ] I2V 首帧在采样前和每步后重锚。

### 实验可靠性

- [ ] P01–P20 schema 全部校验。
- [ ] M2/M3 的 I0 hash 一致，缺图不降级。
- [ ] layer gate 明确为 30 项，窗口使用半开区间。
- [ ] step gate 从实际 sampling step 数动态生成。
- [ ] 每个视频都有 resolved config、manifest、contexts 和性能记录。
- [ ] parity、调用计数、非有限值、连续运行释放测试通过。
- [ ] 先冻结开发设置，再运行 64 视频和 20-prompt 扩展。

满足以上清单后，ACE-Router 才可作为第一部分第 5–8 节所描述的可靠 MVP 基线；否则生成出一个视频并不等同于实现正确。
