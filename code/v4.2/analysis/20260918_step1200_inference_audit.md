# step1200 / demo41 推理适配核验（2026-09-18）

结论：当前推理实现适配 `stability_20260917T181001Z_a83129/step1200`。原96GB和Ada 48GB YAML、共享shell默认值仍指向旧step0100，本次已统一改为指定step1200，并在两个YAML显式设置 `SUITE=demo`、`VARIANT=full`。两种硬件共用 `inference/infer.py` 和相同采样参数，各自输出目录独立。

## 权重与架构

- `config.json`、`ema.pt` 的大小和SHA256均与 `complete.json` 相符，step和EMA更新次数均为1200。
- 通过实际 `load_adapter()` 严格加载333个张量，68,266,498个可训练参数，所有张量有限。推理不读取 `adapter.pt` 或 `training.pt`。
- 当前训练/推理Python文件的完整指纹与checkpoint保存的源码指纹完全一致；本次改动未修改模型或采样Python实现。
- 保存配置启用 `condition_trajectory_v1`，第5/15层各有独立core/writer，两处write gate均存在。没有回退到旧的静态初始化。
- 使用真实EMA、真实P01文本/首图anchor，在CPU执行I2V和T2V初始化：均输出有限的 `[1,16,576,1664]`；首末时间状态差异RMS分别为0.07154和0.08204，时序初始化实际生效。

## 推理流程

1. `prepare` 读取checkpoint保存的配置，准备T5文本和I2V首图VAE条件；缓存存在则复用。
2. `anchors` 仅为I2V准备首图JEPA anchor，不读取真实未来视频。T2V以训练保存的均值、文本、时长和时间位置初始化。
3. `sample` 加载原Wan2.2-TI2V-5B、step1200 EMA和FP32 VAE。每条只计算一次可部署P0，每次solver调用均从同一个P0开始，步内依次更新P5/P15，不将上一步P15接到下一步。
4. 正文本基座分支与带writer分支共享前5层；第5/15层写回均保留gate与幅度约束。最终速度为 `v_negative + 5*(v_base_positive-v_negative) + bounded_delta_v`，adapter残差系数为1，不再被CFG放大5倍。
5. I2V在输入和每次采样更新后固定首个latent切片。UniPC执行50步，shift5、seed42；训练专用STRUCT/WRITE参考支路不参与生成。
6. FP32 VAE逐帧导出MP4，同时保存latent、状态、轨迹、ΔH图表和metadata；完成标记支持同一输出目录续跑。默认 `EVALUATE=0`。

## 41条测试条件

通过调用v4自己的case收集器逐条比较，图片路径、提示词、模式和顺序全部相同：reference I2V 1条，P01–P20各I2V/T2V，共21条I2V、20条T2V。没有加入训练/验证GT案例。

| 设置 | 本次v4.2 |
| --- | --- |
| 帧数与时长条件 | 121帧，5秒时间窗，导出24fps |
| reference画幅 | 384×384 |
| P01–P20画幅 | 512×288 |
| solver | UniPC，50步 |
| shift / CFG / adapter残差系数 / seed | 5 / 5 / 1 / 42 |

121帧的时间点覆盖0–5秒；按24fps封装的容器时长约5.04秒。v4旧YAML采用Euler和101帧，本次复用的是41条测试条件，保持v4.2现有采样方式。

22份文本缓存（含负提示词）、21份首图VAE缓存、21份首图JEPA anchor均存在，并检查了张量形状和有限性。编码器缓存身份与新checkpoint无关，因此可直接复用。

## 验证与执行

- 两种入口的 `PARSE_ONLY=1` 均通过：YAML环境和直接shell启动得到相同的 `prepare → anchors → sample` 命令，均加载指定step1200。
- 三个shell文件通过 `bash -n`。
- `test_independent_correctors`、`test_delta_h_repair` 共8项现有测试全部通过。
- 复现CPU审计：`/home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B analysis/verify_step1200_inference_20260918.py`。
- [机器检查结果](step1200_inference_audit_20260918/result.json)与[完整41条计划](step1200_inference_audit_20260918/cases.json)已保存。

本次环境无法使用CUDA，未提交集群任务、未运行step1200完整GPU采样；48GB显存峰值、实际吞吐和最终画质尚未实测。CPU适配通过不等于完整视频效果已验证。

在v4.2根目录按目标硬件选择一条提交命令：

```bash
det experiment create inference/infer_stability_1x96g.yaml .
# 或 Ada 48GB：
det experiment create inference/infer_stability_1xada48g.yaml .
```

默认输出位于 `inference_outputs/`：

- 96GB：`stability_20260917T181001Z_a83129_step1200_ema_demo_seed42_r3_full/`
- Ada 48GB：`stability_20260917T181001Z_a83129_step1200_ema_demo_1xada48g_seed42_r3_full/`
