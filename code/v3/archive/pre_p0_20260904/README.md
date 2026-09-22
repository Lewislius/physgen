# TRACE-Writer v3

本目录实现 `fixed5-causal-r4`：增强全局语义锚点、阶段实体数量清单、共享锚点的正反最小关系对、封闭世界/生命周期/守恒账本、M3 动态时空 support、M4 operator-saliency support、Boundary-Cosine Trust Region、跨层聚合 SafeCap 和 CFG 后精确 trust region。

输入保持不变：

- M3：`/home/liuzhirui/Project/physGen/code/v1/demo/PXX/PXX-V2-planimg.json` 和同目录的 `*1280x704.jpg/.jpeg/.png`；
- M4：`/home/liuzhirui/Project/physGen/code/v1/demo/PXX/PXX-V2-plan.json`，不读取首帧。

无模型预检全部 20 个样本：

```bash
CHECK_ONLY=1 /home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer_m3.sh
CHECK_ONLY=1 /home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer_m4.sh
```

正式运行：

```bash
/home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer_m3.sh
/home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer_m4.sh
```

提交 Determined：

```bash
det experiment create /home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer_m3.yaml /home/liuzhirui/Project/physGen/code/v3
det experiment create /home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer_m4.yaml /home/liuzhirui/Project/physGen/code/v3
```

`aggregate_strict` 每个去噪步执行 trace conditional、base conditional、unconditional 三次 forward，较原两次 forward 增加约 50% denoiser 计算，用于得到精确 CFG 后 TRACE delta。批量吞吐测试可显式设置 `CAP_MODE=aggregate_estimated`。

当前 JSON 中集合实体的精确数量、真实目标轨迹及质量标定不足时会写入 `compiled.issues`，不会伪造。可在 `/home/liuzhirui/Project/physGen/code/v3/control_overlays/PXX/PXX-m3-control.json` 或 `PXX-m4-control.json` 中补充显式 control 数据。
