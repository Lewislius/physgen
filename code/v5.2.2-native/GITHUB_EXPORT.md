# V5.2.2-native GitHub export

This directory is the source/design/audit export of the native PhysGen experiments.
It is intentionally **not** a runnable copy of every local artifact.

## Included

- Python source, model/fusion/training/inference code
- YAML configs and Determined submission files
- design notes (`DESIGN.zh.txt`, `GITHUB_EXPORT.md`, README files)
- evaluation/audit JSON, CSV and small text reports, including selected
  `inference_outputs/**/scores.json`, manifests and benchmark summaries
- snapshots of source/configs used by named experiments

## Excluded

- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, tensor arrays
- `runs/`, `determined_checkpoints/`
- generated videos and images (`*.mp4`, `*.jpg`, `*.png`, ...), including
  media under `inference_outputs/` and benchmark `outputs/`
- large `.log` and `.jsonl` traces, temporary and cache files

The exclusions are defined in `.gitignore` in this directory.

## Current experiment families

- `experiments/physical_content_v7/request_bound_fusion_r15b/`
  - R15B training: Qwen routing/execution C plus native-T5 request-bound physical read.
- `experiments/physical_content_v7/b400_joint_read_eval/`
  - B400: inference-only joint direct-C + request-bound-C read on the R15B checkpoint.
- `experiments/physical_content_v7/native_response_fusion_r16/`
  - R16 H/I/J/K source, configs, audit and evaluation records.

Large checkpoints and generated videos remain under the corresponding local
`runs/` and `inference_outputs/` directories.
