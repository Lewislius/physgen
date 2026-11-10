# Independent V5 Q256 / 1000-step training

- User authorization 2026-10-29: scale V5 to Q256 and 1000 updates, adapt loss/LR settings from its logs, and submit to one blk-96g RTX6000. Reject H100 before loading models and resubmit until an actual RTX PRO6000/RTX6000 Blackwell is confirmed. This supersedes ancestor Q128/224 limits here.
- Preserve V5: full trainable independent C K/V (not V6 K/V LoRA), full output, Q enhancement, C-only frozen-Qwen semantic CE and native Wan self-attention/FFN LoRA. Keep original CE targets and mean-token reduction, without a (1-sigma) factor. No verifier, dense labels or new losses.
- New project modules/LoRA start fresh from official bases. The old V5 checkpoint is an analysis reference, never initialization. The immutable physical_variants/snapshots/20261029T064043Z provides the exact original model and semantic target methods.
- Original sources, other training jobs and immutable snapshots must not change. Cache fallback roots are read-only; missing caches and locks are local here. Teacher future observations are not used in V5 CE.
- Two stages: 128 interface + 872 joint, accumulation4, seed42. Details in README.txt and reports. Freeze submitted source snapshots and record actual IDs/GPU/retries in reports/latest_submission.json.
- Reuse existing environments and direct det experiment create commands. No environment upgrades or extra GPU reservations. Hyperparameter choices and CPU tests do not establish improved video quality.
