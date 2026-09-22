#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../tools/runtime_env.sh"
config="${CONFIG:-${LORA_ROOT}/configs/train.yaml}"
output="${OUTPUT:-${LORA_ROOT}/inference_outputs/final_step1200_ema_seed42}"
args=(--config "$config" --output "$output" --suite "${SUITE:-final}")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
if [[ "${SUITE:-final}" == single ]]; then
  args+=(--mode "${MODE:-i2v}" --prompt "${PROMPT:?Single inference requires PROMPT}" --duration "${DURATION:-5}")
  if [[ "${MODE:-i2v}" == i2v ]]; then args+=(--image "${IMAGE:?I2V requires IMAGE}"); fi
fi
if [[ "${PORTRAIT:-0}" == 1 ]]; then args+=(--portrait); fi
lora_python "${LORA_ROOT}/inference/infer.py" --phase prepare "${args[@]}"
lora_python "${LORA_ROOT}/inference/infer.py" --config "$config" --phase sample --output "$output"
lora_python "${LORA_ROOT}/inference/infer.py" --config "$config" --phase report --output "$output"
