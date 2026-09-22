#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../tools/runtime_env.sh"
config="${CONFIG:-${LORA_ROOT}/configs/train_4x48g.yaml}"
output="${OUTPUT:-${LORA_ROOT}/inference_outputs/four_gpu/final_step1200_ema_seed42}"
args=(--config "$config" --output "$output" --suite "${SUITE:-final}")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
if [[ "${SUITE:-final}" == single ]]; then
  args+=(--mode "${MODE:-i2v}" --prompt "${PROMPT:?Single inference requires PROMPT}" --duration "${DURATION:-5}")
  if [[ "${MODE:-i2v}" == i2v ]]; then args+=(--image "${IMAGE:?I2V requires IMAGE}"); fi
fi
if [[ "${PORTRAIT:-0}" == 1 ]]; then args+=(--portrait); fi
lora_python "${LORA_ROOT}/four_gpu/infer.py" --phase prepare "${args[@]}"
lora_python "${LORA_ROOT}/four_gpu/infer.py" --config "$config" --phase sample --output "$output"
lora_python "${LORA_ROOT}/four_gpu/infer.py" --config "$config" --phase report --output "$output"
