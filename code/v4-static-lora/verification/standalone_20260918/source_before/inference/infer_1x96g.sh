#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../tools/runtime_env.sh"
export V42_DIRECT_PYTHON="${V42_DIRECT_PYTHON:-1}"
checkpoint="${CHECKPOINT:-${LORA_ROOT}/checkpoints/lora_20260917T140727Z_dae2a0/step0800}"
checkpoint="${checkpoint%/}"
config="${CONFIG:-${checkpoint}/config.json}"
suite="${SUITE:-demo}"
run_dir="$(dirname -- "$checkpoint")"
output="${OUTPUT:-${LORA_ROOT}/inference_outputs/$(basename -- "$run_dir")_$(basename -- "$checkpoint")_ema_${suite}_seed42}"
args=(--config "$config" --checkpoint "$checkpoint" --output "$output" --suite "$suite")
if [[ "$suite" == single ]]; then
  args+=(--mode "${MODE:-i2v}" --prompt "${PROMPT:?Single inference requires PROMPT}" --duration "${DURATION:-5}")
  if [[ "${MODE:-i2v}" == i2v ]]; then args+=(--image "${IMAGE:?I2V requires IMAGE}"); fi
fi
if [[ "${PORTRAIT:-0}" == 1 ]]; then args+=(--portrait); fi
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  lora_python "${LORA_ROOT}/inference/infer.py" --phase check "${args[@]}"
  exit 0
fi
lora_python "${LORA_ROOT}/inference/infer.py" --phase prepare "${args[@]}"
lora_python "${LORA_ROOT}/inference/infer.py" --config "$config" --phase sample --output "$output"
lora_python "${LORA_ROOT}/inference/infer.py" --config "$config" --phase report --output "$output"
