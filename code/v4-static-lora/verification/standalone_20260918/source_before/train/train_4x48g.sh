#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../tools/runtime_env.sh"
config="${CONFIG:-${LORA_ROOT}/configs/train_4x48g.yaml}"
# One process uses all four GPUs. Do not launch torchrun or four replicas.
lora_python "${LORA_ROOT}/four_gpu/train.py" --config "$config" --check-gpu
args=(--config "$config")
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")
[[ -z "${RUN_NAME:-}" ]] || args+=(--run-name "$RUN_NAME")
case "${PREPARE:-1}" in
  1) lora_python "${LORA_ROOT}/four_gpu/train.py" "${args[@]}" --prepare-only ;;
  0) ;;
  *) echo 'PREPARE must be 1 (check/reuse VAE/T5 cache) or 0' >&2; exit 2 ;;
esac
lora_python "${LORA_ROOT}/four_gpu/train.py" "${args[@]}"
