#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime_env.sh"
config="${CONFIG:-${LORA_ROOT}/configs/${1:?Expected a default comparison config filename}}"
entry="${LORA_ROOT}/frame_causal_lora/train.py"
# The Python config selects one GPU or one process owning all four GPUs.
lora_python "$entry" --config "$config" --check-gpu
args=(--config "$config")
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")
[[ -z "${RUN_NAME:-}" ]] || args+=(--run-name "$RUN_NAME")
case "${PREPARE:-1}" in
  1) lora_python "$entry" "${args[@]}" --prepare-only ;;
  0) ;;
  *) echo 'PREPARE must be 1 (check/reuse VAE/T5 cache) or 0' >&2; exit 2 ;;
esac
lora_python "$entry" "${args[@]}"
