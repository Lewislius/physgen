#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../tools/runtime_env.sh"
config="${CONFIG:-${V42_ROOT}/configs/stability_v4_fullwidth3.yaml}"
args=(--config "$config" --phase A1 --hardware 1x96g --run-name "${RUN_NAME:-v42_v4_fullwidth3_with_prior_A1_1x96g}")
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")
[[ -z "${INIT_FROM:-}" ]] || args+=(--init-from "$INIT_FROM")
[[ -z "${CHECK_STEPS:-}" ]] || args+=(--check-steps "$CHECK_STEPS")
[[ -z "${STEPS:-}" ]] || args+=(--steps "$STEPS")
v42_python runtime "${V42_ROOT}/train/train_native_p.py" "${args[@]}" --check-config
if [[ "${CHECK_CONFIG_ONLY:-0}" == 1 ]]; then exit 0; fi
if [[ "${PREPARE:-0}" != 0 ]]; then
  echo 'This recipe reuses the existing 121-frame JEPA32 cache; PREPARE must be 0.' >&2
  exit 2
fi
v42_python runtime -m torch.distributed.run --standalone --nproc-per-node=1 \
  "${V42_ROOT}/train/train_native_p.py" "${args[@]}"
