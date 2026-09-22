#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../tools/runtime_env.sh"
config="${CONFIG:-${V43_ROOT}/configs/flow_oracle.yaml}"
args=(--config "$config")
[[ -z "${RUN_NAME:-}" ]] || args+=(--run-name "$RUN_NAME")
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")
[[ -z "${INIT_FROM:-}" ]] || args+=(--init-from "$INIT_FROM")
[[ -z "${CHECK_STEPS:-}" ]] || args+=(--check-steps "$CHECK_STEPS")
[[ -z "${STEPS:-}" ]] || args+=(--steps "$STEPS")
[[ -z "${WORKERS:-}" ]] || args+=(--workers "$WORKERS")
[[ -z "${SAMPLE_LIMIT:-}" ]] || args+=(--sample-limit "$SAMPLE_LIMIT")
v43_python runtime "${V43_ROOT}/train/train.py" "${args[@]}" --check-config
if [[ "${CHECK_CONFIG_ONLY:-0}" == 1 ]]; then exit 0; fi
v43_python runtime -m torch.distributed.run --standalone --nproc-per-node="${NPROC_PER_NODE:-1}" \
  "${V43_ROOT}/train/train.py" "${args[@]}"
