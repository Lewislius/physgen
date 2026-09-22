#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../tools/runtime_env.sh"
config="${CONFIG:-${V42_ROOT}/configs/stability.yaml}"
# Use the same imports/interpreter as training before spending time preparing data.
v42_python runtime "${V42_ROOT}/train/train.py" --config "$config" --check-gpu
case "${PREPARE:-1}" in
  1)
    prepare_args=(--config "$config")
    [[ -z "${RESUME:-}" ]] || prepare_args+=(--resume)
    v42_python runtime "${V42_ROOT}/tools/bootstrap.py" "${prepare_args[@]}"
    ;;
  0) ;;
  *) echo 'PREPARE must be 1 (automatic source preparation) or 0 (already prepared data)' >&2; exit 2 ;;
esac
args=(--config "$config")
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")
[[ -z "${RUN_NAME:-}" ]] || args+=(--run-name "$RUN_NAME")
v42_python runtime "${V42_ROOT}/train/train.py" "${args[@]}"
