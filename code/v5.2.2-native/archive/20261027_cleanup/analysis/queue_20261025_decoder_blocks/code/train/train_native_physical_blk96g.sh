#!/usr/bin/env bash
set -euo pipefail
NATIVE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
NATIVE_ENV=/home/liuzhirui/miniconda3/envs/moviestory
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONEXECUTABLE VIRTUAL_ENV
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Use the same resident Python environment as the existing Wan jobs.
for native_path in PATH LD_LIBRARY_PATH; do
  IFS=: read -r -a native_parts <<< "${!native_path:-}"
  native_clean=""
  for native_entry in "${native_parts[@]}"; do
    case "$native_entry" in */envs/*) continue ;; esac
    [[ -z "$native_entry" ]] || native_clean="${native_clean:+${native_clean}:}${native_entry}"
  done
  export "${native_path}=${native_clean}"
done
export PATH="${NATIVE_ENV}/bin:${PATH}"
export CONDA_PREFIX="$NATIVE_ENV" CONDA_DEFAULT_ENV=moviestory
args=(--config "${CONFIG:-${NATIVE_ROOT}/configs/native_reasoning_operator_wisa3857.yaml}")
[[ "${SMOKE_TEST:-0}" != 1 ]] || args+=(--smoke)
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")
exec "${NATIVE_ENV}/bin/python" -I -B -u "${NATIVE_ROOT}/train/train.py" "${args[@]}" "$@"
