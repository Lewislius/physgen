#!/usr/bin/env bash
set -euo pipefail
# Same process-local interpreter setup as v5.2/tools/runtime_env.sh.
# This runs AFTER Determined bootstrap; no installation or cluster changes.
NATIVE_ENV=/home/liuzhirui/miniconda3/envs/moviestory
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONEXECUTABLE VIRTUAL_ENV
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
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "${NATIVE_ENV}/bin/python" -I -B -u "$@"
