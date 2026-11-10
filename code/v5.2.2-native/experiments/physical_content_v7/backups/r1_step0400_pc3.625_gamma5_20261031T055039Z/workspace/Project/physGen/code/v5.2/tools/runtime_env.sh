#!/usr/bin/env bash
V52_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export V52_ROOT

v52_python() (
  local prefix="${V52_CONDA_ENVS:-/home/liuzhirui/miniconda3/envs}/moviestory"
  unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONEXECUTABLE VIRTUAL_ENV
  local path_name entry cleaned
  local -a parts
  for path_name in PATH LD_LIBRARY_PATH; do
    IFS=: read -r -a parts <<< "${!path_name:-}"
    cleaned=""
    for entry in "${parts[@]}"; do
      case "$entry" in */envs/*) continue ;; esac
      [[ -z "$entry" ]] || cleaned="${cleaned:+${cleaned}:}${entry}"
    done
    export "${path_name}=${cleaned}"
  done
  export PATH="${prefix}/bin:${PATH}"
  export CONDA_PREFIX="$prefix" CONDA_DEFAULT_ENV=moviestory
  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  exec "${prefix}/bin/python" -I -B -u "$@"
)
