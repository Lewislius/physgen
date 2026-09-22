#!/usr/bin/env bash
# Standalone launcher: no sourcing scripts from another experiment.
LORA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LORA_ROOT
lora_python() {
  local prefix=/home/liuzhirui/miniconda3/envs/moviestory
  local executable="${prefix}/bin/python"
  if [[ ! -x "$executable" ]]; then echo "Required interpreter missing: $executable" >&2; return 1; fi
  local -a command=("$executable" -I -B -u "$@")
  if [[ "${PARSE_ONLY:-0}" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; return 0; fi
  (
    unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONEXECUTABLE PYTHON_EXEC VIRTUAL_ENV
    local path_name entry cleaned
    local -a parts
    for path_name in PATH LD_LIBRARY_PATH; do
      IFS=: read -r -a parts <<< "${!path_name:-}"
      cleaned=""
      for entry in "${parts[@]}"; do
        case "$entry" in
          "${prefix}"/*) ;;
          /home/liuzhirui/miniconda3/envs/*) continue ;;
        esac
        [[ -z "$entry" ]] || cleaned="${cleaned:+${cleaned}:}${entry}"
      done
      export "${path_name}=${cleaned}"
    done
    export PATH="${prefix}/bin:${PATH}"
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" TOKENIZERS_PARALLELISM=false
    export TMPDIR="${LORA_LOCAL_TMP:-/tmp}"
    printf '[lora] Starting %s\n' "$1"
    "${command[@]}"
  )
}
