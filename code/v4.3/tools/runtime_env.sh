#!/usr/bin/env bash
# Source in bash. Never modify the installed model/environment directories.
V43_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export V43_ROOT
v43_python() {
  local role="$1" name
  shift
  case "$role" in
    runtime) name=moviestory ;;
    teacher) name=vjepa2-312 ;;
    *) echo "Unknown environment role: $role" >&2; return 1 ;;
  esac
  local executable="/home/liuzhirui/miniconda3/envs/${name}/bin/python"
  if [[ ! -x "$executable" ]]; then echo "Required interpreter missing: $executable" >&2; return 1; fi
  local -a command=("$executable" -I -B -u "$@")
  if [[ "${PARSE_ONLY:-0}" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; return 0; fi
  (
    unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONEXECUTABLE PYTHON_EXEC VIRTUAL_ENV
    if [[ "${V43_DIRECT_PYTHON:-0}" != 1 ]]; then
      source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
      conda activate "/home/liuzhirui/miniconda3/envs/${name}"
    fi
    local path_name entry cleaned
    local -a parts
    for path_name in PATH LD_LIBRARY_PATH; do
      IFS=: read -r -a parts <<< "${!path_name:-}"
      cleaned=""
      for entry in "${parts[@]}"; do
        case "$entry" in
          "/home/liuzhirui/miniconda3/envs/${name}"/*) ;;
          /home/liuzhirui/miniconda3/envs/*) continue ;;
        esac
        [[ -z "$entry" ]] || cleaned="${cleaned:+${cleaned}:}${entry}"
      done
      export "${path_name}=${cleaned}"
    done
    if [[ "${V43_DIRECT_PYTHON:-0}" == 1 ]]; then
      export PATH="/home/liuzhirui/miniconda3/envs/${name}/bin:${PATH}"
      printf '[inference] Starting %s: %s\n' "$name" "$1"
    fi
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export TOKENIZERS_PARALLELISM=false
    export TMPDIR="${V43_LOCAL_TMP:-/tmp}"
    "${command[@]}"
  )
}
