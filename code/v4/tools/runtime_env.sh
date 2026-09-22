#!/usr/bin/env bash
# Source this file, then call v4_activate runtime or v4_activate teacher.
v4_activate() {
  local role="$1" prefix
  case "${role}" in
    runtime) prefix=/home/liuzhirui/miniconda3/envs/moviestory ;;
    teacher) prefix=/home/liuzhirui/miniconda3/envs/vjepa2-312 ;;
    *) echo "Unknown v4 environment role: ${role}" >&2; return 1 ;;
  esac
  if [ ! -x "${prefix}/bin/python" ]; then
    echo "Required interpreter missing: ${prefix}/bin/python; no fallback is allowed" >&2
    return 1
  fi
  unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONEXECUTABLE PYTHON_EXEC VIRTUAL_ENV VJEPA_ENV
  source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
  conda activate "${prefix}"
  local path_name path_entry cleaned
  local -a path_entries
  for path_name in PATH LD_LIBRARY_PATH; do
    IFS=: read -r -a path_entries <<< "${!path_name}"
    cleaned=""
    for path_entry in "${path_entries[@]}"; do
      case "${path_entry}" in
        "${prefix}"/*) ;;
        /home/liuzhirui/miniconda3/envs/*) continue ;;
      esac
      if [ -n "${path_entry}" ]; then cleaned="${cleaned:+${cleaned}:}${path_entry}"; fi
    done
    export "${path_name}=${cleaned}"
  done
  export V4_PYTHON="${prefix}/bin/python"
  # torchrun honors PYTHON_EXEC even when its own interpreter is correct.
  export PYTHON_EXEC="${V4_PYTHON}"
  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
}

v4_gpu_count() {
  "${V4_PYTHON}" -I -B -c 'import sys; sys.path.insert(0, sys.argv[1]); from physgen_v4.environments import allocated_gpu_count; print(allocated_gpu_count())' "${V4_ROOT}"
}

v4_setup_tmpdir() {
  # Multiprocessing sockets and open temporary files must stay off the shared NFS
  # project mount. Preparation subprocesses inherit the same per-job directory.
  if [ -z "${V4_JOB_TMPDIR:-}" ]; then
    local scratch_root="${V4_LOCAL_TMP_ROOT:-/tmp}"
    mkdir -p "${scratch_root}"
    export V4_JOB_TMPDIR
    V4_JOB_TMPDIR="$(mktemp -d "${scratch_root%/}/physgen-v4.XXXXXXXX")"
  fi
  mkdir -p "${V4_JOB_TMPDIR}"
  export TMPDIR="${V4_JOB_TMPDIR}" TMP="${V4_JOB_TMPDIR}" TEMP="${V4_JOB_TMPDIR}"
  echo "Runtime temporary directory: ${TMPDIR}"
}
