#!/usr/bin/env bash
set -eo pipefail
V4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${2:-${V4_ROOT}/configs/wisa_native_p.yaml}"
source "${V4_ROOT}/tools/runtime_env.sh"
v4_activate runtime
v4_setup_tmpdir
NPROC="${1:-$(v4_gpu_count)}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
"${V4_PYTHON}" -I -B "${V4_ROOT}/tools/prepare_wisa.py" --config "${CONFIG}" --pass index
for ENCODER_PASS in vae text; do
  "${V4_PYTHON}" -I -B -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" "${V4_ROOT}/tools/prepare_wisa.py" --config "${CONFIG}" --pass "${ENCODER_PASS}"
  if [ "${ENCODER_PASS}" = vae ]; then
    "${V4_PYTHON}" -I -B "${V4_ROOT}/tools/prepare_wisa.py" --config "${CONFIG}" --pass filter
  fi
done
# Keep teacher dependencies isolated; the parent stays in moviestory for finalize.
(
  v4_activate teacher
  "${V4_PYTHON}" -I -B -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" "${V4_ROOT}/tools/prepare_wisa.py" --config "${CONFIG}" --pass teacher
)
"${V4_PYTHON}" -I -B "${V4_ROOT}/tools/prepare_wisa.py" --config "${CONFIG}" --pass finalize
