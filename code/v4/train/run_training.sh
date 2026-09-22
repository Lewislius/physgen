#!/usr/bin/env bash
set -eo pipefail
V4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARDWARE="$1"
PHASE="${PHASE:-A1}"
LOG_ROOT="${V4_ROOT}/train/train_log"
mkdir -p "${LOG_ROOT}"
CONSOLE_LOG="${LOG_ROOT}/launcher_${PHASE}_${HARDWARE}_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
exec > >(tee -a "${CONSOLE_LOG}") 2>&1
echo "Console log: ${CONSOLE_LOG}"
source "${V4_ROOT}/tools/runtime_env.sh"
v4_activate runtime
v4_setup_tmpdir
NPROC="$(v4_gpu_count)"
echo "Allocated GPUs / training workers: ${NPROC}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG="${CONFIG:-${V4_ROOT}/configs/wisa_native_p.yaml}"
MONITOR_ARGS=()
case "${WANDB_ENABLED:-}" in
  1) MONITOR_ARGS+=(--wandb) ;;
  0) MONITOR_ARGS+=(--no-wandb) ;;
  '') ;;
  *) echo 'WANDB_ENABLED must be 0 or 1' >&2; exit 1 ;;
esac
if [ "${PREPARE:-1}" = 1 ]; then
  bash "${V4_ROOT}/tools/prepare_wisa.sh" "${NPROC}" "${CONFIG}"
fi
# Each training rank opens its diagnostics here before initializing CUDA/NCCL.
export V4_DIAGNOSTIC_DIR="${V4_DIAGNOSTIC_DIR:-${CONSOLE_LOG%.txt}_diagnostics}"
mkdir -p "${V4_DIAGNOSTIC_DIR}"
case "${TRAIN_NCCL_DEBUG:-0}" in
  1)
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
    export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-${V4_DIAGNOSTIC_DIR}/nccl.%h.%p.log}"
    export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}"
    export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
    export TORCH_NCCL_TRACE_CPP_STACK="${TORCH_NCCL_TRACE_CPP_STACK:-1}"
    ;;
  0) ;;
  *) echo 'TRAIN_NCCL_DEBUG must be 0 or 1' >&2; exit 1 ;;
esac
ARGS=(--config "${CONFIG}" --phase "${PHASE}" --hardware "${HARDWARE}")
case "${PHASE}" in
  A2) PREVIOUS=A1 ;;
  A3) PREVIOUS=A2 ;;
  B1) PREVIOUS=A3 ;;
  AB) PREVIOUS=B1 ;;
esac
if [ -n "${RESUME:-}" ] && [ -n "${INIT_FROM:-}" ]; then
  echo 'RESUME and INIT_FROM are mutually exclusive' >&2
  exit 1
elif [ -n "${RESUME:-}" ]; then
  ARGS+=(--resume "${RESUME}")
elif [ "${PHASE}" != A1 ]; then
  if [ -z "${INIT_FROM:-}" ]; then
    INIT_FROM="$("${V4_PYTHON}" -I -B "${V4_ROOT}/physgen_v4/checkpoint_paths.py" \
      --config "${CONFIG}" --run-name "wisa_native_p_${PREVIOUS}_${HARDWARE}")"
  fi
  echo "Initialization checkpoint: ${INIT_FROM}"
  ARGS+=(--init-from "${INIT_FROM}")
elif [ -n "${INIT_FROM:-}" ]; then
  ARGS+=(--init-from "${INIT_FROM}")
fi
if [ -n "${RUN_NAME:-}" ]; then ARGS+=(--run-name "${RUN_NAME}"); fi
if [ -n "${TRAIN_STEPS:-}" ]; then ARGS+=(--steps "${TRAIN_STEPS}"); fi
if [ -n "${TRAIN_WORKERS:-}" ]; then ARGS+=(--workers "${TRAIN_WORKERS}"); fi
if [ -n "${TRAIN_LOADER_TIMEOUT_SECONDS:-}" ]; then ARGS+=(--loader-timeout-seconds "${TRAIN_LOADER_TIMEOUT_SECONDS}"); fi
if [ -n "${TRAIN_TRACE_STEPS:-}" ]; then ARGS+=(--trace-steps "${TRAIN_TRACE_STEPS}"); fi
if [ -n "${TRAIN_STACK_TIMEOUT_SECONDS:-}" ]; then ARGS+=(--stack-timeout-seconds "${TRAIN_STACK_TIMEOUT_SECONDS}"); fi
if [ -n "${TRAIN_CHECK_STEPS:-}" ]; then ARGS+=(--check-steps "${TRAIN_CHECK_STEPS}"); fi
case "${TRAIN_PIN_MEMORY:-}" in
  1) ARGS+=(--pin-memory) ;;
  0) ARGS+=(--no-pin-memory) ;;
  '') ;;
  *) echo 'TRAIN_PIN_MEMORY must be 0 or 1' >&2; exit 1 ;;
esac
case "${TRAIN_TRACE_WAN_BLOCKS:-}" in
  1) ARGS+=(--trace-wan-blocks) ;;
  0) ARGS+=(--no-trace-wan-blocks) ;;
  '') ;;
  *) echo 'TRAIN_TRACE_WAN_BLOCKS must be 0 or 1' >&2; exit 1 ;;
esac
ARGS+=("${MONITOR_ARGS[@]}")
exec "${V4_PYTHON}" -I -B -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" "${V4_ROOT}/train/train_native_p.py" "${ARGS[@]}"
