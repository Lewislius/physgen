#!/usr/bin/env bash
set -eo pipefail
V4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARDWARE="${1:-1x48g}"
if [ "$#" -gt 0 ]; then shift; fi
case "${HARDWARE}" in
  1x48g|1xada48g|1x80g|1x96g) ;;
  *) echo "Expected 1x48g, 1xada48g, 1x80g or 1x96g; got ${HARDWARE}" >&2; exit 2 ;;
esac
source "${V4_ROOT}/tools/runtime_env.sh"
v4_activate runtime
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ -z "${CHECKPOINT:-}" ] && [ -n "${CHECKPOINT_RUN:-}" ]; then
  CHECKPOINT="$("${V4_PYTHON}" -I -B "${V4_ROOT}/physgen_v4/checkpoint_paths.py" \
    --root "${CHECKPOINT_ROOT:-${V4_ROOT}/checkpoints}" --run-name "${CHECKPOINT_RUN}")"
fi
CHECKPOINT="${CHECKPOINT:-${V4_ROOT}/checkpoints/wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/checkpoint-final}"
echo "Inference checkpoint: ${CHECKPOINT}"
IMAGE="${IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref5.jpg}"
PROMPT="${PROMPT:-The woman smiles and waves at the camera.}"
RUN_TIME="$(date +%Y%m%d-%H%M%S-%N)"
ARGS=(--checkpoint "${CHECKPOINT}" --image "${IMAGE}" --prompt "${PROMPT}"
  --negative-prompt "${NEGATIVE_PROMPT:-}" --device "${DEVICE:-0}"
  --output "${OUTPUT:-${V4_ROOT}/inference_outputs/wisa_native_p_${HARDWARE}/seed${SEED:-42}_${RUN_TIME}.mp4}"
  --frames "${FRAMES:-101}" --fps "${FPS:-24}" --steps "${SAMPLING_STEPS:-50}"
  --shift "${SHIFT:-5}" --guidance "${GUIDANCE:-5}" --solver "${SOLVER:-euler}"
  --variant "${VARIANT:-full}" --seed "${SEED:-42}"
  --mode "${MODE:-i2v}" --demo-modes "${DEMO_MODES:-both}"
  --width "${WIDTH:-512}" --height "${HEIGHT:-288}"
  --suite "${SUITE:-both}" --demo-root "${DEMO_ROOT:-${V4_ROOT}/../v1/demo}"
  --sample-ids "${SAMPLE_IDS:-all}"
  --output-dir "${OUTPUT_DIR:-${V4_ROOT}/inference_outputs/wisa_native_p_${HARDWARE}/final_${RUN_TIME}}")
if [ -n "${CASES:-}" ]; then ARGS+=(--cases "${CASES}"); fi
if [ "${RESUME:-0}" = 1 ]; then
  if [ -z "${OUTPUT_DIR:-}" ]; then
    echo "RESUME=1 requires OUTPUT_DIR pointing to the existing batch directory" >&2
    exit 2
  fi
  ARGS+=(--resume)
fi
if [ -n "${DURATION:-}" ]; then ARGS+=(--duration "${DURATION}"); fi
case "${RESET_STATE:-auto}" in
  auto) ;;
  1) ARGS+=(--reset-state) ;;
  0) ARGS+=(--persistent-state) ;;
  *) echo "RESET_STATE must be auto, 0 or 1" >&2; exit 2 ;;
esac
if [ -n "${WRITER_OFF:-}" ]; then
  read -r -a WRITERS <<< "${WRITER_OFF}"
  ARGS+=(--writer-off "${WRITERS[@]}")
fi
if [ "${CHECK_ONLY:-0}" = 1 ]; then ARGS+=(--check-only); fi
if [ -n "${REPORT:-}" ]; then ARGS+=(--report "${REPORT}"); fi
# Trailing CLI arguments are forwarded; scalar arguments override environment defaults.
ARGS+=("$@")
if [ "${PARSE_ONLY:-0}" = 1 ]; then
  printf '%q ' "${V4_PYTHON}" -I -B "${V4_ROOT}/inference/infer_native_p.py" "${ARGS[@]}"
  printf '\n'
  exit 0
fi
v4_setup_tmpdir
echo "Single GPU inference: ${HARDWARE}; Python: ${V4_PYTHON}"
exec "${V4_PYTHON}" -I -B "${V4_ROOT}/inference/infer_native_p.py" "${ARGS[@]}"
