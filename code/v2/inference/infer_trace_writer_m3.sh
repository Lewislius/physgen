#!/usr/bin/env bash
set -euo pipefail

INFERENCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${INFERENCE_ROOT}/.." && pwd)"
mkdir -p "${V2_ROOT}/tmp/trace_m3" "${V2_ROOT}/.cache/huggingface" "${V2_ROOT}/.cache/torch"

source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

WAN_REPO="${WAN_REPO:-/home/liuzhirui/model/Wan2.2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
DEMO_ROOT="${DEMO_ROOT:-/home/liuzhirui/Project/physGen/code/v1/demo}"
PLAN_ROOT="${PLAN_ROOT:-${DEMO_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/Project/physGen/code/v2/outputs/trace_writer}"
RUN_ID="${RUN_ID:-trace-m3-$(date -u +%Y%m%d-%H%M%S)}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${WAN_REPO}:${V2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${V2_ROOT}/.cache/huggingface" TORCH_HOME="${V2_ROOT}/.cache/torch"
export XDG_CACHE_HOME="${V2_ROOT}/.cache" TMPDIR="${V2_ROOT}/tmp/trace_m3"
export WAN_FLASH_ATTN_FORCE_VERSION="${WAN_FLASH_ATTN_FORCE_VERSION:-2}"
export WAN_FLASH_ATTN_FORCE_CONTIGUOUS="${WAN_FLASH_ATTN_FORCE_CONTIGUOUS:-1}"

cmd=(python -B "${INFERENCE_ROOT}/infer_trace_writer.py"
  --mode i2v
  --wan_repo "${WAN_REPO}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --demo_root "${DEMO_ROOT}"
  --plan_root "${PLAN_ROOT}"
  --output_root "${OUTPUT_ROOT}"
  --run_id "${RUN_ID}"
  --sample_ids "${SAMPLE_IDS:-all}"
  --seeds "${SEEDS:-42}"
  --crossfade_tokens "${CROSSFADE_TOKENS:-1}"
  --spatial_mode "${SPATIAL_MODE:-spacetime}"
  --reader_mode "${READER_MODE:-off}"
  --width "${WIDTH:-1280}"
  --height "${HEIGHT:-704}"
  --max_area "${MAX_AREA:-901120}"
  --frame_num "${FRAME_NUM:-97}"
  --sampling_steps "${SAMPLING_STEPS:-50}"
  --sample_solver "${SAMPLE_SOLVER:-unipc}"
  --guide_scale "${GUIDE_SCALE:-3.5}"
  --shift "${SHIFT:-5.0}"
  --fps "${FPS:-24}"
  --lambda0 "${LAMBDA0:-0.10}"
  --token_cap_ratio "${TOKEN_CAP_RATIO:-0.10}"
  --global_cap_ratio "${GLOBAL_CAP_RATIO:-0.02}"
  --device_id "${INFERENCE_DEVICE:-0}"
)
[[ "${OFFLOAD_MODEL:-1}" == "1" ]] && cmd+=(--offload_model) || cmd+=(--no-offload_model)
[[ "${T5_CPU:-1}" == "1" ]] && cmd+=(--t5_cpu) || cmd+=(--no-t5_cpu)
[[ "${CONVERT_MODEL_DTYPE:-1}" == "1" ]] && cmd+=(--convert_model_dtype) || cmd+=(--no-convert_model_dtype)
[[ "${SHOW_PROGRESS:-1}" == "1" ]] && cmd+=(--show_progress) || cmd+=(--no-show_progress)
[[ "${CONTINUE_ON_ERROR:-1}" == "1" ]] && cmd+=(--continue_on_error)
[[ "${ALLOW_UPSTREAM_DRIFT:-0}" == "1" ]] && cmd+=(--allow_upstream_drift)
[[ "${PARSE_ONLY:-0}" == "1" ]] && cmd+=(--parse_only)
[[ "${CHECK_ONLY:-0}" == "1" ]] && cmd+=(--check_only)

printf '[RUN] TRACE-M3 I2V fixed5: planimg + PXX-i0-1280x704 image\n'
printf '[RUN] samples=%s seeds=%s crossfade=%s spatial=%s\n' "${SAMPLE_IDS:-all}" "${SEEDS:-42}" "${CROSSFADE_TOKENS:-1}" "${SPATIAL_MODE:-spacetime}"
printf '[RUN] plans=%s output=%s/%s\n' "${PLAN_ROOT}" "${OUTPUT_ROOT}" "${RUN_ID}"
exec "${cmd[@]}"
