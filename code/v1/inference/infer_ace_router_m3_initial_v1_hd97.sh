#!/usr/bin/env bash
set -euo pipefail

INFERENCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V1_ROOT="$(cd "${INFERENCE_ROOT}/.." && pwd)"
RUNTIME_TMPDIR="${V1_ROOT}/tmp/ace_router_m3_initial_v1_hd97"
mkdir -p "${RUNTIME_TMPDIR}" "${V1_ROOT}/.cache/huggingface" "${V1_ROOT}/.cache/torch"

source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

WAN_REPO="${WAN_REPO:-/home/liuzhirui/model/Wan2.2}"
WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${WAN_REPO}:${V1_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${V1_ROOT}/.cache/huggingface"
export TORCH_HOME="${V1_ROOT}/.cache/torch"
export XDG_CACHE_HOME="${V1_ROOT}/.cache"
export TMPDIR="${RUNTIME_TMPDIR}"
export WAN_FLASH_ATTN_FORCE_VERSION="${WAN_FLASH_ATTN_FORCE_VERSION:-2}"
export WAN_FLASH_ATTN_FORCE_CONTIGUOUS="${WAN_FLASH_ATTN_FORCE_CONTIGUOUS:-1}"

DEMO_ROOT="${DEMO_ROOT:-${V1_ROOT}/demo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${V1_ROOT}/outputs/ace_router_initial_v1_hd97}"
SAMPLE_IDS="${SAMPLE_IDS:-all}"
SEEDS="${SEEDS:-42}"
RUN_ID="${RUN_ID:-m3-initial-v1-hd97-$(date -u +%Y%m%d-%H%M%S)}"

FRAME_NUM="${FRAME_NUM:-97}"
MAX_AREA="${MAX_AREA:-901120}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
SHIFT="${SHIFT:-5.0}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"
FPS="${FPS:-24}"
LAMBDA0="${LAMBDA0:-0.5}"
LAYER_PRESET="${LAYER_PRESET:-mvp_mid16}"
STEP_PRESET="${STEP_PRESET:-legacy}"
DIAGNOSTICS="${DIAGNOSTICS:-summary}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-0}"
MINIMUM_CONFIDENCE="${MINIMUM_CONFIDENCE:-0.7}"

OFFLOAD_MODEL="${OFFLOAD_MODEL:-1}"
T5_CPU="${T5_CPU:-1}"
CONVERT_MODEL_DTYPE="${CONVERT_MODEL_DTYPE:-1}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
ALLOW_UPSTREAM_DRIFT="${ALLOW_UPSTREAM_DRIFT:-0}"
REQUIRE_SELECTION_MANIFEST="${REQUIRE_SELECTION_MANIFEST:-0}"
PARSE_ONLY="${PARSE_ONLY:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"

cmd=(
  python -B "${INFERENCE_ROOT}/infer_ace_router.py"
  --method M3
  --wan_repo "${WAN_REPO}"
  --checkpoint_dir "${WAN_CHECKPOINT_DIR}"
  --demo_root "${DEMO_ROOT}"
  --output_root "${OUTPUT_ROOT}"
  --run_id "${RUN_ID}"
  --sample_ids "${SAMPLE_IDS}"
  --seeds "${SEEDS}"
  --frame_num "${FRAME_NUM}"
  --max_area "${MAX_AREA}"
  --sampling_steps "${SAMPLING_STEPS}"
  --sample_solver "${SAMPLE_SOLVER}"
  --guide_scale "${GUIDE_SCALE}"
  --shift "${SHIFT}"
  --fps "${FPS}"
  --lambda0 "${LAMBDA0}"
  --layer_preset "${LAYER_PRESET}"
  --step_preset "${STEP_PRESET}"
  --conditioning_mode initial_v1
  --cfg_negative_mode wan_default
  --diagnostics "${DIAGNOSTICS}"
  --device_id "${INFERENCE_DEVICE}"
  --minimum_confidence "${MINIMUM_CONFIDENCE}"
)

if [[ "${OFFLOAD_MODEL}" == "1" ]]; then cmd+=(--offload_model); else cmd+=(--no-offload_model); fi
if [[ "${T5_CPU}" == "1" ]]; then cmd+=(--t5_cpu); else cmd+=(--no-t5_cpu); fi
if [[ "${CONVERT_MODEL_DTYPE}" == "1" ]]; then cmd+=(--convert_model_dtype); else cmd+=(--no-convert_model_dtype); fi
if [[ "${SHOW_PROGRESS}" == "1" ]]; then cmd+=(--show_progress); else cmd+=(--no-show_progress); fi
if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then cmd+=(--continue_on_error); fi
if [[ "${ALLOW_UPSTREAM_DRIFT}" == "1" ]]; then cmd+=(--allow_upstream_drift); fi
if [[ "${REQUIRE_SELECTION_MANIFEST}" == "1" ]]; then cmd+=(--require_selection_manifest); fi
if [[ "${PARSE_ONLY}" == "1" ]]; then cmd+=(--parse_only); exec "${cmd[@]}"; fi
if [[ "${CHECK_ONLY}" == "1" ]]; then cmd+=(--check_only); exec "${cmd[@]}"; fi

printf '[RUN] M3 initial ACE-Router V1 compatibility ablation at 1280x704 / 97 frames\n'
printf '[RUN] original short semantic; origin-prefixed cplus/cminus; Wan default CFG negative\n'
printf '[RUN] lambda0=%s cap=none layer=%s step=%s cfg=%s\n' "${LAMBDA0}" "${LAYER_PRESET}" "${STEP_PRESET}" "${GUIDE_SCALE}"
printf '[RUN] preferred I0=Pxx-i0-1280x704.png; samples=%s seeds=%s\n' "${SAMPLE_IDS}" "${SEEDS}"
printf '[RUN] output=%s/%s\n' "${OUTPUT_ROOT}" "${RUN_ID}"
exec "${cmd[@]}"
