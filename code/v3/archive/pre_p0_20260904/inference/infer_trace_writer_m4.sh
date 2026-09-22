#!/usr/bin/env bash
set -euo pipefail

V3_ROOT=/home/liuzhirui/Project/physGen/code/v3
CODE_ROOT=/home/liuzhirui/Project/physGen/code
INFERENCE_SCRIPT=/home/liuzhirui/Project/physGen/code/v3/inference/infer_trace_writer.py
mkdir -p /home/liuzhirui/Project/physGen/code/v3/tmp/trace_m4
mkdir -p /home/liuzhirui/Project/physGen/code/v3/.cache/huggingface
mkdir -p /home/liuzhirui/Project/physGen/code/v3/.cache/torch

source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

WAN_REPO=${WAN_REPO:-/home/liuzhirui/model/Wan2.2}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}
DEMO_ROOT=${DEMO_ROOT:-/home/liuzhirui/Project/physGen/code/v1/demo}
PLAN_ROOT=${PLAN_ROOT:-/home/liuzhirui/Project/physGen/code/v1/demo}
CONTROL_OVERLAY_ROOT=${CONTROL_OVERLAY_ROOT:-/home/liuzhirui/Project/physGen/code/v3/control_overlays}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/liuzhirui/Project/physGen/code/v3/outputs/trace_writer}
RUN_ID=${RUN_ID:-trace-v3-m4-$(date -u +%Y%m%d-%H%M%S)}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${CODE_ROOT}:${V3_ROOT}:${WAN_REPO}${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/home/liuzhirui/Project/physGen/code/v3/.cache/huggingface
export TORCH_HOME=/home/liuzhirui/Project/physGen/code/v3/.cache/torch
export XDG_CACHE_HOME=/home/liuzhirui/Project/physGen/code/v3/.cache
export TMPDIR=/home/liuzhirui/Project/physGen/code/v3/tmp/trace_m4
export WAN_FLASH_ATTN_FORCE_VERSION=${WAN_FLASH_ATTN_FORCE_VERSION:-2}
export WAN_FLASH_ATTN_FORCE_CONTIGUOUS=${WAN_FLASH_ATTN_FORCE_CONTIGUOUS:-1}

command=(python -B ${INFERENCE_SCRIPT}
  --mode t2v
  --wan_repo ${WAN_REPO}
  --checkpoint_dir ${CHECKPOINT_DIR}
  --demo_root ${DEMO_ROOT}
  --plan_root ${PLAN_ROOT}
  --control_overlay_root ${CONTROL_OVERLAY_ROOT}
  --output_root ${OUTPUT_ROOT}
  --run_id ${RUN_ID}
  --sample_ids ${SAMPLE_IDS:-all}
  --seeds ${SEEDS:-42}
  --minimal_pair_mode ${MINIMAL_PAIR_MODE:-strict}
  --crossfade_tokens ${CROSSFADE_TOKENS:-2}
  --event_clock_mode ${EVENT_CLOCK_MODE:-plan_prior}
  --boundary_morph ${BOUNDARY_MORPH:-cosine_trust}
  --dynamic_support_mode ${DYNAMIC_SUPPORT_MODE:-planned_saliency}
  --cap_mode ${CAP_MODE:-aggregate_strict}
  --reader_mode ${READER_MODE:-audit}
  --verify_mode ${VERIFY_MODE:-audit}
  --width ${WIDTH:-1280}
  --height ${HEIGHT:-704}
  --max_area ${MAX_AREA:-901120}
  --frame_num ${FRAME_NUM:-97}
  --sampling_steps ${SAMPLING_STEPS:-50}
  --sample_solver ${SAMPLE_SOLVER:-unipc}
  --guide_scale ${GUIDE_SCALE:-3.5}
  --shift ${SHIFT:-5.0}
  --fps ${FPS:-24}
  --lambda0 ${LAMBDA0:-0.05}
  --token_cap_ratio ${TOKEN_CAP_RATIO:-0.05}
  --layer_cap_ratio ${LAYER_CAP_RATIO:-0.01}
  --group_cap_ratio ${GROUP_CAP_RATIO:-0.01}
  --cfg_cap_ratio ${CFG_CAP_RATIO:-0.01}
  --cfg_outside_cap_ratio ${CFG_OUTSIDE_CAP_RATIO:-0.0025}
  --temporal_derivative_ratio ${TEMPORAL_DERIVATIVE_RATIO:-0.05}
  --device_id ${INFERENCE_DEVICE:-0})

[[ ${OFFLOAD_MODEL:-1} == 1 ]] && command+=(--offload_model) || command+=(--no-offload_model)
[[ ${T5_CPU:-1} == 1 ]] && command+=(--t5_cpu) || command+=(--no-t5_cpu)
[[ ${CONVERT_MODEL_DTYPE:-1} == 1 ]] && command+=(--convert_model_dtype) || command+=(--no-convert_model_dtype)
[[ ${SHOW_PROGRESS:-1} == 1 ]] && command+=(--show_progress) || command+=(--no-show_progress)
[[ ${CONTINUE_ON_ERROR:-1} == 1 ]] && command+=(--continue_on_error) || command+=(--no-continue_on_error)
[[ ${ALLOW_UPSTREAM_DRIFT:-0} == 1 ]] && command+=(--allow_upstream_drift)
[[ ${PARSE_ONLY:-0} == 1 ]] && command+=(--parse_only)
[[ ${CHECK_ONLY:-0} == 1 ]] && command+=(--check_only)

printf '[RUN] TRACE-v3 M4/T2V: 20 plan demos, no first-frame input\n'
printf '[RUN] samples=%s cap=%s support=%s output=%s/%s\n' "${SAMPLE_IDS:-all}" "${CAP_MODE:-aggregate_strict}" "${DYNAMIC_SUPPORT_MODE:-planned_saliency}" "${OUTPUT_ROOT}" "${RUN_ID}"
exec "${command[@]}"
