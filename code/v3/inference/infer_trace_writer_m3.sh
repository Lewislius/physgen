#!/usr/bin/env bash
set -euo pipefail

V3_ROOT=/home/liuzhirui/Project/physGen/code/v3
INFERENCE_SCRIPT=${V3_ROOT}/inference/infer_trace_writer.py
mkdir -p ${V3_ROOT}/tmp/trace_p0_i2v ${V3_ROOT}/.cache/huggingface ${V3_ROOT}/.cache/torch

source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

WAN_REPO=${WAN_REPO:-/home/liuzhirui/model/Wan2.2}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}
DEMO_ROOT=${DEMO_ROOT:-/home/liuzhirui/Project/physGen/code/v1/demo}
PLAN_ROOT=${PLAN_ROOT:-/home/liuzhirui/Project/physGen/code/v1/demo}
OUTPUT_ROOT=${OUTPUT_ROOT:-${V3_ROOT}/outputs/trace_writer}
RUN_ID=${RUN_ID:-trace-v3-p0-i2v-$(date -u +%Y%m%d-%H%M%S)}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${WAN_REPO}${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=${V3_ROOT}/.cache/huggingface
export TORCH_HOME=${V3_ROOT}/.cache/torch
export XDG_CACHE_HOME=${V3_ROOT}/.cache
export TMPDIR=${V3_ROOT}/tmp/trace_p0_i2v
export WAN_FLASH_ATTN_FORCE_VERSION=${WAN_FLASH_ATTN_FORCE_VERSION:-2}
export WAN_FLASH_ATTN_FORCE_CONTIGUOUS=${WAN_FLASH_ATTN_FORCE_CONTIGUOUS:-1}

command=(python -B ${INFERENCE_SCRIPT}
  --mode i2v
  --wan_repo ${WAN_REPO}
  --checkpoint_dir ${CHECKPOINT_DIR}
  --demo_root ${DEMO_ROOT}
  --plan_root ${PLAN_ROOT}
  --output_root ${OUTPUT_ROOT}
  --run_id ${RUN_ID}
  --sample_ids ${SAMPLE_IDS:-all}
  --seeds ${SEEDS:-42}
  --width 1280
  --height 704
  --max_area 901120
  --frame_num ${FRAME_NUM:-97}
  --sampling_steps ${SAMPLING_STEPS:-50}
  --sample_solver ${SAMPLE_SOLVER:-unipc}
  --guide_scale ${GUIDE_SCALE:-5.0}
  --cfg_mode ${CFG_MODE:-legacy}
  --phys_guidance_scale ${PHYS_GUIDANCE_SCALE:-1.0}
  --shift ${SHIFT:-5.0}
  --fps ${FPS:-24}
  --stage_strength ${STAGE_STRENGTH:-0.05}
  --conditioning_variant ${CONDITIONING_VARIANT:-baseline}
  --stage_ratio_to_global ${STAGE_RATIO_TO_GLOBAL:-1.0}
  --json_global_ratio_to_global ${JSON_GLOBAL_RATIO_TO_GLOBAL:-0.25}
  --json_stage_ratio_to_global ${JSON_STAGE_RATIO_TO_GLOBAL:-0.50}
  --condition_extra_cap_ratio ${CONDITION_EXTRA_CAP_RATIO:-1.75}
  --condition_norm_max_scale ${CONDITION_NORM_MAX_SCALE:-10.0}
  --stage_block_start ${STAGE_BLOCK_START:-14}
  --stage_block_stop ${STAGE_BLOCK_STOP:-24}
  --device_id ${INFERENCE_DEVICE:-0})

[[ ${OFFLOAD_MODEL:-1} == 1 ]] && command+=(--offload_model) || command+=(--no-offload_model)
[[ ${T5_CPU:-1} == 1 ]] && command+=(--t5_cpu) || command+=(--no-t5_cpu)
[[ ${CONVERT_MODEL_DTYPE:-1} == 1 ]] && command+=(--convert_model_dtype) || command+=(--no-convert_model_dtype)
[[ ${CONTINUE_ON_ERROR:-1} == 1 ]] && command+=(--continue_on_error) || command+=(--no-continue_on_error)
[[ ${SHOW_PROGRESS:-1} == 1 ]] && command+=(--show_progress) || command+=(--no-show_progress)
[[ ${CONDITIONING_AUDIT:-0} == 1 ]] && command+=(--conditioning_audit) || command+=(--no-conditioning_audit)
[[ ${PARSE_ONLY:-0} == 1 ]] && command+=(--parse_only)
[[ ${CHECK_ONLY:-0} == 1 ]] && command+=(--check_only)

printf '[RUN] staged I2V variant=%s: global + time-local stage c+%s\n' \
  "${CONDITIONING_VARIANT:-baseline}" \
  "$([[ ${CONDITIONING_VARIANT:-baseline} == strong_stage_json ]] && printf ' + JSON facts' || true)"
exec "${command[@]}"
