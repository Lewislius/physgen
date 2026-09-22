#!/usr/bin/env bash
set -euo pipefail

V3_ROOT=/home/liuzhirui/Project/physGen/code/v3
DEMO_TAG=${DEMO_TAG:-$(date -u +%Y%m%d-%H%M%S)}

run_variant() (
  label=$1
  stage_prompt=$2
  export SAMPLE_IDS=P01
  export SEEDS=42
  export CFG_MODE=separate
  export GUIDE_SCALE=5.0
  export PHYS_GUIDANCE_SCALE=2.0
  export CONDITIONING_VARIANT=strong_stage
  export STAGE_STRENGTH=0.0
  export STAGE_RATIO_TO_GLOBAL=0.05
  export JSON_GLOBAL_RATIO_TO_GLOBAL=0.0
  export JSON_STAGE_RATIO_TO_GLOBAL=0.0
  export CONDITION_EXTRA_CAP_RATIO=0.10
  export CONDITION_NORM_MAX_SCALE=10.0
  export STAGE_BLOCK_START=14
  export STAGE_BLOCK_STOP=24
  export CONDITIONING_AUDIT=1
  export GLOBAL_PROMPT_OVERRIDE="Static side view of the same black electric kick scooter with its front wheel touching the same upright black metal trash can. Both original objects remain fully visible and intact."
  export STAGE_PROMPT_OVERRIDE=${stage_prompt}
  export RUN_ID=trace-v3-demo-test-D-${label}-${DEMO_TAG}
  "${V3_ROOT}/inference/infer_trace_writer_m4.sh"
)

run_variant fall "The same scooter rotates down onto its right side and ends fully lying motionless beside the upright trash can."
run_variant upright "The same scooter remains completely upright and motionless beside the upright trash can through the final frames."
