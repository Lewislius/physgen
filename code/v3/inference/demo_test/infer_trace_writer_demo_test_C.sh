#!/usr/bin/env bash
set -euo pipefail

V3_ROOT=/home/liuzhirui/Project/physGen/code/v3
DEMO_TAG=${DEMO_TAG:-$(date -u +%Y%m%d-%H%M%S)}

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
export GLOBAL_PROMPT_OVERRIDE="Static side view beginning with one black electric kick scooter's front wheel already touching one upright black metal trash can. The same scooter immediately rotates down onto its right side and ends fully lying motionless beside the can, while the trash can remains upright."
export STAGE_PROMPT_OVERRIDE="The scooter starts in visible front-wheel contact with the trash can and immediately rotates down onto its right side, ending fully lying motionless beside the upright trash can."
export RUN_ID=trace-v3-demo-test-C-contact-at-start-${DEMO_TAG}

exec "${V3_ROOT}/inference/infer_trace_writer_m4.sh"
