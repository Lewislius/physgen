#!/usr/bin/env bash
set -euo pipefail

V3_ROOT=/home/liuzhirui/Project/physGen/code/v3
DEMO_TAG=${DEMO_TAG:-$(date -u +%Y%m%d-%H%M%S)}

export SAMPLE_IDS=P01
export SEEDS=42
export CFG_MODE=separate
export GUIDE_SCALE=5.0
export PHYS_GUIDANCE_SCALE=2.0
export CONDITIONING_VARIANT=baseline
export STAGE_STRENGTH=0.0
export CONDITIONING_AUDIT=1
export GLOBAL_PROMPT_OVERRIDE="Fixed side view of one black electric kick scooter rolling right toward one upright black metal trash can. The scooter's front wheel visibly hits the trash can, the same scooter immediately rotates down onto its right side, and it finishes fully lying motionless beside the upright trash can."
unset STAGE_PROMPT_OVERRIDE
export RUN_ID=trace-v3-demo-test-A-global-only-${DEMO_TAG}

exec "${V3_ROOT}/inference/infer_trace_writer_m4.sh"
