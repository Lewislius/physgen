#!/usr/bin/env bash
set -euo pipefail

# Same 41 conditions as the original V4.2 demo: reference I2V + 20 I2V/T2V pairs.
# CFG uses an empty text branch, matching this checkpoint's text-dropout training.
export CHECKPOINT="${CHECKPOINT:-/home/liuzhirui/Project/physGen/code/v4.3/checkpoints/v43_native_joint_flow_oracle_20260918T143237Z-e47bdd74/checkpoint-final}"
export SUITE="${SUITE:-both}"
export DEMO_MODES="${DEMO_MODES:-both}"
export VARIANT="${VARIANT:-full}"
export SOLVER="${SOLVER:-unipc}"

exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_inference.sh" 1x96g "$@"
