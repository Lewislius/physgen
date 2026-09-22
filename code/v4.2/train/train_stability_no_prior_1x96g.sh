#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../tools/runtime_env.sh"
export CONFIG="${CONFIG:-${V42_ROOT}/configs/stability_v4_fullwidth3_no_prior.yaml}"
export RUN_NAME="${RUN_NAME:-v42_v4_fullwidth3_without_prior_A1_1x96g}"
exec bash "${V42_ROOT}/train/train_stability_1x96g.sh"
