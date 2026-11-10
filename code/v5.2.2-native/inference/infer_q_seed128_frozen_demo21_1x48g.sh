#!/usr/bin/env bash
set -euo pipefail
DEMO_ROOT_NATIVE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DEMO_RESOURCE_POOL=amp-48g
export CPU_OFFLOAD="${CPU_OFFLOAD:-1}"
exec bash "${DEMO_ROOT_NATIVE}/inference/infer_q_seed128_frozen_demo21_1x96g.sh" "$@"
