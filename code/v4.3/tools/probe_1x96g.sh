#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime_env.sh"
args=(--config "${CONFIG:-${V43_ROOT}/configs/flow_oracle.yaml}"
      --sample-index "${SAMPLE_INDEX:-0}" --report "${REPORT:-${V43_ROOT}/analysis/checks/real_wan_probe.json}")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
v43_python runtime "${V43_ROOT}/tools/probe_flow_oracle.py" "${args[@]}" "$@"
