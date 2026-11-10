#!/usr/bin/env bash
set -euo pipefail
DEMO_ROOT_NATIVE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MESSAGE_MODE=q_seed128_message
export CHECKPOINT="${CHECKPOINT:-${DEMO_ROOT_NATIVE}/checkpoints/v522-q-seed128-message-verifier-frozen-simple_66353/latest.pt}"
exec bash "${DEMO_ROOT_NATIVE}/inference/message_demo21.sh" "$@"
