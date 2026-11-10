#!/usr/bin/env bash
set -euo pipefail
DEMO_ROOT_NATIVE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MESSAGE_MODE=h_content_message
export CHECKPOINT="${CHECKPOINT:-${DEMO_ROOT_NATIVE}/checkpoints/v522-h-content-message-verifier-frozen-simple_66356/latest.pt}"
exec bash "${DEMO_ROOT_NATIVE}/inference/message_demo21.sh" "$@"
