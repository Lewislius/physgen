#!/usr/bin/env bash
set -eo pipefail
INFERENCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${INFERENCE_ROOT}/run_inference.sh" 1x96g "$@"
