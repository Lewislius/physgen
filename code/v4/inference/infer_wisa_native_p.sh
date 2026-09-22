#!/usr/bin/env bash
# Backward-compatible default: the single-GPU amp-48g profile.
set -eo pipefail
INFERENCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${INFERENCE_ROOT}/infer_wisa_native_p_1x48g.sh" "$@"
