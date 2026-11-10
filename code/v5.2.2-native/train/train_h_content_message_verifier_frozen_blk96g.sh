#!/usr/bin/env bash
set -euo pipefail
SIMPLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${SIMPLE_ROOT}/train/simple_verifier_blk96g.sh" h_content_message frozen "$@"
