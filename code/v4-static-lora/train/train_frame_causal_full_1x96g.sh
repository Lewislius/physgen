#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../tools/train_frame_causal.sh" "train_frame_causal_full_1x96g.yaml"
