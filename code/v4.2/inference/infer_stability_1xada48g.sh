#!/usr/bin/env bash
set -euo pipefail
export CHECKPOINT="${CHECKPOINT:-/home/liuzhirui/Project/physGen/code/v4.2/checkpoints/v4_joint_20260918T034836Z_16183b/step0200}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_inference.sh" 1xada48g
