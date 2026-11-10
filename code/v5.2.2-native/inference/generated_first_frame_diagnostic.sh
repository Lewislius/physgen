#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/liuzhirui/Project/physGen/code/v5.2.2-native
exec bash "$ROOT/tools/runtime_python.sh" "$ROOT/inference/run_generated_first_frame_diagnostic.py" \
  --config "$ROOT/inference/generated_first_frame_diagnostic.yaml" "$@"
