#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/liuzhirui/Project/physGen/code/v5.2.2-native
exec bash "$ROOT/tools/runtime_python.sh" "$ROOT/inference/infer_native_wan_official_recipe.py" \
  --config "$ROOT/inference/native_wan_official_recipe.yaml" "$@"
