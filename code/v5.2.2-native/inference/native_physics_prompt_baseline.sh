#!/usr/bin/env bash
set -euo pipefail
BASELINE_ROOT=/home/liuzhirui/Project/physGen/code/v5.2.2-native
BASELINE_CONFIG="${1:-${BASELINE_ROOT}/inference/native_physics_prompt_baseline.yaml}"
cd "$BASELINE_ROOT"
bash tools/runtime_python.sh inference/infer_native_physics_prompt_baseline.py --config "$BASELINE_CONFIG" --check-only
bash tools/runtime_python.sh inference/infer_native_physics_prompt_baseline.py --config "$BASELINE_CONFIG" --stage prepare
bash tools/runtime_python.sh inference/infer_native_physics_prompt_baseline.py --config "$BASELINE_CONFIG" --stage generate
# Separate interpreter/process: release all generation models before AutoEval.
bash tools/runtime_python.sh inference/run_physics_prompt_evaluation.py --config "$BASELINE_CONFIG"
bash tools/runtime_python.sh inference/infer_native_physics_prompt_baseline.py --config "$BASELINE_CONFIG" --stage compare
