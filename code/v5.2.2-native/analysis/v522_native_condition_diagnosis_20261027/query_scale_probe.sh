#!/usr/bin/env bash
set -euo pipefail
exec bash /home/liuzhirui/Project/physGen/code/v5.2.2-native/tools/runtime_python.sh \
  /home/liuzhirui/Project/physGen/code/v5.2.2-native/analysis/v522_native_condition_diagnosis_20261027/query_scale_probe.py "$@"
