#!/usr/bin/env bash
set -euo pipefail
audit_dir=/home/liuzhirui/Project/physGen/code/v5.2.2-native/analysis/physical_scaling_audit_20261030
bash /home/liuzhirui/Project/physGen/code/v5.2.2-native/tools/runtime_python.sh "$audit_dir/gpu_ablation.py"
