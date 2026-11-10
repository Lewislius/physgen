#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' '{"kind":"startup","stage":"shell_entry"}'
exec bash /home/liuzhirui/Project/physGen/code/v5.2.2-native/tools/runtime_python.sh /home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/state_response_fusion_r15e/diagnostic_snapshots/20261106T090919Z_rtx_recovery/run_readonly_probe.py
