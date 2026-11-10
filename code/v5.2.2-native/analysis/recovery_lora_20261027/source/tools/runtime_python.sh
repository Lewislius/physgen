#!/usr/bin/env bash
set -euo pipefail
# Reuse the exact established v5.2 launcher after Determined bootstrap.
# The shared file is read-only; no environment installation or reconfiguration.
source /home/liuzhirui/Project/physGen/code/v5.2/tools/runtime_env.sh
v52_python "$@"
