#!/usr/bin/env bash
set -euo pipefail
v5_timer_root=/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_v5_scaling_q256
nohup /usr/bin/python3 -u "$v5_timer_root/tools/rtx_submission_timer.py" \
  > "$v5_timer_root/reports/rtx_submission_timer.log" 2>&1 < /dev/null &
v5_timer_pid=$!
printf 'Started V5 RTX6000 submission timer, PID %s\n' "$v5_timer_pid"
