#!/usr/bin/env bash
set -euo pipefail
scaling_train_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scaling_root="$(cd -- "$scaling_train_dir/.." && pwd)"
scaling_runtime=/home/liuzhirui/Project/physGen/code/v5.2.2-native/tools/runtime_python.sh
scaling_action="${1:---check}"
if [[ $# -gt 0 ]]; then shift; fi
case "$scaling_action" in
  --check) exec bash "$scaling_runtime" "$scaling_root/train.py" --config "$scaling_root/configs/v6_write_q256_s1000.yaml" --check-only "$@" ;;
  --local) exec bash "$scaling_runtime" "$scaling_root/train.py" --config "$scaling_root/configs/v6_write_q256_s1000.yaml" "$@" ;;
  --submit)
    bash "$scaling_runtime" "$scaling_root/train.py" --config "$scaling_root/configs/v6_write_q256_s1000.yaml" --check-only
    exec det -m http://10.130.130.5:4096 experiment create "$scaling_train_dir/v6_write_q256_s1000_blk96g.yaml" "$scaling_root/submission_context" ;;
  *) printf '%s\n' 'Usage: --check | --local [--resume CKPT --run-id RUN] | --submit' >&2; exit 2 ;;
esac
