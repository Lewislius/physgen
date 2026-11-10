#!/usr/bin/env bash
set -euo pipefail
physical_infer_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
physical_runtime=/home/liuzhirui/Project/physGen/code/v5.2.2-native/tools/runtime_python.sh
case "${1:---check}" in
  --check) exec bash "$physical_runtime" "$physical_infer_dir/pipeline.py" --variant v5_ce --check-only ;;
  --local) exec bash "$physical_runtime" "$physical_infer_dir/pipeline.py" --variant v5_ce ;;
  --submit)
    bash "$physical_runtime" "$physical_infer_dir/pipeline.py" --variant v5_ce --check-only
    exec det -m http://10.130.130.5:4096 experiment create "$physical_infer_dir/v5_ce_blk96g.yaml" /home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_variants/submission_context ;;
  *) printf '%s\n' 'Usage: --check | --local | --submit' >&2; exit 2 ;;
esac
