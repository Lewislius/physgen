#!/usr/bin/env bash
set -euo pipefail
workspace_infer_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_infer_action="${1:---check}"
case "$workspace_infer_action" in
  --check)
    exec bash "$workspace_infer_dir/../tools/runtime_python.sh" "$workspace_infer_dir/pipeline.py" --variant d-i --check-only ;;
  --local)
    exec bash "$workspace_infer_dir/../tools/runtime_python.sh" "$workspace_infer_dir/pipeline.py" --variant d-i ;;
  --submit)
    bash "$workspace_infer_dir/../tools/runtime_python.sh" "$workspace_infer_dir/pipeline.py" --variant d-i --check-only
    exec det -m http://10.130.130.5:4096 experiment create "$workspace_infer_dir/d_i_amp80g.yaml" "$workspace_infer_dir/../submission_context" ;;
  *) printf '%s\n' 'Usage: --check | --submit | --local' >&2; exit 2 ;;
esac
