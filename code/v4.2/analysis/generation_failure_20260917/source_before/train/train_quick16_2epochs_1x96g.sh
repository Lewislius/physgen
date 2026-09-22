#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../tools/runtime_env.sh"
config="${CONFIG:-${V42_ROOT}/configs/quick16.yaml}"
runner="${V42_ROOT}/quick16/run.py"
for phase in index vae text teacher finalize train infer; do
  role=runtime
  [[ "$phase" != teacher ]] || role=teacher
  v42_python "$role" "$runner" --config "$config" --phase "$phase"
done
