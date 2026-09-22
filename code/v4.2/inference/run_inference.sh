#!/usr/bin/env bash
set -euo pipefail
hardware="${1:-1x96g}"
case "$hardware" in
  1x96g|1xada48g) ;;
  *) echo "Expected 1x96g or 1xada48g; got $hardware" >&2; exit 2 ;;
esac
source "$(dirname -- "${BASH_SOURCE[0]}")/../tools/runtime_env.sh"
export V42_DIRECT_PYTHON=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
checkpoint="${CHECKPOINT:-latest}"
suite="${SUITE:-demo}"
variant="${VARIANT:-full}"
case "$variant" in full|wan|A) ;; *) echo 'VARIANT must be full, wan, or A' >&2; exit 2 ;; esac
output="${OUTPUT:-${V42_ROOT}/inference_outputs/v42_fullwidth3_write_$(date -u +%Y%m%dT%H%M%SZ)_${hardware}}"
args=(--checkpoint "$checkpoint" --suite "$suite" --variant "$variant"
  --frames 121 --fps 24 --duration 5 --steps 50 --shift 5 --guidance 5 --solver euler
  --negative-prompt '' --seed "${SEED:-42}")
if [[ "$suite" == single ]]; then
  args+=(--output "${output%.mp4}.mp4" --mode "${MODE:-i2v}" --prompt "${PROMPT:?Single inference requires PROMPT}")
  if [[ "${MODE:-i2v}" == i2v ]]; then args+=(--image "${IMAGE:?I2V requires IMAGE}"); fi
else
  args+=(--output-dir "$output" --sample-ids "${SAMPLE_IDS:-all}" --demo-modes "${DEMO_MODES:-i2v}")
  if [[ "${RESUME_INFERENCE:-0}" == 1 ]]; then args+=(--resume); fi
fi
v42_python runtime "${V42_ROOT}/inference/infer_native_p.py" "${args[@]}"
