#!/usr/bin/env bash
set -euo pipefail
hardware="${1:-1x96g}"
if [[ $# -gt 0 ]]; then shift; fi
case "$hardware" in
  1x96g|1xada48g) ;;
  *) echo "Expected 1x96g or 1xada48g; got $hardware" >&2; exit 2 ;;
esac
source "$(dirname -- "${BASH_SOURCE[0]}")/../tools/runtime_env.sh"
export V43_DIRECT_PYTHON=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
checkpoint="${CHECKPOINT:-latest}"
suite="${SUITE:-demo}"
case "$suite" in single|demo|both) ;; *) echo 'SUITE must be single, demo, or both' >&2; exit 2 ;; esac
variant="${VARIANT:-full}"
case "$variant" in full|wan|A) ;; *) echo 'VARIANT must be full, wan, or A' >&2; exit 2 ;; esac
output="${OUTPUT:-${V43_ROOT}/inference_outputs/v43_flow_oracle_$(date -u +%Y%m%dT%H%M%SZ)_${hardware}}"
args=(--checkpoint "$checkpoint" --suite "$suite" --variant "$variant"
  --frames "${FRAMES:-121}" --fps "${FPS:-24}" --duration "${DURATION:-5}"
  --steps "${SAMPLING_STEPS:-50}" --shift "${SHIFT:-5}" --guidance "${GUIDANCE:-5}"
  --solver "${SOLVER:-euler}" --negative-prompt '' --seed "${SEED:-42}"
  --device "${DEVICE:-0}" --width "${WIDTH:-512}" --height "${HEIGHT:-288}")
if [[ "$suite" == single ]]; then
  args+=(--output "${output%.mp4}.mp4" --mode "${MODE:-i2v}" --prompt "${PROMPT:?Single inference requires PROMPT}")
  if [[ "${MODE:-i2v}" == i2v ]]; then args+=(--image "${IMAGE:?I2V requires IMAGE}"); fi
else
  args+=(--output-dir "$output" --sample-ids "${SAMPLE_IDS:-all}" --demo-modes "${DEMO_MODES:-i2v}"
    --demo-root "${DEMO_ROOT:-${V43_ROOT}/../v1/demo}")
  if [[ "$suite" == both ]]; then
    if [[ -n "${CASES:-}" ]]; then
      args+=(--cases "$CASES")
    else
      args+=(--mode "${MODE:-i2v}"
        --image "${IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref5.jpg}"
        --prompt "${PROMPT:-The woman smiles and waves at the camera.}")
    fi
  fi
  if [[ "${RESUME_INFERENCE:-0}" == 1 ]]; then args+=(--resume); fi
fi
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then args+=(--check-only); fi
if [[ -n "${REPORT:-}" ]]; then args+=(--report "$REPORT"); fi
# Scalar CLI arguments may override environment defaults.
args+=("$@")
v43_python runtime "${V43_ROOT}/inference/infer.py" "${args[@]}"
