#!/usr/bin/env bash
set -euo pipefail
if [[ "${PARSE_ONLY:-0}" != 1 ]]; then printf '[inference] Launcher started.\n'; fi
hardware="${1:-1x96g}"
case "$hardware" in
  1x96g) output_suffix="" ;;
  1xada48g) output_suffix="_1xada48g" ;;
  *) echo "Expected 1x96g or 1xada48g; got $hardware" >&2; exit 2 ;;
esac
source "$(dirname -- "${BASH_SOURCE[0]}")/../tools/runtime_env.sh"
export V42_DIRECT_PYTHON=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
checkpoint="${CHECKPOINT:-${V42_ROOT}/checkpoints/v4_joint_20260918T034836Z_16183b/step0400}"
suite="${SUITE:-demo}"
variant="${VARIANT:-full}"
case "$variant" in full|base|half|writer5_only|writer15_only) ;; *) echo 'Invalid writer intervention VARIANT' >&2; exit 2 ;; esac
run_name="$(basename -- "$(dirname -- "$checkpoint")")"
step_name="$(basename -- "$checkpoint")"
output="${OUTPUT:-${V42_ROOT}/inference_outputs/${run_name}_${step_name}_ema_${suite}${output_suffix}_seed42_r3_${variant}}"
args=(--checkpoint "$checkpoint" --output "$output" --suite "$suite" --variant "$variant")
[[ -z "${CONFIG:-}" ]] || args+=(--config "$CONFIG")
if [[ "$suite" == single ]]; then
  args+=(--mode "${MODE:-i2v}" --prompt "${PROMPT:?Single inference requires PROMPT}" --duration "${DURATION:-5}")
  if [[ "${MODE:-i2v}" == i2v ]]; then args+=(--image "${IMAGE:?I2V requires IMAGE}"); fi
fi
if [[ "${PORTRAIT:-0}" == 1 ]]; then args+=(--portrait); fi
if [[ "${PARSE_ONLY:-0}" != 1 ]]; then
  printf '[inference] %s | checkpoint=%s | suite=%s\n[inference] output=%s\n' "$hardware" "$checkpoint" "$suite" "$output"
fi
v42_python runtime "${V42_ROOT}/inference/infer.py" --phase prepare "${args[@]}"
if [[ "$variant" != base ]]; then
  v42_python teacher "${V42_ROOT}/tools/teacher.py" --phase anchors --output "$output"
fi
v42_python runtime "${V42_ROOT}/inference/infer.py" --phase sample --output "$output"
if [[ "${EVALUATE:-0}" == 1 ]]; then
  v42_python teacher "${V42_ROOT}/tools/teacher.py" --phase generated --output "$output"
  v42_python runtime "${V42_ROOT}/inference/report.py" --output "$output"
fi
