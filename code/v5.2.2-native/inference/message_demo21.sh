#!/usr/bin/env bash
set -euo pipefail
DEMO_ROOT_NATIVE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
demo_action="${1:---local}"
case "$demo_action" in --check|--submit|--local) shift "$(( $# > 0 ? 1 : 0 ))" ;; *) demo_action=--local ;; esac
case "${MESSAGE_MODE:?Use an H or Q launcher}" in
  h_content_message) demo_stem=infer_h_content_frozen_demo21 ;;
  q_seed128_message) demo_stem=infer_q_seed128_frozen_demo21 ;;
  *) printf 'Unsupported mode: %s\n' "$MESSAGE_MODE" >&2; exit 2 ;;
esac
case "${DEMO_RESOURCE_POOL:-blk-96g}" in
  blk-96g) demo_job="${demo_stem}_1x96g.yaml" ;;
  amp-48g) demo_job="${demo_stem}_1x48g.yaml" ;;
  *) printf 'Unsupported resource pool: %s\n' "$DEMO_RESOURCE_POOL" >&2; exit 2 ;;
esac
if [[ "$demo_action" == --submit ]]; then
  [[ $# -eq 0 ]] || { printf '%s\n' '--submit uses the explicit job YAML; put overrides there.' >&2; exit 2; }
  exec det -m "${DET_MASTER:-http://10.130.130.5:4096}" experiment create "${DEMO_ROOT_NATIVE}/inference/${demo_job}"
fi
demo_args=(--checkpoint "${CHECKPOINT:?}" --mode "$MESSAGE_MODE"
  --prepared-root "${PREPARED_ROOT:-${DEMO_ROOT_NATIVE}/../v5.2/inference_outputs/1d-v52-lora-demo21_20261010}"
  --sample-ids "${SAMPLE_IDS:-all}" --seed "${SEED:-42}" --fps "${FPS:-24}"
  --steps "${SAMPLING_STEPS:-50}" --shift "${SHIFT:-5}" --guidance "${GUIDANCE:-5}")
[[ -z "${OUTPUT:-}" ]] || demo_args+=(--output "$OUTPUT")
if [[ "$demo_action" == --check || "${CHECK_ONLY:-0}" == 1 ]]; then demo_args+=(--check-only); fi
if [[ "${RESUME:-0}" == 1 ]] || { [[ "${RESUME:-0}" == auto && -n "${OUTPUT:-}" ]] &&
  [[ -f "$OUTPUT/checkpoint.pt" && -f "$OUTPUT/manifest.json" ]]; }; then demo_args+=(--resume); fi
[[ -z "${EXCLUDE_GPU_UUID:-}" ]] || demo_args+=(--exclude-gpu-uuid "$EXCLUDE_GPU_UUID")
if [[ "${CPU_OFFLOAD:-0}" == 1 ]]; then demo_args+=(--cpu-offload); fi
exec bash "${DEMO_ROOT_NATIVE}/tools/runtime_python.sh" "${DEMO_ROOT_NATIVE}/inference/infer_message_demo21.py" "${demo_args[@]}" "$@"
