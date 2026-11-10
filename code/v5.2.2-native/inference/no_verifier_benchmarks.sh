#!/usr/bin/env bash
set -euo pipefail
native_infer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
native_infer_job="${1:?Provide h_demo21, q_demo21, h_videophy40, or q_videophy40}"
shift
native_infer_action="${1:---local}"
case "$native_infer_action" in
  --check|--submit|--local) if [[ $# -gt 0 ]]; then shift; fi ;;
  *) native_infer_action=--local ;;
esac
if [[ "$native_infer_action" == --submit ]]; then
  [[ $# -eq 0 ]] || { printf '%s\n' '--submit uses the checked job YAML.' >&2; exit 2; }
  exec /usr/bin/python3 "${native_infer_root}/inference/submit_no_verifier_benchmarks.py" --jobs "$native_infer_job" --background
fi
native_infer_args=(--job "$native_infer_job")
if [[ "$native_infer_action" == --check ]]; then native_infer_args+=(--check-only); fi
exec bash "${native_infer_root}/tools/runtime_python.sh" \
  "${native_infer_root}/inference/infer_no_verifier_benchmarks.py" "${native_infer_args[@]}" "$@"

