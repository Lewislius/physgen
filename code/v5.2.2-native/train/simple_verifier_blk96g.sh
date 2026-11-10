#!/usr/bin/env bash
set -euo pipefail
SIMPLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
simple_mode="${1:?Specify h_content_message or q_seed128_message}"
simple_adapter="${2:?Specify frozen or lora}"
shift 2
case "$simple_mode/$simple_adapter" in
  h_content_message/frozen|h_content_message/lora|q_seed128_message/frozen|q_seed128_message/lora) ;;
  *) printf 'Unknown message/verifier combination: %s/%s\n' "$simple_mode" "$simple_adapter" >&2; exit 2 ;;
esac
simple_action="${1:---check}"
if [[ $# -gt 0 ]]; then shift; fi
simple_stem="native_${simple_mode}_verifier_${simple_adapter}_simple"
if [[ "$simple_adapter" == frozen ]]; then simple_stem="native_${simple_mode}_no_verifier_simple"; fi
simple_config="${SIMPLE_ROOT}/configs/${simple_stem}_wisa3857.yaml"
simple_job="${SIMPLE_ROOT}/train/${simple_stem}_1x96g.yaml"
simple_initial="${SIMPLE_ROOT}/checkpoints/initialization/generator.pt"
case "$simple_action" in
  --check)
    [[ $# -eq 0 ]] || { printf '%s\n' '--check takes no extra arguments' >&2; exit 2; }
    [[ -r "$simple_initial" ]] || { printf 'Missing initialization: %s\n' "$simple_initial" >&2; exit 1; }
    exec bash "${SIMPLE_ROOT}/tools/runtime_python.sh" "${SIMPLE_ROOT}/train/train.py" --config "$simple_config" --check-only
    ;;
  --submit)
    [[ $# -eq 0 ]] || { printf '%s\n' '--submit takes no extra arguments; edit the explicit job YAML if needed' >&2; exit 2; }
    [[ -r "$simple_initial" ]] || { printf 'Missing initialization: %s\n' "$simple_initial" >&2; exit 1; }
    bash "${SIMPLE_ROOT}/tools/runtime_python.sh" "${SIMPLE_ROOT}/train/train.py" --config "$simple_config" --check-only
    # The established job reads the shared mount; do not upload caches or runs.
    exec det -m "${DET_MASTER:-http://10.130.130.5:4096}" experiment create "$simple_job"
    ;;
  --local)
    # Use only inside an already allocated single-GPU environment.
    simple_init_args=(--initialize-generator "$simple_initial")
    for simple_arg in "$@"; do
      case "$simple_arg" in --resume|--resume=*) simple_init_args=(); break ;; esac
    done
    exec bash "${SIMPLE_ROOT}/tools/runtime_python.sh" "${SIMPLE_ROOT}/train/train.py" \
      --config "$simple_config" "${simple_init_args[@]}" "$@"
    ;;
  *)
    printf '%s\n' 'Usage: <variant launcher> [--check | --submit | --local [train.py options]]' >&2
    exit 2
    ;;
esac
