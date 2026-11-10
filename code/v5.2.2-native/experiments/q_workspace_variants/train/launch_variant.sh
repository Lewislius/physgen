#!/usr/bin/env bash
set -euo pipefail
variant_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
variant_name="${1:?Specify b, c, d-n, d-s or d-i}"
shift
case "$variant_name" in b|c|d-n|d-s|d-i) ;; *) printf 'Unknown variant: %s\n' "$variant_name" >&2; exit 2 ;; esac
variant_file="${variant_name//-/_}"
variant_config="${variant_root}/configs/${variant_file}.yaml"
variant_job="${variant_root}/train/${variant_file}_1x96g.yaml"
if [[ "$variant_name" == b ]]; then variant_job="${variant_root}/train/b_1x80g.yaml"; fi
variant_initial="/home/liuzhirui/Project/physGen/code/v5.2.2-native/checkpoints/initialization/generator.pt"
variant_action="${1:---check}"
if [[ $# -gt 0 ]]; then shift; fi
case "$variant_action" in
  --check)
    [[ $# -eq 0 ]] || { printf '%s\n' '--check takes no additional arguments' >&2; exit 2; }
    exec bash "${variant_root}/tools/runtime_python.sh" "${variant_root}/tools/check_config.py" \
      --config "$variant_config" --initialize-generator "$variant_initial"
    ;;
  --submit)
    [[ $# -eq 0 ]] || { printf '%s\n' '--submit takes no additional arguments; resume uses the saved config' >&2; exit 2; }
    bash "${variant_root}/tools/runtime_python.sh" "${variant_root}/tools/check_config.py" \
      --config "$variant_config" --initialize-generator "$variant_initial"
    # An explicit empty context avoids uploading model/cache/output directories.
    exec det -m "${DET_MASTER:-http://10.130.130.5:4096}" experiment create \
      "$variant_job" "${variant_root}/submission_context"
    ;;
  --local)
    for variant_arg in "$@"; do
      case "$variant_arg" in
        --config|--config=*|--resume|--resume=*|--initialize-*)
          printf '%s\n' 'Use --resume RUN_DIR for continuation; this launcher supplies its own config and initializer.' >&2
          exit 2 ;;
      esac
    done
    exec bash "${variant_root}/tools/runtime_python.sh" "${variant_root}/train/train.py" \
      --config "$variant_config" --initialize-generator "$variant_initial" "$@"
    ;;
  --resume)
    variant_resume="${1:?Usage: --resume RUN_DIR [train.py options]}"
    shift
    exec bash "${variant_root}/tools/runtime_python.sh" "${variant_root}/tools/resume_variant.py" \
      --variant "$variant_name" --run "$variant_resume" "$@"
    ;;
  *)
    printf '%s\n' 'Usage: train_<variant>.sh [--check | --submit | --local [train.py options] | --resume RUN_DIR [train.py options]]' >&2
    exit 2
    ;;
esac
