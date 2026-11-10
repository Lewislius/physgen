#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/no_verifier_benchmarks.sh" q_videophy40 "$@"
