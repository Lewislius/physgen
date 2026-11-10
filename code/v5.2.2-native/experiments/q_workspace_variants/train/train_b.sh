#!/usr/bin/env bash
set -euo pipefail
variant_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${variant_script_dir}/launch_variant.sh" b "$@"
