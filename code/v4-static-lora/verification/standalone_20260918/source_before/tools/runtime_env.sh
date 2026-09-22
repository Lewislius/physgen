#!/usr/bin/env bash
# Reuse the formal experiment's isolated, installed Python environment.
LORA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LORA_ROOT
source "${LORA_ROOT}/../v4.2/tools/runtime_env.sh"
lora_python() { v42_python runtime "$@"; }
