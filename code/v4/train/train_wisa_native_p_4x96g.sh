#!/usr/bin/env bash
set -e
TRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${TRAIN_ROOT}/run_training.sh" 4x96g
