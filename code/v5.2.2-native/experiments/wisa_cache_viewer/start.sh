#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${WISA_VIEWER_PYTHON:-/home/liuzhirui/miniconda3/envs/physvid/bin/python}"
exec "$PYTHON" server.py "$@"
