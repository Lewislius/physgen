#!/bin/bash
set -euo pipefail
cd /home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/r20ft_wisa20k
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
exec /home/liuzhirui/miniconda3/envs/qwen3vl/bin/python -I -B -u pilot_expand.py \
  --config /home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/r20ft_wisa20k/configs/r20ft_wisa20k.yaml
