#!/bin/bash
set -euo pipefail
cd /home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/r20ft_wisa20k
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
SHARD="${SHARD_INDEX:-0}"
GPUS="${GPUS:-4}"
exec /home/liuzhirui/miniconda3/envs/moviestory/bin/python -I -B -u run_shard.py \
  --config /home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/r20ft_wisa20k/configs/r20ft_wisa20k.yaml \
  --shard "${SHARD}" --gpus "${GPUS}" --index-workers 16
