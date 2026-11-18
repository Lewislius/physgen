"""Stable import paths for the independent NoChunks-v2 implementation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
SCALING = BASE / 'experiments/physical_scaling_q256/snapshots/20261029T140406Z'
FOUNDATION = BASE / 'experiments/physical_variants/snapshots/20261029T064043Z'

sys.path[:0] = [str(ROOT), str(SCALING), str(FOUNDATION), str(BASE)]
