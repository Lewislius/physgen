"""Import paths for the self-contained R16Gt T2V implementation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
SCALING = BASE / 'experiments/physical_scaling_q256/snapshots/20261029T140406Z'
sys.path[:0] = [str(ROOT), str(BASE), str(SCALING)]
