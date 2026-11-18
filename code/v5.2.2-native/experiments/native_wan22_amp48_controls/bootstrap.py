"""Import paths for the official Wan2.2-TI2V-5B T2V baseline."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
SCALING = BASE / 'experiments/physical_scaling_q256/snapshots/20261029T140406Z'
R15BT = BASE / 'experiments/physical_content_v7/r15bt'
sys.path[:0] = [str(ROOT), str(BASE), str(SCALING), str(R15BT)]
