"""Stable import paths for the R20FT adaptation to Wan2.1-T2V-1.3B.

The installed ``wan`` package in the shared training image is the Wan2.2
package.  Importing it here would silently leave Wan2.2 operators (especially
``rope_apply``/``flash_attention``) inside the Wan2.1 candidate.  Therefore the
Wan2.1 source namespace is installed *before* any ``physical_content`` import.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
SCALING = BASE / 'experiments/physical_scaling_q256/snapshots/20261029T140406Z'
FOUNDATION = BASE / 'experiments/physical_variants/snapshots/20261029T064043Z'
WAN21_CODE = Path(os.getenv('WAN_CODE', '/home/liuzhirui/model/Wan2.1'))

sys.path[:0] = [str(ROOT), str(SCALING), str(FOUNDATION), str(BASE)]

# Must precede every ``from physical_content...`` import.  The local module
# only installs a ``wan`` namespace with the Wan2.1 source path; it does not
# execute Wan's optional speech modules.
from physical_content.wan21_runtime import install_wan21_namespace  # noqa: E402

install_wan21_namespace(WAN21_CODE)
