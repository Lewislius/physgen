#!/usr/bin/env python3
"""Independent Q256/1000-step training entrypoint; official bases, fresh modules."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
FOUNDATION_SOURCE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_variants/snapshots/20261029T064043Z')
sys.path.insert(0,str(FOUNDATION_SOURCE))
sys.path.insert(0,str(ROOT))
from physical_scaling.train import main
if __name__=='__main__':main()
