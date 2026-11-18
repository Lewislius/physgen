"""Accelerate the existing R7 final8, then finish its already-declared two control modes."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/native_coordinate_r7')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    commands=[
        [sys.executable,'-I','-B','-u',Path(__file__).with_name('run_primary_tail.py'),
         '--variant','native_coordinate_r7','--run','physical_content_66920',
         '--source',ROOT/'snapshots/20261031T120744Z'],
        [sys.executable,'-I','-B','-u',ROOT/'control_dispatch/snapshots/20261031T125716Z/run_pair.py']]
    for command in commands:
        if args.describe:command+=['--describe']
        result=subprocess.run(list(map(str,command)))
        if result.returncode:return result.returncode
    return 0

if __name__=='__main__':raise SystemExit(main())
