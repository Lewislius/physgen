"""Score already completed controls on an idle RTX, reusing normal score caches."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
from physical_content.common import atomic_json, gpu_guard
from pipeline import evaluator_environment, run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--modes', required=True, nargs='+')
    args = parser.parse_args()
    job = Path(__file__).parent/'reports/scoring_jobs'/os.environ['DET_EXPERIMENT_ID']
    gpu_guard(job)
    folders = [args.root/'controls'/mode/'videophy40' for mode in args.modes]
    for folder in folders:
        if len(list((folder/'cases').glob('*/planned.json'))) != 16:
            raise ValueError(f'Only score a completed fixed16: {folder}')
    atomic_json(dict(status='scoring', modes=args.modes), job/'status.json')
    run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B',
               str(bootstrap.BASE/'tools/evaluate_videophy2.py'),
               *[str(p) for p in folders]], job/'scoring.log', evaluator_environment())
    subprocess.run([os.sys.executable, '-I', '-B',
                    str(Path(__file__).parent/'summarize_controls.py'),
                    '--root', str(args.root)], check=True)
    atomic_json(dict(status='complete', modes=args.modes), job/'status.json')


if __name__ == '__main__':
    main()
