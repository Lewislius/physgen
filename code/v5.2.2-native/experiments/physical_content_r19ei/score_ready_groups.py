"""Score complete fixed16 groups while generation continues on separate GPUs."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
from physical_content.common import atomic_json, gpu_guard
from pipeline import evaluator_environment, run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folders', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    job = Path(__file__).parent / 'reports/scoring_jobs' / os.environ['DET_EXPERIMENT_ID']
    gpu_guard(job)
    pending = []
    for folder in args.folders:
        manifest = json.loads((folder / 'manifest.json').read_text())
        cases = manifest['cases']
        if len(cases) != 16 or len({case['id'] for case in cases}) != 16:
            raise ValueError(f'Expected a complete fixed16 manifest: {folder}')
        for case in cases:
            directory = folder / 'cases' / case['id']
            if not (directory / 'planned.json').is_file() or not (directory / 'planned.mp4').is_file():
                raise ValueError(f'Generation is incomplete: {directory}')
        if not (folder / 'scores.json').exists():
            pending.append(folder)
    atomic_json(dict(status='scoring', folders=[str(p) for p in pending]), job / 'status.json')
    if pending:
        run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
                   str(bootstrap.BASE / 'tools/evaluate_videophy2.py'),
                   *[str(folder) for folder in pending]],
                  job / 'scoring.log', evaluator_environment())
    atomic_json(dict(status='complete', folders=[str(p) for p in args.folders]), job / 'status.json')


if __name__ == '__main__':
    main()
