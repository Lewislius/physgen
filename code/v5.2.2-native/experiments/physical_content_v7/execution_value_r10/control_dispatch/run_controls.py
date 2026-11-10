"""Finish selected already-declared controls after the own400 checkpoint is ready."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/execution_value_r10')
RUN = 'physical_content_67003'
MODES = ['global_train_mean', 'routing_only', 'anchor_only', 'no_c']
DECLARED_MODES = ['reasoning_swapped', 'empty_reasoning', *MODES]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--modes', nargs='+', choices=DECLARED_MODES, default=MODES)
    args = parser.parse_args()
    if len(args.modes) != len(set(args.modes)):
        raise ValueError('Do not run the same declared intervention twice')
    source = Path(__file__).with_name('run_control.py')
    run = ROOT / 'runs' / RUN
    # global_train_mean prepares/reuses its original train-only64-per-arm mean
    # itself; do not reserve a GPU waiting for the primary runner's preparation.
    required = [run / 'step0400.pt']
    status_file = run / 'status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    ready = (all(p.is_file() for p in required) and status.get('step') == 400
             and status.get('status') in ['paused', 'complete'])
    if args.describe:
        print(json.dumps(dict(training_run=RUN, ready=ready, required=list(map(str, required)),
                              cases_per_mode=16, modes=args.modes, added_cases=0, added_seeds=0)))
        for mode in args.modes:
            subprocess.run([sys.executable, '-I', '-B', source, '--mode', mode, '--describe'], check=True)
        return
    if not ready:
        raise RuntimeError('Submit only after the declared400 checkpoint is fully saved')
    results = {}
    job = ROOT / 'control_dispatch/reports/jobs' / os.getenv('DET_EXPERIMENT_ID', 'local')
    for mode in args.modes:
        result = subprocess.run([sys.executable, '-I', '-B', '-u', source, '--mode', mode])
        if result.returncode:
            sys.exit(result.returncode)
        results[mode] = json.loads((job / mode / 'status.json').read_text())
    (job / 'status.json').write_text(json.dumps(dict(status='finished',
        recorded_utc=datetime.now(timezone.utc).isoformat(), training_run=RUN,
        modes=results, additional_cases=0, additional_seeds=0), indent=2)+'\n')


if __name__ == '__main__':
    main()
