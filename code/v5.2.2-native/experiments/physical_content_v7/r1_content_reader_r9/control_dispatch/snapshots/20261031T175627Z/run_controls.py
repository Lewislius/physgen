"""Finish the two already-declared late controls after the trained checkpoint is ready."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/r1_content_reader_r9')
RUN = 'physical_content_66968'
MODES = ['anchor_only', 'no_c']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).with_name('run_control.py')
    run = ROOT / 'runs' / RUN
    # These two interventions do not use a training-derived mean.
    required = [run / 'step0400.pt']
    status_file = run / 'status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    ready = (all(p.is_file() for p in required) and status.get('step') == 400
             and status.get('status') in ['paused', 'complete'])
    if args.describe:
        print(json.dumps(dict(training_run=RUN, ready=ready, required=list(map(str, required)),
                              cases_per_mode=16, modes=MODES, added_cases=0, added_seeds=0)))
        for mode in MODES:
            subprocess.run([sys.executable, '-I', '-B', source, '--mode', mode, '--describe'], check=True)
        return
    if not ready:
        raise RuntimeError('Submit only after the declared400 checkpoint is fully saved; these controls require no mean preparation')
    results = {}
    job = ROOT / 'control_dispatch/reports/jobs' / os.getenv('DET_EXPERIMENT_ID', 'local')
    for mode in MODES:
        result = subprocess.run([sys.executable, '-I', '-B', '-u', source, '--mode', mode])
        if result.returncode:
            sys.exit(result.returncode)
        results[mode] = json.loads((job / mode / 'status.json').read_text())
    (job / 'status.json').write_text(json.dumps(dict(status='finished',
        recorded_utc=datetime.now(timezone.utc).isoformat(), training_run=RUN,
        modes=results, additional_cases=0, additional_seeds=0), indent=2)+'\n')


if __name__ == '__main__':
    main()
