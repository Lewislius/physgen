"""A resumable naive-condition training prefix and the same heldout outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physgen_native.config import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = read_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    experiment = os.environ['DET_EXPERIMENT_ID']
    run = Path(cfg['paths']['runs'])/f'{cfg["run_name"]}_{experiment}'
    stages = [
        ('normal_training_16', [str(ROOT/'train/train.py'), '--config', str(args.config), '--stop-after', '16']),
        ('matched_conditions', [str(ROOT/'tools/evaluate_condition_use.py'), '--checkpoint', str(run/'latest.pt'),
            '--output', str(args.output/'comparison'), '--case-ids', 'source0000042', 'source0000086',
            '--seeds', '42', '314159', '--paired-variants', 'correct', '--full-samples', '2',
            '--sampling-seeds', '42', '--sampling-variants', 'correct'])]
    records = []
    for name, command in stages:
        records.append(dict(name=name, state='running', command=command))
        (args.output/'pipeline.json').write_text(json.dumps(records, indent=2)+'\n')
        result = subprocess.run([sys.executable, '-I', '-B', '-u', *command])
        records[-1].update(state='complete' if result.returncode == 0 else 'failed', returncode=result.returncode)
        (args.output/'pipeline.json').write_text(json.dumps(records, indent=2)+'\n')
        if result.returncode:
            raise SystemExit(result.returncode if result.returncode > 0 else 128-result.returncode)


if __name__ == '__main__': main()
