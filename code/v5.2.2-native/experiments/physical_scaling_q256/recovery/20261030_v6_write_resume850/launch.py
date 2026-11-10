"""Resume the exact scaled V6-WRITE checkpoint and retain process evidence."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_recorded import run

PROJECT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
OUTPUT = PROJECT / 'experiments/physical_scaling_q256/runs'
SOURCE = OUTPUT / 'v6_write_66674/latest.pt'


def main():
    experiment = os.environ['DET_EXPERIMENT_ID']
    name = f'v6_write_resume850_{experiment}'
    target = OUTPUT / name
    checkpoint = target / 'latest.pt'
    checkpoint = checkpoint if checkpoint.is_file() else SOURCE
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    command = ['bash', str(PROJECT/'tools/runtime_python.sh'), str(ROOT/'train.py'),
               '--config', str(ROOT/'configs/v6_write_q256_s1000.yaml'),
               '--resume', str(checkpoint), '--run-id', name, '--checkpoint-every', '10']
    print(json.dumps(dict(kind='recovery_launch',source_experiment=66674,
                         checkpoint=str(checkpoint),run=str(target))),flush=True)
    return run(command, target/'process.resources.jsonl')


if __name__ == '__main__':
    sys.exit(main())
