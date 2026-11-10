"""Disjoint generation shards; shard zero scores the completed paired pilot."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--index', type=int, required=True)
parser.add_argument('--shards', type=int, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(Path(args.config).read_text())
out = Path(cfg['output'])
if json.loads((out/'preparation.json').read_text())['status'] != 'complete':
    raise RuntimeError('Complete prompt preparation once before launching generation shards')
subprocess.run([sys.executable, '-I', '-B', str(root/'inference/infer_native_physics_prompt_baseline.py'),
    '--config', args.config, '--stage', 'generate', '--shard-index', str(args.index), '--shards', str(args.shards)], check=True)
if args.index == 0:
    expected = [out/arm/'cases'/cid/'planned.complete.json'
                for cid in cfg['case_ids'] for arm in ('original', 'physics_text')]
    deadline = time.monotonic() + 3600
    while not all(p.is_file() for p in expected):
        if time.monotonic() > deadline:
            raise RuntimeError('Other shards incomplete; preserve all finished videos for resume')
        print(json.dumps(dict(kind='waiting_for_shards', completed=sum(p.is_file() for p in expected),
                              total=len(expected))), flush=True)
        time.sleep(30)
    subprocess.run([sys.executable, '-I', '-B', str(root/'inference/run_physics_prompt_evaluation.py'),
                    '--config', args.config], check=True)
    subprocess.run([sys.executable, '-I', '-B', str(root/'inference/infer_native_physics_prompt_baseline.py'),
                    '--config', args.config, '--stage', 'compare'], check=True)
