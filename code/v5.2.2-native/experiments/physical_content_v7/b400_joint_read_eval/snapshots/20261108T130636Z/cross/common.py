"""Shared protocol utilities; no model imports or network access at import time."""
from pathlib import Path
import json
import os
import sys
import time

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT = BASE / 'experiments/physical_content_v7/cross_benchmark_70_60'
SCALING = BASE / 'experiments/physical_scaling_q256/snapshots/20261029T140406Z'
FOUNDATION = BASE / 'experiments/physical_variants/snapshots/20261029T064043Z'
R1_SOURCE = BASE / 'experiments/physical_content_v7/snapshots/20261030T141432Z'
B_SOURCE = BASE / 'experiments/physical_content_v7/grounded_execution_r10b/snapshots/20261101T054248Z'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def save_tensor(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def activate(source=R1_SOURCE):
    sys.path[:0] = [str(source), str(SCALING), str(FOUNDATION), str(BASE)]


def plan():
    frozen = Path(__file__).resolve().parent / 'protocol.json'
    return read(frozen if frozen.exists() else ROOT / 'protocol.json')


def owned_cases(model, limit=None):
    cases = [c for c in plan()['cases'] if c['preparation_owner'] == model]
    return cases[:limit] if limit else cases


def shared(case):
    return ROOT / 'shared' / case['id']


def gpu_guard(model, resource_pool=None):
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    from gpu_runtime import gpu_guard as check
    return check(Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/b400_joint_read_eval')/'reports/jobs'/os.getenv('DET_EXPERIMENT_ID','local'))


def case_seed(seed, index):
    # Order/sharding independent; paired five-model draws are bitwise equal on CPU.
    return (int(seed) + int(index)) % (2**63 - 1)


BENCH = Path("/home/liuzhirui/benchmark")
