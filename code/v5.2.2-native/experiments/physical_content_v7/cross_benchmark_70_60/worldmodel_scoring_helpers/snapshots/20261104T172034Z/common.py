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
MODELS = ('wan_lora', 'r1', 'r10a')


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
    return read(ROOT / 'protocol.json')


def owned_cases(model, limit=None):
    cases = [c for c in plan()['cases'] if c['preparation_owner'] == model]
    return cases[:limit] if limit else cases


def shared(case):
    return ROOT / 'shared' / case['id']


def gpu_guard(model, resource_pool=None):
    import torch
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if count else 'NONE'
    total = torch.cuda.get_device_properties(0).total_memory if count else 0
    pool = resource_pool or plan()['models'][model]['resource_pool']
    compact = ''.join(c for c in name.upper() if c.isalnum())
    if pool == 'blk-96g':
        correct = 'RTXPRO6000' in compact or 'RTX6000BLACKWELL' in compact
    elif pool == 'amp-80g':
        correct = ('A100' in compact or 'A800' in compact) and total >= 75*1024**3
    elif pool == 'ada-48g':
        # The installed48GB L40S exposes44.39GiB after device reservation.
        correct = any(v in compact for v in ('L40','RTX6000ADA','RTX4090')) and total >= 44*1024**3
    else:
        correct = False
    accepted = count == 1 and 'H100' not in compact and correct
    info = dict(model=model, pool=pool, name=name, memory_bytes=total,
                visible_devices=count, accepted=accepted,
                experiment_id=os.getenv('DET_EXPERIMENT_ID'), time=time.time())
    save(ROOT / 'allocations' / os.getenv('DET_EXPERIMENT_ID', 'local') / 'gpu.json', info)
    print(json.dumps(info), flush=True)
    if not accepted:
        raise SystemExit(86)
    return info


def case_seed(seed, index):
    # Order/sharding independent; paired five-model draws are bitwise equal on CPU.
    return (int(seed) + int(index)) % (2**63 - 1)


BENCH = Path("/home/liuzhirui/benchmark")
