"""Shared paths and protocol utilities for the ProPhy-style 600/180 replay."""
from pathlib import Path
import json
import os
import sys
import time

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT = BASE / 'experiments/physical_content_v7/prophy_videophy2_600'
SHARED = BASE / 'experiments/physical_content_v7/physvid_i2v_599/shared'
SCALING = BASE / 'experiments/physical_scaling_q256/snapshots/20261029T140406Z'
FOUNDATION = BASE / 'experiments/physical_variants/snapshots/20261029T064043Z'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def activate(source):
    sys.path[:0] = [str(source), str(SCALING), str(FOUNDATION), str(BASE)]


def plan():
    return read(ROOT / 'protocol.json')


def case_seed(seed, index):
    return (int(seed) + int(index)) % (2**63 - 1)


def aggregate(rows):
    n = len(rows)
    if not n:
        raise ValueError('No scores')
    if any(r[k] not in range(6) for r in rows for k in ('sa', 'pc')):
        raise ValueError('Expected evaluator scores in 0..5')
    sa_count = sum(r['sa'] >= 4 for r in rows)
    pc_count = sum(r['pc'] >= 4 for r in rows)
    joint = sum(r['sa'] >= 4 and r['pc'] >= 4 for r in rows)
    return dict(count=n, sa_mean=sum(r['sa'] for r in rows)/n,
                pc_mean=sum(r['pc'] for r in rows)/n,
                sa_pass_count=sa_count, sa_pass_rate=sa_count/n,
                pc_pass_count=pc_count, pc_pass_rate=pc_count/n,
                joint_pass_count=joint, joint_pass_rate=joint/n)


def physvid_score(raw):
    """Official VideoPhy2 AutoEval parser with its documented zero fallback."""
    text = raw.lower().strip()
    numbers = dict(zip(('zero', 'one', 'two', 'three', 'four', 'five'), range(6)))
    numbers.update({str(i): i for i in range(6)})
    for key, value in numbers.items():
        if key in text:
            return value
    digits = ''.join(c for c in text if c.isdigit())
    return int(digits) if digits and int(digits) in numbers.values() else 0


def gpu_guard(run):
    """This replay is authorized on one ada-48g GPU; H100 is never accepted."""
    import torch
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if count else 'NONE'
    total = torch.cuda.get_device_properties(0).total_memory if count else 0
    compact = ''.join(c for c in name.upper() if c.isalnum())
    accepted = count == 1 and 'H100' not in compact and (
        'L40S' in compact or 'RTX4090' in compact or 'RTXPRO6000' in compact or 'RTX6000BLACKWELL' in compact)
    record = dict(kind='gpu_policy', actual=name, visible_count=count, accepted=accepted,
                  total_memory_bytes=total, resource_pool='ada-48g',
                  experiment_id=os.getenv('DET_EXPERIMENT_ID'), time=time.time())
    save(Path(run) / 'gpu.json', record)
    print(json.dumps(record), flush=True)
    if not accepted:
        raise SystemExit(86)
    return record
