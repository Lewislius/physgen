"""Shared protocol utilities; no model imports or network access at import time."""
from pathlib import Path
import json
import os
import sys
import time

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT = BASE / 'experiments/physical_content_v7/physvid_i2v_599'
SCALING = BASE / 'experiments/physical_scaling_q256/snapshots/20261029T140406Z'
FOUNDATION = BASE / 'experiments/physical_variants/snapshots/20261029T064043Z'
R1_SOURCE = BASE / 'experiments/physical_content_v7/snapshots/20261030T141432Z'
B_SOURCE = BASE / 'experiments/physical_content_v7/grounded_execution_r10b/snapshots/20261101T054248Z'
MODELS = ('wan_lora', 'r1', 'r9', 'r10a', 'r10b')


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


def gpu_guard(model):
    import torch
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if count else 'NONE'
    total = torch.cuda.get_device_properties(0).total_memory if count else 0
    pool = plan()['models'][model]['resource_pool']
    if pool == 'blk-96g':
        compact = ''.join(c for c in name.upper() if c.isalnum())
        accepted = count == 1 and 'H100' not in compact and (
            'RTXPRO6000' in compact or 'RTX6000BLACKWELL' in compact)
    else:
        accepted = count == 1 and 'A100' in name.upper() and total >= 75 * 1024**3
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


def aggregate(rows):
    n = len(rows)
    if not n:
        raise ValueError('No scores')
    if any(r[k] not in range(6) for r in rows for k in ('sa', 'pc')):
        raise ValueError('Expected original evaluator integers 0..5')
    counts = {k + '_pass_count': sum(r[k] >= 4 for r in rows) for k in ('sa', 'pc')}
    joint = sum(r['sa'] >= 4 and r['pc'] >= 4 for r in rows)
    return dict(count=n, sa_mean=sum(r['sa'] for r in rows)/n,
                pc_mean=sum(r['pc'] for r in rows)/n, **counts,
                sa_pass_rate=counts['sa_pass_count']/n,
                pc_pass_rate=counts['pc_pass_count']/n,
                joint_pass_count=joint, joint_pass_rate=joint/n)


def physvid_score(raw):
    """Exact author's substring parser, including its zero fallback."""
    text = raw.lower().strip()
    numbers = dict(zip(('zero','one','two','three','four','five'), range(6)))
    numbers.update({str(i): i for i in range(6)})
    for key, value in numbers.items():
        if key in text:
            return value
    digits = ''.join(c for c in text if c.isdigit())
    return int(digits) if digits and int(digits) in numbers.values() else 0
