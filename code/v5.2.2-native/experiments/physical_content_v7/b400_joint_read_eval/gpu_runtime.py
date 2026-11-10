"""Guard only the already allocated single Ada48/RTX6000 slot; never H100."""
from pathlib import Path
import json
import os


def gpu_guard(folder):
    import torch
    count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0) if count else None
    actual = props.name if props else 'NONE'
    compact = ''.join(c for c in actual.upper() if c.isalnum())
    pool = os.environ.get('B400_EVAL_POOL', 'ada-48g')
    memory = props.total_memory if props else 0
    if pool == 'ada-48g':
        correct = any(s in compact for s in ('L40', 'RTX6000ADA', 'RTX4090')) and memory >= 44 * 1024**3
    elif pool == 'blk-96g':
        correct = 'RTXPRO6000' in compact or 'RTX6000BLACKWELL' in compact
    else:
        correct = False
    accepted = count == 1 and 'H100' not in compact and correct
    record = dict(actual=actual, visible_count=count, memory_bytes=memory,
                  resource_pool=pool, accepted=accepted, experiment_id=os.getenv('DET_EXPERIMENT_ID'))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'gpu.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    if not accepted:
        raise SystemExit(86)
    return record
