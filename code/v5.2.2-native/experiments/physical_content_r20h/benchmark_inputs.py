"""Fixed VideoPhy2 60 prompts and persisted paired seed42 noise.

I2V reuses the historical first-image/T5/noise protocol.  T2V uses the same
captions and noise but deliberately consumes no first-image tensor.
"""
from pathlib import Path
import json
import sys

import bootstrap

BASE = bootstrap.BASE
ROOT = BASE / 'experiments/physical_content_v7'
PINNED = BASE / 'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
sys.path.insert(0, str(PINNED))
from protocol import load_inputs, load_noise, verify_video  # noqa: E402,F401


def reference():
    return json.loads((ROOT / 'videophy60/protocol.json').read_text())


def _suite(case):
    return 'videophy40' if int(case['id'][2:]) <= 40 else 'videophy60'


def case_input(case, task):
    suite = _suite(case)
    if task == 't2v':
        import torch
        folder = Path(case['prepared_inputs'])
        stored = torch.load(folder / 'conditions.pt', map_location='cpu',
                            weights_only=True, mmap=True)
        sample = {key: stored[key].float() for key in ('text', 'null_text', 'times')}
        item = dict(id=f'{suite}_{case["id"]}', case=case['id'],
                    caption=case['caption'], duration=float(sample['times'][-1]),
                    image=None)
        return sample, item
    if task != 'i2v':
        raise ValueError(task)
    inputs, canvas = load_inputs(case)
    item = dict(id=f'{suite}_{case["id"]}', case=case['id'],
                caption=case['caption'], image=str(canvas.resolve()),
                duration=float(inputs['times'][-1]))
    return inputs, item
