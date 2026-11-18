"""Fixed VideoPhy2 60 prompts, native T5 text, and persisted seed42 noise.

T2V consumes no first-frame tensor, no canvas, and no image path.  The native
T5 text and the persisted noise are the same matched inputs used by the
existing T2V branch, so generations are comparable across T2V candidates.
"""
from pathlib import Path
import json
import sys

import bootstrap

BASE = bootstrap.BASE
ROOT = BASE / 'experiments/physical_content_v7'
PINNED = BASE / 'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
sys.path.insert(0, str(PINNED))
from protocol import load_noise, verify_video  # noqa: E402,F401

T2V_SCOPE = ('Fixed60 historical VideoPhy2 captions; R16Gt T2V uses no first '
             'image; seed42. Not the official 590-caption benchmark.')


def reference(cfg):
    return json.loads(Path(cfg['paths']['eval_protocol']).read_text())


def case_input(case, native_root):
    import torch
    # T2V reads its own copied native T5/time tensors, never the I2V prepared
    # folder or any first-frame file.  The root comes from the config so an
    # immutable code snapshot still reads the declared working-directory copy.
    folder = Path(native_root) / case['id']
    stored = torch.load(folder / 'conditions.pt', map_location='cpu',
                        weights_only=True, mmap=True)
    sample = {key: stored[key].float() for key in ('text', 'null_text', 'times')}
    text, null = sample['text'], sample['null_text']
    if (text.ndim != 3 or text.shape[0] != 1 or not 0 < text.shape[1] <= 512
            or text.shape[-1] != 4096
            or null.ndim != 3 or null.shape[0] != 1 or null.shape[-1] != 4096):
        raise ValueError(f'{case["id"]}: wrong native T5 dimensions')
    if tuple(sample['times'].shape) != (21,):
        raise ValueError(f'{case["id"]}: wrong time grid')
    if not torch.allclose(sample['times'],
                          torch.arange(21).float() / 6, atol=1e-5, rtol=0):
        raise ValueError(f'{case["id"]}: PTS must match 24 fps')
    if not all(bool(torch.isfinite(value).all()) for value in sample.values()):
        raise ValueError(f'{case["id"]}: nonfinite native input')
    info = json.loads((folder / 'input.json').read_text())
    if info['prompt'] != case['caption']:
        raise ValueError(f'{case["id"]}: prepared prompt differs')
    # Deliberately do not expose the stored first latent or canvas: sample is
    # built only from the three native T5/time tensors above.
    suite = 'videophy40' if int(case['id'][2:]) <= 40 else 'videophy60'
    item = dict(id=f'{suite}_{case["id"]}', case=case['id'],
                caption=case['caption'], duration=float(sample['times'][-1]))
    return sample, item
