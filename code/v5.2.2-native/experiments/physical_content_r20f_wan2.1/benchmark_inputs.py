"""Fixed VideoPhy2 60 prompts and Wan2.1-shaped seed42 noise.

The fixed prompts, captions and positive/null UMT5 text tensors are shared with
the project protocol.  The Wan2.2 persisted noise is intentionally not reused:
Wan2.1 uses a 16-channel, 4x8x8 VAE and therefore a different noise tensor.
The Wan2.1 noise is drawn exactly as in the native Wan2.1 baseline, one
generation seed per case.
"""
from pathlib import Path
import json
import sys

import bootstrap

BASE = bootstrap.BASE
ROOT = BASE / 'experiments/physical_content_v7'
PINNED = BASE / 'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
sys.path.insert(0, str(PINNED))
from protocol import verify_video  # noqa: E402,F401


def reference():
    return json.loads((ROOT / 'videophy60/protocol.json').read_text())


def _suite(case):
    return 'videophy40' if int(case['id'][2:]) <= 40 else 'videophy60'


def case_input(case, task):
    if task != 't2v':
        raise ValueError('R20FT-Wan2.1 is text-to-video only')
    import torch
    folder = Path(case['prepared_inputs'])
    stored = torch.load(folder / 'conditions.pt', map_location='cpu',
                        weights_only=True, mmap=True)
    sample = {key: stored[key].float() for key in ('text', 'null_text', 'times')}
    if tuple(sample['text'].shape[-1:]) != (4096,):
        raise ValueError('The fixed protocol must provide 4096-wide UMT5 features')
    if sample['times'].numel() != 21:
        raise ValueError('Wan2.1 81-frame/4x temporal compression needs 21 latent times')
    item = dict(id=f'{_suite(case)}_{case["id"]}', case=case['id'],
                caption=case['caption'], duration=float(sample['times'][-1]),
                image=None)
    return sample, item


def wan21_noise(case, cfg):
    """Draw the Wan2.1 latent noise with the case's fixed generation seed.

    The native Wan2.1 baseline uses this exact shape [16, 21, 60, 104] and
    generator seed.  A leading singleton batch dimension is added for the
    R20FT velocity interface without changing the random draw.
    """
    import torch
    shape = tuple(int(value) for value in cfg['data']['latent_shape'])
    if shape != (16, 21, 60, 104):
        raise ValueError(f'Unexpected Wan2.1 latent shape: {shape}')
    seed = int(case.get('generation_seed', cfg['inference']['seed']))
    generator = torch.Generator(device='cuda').manual_seed(seed)
    value = torch.randn(shape, device='cuda', dtype=torch.float32,
                        generator=generator)
    return value[None]
