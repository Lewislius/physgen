"""Fixed VideoPhy2 60 prompts, Wan2.1 noise and Wan2.1-native fps verification.

The prompts, captions and positive/null UMT5 text tensors are shared with the
project protocol.  The Wan2.2 persisted noise is intentionally not reused:
Wan2.1 uses a 16-channel, 4x8x8 VAE and therefore a different noise tensor.
The formal Wan2.1 protocol writes and verifies 16 fps; Wan2.2 candidates remain
on 24 fps and do not use this module.
"""
from pathlib import Path
from fractions import Fraction
import json
import math
import subprocess
import sys

import bootstrap

BASE = bootstrap.BASE
ROOT = BASE / 'experiments/physical_content_v7'


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


def verify_video(path, fps):
    """Verify Wan2.1 geometry, 81 frames and the configured native fps."""
    value = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
         '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate',
         '-of', 'json', str(path)],
        check=True, capture_output=True, text=True)
    stream, = json.loads(value.stdout)['streams']
    if (int(stream['width']), int(stream['height']),
            int(stream['nb_read_frames'])) != (832, 480, 81):
        raise ValueError(f'{path}: invalid geometry/length')
    if not math.isclose(float(Fraction(stream['avg_frame_rate'])), float(fps),
                        rel_tol=1e-6):
        raise ValueError(f'{path}: expected {fps} fps, got {stream["avg_frame_rate"]}')
    return stream
