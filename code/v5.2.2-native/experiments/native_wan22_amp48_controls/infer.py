"""Official Wan2.2-TI2V-5B T2V generation on the fixed VideoPhy2 40/60 cases.

No project LoRA, no physical modules, no first image.  The native T5 text,
null text, time grid and persisted seed42 noise are the same protocol tensors
used by the R18E/G and R15Bt T2V evaluations.
"""
from pathlib import Path
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

import numpy as np
import torch
import yaml

from native_control.common import atomic_json, gpu_guard_amp48
from physgen_native.context_alignment import native_text_context
from physgen_native.decoder import VideoDecoder
from physgen_native.generator import load_wan, sinusoidal_embedding_1d
from generate import write_video


SCOPE = ('Fixed60 historical VideoPhy2 captions; official Wan2.2-TI2V-5B T2V '
         'baseline, no project LoRA or physical module, no first image; seed42. '
         'Not the official 590-caption benchmark.')


def read_cfg(path):
    return yaml.safe_load(Path(path).read_text())


def protocol(cfg):
    return json.loads(Path(cfg['paths']['eval_protocol']).read_text())


def case_input(cfg, case):
    folder = Path(cfg['paths']['eval_native_inputs']) / case['id']
    stored = torch.load(folder / 'conditions.pt', map_location='cpu',
                        weights_only=True, mmap=True)
    sample = {key: stored[key].float() for key in ('text', 'null_text', 'times')}
    info = json.loads((folder / 'input.json').read_text())
    if info['prompt'] != case['caption'] or info['image'] or info['first_latent']:
        raise ValueError(f'{case["id"]}: invalid T2V native input')
    text, null, times = sample['text'], sample['null_text'], sample['times']
    if (text.ndim != 3 or text.shape[0] != 1 or not 0 < text.shape[1] <= 512
            or text.shape[-1] != 4096 or null.ndim != 3 or null.shape[0] != 1
            or null.shape[-1] != 4096 or tuple(times.shape) != (21,)):
        raise ValueError(f'{case["id"]}: wrong native T5/time shape')
    if not torch.allclose(times, torch.arange(21).float() / 6, atol=1e-5, rtol=0):
        raise ValueError(f'{case["id"]}: time grid differs')
    return sample


def noise_for(case):
    value = torch.load(case['initial_noise'], map_location='cpu',
                       weights_only=True, mmap=True)
    if tuple(value.shape) != (1, 48, 21, 30, 52) or value.dtype != torch.float32:
        raise ValueError(f'{case["id"]}: wrong persisted noise')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{case["id"]}: nonfinite persisted noise')
    return value


@torch.no_grad()
def velocity(wan, noisy, text, sigma):
    if wan.freqs is not None and wan.freqs.device != noisy.device:
        wan.freqs = wan.freqs.to(noisy.device)
    x = wan.patch_embedding(noisy)
    geometry = tuple(x.shape[2:])
    x = x.flatten(2).transpose(1, 2)
    length = x.shape[1]
    position = sigma.expand(length) * 1000
    with torch.autocast(noisy.device.type, enabled=False):
        e = wan.time_embedding(
            sinusoidal_embedding_1d(wan.freq_dim, position).float()[None])
        modulation = wan.time_projection(e).unflatten(-1, (6, wan.dim))
    context = native_text_context(wan, text)
    grid = torch.tensor([geometry], dtype=torch.long)
    lengths = torch.tensor([length], dtype=torch.long)
    for block in wan.blocks:
        x = block(x, modulation, lengths, grid, wan.freqs, context, None)
    return wan.unpatchify(wan.head(x, e), grid)[0][None].float()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--suite', choices=['videophy40', 'videophy60'], required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = read_cfg(args.config)
    base = protocol(cfg)
    selected = [case for case in base['cases'] if case['id'] in args.cases]
    if len(selected) != len(set(args.cases)):
        raise ValueError('Case selection differs')
    expected = {f'VP{i:03d}' for i in
                (range(1, 41) if args.suite == 'videophy40' else range(41, 61))}
    if set(args.cases) != expected:
        raise ValueError('Use the complete declared 40 or disjoint 20')
    root = args.output
    gpu_guard_amp48(root / 'allocations' / os.getenv('DET_EXPERIMENT_ID', 'local'))
    random_state = torch.get_rng_state()
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    wan = load_wan(cfg, 'cuda').eval().requires_grad_(False)
    decoder = VideoDecoder(cfg, device='cuda').eval()
    directory = root / args.suite
    directory.mkdir(parents=True, exist_ok=True)
    manifest_cases = copy.deepcopy(selected)
    for case in manifest_cases:
        for key in ('image', 'canvas', 'first_prompt', 'image_seed',
                    'prepared_inputs'):
            case.pop(key, None)
        case['generation_seed'] = 42
    plan = copy.deepcopy(base)
    plan.update(
        cases=manifest_cases, suite=args.suite,
        checkpoint='official Wan2.2-TI2V-5B', checkpoint_step=None,
        source_checkpoint='official Wan2.2-TI2V-5B', variant='native_want2v',
        task='t2v', first_frame_used=False, known_latent_frames=0,
        conditioning='Official Wan2.2 native T5 text and null text; no first image',
        scope=SCOPE, generation_seed=42, mode='real',
        sampler=cfg['sampling'], evaluation=cfg['evaluation'],
        t2v_input_contract=dict(keys=['text', 'null_text', 'times'],
                                image=False, first_latent=False))
    for key in ('prepared_root', 'physical_input_image', 'input_keys'):
        plan.pop(key, None)
    atomic_json(plan, directory / 'manifest.json')
    sampling = cfg['sampling']
    grid = torch.linspace(1, 0, sampling['steps'] + 1, device='cuda')
    grid = sampling['shift'] * grid / (1 + (sampling['shift'] - 1) * grid)
    completed = []
    for case in selected:
        path = directory / 'cases' / case['id'] / 'planned.mp4'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.with_suffix('.json').is_file():
            completed.append(case['id'])
            continue
        begin = time.monotonic()
        sample = {key: value.cuda() for key, value in case_input(cfg, case).items()}
        noise = noise_for(case)
        x = noise.cuda().clone()
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            for index in range(sampling['steps']):
                pos = velocity(wan, x, sample['text'], grid[index])
                neg = velocity(wan, x, sample['null_text'], grid[index])
                x = x + (grid[index + 1] - grid[index]) * (
                    neg + sampling['guidance'] * (pos - neg))
            frames = decoder(x)
        frames = ((frames.permute(0, 2, 3, 1).clamp(-1, 1) + 1) * 127.5
                  ).round().byte().cpu().numpy()
        if frames.shape != (81, 480, 832, 3):
            raise ValueError(f'Unexpected decoded video shape: {frames.shape}')
        write_video(frames, path)
        atomic_json(dict(
            checkpoint='official Wan2.2-TI2V-5B', seed=42, case=case['id'],
            caption=case['caption'], task='t2v', first_frame_used=False,
            conditioning='official native T5 text/null text; no image',
            sampler=sampling, scope=SCOPE,
            seconds=time.monotonic() - begin), path.with_suffix('.json'))
        completed.append(case['id'])
        atomic_json(dict(status='running', completed_cases=len(completed)),
                    directory / 'summary.json')
        print(json.dumps(dict(stage='video_complete', case=case['id'],
                              seconds=time.monotonic() - begin)), flush=True)
    atomic_json(dict(status='completed', completed_cases=len(completed)),
                directory / 'summary.json')
    atomic_json(dict(status='completed', suite=args.suite,
                     cases=len(completed), mode='native_wan_t2v',
                     first_frame_used=False), root / 'generation_status.json')
    torch.set_rng_state(random_state)


if __name__ == '__main__':
    main()
