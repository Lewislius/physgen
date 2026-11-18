"""Fixed two-arm I2V inference for one R18 checkpoint."""
from pathlib import Path
import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap

import numpy as np
import torch

from benchmark_inputs import case_input, load_noise, reference, verify_video
from condition_controls import InferenceCache
from generate import write_video
from guidance import description
from physical_content.common import atomic_json
from physical_content.model import PhysicalModel


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=['videophy40', 'videophy60'], required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    args = parser.parse_args(argv)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = saved['config']
    step = int(saved.get('update', saved.get('step', -1)))
    expected_revision = cfg.get('revision') or saved.get('revision')
    if not expected_revision or saved.get('revision') != expected_revision:
        raise ValueError('Not an R18 checkpoint for its declared revision')
    max_updates = int(cfg['train']['max_updates'])
    if step != max_updates:
        raise ValueError(f'Use the final step{max_updates:04d} checkpoint')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    model = PhysicalModel(cfg)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    if set(saved['model']) != set(model.learned_names):
        raise ValueError('Checkpoint ownership differs from the R18 model')
    with torch.no_grad():
        for name, value in saved['model'].items():
            parameters[name].copy_(value)
        for name, value in saved['constants'].items():
            buffers[name].copy_(value)
    del saved
    model.eval().requires_grad_(False)
    from physgen_native.decoder import VideoDecoder
    decoder = VideoDecoder(cfg).eval()
    cache = InferenceCache(cfg['paths']['inference_evidence'])
    protocol = reference()
    selected = [case for case in protocol['cases'] if case['id'] in args.cases]
    if len(selected) != len(set(args.cases)):
        raise ValueError('Case selection differs from the fixed protocol')
    output = args.output / args.suite
    output.mkdir(parents=True, exist_ok=True)
    plan = copy.deepcopy(protocol)
    plan.update(
        cases=copy.deepcopy(selected), suite=args.suite, checkpoint=str(args.checkpoint),
        checkpoint_step=step, revision=expected_revision, variant=cfg['variant_id'],
        data_revision=cfg['train'].get('data_revision'), generation_seed=42,
        guidance=description(5.0),
        inference_evidence=cfg['paths']['inference_evidence'],
        conditioning='original native T5 plus original first-image/request Qwen C',
        architecture=dict(
            width=cfg['model']['width'], q_tokens=cfg['model']['q_tokens'],
            sites=cfg['model']['sites_zero_based'],
            video_rope='native_grid_only', static_writer_multiplier=1.0),
    )
    for case in plan['cases']:
        case['generation_seed'] = 42
    atomic_json(plan, output / 'manifest.json')
    completed = []
    for case in plan['cases']:
        path = output / 'cases' / case['id'] / 'planned.mp4'
        path.parent.mkdir(parents=True, exist_ok=True)
        provenance = dict(checkpoint=str(args.checkpoint), step=step, case=case['id'],
                          seed=42, guidance=description(5.0), suite=args.suite)
        if path.is_file() and path.with_suffix('.json').is_file():
            previous = json.loads(path.with_suffix('.json').read_text())
            if any(previous.get(key) != value for key, value in provenance.items()):
                raise ValueError('Different output provenance')
            completed.append(case['id'])
            continue
        begin = time.monotonic()
        inputs, item = case_input(case)
        sample = {key: value.cuda() for key, value in inputs.items()}
        sample['times'] = sample['times'][None]
        positive = cache.input(item, image=item['image'], duration=item['duration'])
        negative = cache.input(item, image=item['image'], duration=item['duration'], drop=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            # C memory depends only on the condition arm, not on sigma or H.
            content, groups = model.prepare_content(positive, sample['text'])
            null_content, null_groups = model.prepare_content(negative, sample['null_text'])
            noise = load_noise(case)
            if noise is None:
                raise ValueError('Use the persisted matched seed42 noise')
            x = noise.cuda().clone()
            x[:, :, :1] = sample['first']
            policy = cfg['inference']
            grid = torch.linspace(1, 0, policy['steps'] + 1, device='cuda')
            grid = policy['shift'] * grid / (1 + (policy['shift'] - 1) * grid)
            for index in range(policy['steps']):
                pos, _ = model.velocity(
                    x, sample['first'], sample['text'], grid[index],
                    content, groups, sample['times'])
                neg, _ = model.velocity(
                    x, sample['first'], sample['null_text'], grid[index],
                    null_content, null_groups, sample['times'])
                velocity = neg + policy['cfg'] * (pos - neg)
                x = x + (grid[index + 1] - grid[index]) * velocity
                x[:, :, :1] = sample['first']
            frames = decoder(x)
        frames = ((frames.permute(0, 2, 3, 1).clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
        write_video(frames, path)
        verify_video(path)
        atomic_json(dict(
            **provenance, caption=case['caption'], canvas=item['image'],
            sampler=policy, positive_analysis=positive.get('analysis'),
            negative_analysis=negative.get('analysis'),
            scope='own image, original request, positive/negative arms kept separate',
            seconds=time.monotonic() - begin), path.with_suffix('.json'))
        completed.append(case['id'])
        atomic_json(dict(status='running', completed_cases=len(completed)), output / 'summary.json')
        del sample, positive, negative, content, null_content, x, pos, neg, velocity, frames
    atomic_json(dict(status='completed', checkpoint=str(args.checkpoint),
                     completed_cases=len(completed)), output / 'summary.json')
    atomic_json(dict(status='completed', checkpoint=str(args.checkpoint),
                     suite=args.suite, cases=len(completed)), args.output / 'generation_status.json')


if __name__ == '__main__':
    main()
