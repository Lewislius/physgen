"""R18G T2V fixed40 with shift=3.0, seed=42, empty negative prompt.

This is an inference-only protocol variant of the trained R18G checkpoint.  It
uses the existing R18G prepared evidence and persisted seed42 noise; only the
scheduler shift is changed and the native negative text is the protocol's
empty prompt (1 token in conditions.pt).  Official VideoPhy2 AutoEval follows.
"""
from pathlib import Path
import argparse
import copy
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

import numpy as np
import torch

from benchmark_inputs import case_input, load_noise, reference, verify_video
from condition_controls import InferenceCache
from generate import write_video
from guidance import description
from physical_content.common import atomic_json, gpu_guard
from physical_content.model import PhysicalModel
from pipeline import evaluator_environment

SCOPE = ('VideoPhy2 fixed40 T2V, R18G checkpoint, shift=3.0, seed=42, '
         'empty native negative prompt; not the ProPhy 600/180 benchmark.')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path(__file__).resolve().parent /
                        'runs/r18g_nchunks400/step0400.pt')
    parser.add_argument('--shift', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    experiment = os.getenv('DET_EXPERIMENT_ID', 'local')
    output = args.output or (Path(__file__).resolve().parent / 'inference_outputs' /
                             'r18g_nchunks400' /
                             f'step0400_fixed40_shift{args.shift:g}_seed{args.seed}_emptyneg')
    job = Path(__file__).resolve().parent / 'reports/jobs' / experiment
    gpu_guard(job)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = saved['config']
    if cfg['task'] != 't2v' or cfg['plan'] != 'G':
        raise ValueError('Use the R18G T2V checkpoint')
    if saved['step'] != 400:
        raise ValueError('Use step0400')
    policy = dict(cfg['inference'])
    policy.update(shift=args.shift, seed=args.seed)
    random_state = torch.get_rng_state()
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    model = PhysicalModel(cfg)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    if set(saved['model']) != set(model.learned_names):
        raise ValueError('Checkpoint ownership differs')
    with torch.no_grad():
        for name, value in saved['model'].items():
            parameters[name].copy_(value)
        for name, value in saved['constants'].items():
            buffers[name].copy_(value)
    del saved
    model.eval().requires_grad_(False)
    from physgen_native.decoder import VideoDecoder
    decoder = VideoDecoder(cfg).eval()
    cache = InferenceCache(cfg['paths']['inference_evidence'], 't2v')
    base = reference()
    selected = [case for case in base['cases'] if int(case['id'][2:]) <= 40]
    if {case['id'] for case in selected} != {f'VP{i:03d}' for i in range(1, 41)}:
        raise ValueError('Use the complete fixed40')
    directory = output / 'videophy40'
    directory.mkdir(parents=True, exist_ok=True)
    manifest_cases = copy.deepcopy(selected)
    for case in manifest_cases:
        for key in ('image', 'canvas', 'first_prompt', 'image_seed',
                    'prepared_inputs'):
            case.pop(key, None)
        case['generation_seed'] = args.seed
    plan = copy.deepcopy(base)
    plan.update(cases=manifest_cases, suite='videophy40', checkpoint=str(args.checkpoint),
                checkpoint_step=400, variant='R18G', plan='G', design_plan='T2',
                task='t2v', first_frame_used=False, known_latent_frames=0,
                conditioning='request-only evidence; empty native negative text',
                scope=SCOPE, mode='real', generation_seed=args.seed,
                guidance=description(5.0, 't2v'),
                sampling=dict(steps=50, shift=args.shift, cfg=5.0, seed=args.seed),
                intervention_scope='R18G shift3 seed42 empty-negative fixed40')
    for key in ('prepared_root', 'physical_input_image', 'input_keys'):
        plan.pop(key, None)
    atomic_json(plan, directory / 'manifest.json')
    grid = torch.linspace(1, 0, policy['steps'] + 1, device='cuda')
    grid = policy['shift'] * grid / (1 + (policy['shift'] - 1) * grid)
    completed = []
    for case in selected:
        path = directory / 'cases' / case['id'] / 'planned.mp4'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.with_suffix('.json').is_file():
            completed.append(case['id'])
            continue
        begin = time.monotonic()
        inputs, item = case_input(case, 't2v')
        sample = {key: value.cuda() for key, value in inputs.items()}
        sample['times'] = sample['times'][None]
        if sample['null_text'].shape[1] != 1:
            raise ValueError('Expected the protocol empty native negative prompt')
        positive = cache.input(item, image=None, duration=item['duration'])
        negative = cache.input(item, image=None, duration=item['duration'], drop=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            content, groups = model.prepare_content(positive, sample['text'])
            null_content, null_groups = model.prepare_content(negative, sample['null_text'])
            noise = load_noise(case)
            if noise is None:
                raise ValueError('Missing persisted seed42 noise')
            x = noise.cuda().clone()
            for index in range(policy['steps']):
                pos, _ = model.velocity(x, None, sample['text'], grid[index],
                                        content, groups, sample['times'])
                neg, _ = model.velocity(x, None, sample['null_text'], grid[index],
                                        null_content, null_groups, sample['times'])
                x = x + (grid[index + 1] - grid[index]) * (
                    neg + policy['cfg'] * (pos - neg))
            frames = decoder(x)
        frames = ((frames.permute(0, 2, 3, 1).clamp(-1, 1) + 1) * 127.5
                  ).round().byte().cpu().numpy()
        write_video(frames, path)
        verify_video(path)
        atomic_json(dict(checkpoint=str(args.checkpoint), step=400, case=case['id'],
                         caption=case['caption'], seed=args.seed, task='t2v',
                         first_frame_used=False, shift=args.shift,
                         negative_prompt='empty', sampler=policy,
                         seconds=time.monotonic() - begin, scope=SCOPE),
                    path.with_suffix('.json'))
        completed.append(case['id'])
        atomic_json(dict(status='running', completed_cases=len(completed)),
                    directory / 'summary.json')
        print(json.dumps(dict(stage='video_complete', case=case['id'],
                              seconds=time.monotonic() - begin)), flush=True)
    atomic_json(dict(status='completed', completed_cases=len(completed)),
                directory / 'summary.json')
    scorer = '/home/liuzhirui/miniconda3/envs/physvid/bin/python'
    subprocess.run([scorer, '-I', '-B', '-u',
                    str(bootstrap.BASE / 'tools/evaluate_videophy2.py'),
                    str(directory)], check=True, env=evaluator_environment())
    scores = json.loads((directory / 'scores.json').read_text())
    atomic_json(dict(status='completed', shift=args.shift, seed=args.seed,
                     negative_prompt='empty', scores=scores['summary'],
                     output=str(output)), job / 'status.json')
    print(json.dumps(scores['summary']), flush=True)
    torch.set_rng_state(random_state)


if __name__ == '__main__':
    main()
