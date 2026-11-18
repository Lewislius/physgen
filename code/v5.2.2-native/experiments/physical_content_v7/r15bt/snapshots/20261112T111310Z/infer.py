"""Matched R15Bt T2V generation on the fixed VideoPhy2 40/60 prompts.

No first image, canvas, or first latent is read.  All 21 latent frames start
from the persisted seed42 noise and every frame is denoised by the same
request-bound joint physical read.
"""
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

from benchmark_inputs import (T2V_SCOPE, case_input, load_noise, reference,
                             verify_video)
from condition_controls import InferenceCache
from generate import write_video
from guidance import description

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=['videophy40', 'videophy60'], required=True)
    parser.add_argument('--modes', nargs='+', default=['real'],
                        choices=['real', 'no_c', 'direct_physical_read', 'state_value_off'])
    parser.add_argument('--cases', nargs='+', required=True)
    args = parser.parse_args()
    from physical_content.common import REVISION, atomic_json, gpu_guard
    from physical_content.model import PhysicalModel
    from physgen_native.decoder import VideoDecoder

    root = args.output
    gpu_guard(root / 'allocations' / os.getenv('DET_EXPERIMENT_ID', 'local'))
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = saved['config']
    step = saved['step']
    if saved['revision'] != REVISION or cfg['revision'] != REVISION or step != 400:
        raise ValueError('Use this R15Bt candidate own400 weights')
    interface = {}
    for name in ('mean', 'scale', 'weight', 'bias', 'target_scale'):
        key = f'enhancement.anchor_{name}'
        if key not in saved['constants']:
            raise KeyError(f'Checkpoint lacks interface buffer {key}')
        interface[name] = saved['constants'][key]
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    model = PhysicalModel(cfg, interface=interface)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    if set(saved['model']) != set(model.learned_names):
        raise ValueError('Checkpoint ownership mismatch')
    for name, value in saved['model'].items():
        parameters[name].copy_(value)
    for name, value in saved['constants'].items():
        buffers[name].copy_(value)
    del saved
    model.eval().requires_grad_(False)
    decoder = VideoDecoder(cfg).eval()
    reader = InferenceCache(cfg['paths']['inference_evidence'])
    base = reference(cfg)
    selected = [case for case in base['cases'] if case['id'] in args.cases]
    if len(selected) != len(set(args.cases)):
        raise ValueError('Case selection differs')
    expected = {f'VP{i:03d}' for i in
                (range(1, 41) if args.suite == 'videophy40' else range(41, 61))}
    if set(args.cases) != expected:
        raise ValueError('Use the complete declared 40 or disjoint 20')
    scopes = dict(
        real='Request-only evidence; B400 joint direct and request-bound physical read; no first image',
        no_c='Native generation branch: physical fusion bypassed in both CFG arms',
        direct_physical_read='Bypass request binding and restore the R10A direct C read using this checkpoint weights in both CFG arms',
        state_value_off='Bypass only the state-value interaction after the request-bound physical read in both CFG arms')
    def log(kind, **values):
        row = dict(kind=kind, time=time.time(), **values)
        print(json.dumps(row), flush=True)
        with (root / 'generation.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
    for mode in args.modes:
        if mode not in scopes:
            raise ValueError(mode)
        directory = root / args.suite if mode == 'real' else root / 'controls' / mode / args.suite
        directory.mkdir(parents=True, exist_ok=True)
        plan = copy.deepcopy(base)
        generation_cases = copy.deepcopy(selected)
        for case in generation_cases:
            # T2V consumes only caption/id/native T5/times from the T2V-only
            # eval_native_inputs copy.  Remove every I2V path/metadata so no
            # runtime or manifest object can imply a first-frame dependency.
            for key in ('image', 'canvas', 'first_prompt', 'image_seed',
                        'prepared_inputs'):
                case.pop(key, None)
            case['generation_seed'] = 42
        manifest_cases = copy.deepcopy(generation_cases)
        plan.update(
            cases=manifest_cases, suite=args.suite,
            checkpoint=str(args.checkpoint), source_checkpoint=str(args.checkpoint),
            checkpoint_step=step, revision=cfg['revision'],
            variant='physical_content', training_source=cfg['source_snapshot'],
            training_experiment_id=os.getenv('DET_EXPERIMENT_ID',
                                             args.checkpoint.parent.name),
            mode=mode, generation_seed=42, task='t2v',
            conditioning='Request-only frozen Qwen physical evidence; no first image',
            scope=T2V_SCOPE, guidance=description(5., 5.),
            intervention_scope=scopes[mode], data_revision=cfg['data_revision'],
            inference_evidence=cfg['paths']['inference_evidence'],
            fusion_order=cfg['model']['fusion_order'],
            architecture=dict(c_tokens=256, enhancement_width=4096,
                              wan_width=3072, fusion_layers=list(range(3, 31, 3))),
            inference_revision='r15bt_joint_read_t2v_v1',
            first_frame_used=False, known_latent_frames=0,
            physical_read='B400 joint direct and request-bound memory',
            t2v_input_contract=dict(keys=['text', 'null_text', 'times'],
                                    image=False, first_latent=False))
        for key in ('prepared_root', 'physical_input_image', 'input_keys'):
            plan.pop(key, None)
        atomic_json(plan, directory / 'manifest.json')
        completed = []
        for case in generation_cases:
            output = directory / 'cases' / case['id'] / 'planned.mp4'
            output.parent.mkdir(parents=True, exist_ok=True)
            provenance = dict(
                checkpoint=str(args.checkpoint), step=step, mode=mode,
                case=case['id'], seed=42, task='t2v', first_frame_used=False,
                inference_revision='r15bt_joint_read_t2v_v1',
                inference_source=str(Path(__file__).resolve().parent),
                inference_evidence=cfg['paths']['inference_evidence'],
                guidance=description(5., 5.))
            if output.is_file() and output.with_suffix('.json').is_file():
                previous = json.loads(output.with_suffix('.json').read_text())
                if any(previous.get(key) != value for key, value in provenance.items()):
                    raise ValueError('Different output provenance')
                completed.append(case['id'])
                continue
            begin = time.monotonic()
            inputs, item = case_input(case, cfg['paths']['eval_native_inputs'])
            sample = {key: value.cuda() for key, value in inputs.items()}
            sample['times'] = sample['times'][None]
            positive = reader.input(item)
            negative = reader.input(item, drop=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                c, groups = model.enhancement(positive)
                nc, ngroups = model.enhancement(negative)
                noise = load_noise(case)
                if noise is None:
                    raise ValueError('Use the persisted matched seed42 noise')
                x = noise.cuda().clone()
                if tuple(x.shape) != (1, 48, 21, 30, 52):
                    raise ValueError(f'Unexpected latent shape: {tuple(x.shape)}')
                policy = cfg['inference']
                grid = torch.linspace(1, 0, policy['steps'] + 1, device='cuda')
                grid = policy['shift'] * grid / (1 + (policy['shift'] - 1) * grid)
                execution = dict(
                    physical_enabled=mode != 'no_c',
                    direct_physical_read=mode == 'direct_physical_read',
                    state_value_enabled=mode != 'state_value_off')
                for index in range(policy['steps']):
                    pos, _ = model.velocity(
                        x, None, sample['text'], grid[index], c,
                        sample['times'], groups, **execution)
                    neg, _ = model.velocity(
                        x, None, sample['null_text'], grid[index], nc,
                        sample['times'], ngroups, **execution)
                    velocity = neg + policy['guidance'] * (pos - neg)
                    x = x + (grid[index + 1] - grid[index]) * velocity
                    if (index + 1) % 10 == 0:
                        log('sampling', case=case['id'], mode=mode, step=index + 1)
                frames = decoder(x)
            frames = ((frames.permute(0, 2, 3, 1).clamp(-1, 1) + 1) * 127.5
                      ).round().byte().cpu().numpy()
            write_video(frames, output)
            verify_video(output)
            atomic_json(dict(
                **provenance, caption=case['caption'],
                sampler=policy,
                positive_analysis=positive.get('analysis'),
                positive_relations=positive.get('relations'),
                negative_analysis=negative.get('analysis'),
                scope=scopes[mode], seconds=time.monotonic() - begin,
                first_frame_used=False, known_latent_frames=0,
                conditioning='request only; no image'),
                output.with_suffix('.json'))
            completed.append(case['id'])
            log('video_complete', case=case['id'], mode=mode,
                seconds=time.monotonic() - begin)
            atomic_json(dict(status='running', mode=mode,
                             completed_cases=len(completed)), directory / 'summary.json')
            del sample, positive, negative, c, nc, x, pos, neg, velocity, frames
        atomic_json(dict(status='completed', mode=mode,
                         completed_cases=len(completed)), directory / 'summary.json')
    atomic_json(dict(status='completed', checkpoint=str(args.checkpoint),
                     modes=args.modes, task='t2v', first_frame_used=False),
                root / 'generation_status.json')


if __name__ == '__main__':
    main()
