"""Fixed two-arm inference for one NoChunks-v2 checkpoint."""
from pathlib import Path
import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

import numpy as np
import torch

from benchmark_inputs import case_input, load_noise, reference, verify_video
from condition_controls import InferenceCache, prepared_i2v_control, validate_control_input
from generate import write_video
from guidance import description
from physical_content.common import REVISION, atomic_json
from physical_content.model import PhysicalModel
from physical_content.interventions import MODES, fixed_content, physics_evidence, velocity_mode


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=['videophy40', 'videophy60'], required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--control', choices=MODES, default='normal')
    parser.add_argument('--all-controls', action='store_true')
    parser.add_argument('--controls', nargs='+', choices=MODES)
    parser.add_argument('--control-cache', type=Path)
    args = parser.parse_args(argv)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = saved['config']
    task = cfg['task']
    if args.control_cache is not None and task != 'i2v':
        raise ValueError('The prepared image-conditioned controls cannot enter T2V')
    if task == 'i2v' and (args.all_controls or args.controls or args.control in
                         ('empty_reasoning', 'reasoning_swapped')) and args.control_cache is None:
        selected_modes = args.controls or ([m for m in MODES if m != 'normal']
                                          if args.all_controls else [args.control])
        if {'empty_reasoning', 'reasoning_swapped'} & set(selected_modes):
            raise ValueError('I2V controls require own-image re-encoded controls, not the negative arm')
    step = int(saved.get('update', saved.get('step', -1)))
    expected_revision = cfg.get('revision') or saved.get('revision')
    if not expected_revision or saved.get('revision') != expected_revision:
        raise ValueError('Not a NoChunks checkpoint for its declared revision')
    if expected_revision != REVISION:
        raise ValueError('Checkpoint revision differs from the NoChunks-v2 design')
    maximum = int(cfg['train']['max_updates'])
    if step != maximum:
        raise ValueError(f'Use the final step{maximum:04d} checkpoint')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    model = PhysicalModel(cfg)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    if set(saved['model']) != set(model.learned_names):
        raise ValueError('Checkpoint ownership differs from the model')
    with torch.no_grad():
        for name, value in saved['model'].items():
            parameters[name].copy_(value)
        for name, value in saved['constants'].items():
            buffers[name].copy_(value)
    del saved
    model.eval().requires_grad_(False)
    from physgen_native.decoder import VideoDecoder
    decoder = VideoDecoder(cfg).eval()
    cache = InferenceCache(cfg['paths']['inference_evidence'], task)
    protocol = reference()
    selected = [case for case in protocol['cases'] if case['id'] in args.cases]
    if len(selected) != len(set(args.cases)):
        raise ValueError('Case selection differs from the fixed protocol')
    modes = (args.controls or ([mode for mode in MODES if mode != 'normal']
                             if args.all_controls else [args.control]))
    mean_content = None
    if 'global_train_mean' in modes:
        from prepare_train_mean import prepare_train_mean
        mean_content = prepare_train_mean(
            model, cfg, args.checkpoint, args.output / 'training_mean_c.pt')
    for mode in modes:
        mode_output = args.output if mode == 'normal' else args.output / 'controls' / mode
        generate_selected(model, decoder, cache, cfg, protocol, selected, args,
                          step, expected_revision, mode, mode_output, mean_content)


@torch.no_grad()
def generate_selected(model, decoder, cache, cfg, protocol, selected, args,
                      step, expected_revision, mode, mode_output, mean_content):
    task = cfg['task']
    intervention = velocity_mode(mode)
    output = mode_output / args.suite
    output.mkdir(parents=True, exist_ok=True)
    prepared = ({case['id']: prepared_i2v_control(args.control_cache, case['id'], mode)
                 for case in selected}
                if args.control_cache and mode in ('empty_reasoning', 'reasoning_swapped') else {})
    plan = copy.deepcopy(protocol)
    plan.update(
        cases=copy.deepcopy(selected), suite=args.suite, checkpoint=str(args.checkpoint),
        source_checkpoint=str(args.checkpoint), mode=mode,
        checkpoint_step=step, revision=expected_revision, plan=cfg['plan'],
        design_plan=cfg['design_plan'], task=task,
        mechanism=cfg['mechanism'], variant=cfg['variant_id'],
        training_source=cfg['train']['sample_sequence'],
        scope=('T2V: request only; fixed protocol prompts/noise, no initial image'
               if task == 't2v' else 'I2V: fixed shared initial image and original request'),
        data_revision=cfg['train'].get('data_revision'), generation_seed=42,
        guidance=description(5.0, task),
        inference_evidence=cfg['paths']['inference_evidence'],
        conditioning=('request-only Qwen reasoning plus frozen T5/Wan encoding of that reasoning; no image'
                      if task == 't2v' else
                      'own first-image/request Qwen physics plus frozen T5/Wan encoding of the same analysis'),
    )
    if mode != 'normal':
        plan.update(intervention=mode, null_arm='own original condition; module control applied to both arms',
                    request_preserved=True, native_input_preserved=True,
                    donors={case['id']: selected[(i + 1) % len(selected)]['id']
                            for i, case in enumerate(selected)} if mode == 'reasoning_swapped' else None)
        if prepared:
            plan.update(control_cache=str(args.control_cache),
                        control_encoding='re-encoded with own image and own request; native inputs unchanged')
            if mode == 'reasoning_swapped':
                plan['donors'] = {case: value['donor'] for case, value in prepared.items()}
    for case in plan['cases']:
        case['generation_seed'] = 42
    atomic_json(plan, output / 'manifest.json')
    completed = []
    for case in plan['cases']:
        path = output / 'cases' / case['id'] / 'planned.mp4'
        path.parent.mkdir(parents=True, exist_ok=True)
        provenance = dict(checkpoint=str(args.checkpoint), step=step, case=case['id'],
                          seed=42, guidance=description(5.0, task), suite=args.suite,
                          task=task)
        if mode != 'normal':
            provenance['intervention'] = mode
            if prepared:
                provenance['control_source'] = prepared[case['id']]['control_source']
            if mode == 'reasoning_swapped':
                provenance['donor'] = plan['donors'][case['id']]
        if path.is_file() and path.with_suffix('.json').is_file():
            previous = json.loads(path.with_suffix('.json').read_text())
            if any(previous.get(key) != value for key, value in provenance.items()):
                raise ValueError('Different output provenance')
            completed.append(case['id'])
            continue
        begin = time.monotonic()
        inputs, item = case_input(case, task)
        sample = {key: value.cuda() for key, value in inputs.items()}
        sample['times'] = sample['times'][None]
        positive = cache.input(item, image=item.get('image'),
                               duration=item['duration'])
        negative = cache.input(item, image=item.get('image'),
                               duration=item['duration'], drop=True)
        physical_evidence = positive
        if prepared:
            replacement = prepared[case['id']]
            validate_control_input(replacement, item)
            replacement = {key: value.cuda() if torch.is_tensor(value) else value
                           for key, value in replacement.items()}
            physical_evidence = physics_evidence(positive, replacement)
        elif mode == 'reasoning_swapped':
            donor_case = next(row for row in selected if row['id'] == provenance['donor'])
            _, donor_item = case_input(donor_case, task)
            donor_evidence = cache.input(donor_item, image=donor_item.get('image'),
                                         duration=donor_item['duration'])
            physical_evidence = physics_evidence(positive, donor_evidence)
        elif mode == 'empty_reasoning':
            physical_evidence = physics_evidence(positive, negative)
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            anchor_source = 'estimated' if mode == 'estimated_anchor' else 'native'
            content, groups = model.prepare_content(
                physical_evidence, sample['text'], anchor_source=anchor_source)
            if mode == 'global_train_mean':
                content = fixed_content(content, mean_content)
                groups = mean_content['groups']
            null_content, null_groups = model.prepare_content(
                negative, sample['null_text'], anchor_source=anchor_source)
            noise = load_noise(case)
            if noise is None:
                raise ValueError('Use the persisted matched seed42 noise')
            x = noise.cuda().clone()
            first = sample.get('first')
            if task == 'i2v':
                if first is None:
                    raise ValueError('I2V inference lacks its first latent')
                x[:, :, :1] = first
            policy = cfg['inference']
            grid = torch.linspace(1, 0, policy['steps'] + 1, device='cuda')
            grid = policy['shift'] * grid / (1 + (policy['shift'] - 1) * grid)
            for index in range(policy['steps']):
                pos, _ = model.velocity(
                    x, first, sample['text'], grid[index],
                    content, groups, sample['times'], physical_mode=intervention)
                neg, _ = model.velocity(
                    x, first, sample['null_text'], grid[index],
                    null_content, null_groups, sample['times'], physical_mode=intervention)
                velocity = neg + policy['cfg'] * (pos - neg)
                x = x + (grid[index + 1] - grid[index]) * velocity
                if task == 'i2v':
                    x[:, :, :1] = first
                if (index + 1) % 10 == 0:
                    print(json.dumps(dict(kind='sampling', case=case['id'], mode=mode,
                                          step=index + 1)), flush=True)
            frames = decoder(x)
        frames = ((frames.permute(0, 2, 3, 1).clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
        write_video(frames, path)
        verify_video(path)
        atomic_json(dict(
            **provenance, caption=case['caption'], canvas=item.get('image'),
            sampler=policy, positive_analysis=positive.get('analysis'),
            physical_analysis=physical_evidence.get('analysis'),
            negative_analysis=negative.get('analysis'),
            scope='own image/request' if task == 'i2v' else 'request only; no image',
            seconds=time.monotonic() - begin), path.with_suffix('.json'))
        print(json.dumps(dict(kind='video_complete', case=case['id'], mode=mode,
                              seconds=time.monotonic() - begin)), flush=True)
        completed.append(case['id'])
        atomic_json(dict(status='running', completed_cases=len(completed)),
                    output / 'summary.json')
        del sample, positive, negative, content, null_content, x, pos, neg, velocity, frames
    atomic_json(dict(status='completed', checkpoint=str(args.checkpoint),
                     completed_cases=len(completed)), output / 'summary.json')
    atomic_json(dict(status='completed', checkpoint=str(args.checkpoint),
                     suite=args.suite, cases=len(completed)),
                mode_output / 'generation_status.json')


if __name__ == '__main__':
    main()
