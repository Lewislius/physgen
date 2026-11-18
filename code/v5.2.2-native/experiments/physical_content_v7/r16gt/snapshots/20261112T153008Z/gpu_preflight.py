"""Disposable full-size R16Gt T2V preflight: no formal weights or evaluation."""
from pathlib import Path
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap

import numpy as np
import torch

from physical_content.common import ARCHITECTURE, atomic_json, gpu_guard, load_config
from physical_content.data import Data
from physical_content.model import PhysicalModel
from physical_content.supervision import generation_loss
from physical_content.teacher import Teacher
from physgen_native.data import SampleOrder
from physical_scaling.schedule import auxiliary_coefficient, rates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    folder = args.output
    gpu = gpu_guard(folder)
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    atomic_json(dict(status='loading_fresh_official_model', gpu=gpu), folder / 'status.json')
    try:
        model = PhysicalModel(cfg).train()
        model.wan.eval()
        model.set_phase(0)
        teacher = Teacher(cfg)
        named = dict(model.named_parameters())
        groups = model.optim_groups()
        count = sum(named[n].numel() for n in model.learned_names)
        if any(p.requires_grad for block in model.wan.blocks
               for p in block.cross_attn.parameters()):
            raise AssertionError('Native text projections must stay frozen')
        if any(p.requires_grad for name, p in model.wan.named_parameters()
               if name.endswith(('q.weight', 'k.weight', 'v.weight', 'o.weight'))
               and 'cross_attn' in name):
            raise AssertionError('Frozen native text Q/K/V/O violated')
        for site, fusion in model.fusions.items():
            for name in ('output', 'condition_query', 'request_condition_query',
                         'joint_condition_query', 'time', 'direct_output',
                         'request_output', 'joint_output', 'direct_state_value',
                         'request_state_value', 'joint_state_value',
                         'gate_direct', 'gate_request', 'gate_joint', 'gate_single',
                         'local_value_gate', 'fused_output', 'fused_state_value',
                         'final_local_gate', 'final_global_gate'):
                layer = getattr(fusion, name, None)
                if layer is not None and torch.count_nonzero(layer.weight):
                    raise AssertionError(f'{site}: declared zero-init module is nonzero: {name}')
        if torch.count_nonzero(model.value_correction.weight):
            raise AssertionError('Declared shared V correction is nonzero at start')
        optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=.01,
                                      foreach=False)
        data = Data(cfg)
        index = next(iter(SampleOrder(len(data), cfg['train']['seed'], 0)))
        sample = data.get(index)
        if 'first' in sample or 'first_rgb' in sample:
            raise AssertionError('T2V data view exposed a first frame')
        evidence = teacher.input(sample)
        noise = torch.randn(sample['latent'].shape, device='cuda',
                            generator=torch.Generator(device='cuda').manual_seed(42))
        sigma = torch.tensor(.6, device='cuda')
        rows = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                            cache_enabled=False):
            initial, relations = model.enhancement(evidence)
            noisy = (1 - sigma) * sample['latent'] + sigma * noise
            initial_full, _ = model.velocity(noisy, None, sample['text'], sigma,
                                             initial, sample['times'], relations)
            initial_native, _ = model.velocity(noisy, None, sample['text'], sigma,
                                               initial, sample['times'], relations,
                                               physical_enabled=False)
            initial_native_max_abs = float((initial_full - initial_native).abs().max())
            torch.testing.assert_close(initial_full, initial_native, rtol=2e-5, atol=2e-5)
            del initial, initial_full, initial_native, noisy
        for iteration, step in enumerate((0, 1, 2, 128)):
            phase = model.set_phase(step)
            rates(optimizer, cfg, step)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                c, groups_i = model.enhancement(evidence)
                noisy = (1 - sigma) * sample['latent'] + sigma * noise
                pred, _ = model.velocity(noisy, None, sample['text'], sigma,
                                         c, sample['times'], groups_i)
                generation, parts = generation_loss(
                    pred, noise, sample['latent'], sample['valid'],
                    known_frames=0, **cfg['loss']['generation'])
                aux = model.enhancement.content_loss(c, evidence)
                objective = generation + auxiliary_coefficient(cfg, step) * aux
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite training objective')
            objective.backward()
            fusion = model.fusions['2']
            grad = lambda p: None if p.grad is None else float(p.grad.detach().float().norm())
            def fgrad(name):
                layer = getattr(fusion, name, None)
                return grad(layer.weight) if layer is not None else None
            relevant = dict(
                output=fgrad('output'), condition_query=fgrad('condition_query'),
                direct_output=fgrad('direct_output'), request_output=fgrad('request_output'),
                joint_output=fgrad('joint_output'),
                direct_state_value=fgrad('direct_state_value'),
                request_state_value=fgrad('request_state_value'),
                joint_state_value=fgrad('joint_state_value'),
                video_q=fgrad('video_q'), video_k=fgrad('video_k'),
                video_v=fgrad('video_v'), key=fgrad('key'), value=fgrad('value'),
                routing_C=grad(model.enhancement.project.weight),
                execution_C=grad(model.enhancement.execution_project.weight))
            if not relevant['output'] > 0:
                raise AssertionError('FM gradient must reach the zero-initialized local output')
            for name in ('direct', 'request', 'joint'):
                if not relevant[name + '_output'] > 0:
                    raise AssertionError(f'Global {name} output did not receive generation gradient')
            if iteration >= 1:
                for key in ('condition_query', 'video_q', 'video_k', 'video_v', 'key', 'value'):
                    if not relevant[key] > 0:
                        raise AssertionError(f'Generation gradient did not reach {key}')
                for name in ('direct', 'request', 'joint'):
                    if not relevant[name + '_state_value'] > 0:
                        raise AssertionError(f'Global {name} state gate did not receive gradient')
            for name, param in model.wan.named_parameters():
                if param.grad is not None and (
                        phase == 'interface' or not name.endswith(('lora_a', 'lora_b'))):
                    raise AssertionError(f'Unexpected native gradient: {phase}/{name}')
            norms = {}
            for group in optimizer.param_groups:
                params = [p for p in group['params'] if p.grad is not None]
                norms[group['name']] = float(torch.nn.utils.clip_grad_norm_(
                    params, cfg['train']['grad_clip'][group['name']],
                    error_if_nonfinite=True)) if params else 0.
            if phase == 'joint' and not norms['wan_lora'] > 0:
                raise AssertionError('Joint phase has no native LoRA gradient')
            optimizer.step()
            torch.cuda.synchronize()
            row = dict(disposable_iteration=iteration + 1, phase=phase,
                       sample=sample['id'], seconds=time.monotonic() - start,
                       peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                       gradients=relevant, group_gradients=norms,
                       generation=float(generation.detach()),
                       aux=float(aux.detach()))
            rows.append(row)
            atomic_json(dict(status='running', rows=len(rows), last=row),
                        folder / 'status.json')
            print(json.dumps(row), flush=True)
            if iteration == 1:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                    cache_enabled=False):
                    c, relations = model.enhancement(evidence)
                    full, _ = model.velocity(noisy, None, sample['text'], sigma,
                                             c, sample['times'], relations)
                    native, _ = model.velocity(noisy, None, sample['text'], sigma,
                                               c, sample['times'], relations,
                                               physical_enabled=False)
                    per_frame = (full - native).abs().amax(dim=(0, 1, 3, 4))
                    if not bool((per_frame > 1e-7).all()):
                        raise AssertionError(
                            f'T2V physical write missed a frame: {per_frame.tolist()}')
        result = dict(
            status='passed', gpu=gpu, parameters=count,
            parameter_groups={group['name']: sum(p.numel() for p in group['params'])
                              for group in groups},
            response_mode=cfg['model']['response_mode'],
            global_execution=cfg['model']['global_execution'],
            architecture=ARCHITECTURE,
            first_frame_used=False, known_latent_frames=0,
            initial_native_max_abs=initial_native_max_abs,
            memory_policy='one video per GPU; gradient accumulation 4',
            rows=rows)
        atomic_json(result, folder / 'report.json')
        atomic_json(dict(status='passed', report=str(folder / 'report.json')),
                    folder / 'status.json')
        print(json.dumps({key: result[key] for key in
                          ('status', 'parameters', 'initial_native_max_abs')}))
    except BaseException as error:
        atomic_json(dict(status='failed', error=repr(error)), folder / 'status.json')
        raise


if __name__ == '__main__':
    main()
