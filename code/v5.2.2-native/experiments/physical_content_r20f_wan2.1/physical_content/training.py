"""The single official-fresh compressed-400, effective-batch-4 training loop."""
from pathlib import Path
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from physical_scaling.common import Log
from physgen_native.data import SampleOrder

from .checkpointing import provenance, restore_checkpoint, save_checkpoint
from .common import atomic_json, audit, design_hash, gpu_guard, load_config
from .data import TrainingData
from .model import PhysicalModel
from .objectives import backward_micro
from .schedule import route_coefficient, set_lrs
from .teacher import Teacher


def _run_dir(cfg, run_id):
    if run_id is None:
        return Path(cfg['paths']['run_dir'])
    return Path(cfg['paths']['run_dir']).parent / run_id


def _status(run, **value):
    atomic_json(value, run / 'status.json')


def train(cfg, run, *, resume=None, stop_after=None, log=None):
    if cfg['status'] != 'ready_for_training':
        raise ValueError('FT design has not been finalized')
    if log is None:
        log = Log(run)
    report = audit(cfg)
    log('audit', **report)
    run.mkdir(parents=True, exist_ok=True)
    gpu = gpu_guard(run, cfg['train'].get('gpu_policy', 'any48'))
    log('gpu', **{key: value for key, value in gpu.items() if key != 'kind'})
    random.seed(cfg['train']['seed'])
    np.random.seed(cfg['train']['seed'])
    torch.manual_seed(cfg['train']['seed'])
    torch.cuda.manual_seed_all(cfg['train']['seed'])
    torch.set_num_threads(4)

    _status(run, status='loading', variant=cfg['variant_id'], update=0)
    model = PhysicalModel(cfg).train()
    model.wan.eval()
    teacher = Teacher(cfg)
    optimizer = torch.optim.AdamW(
        model.optim_groups(), betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, foreach=False)
    data = TrainingData(cfg)
    null = torch.load(
        Path(cfg['paths']['cache']) / 'base/null_text.pt',
        map_location='cpu', weights_only=True).float()[None].cuda()
    if resume is not None:
        completed, cursor, saved_provenance = restore_checkpoint(
            model, optimizer, resume, cfg)
        for key in ('interface_hash', 'wan_checkpoint_hash', 'train_manifest_hash'):
            if saved_provenance.get(key) != report[key]:
                raise ValueError(f'Resume evidence differs: {key}')
        log('resumed', update=completed, cursor=cursor,
            checkpoint=str(resume), provenance=saved_provenance)
    else:
        completed, cursor = 0, 0
    if completed > cfg['train']['max_updates']:
        raise ValueError('Resume passed the configured maximum updates')
    if completed == cfg['train']['max_updates']:
        _status(run, status='complete', update=completed,
                checkpoint=str(Path(run) / 'latest.pt'))
        log('complete', update=completed,
            reason=f'already at {cfg["train"]["max_updates"]}; no further optimizer.step')
        return

    order = (iter(range(cursor, len(data))) if data.fixed_sequence else
             iter(SampleOrder(len(data), cfg['train']['seed'], cursor)))
    total = int(cfg['train']['max_updates'])
    stop = total if stop_after is None else min(total, int(stop_after))
    if stop < completed:
        raise ValueError('stop-after lies before the resumed update')
    parameter_counts = model.trainable_count()
    log('model_ready', plan=cfg['plan'],
        parameters=sum(parameter_counts.values()),
        groups={group['name']: sum(p.numel() for p in group['params'])
                for group in optimizer.param_groups},
        target_count=len(model.video_lora_targets),
        initialization='official foundations only',
        design_hash=design_hash(cfg))
    _status(run, status='training', update=completed, variant=cfg['variant_id'])
    base_before = model.base_bank.hashes(full=False)
    final = completed

    for update in range(completed + 1, stop + 1):
        phase = model.set_phase(update - 1)
        multiplier = set_lrs(optimizer, cfg, update)
        coefficient = route_coefficient(update, cfg)
        optimizer.zero_grad(set_to_none=True)
        sums = {key: 0.0 for key in
                ('fm', 'tail', 'generation', 'route', 'weighted_route',
                 'c_pair_cosine', 'c_centered_energy',
                 'c_update_over_train_source_rms', 'write_relative_rms_mean', 'write_relative_rms_max')}
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for micro in range(cfg['train']['accumulation']):
            sample = data.get(next(order))
            draws = torch.rand(
                2, generator=torch.Generator().manual_seed(
                    cfg['train']['seed'] * 1000003 + cursor))
            drop = bool(draws[0] < cfg['train']['condition_dropout'])
            sigma = (cfg['train']['sigma_min'] + (cfg['train']['sigma_max'] - cfg['train']['sigma_min']) * draws[1]).to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                evidence = teacher.input(sample, drop=drop)
                text = null if drop else sample['text']
                noise = torch.randn(
                    sample['latent'].shape, device='cuda',
                    generator=torch.Generator(device='cuda').manual_seed(
                        cfg['train']['seed'] * 2000003 + cursor))
                values = backward_micro(
                    model, evidence, sample, text, sigma, noise,
                    coefficient, cfg['train']['accumulation'])
            for key in sums:
                sums[key] += values[key]
            cursor += 1
            log('micro', update=update, micro=micro + 1, sample=sample['id'],
                sigma=float(sigma), condition_dropped=drop,
                route_active=coefficient > 0, route_coefficient=coefficient,
                **values, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            del sample, evidence, noise

        grad, clip_scales = {}, {}
        by_role = {}
        for group in optimizer.param_groups:
            by_role.setdefault(group['role'], []).extend(group['params'])
        threshold = float(cfg['train']['grad_clip_per_group'])
        for role, parameters in by_role.items():
            parameters = [p for p in parameters if p.grad is not None]
            if not parameters:
                grad[role], clip_scales[role] = 0.0, 1.0
                continue
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, threshold, error_if_nonfinite=True)
            grad[role] = float(norm)
            clip_scales[role] = min(1.0, threshold / (float(norm) + 1e-6))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        log('train', update=update, phase=phase, multiplier=multiplier,
            route_coefficient=coefficient, cursor=cursor,
            seconds=time.monotonic() - start,
            **{key: value / cfg['train']['accumulation'] for key, value in sums.items()},
            gradients=grad, clip_scales=clip_scales,
            learning_rates={group['name']: group['lr'] for group in optimizer.param_groups},
            peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        _status(run, status='training', update=update, variant=cfg['variant_id'],
                cursor=cursor)
        if (update % cfg['train']['save_every'] == 0 or update in cfg['train']['keep_updates']
                or update == stop):
            model.assert_base_unchanged(base_before, full=False)
            save_checkpoint(model, optimizer, cfg, run, update, cursor,
                            provenance=provenance(cfg, report))
            log('checkpoint', update=update, path=str(Path(run) / 'latest.pt'))
        final = update

    model.assert_base_unchanged(base_before, full=False)
    status = 'complete' if final == total else 'paused'
    _status(run, status=status, update=final, variant=cfg['variant_id'],
            checkpoint=str(Path(run) / 'latest.pt'))
    log(status, update=final, checkpoint=str(Path(run) / 'latest.pt'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config, require_paths=not args.check_only)
    if args.check_only:
        print(json.dumps(audit(cfg), ensure_ascii=False, indent=2))
        return
    run = _run_dir(cfg, args.run_id)
    run.mkdir(parents=True, exist_ok=True)
    if args.resume is None and (run / 'latest.pt').exists():
        raise ValueError('Fresh initialization refuses an existing run; use --resume or a new run id')
    if args.resume is not None:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved.get('revision') != cfg['revision']:
            raise ValueError('Resume checkpoint revision differs')
    atomic_json(cfg, run / 'config.json')
    train(cfg, run, resume=args.resume, stop_after=args.stop_after)


if __name__ == '__main__':
    main()
