"""Train only the official Wan2.2 video-LoRA on the exact R20 fixed1600 draws."""
from pathlib import Path
import argparse
import copy
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from physgen_native.context_alignment import native_text_context
from physgen_native.generator import load_wan, sinusoidal_embedding_1d

from native_control.common import (atomic_json, atomic_torch, canonical_hash,
                                   data_audit, gpu_guard_amp48, load_yaml,
                                   validate_lora_config)
from native_control.data import TrainingData
from native_control.lora import (assert_native_text_frozen, install_video_lora,
                                 video_lora_parameters)
from native_control.supervision import generation_loss
from native_control.wan_loader import ensure_wan_namespace


def _status(run, **value):
    atomic_json(value, run / 'status.json')


def _log(log_path, **value):
    value = dict(time=time.time(), **value)
    with Path(log_path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def _lr_multiplier(update, cfg):
    t = cfg['train']
    if update <= int(t['warmup_updates']):
        return update / int(t['warmup_updates'])
    if update <= int(t['peak_end_update']):
        return 1.0
    minimum = float(t['min_lr_ratio'])
    total = int(t['max_updates'])
    progress = min(1.0, (update - int(t['peak_end_update']))
                   / max(1, total - int(t['peak_end_update'])))
    return minimum + (1.0 - minimum) * .5 * (1.0 + math.cos(math.pi * progress))


def _velocity(wan, noisy, text, sigma, checkpointing=True):
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
        args = (x, modulation, lengths, grid, wan.freqs, context, None)
        x = (checkpoint(block, *args, use_reentrant=False)
             if checkpointing and torch.is_grad_enabled() else block(*args))
    return wan.unpatchify(wan.head(x, e), grid)[0][None].float()


def _save(wan, optimizer, cfg, run, update, cursor, audit):
    lora = {name: value.detach().cpu()
            for name, value in video_lora_parameters(wan)}
    state = dict(
        format='native-wan22-video-lora-v1',
        revision=cfg['revision'], update=update, cursor=cursor,
        config=copy.deepcopy(cfg), lora=lora,
        target_names=install_target_names(cfg),
        optimizer=optimizer.state_dict(), audit=audit,
        provenance=dict(
            initialization='official Wan2.2-TI2V-5B; video LoRA only',
            no_project_checkpoint=True,
            train_manifest_hash=audit['train_manifest_hash'],
            sample_sequence_hash=audit['sample_sequence_hash'],
            wan_index_hash=audit['wan_index_hash'],
            data_revision=audit['data_revision'],
            config_hash=canonical_hash(cfg),
        ),
    )
    latest = run / 'latest.pt'
    atomic_torch(state, latest)
    if update in cfg['train']['keep_updates'] or update == cfg['train']['max_updates']:
        target = run / f'step{update:04d}.pt'
        if target.exists():
            target.unlink()
        os.link(latest, target)


def install_target_names(cfg):
    # This is the fixed Wan2.2 video target allowlist, independent of weights.
    names = []
    for block in range(30):
        for name in ('self_attn.q', 'self_attn.k', 'self_attn.v',
                     'self_attn.o', 'ffn.0', 'ffn.2'):
            for suffix in ('lora_a', 'lora_b'):
                names.append(f'blocks.{block}.{name}.{suffix}')
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    cfg = validate_lora_config(load_yaml(args.config))
    run = Path(cfg['paths']['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    if args.resume is None and (run / 'latest.pt').exists():
        raise RuntimeError(f'Fresh LoRA run refuses existing checkpoint: {run / "latest.pt"}')
    audit = data_audit(cfg)
    _status(run, status='auditing', update=0, variant='wan22_video_lora400')
    gpu_guard_amp48(run)
    atomic_json(cfg, run / 'config.json')
    random.seed(cfg['train']['seed'])
    np.random.seed(cfg['train']['seed'])
    torch.manual_seed(cfg['train']['seed'])
    torch.cuda.manual_seed_all(cfg['train']['seed'])
    torch.set_num_threads(4)

    ensure_wan_namespace(cfg)
    wan = load_wan(cfg, 'cuda')
    wan.requires_grad_(False).eval()
    targets = install_video_lora(wan, cfg)
    assert_native_text_frozen(wan)
    if len(targets) != 180 or len(set(targets)) != 180:
        raise AssertionError('Video LoRA target count differs')
    parameters = [value for _, value in video_lora_parameters(wan)]
    if not parameters:
        raise RuntimeError('No trainable video LoRA parameters')
    optimizer = torch.optim.AdamW(
        parameters, lr=0.0, betas=tuple(cfg['train']['betas']),
        eps=float(cfg['train']['eps']), weight_decay=0.0, foreach=False)
    data = TrainingData(cfg)
    null = torch.load(Path(cfg['paths']['cache']) / 'base/null_text.pt',
                      map_location='cpu', weights_only=True).float()[None].cuda()
    completed, cursor = 0, 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved.get('format') != 'native-wan22-video-lora-v1':
            raise ValueError('Resume is not a native Wan2.2 LoRA checkpoint')
        if saved.get('revision') != cfg['revision']:
            raise ValueError('Resume revision differs')
        named = dict(wan.named_parameters())
        if set(saved['lora']) != {name for name, _ in video_lora_parameters(wan)}:
            raise ValueError('Resume LoRA parameter names differ')
        with torch.no_grad():
            for name, value in saved['lora'].items():
                named[name].copy_(value.to(device=named[name].device))
        optimizer.load_state_dict(saved['optimizer'])
        completed, cursor = int(saved['update']), int(saved['cursor'])
        _log(run / 'train.jsonl', kind='resumed', update=completed,
             cursor=cursor, checkpoint=str(args.resume))
    if completed == int(cfg['train']['max_updates']):
        _status(run, status='complete', update=completed,
                variant='wan22_video_lora400', checkpoint=str(run / 'latest.pt'))
        _log(run / 'train.jsonl', kind='complete', update=completed,
             reason='already at maximum')
        return

    _log(run / 'train.jsonl', kind='audit', **audit)
    _log(run / 'train.jsonl', kind='model_ready',
         parameters=sum(p.numel() for p in parameters),
         targets=len(targets), base_model=cfg['base']['model'],
         initialization='official Wan2.2-TI2V-5B; video LoRA only')
    _status(run, status='training', update=completed,
            variant='wan22_video_lora400', cursor=cursor)
    order = iter(range(cursor, len(data)))
    total = int(cfg['train']['max_updates'])
    final = completed
    for update in range(completed + 1, total + 1):
        lr = float(cfg['train']['peak_lr']) * _lr_multiplier(update, cfg)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        sums = dict(fm=0.0, tail=0.0, generation=0.0)
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for micro in range(int(cfg['train']['gradient_accumulation'])):
            sample = data.get(next(order))
            draws = torch.rand(
                2, generator=torch.Generator().manual_seed(
                    cfg['train']['seed'] * 1000003 + cursor))
            drop = bool(draws[0] < float(cfg['train']['condition_dropout']))
            sigma = (float(cfg['train']['sigma_min'])
                     + (float(cfg['train']['sigma_max'])
                        - float(cfg['train']['sigma_min'])) * draws[1]).to('cuda')
            text = null if drop else sample['text']
            noise = torch.randn(
                sample['latent'].shape, device='cuda',
                generator=torch.Generator(device='cuda').manual_seed(
                    cfg['train']['seed'] * 2000003 + cursor))
            noisy = (1 - sigma) * sample['latent'] + sigma * noise
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                prediction = _velocity(wan, noisy, text, sigma,
                                       checkpointing=bool(cfg['train']['gradient_checkpointing']))
                generation, parts = generation_loss(
                    prediction, noise, sample['latent'], sample['valid'], 't2v',
                    tail_weight=float(cfg['loss']['tail']),
                    tail_fraction=float(cfg['loss']['tail_fraction']))
            if not torch.isfinite(generation):
                raise FloatingPointError('Nonfinite native LoRA generation loss')
            (generation / int(cfg['train']['gradient_accumulation'])).backward()
            sums['fm'] += float(parts['fm'].detach())
            sums['tail'] += float(parts['tail'].detach())
            sums['generation'] += float(generation.detach())
            cursor += 1
            _log(run / 'train.jsonl', kind='micro', update=update,
                 micro=micro + 1, sample=sample['id'], sigma=float(sigma),
                 condition_dropped=drop,
                 fm=float(parts['fm'].detach()),
                 tail=float(parts['tail'].detach()),
                 generation=float(generation.detach()),
                 peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            del sample, noise, noisy, prediction
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(cfg['train']['grad_clip']),
            error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        _log(run / 'train.jsonl', kind='train', update=update,
             phase='video_lora_only', lr=lr, cursor=cursor,
             seconds=time.monotonic() - start,
             fm=sums['fm'] / int(cfg['train']['gradient_accumulation']),
             tail=sums['tail'] / int(cfg['train']['gradient_accumulation']),
             generation=sums['generation'] / int(cfg['train']['gradient_accumulation']),
             grad_norm=float(norm),
             peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        _status(run, status='training', update=update,
                variant='wan22_video_lora400', cursor=cursor)
        if update % int(cfg['train']['save_every']) == 0 or update == total:
            allowed = {id(p) for p in parameters}
            if any(p.requires_grad and id(p) not in allowed
                   for p in wan.parameters()):
                raise AssertionError('A non-LoRA Wan parameter became trainable')
            _save(wan, optimizer, cfg, run, update, cursor, audit)
            _log(run / 'train.jsonl', kind='checkpoint', update=update,
                 path=str(run / 'latest.pt'))
        final = update

    _status(run, status='complete', update=final,
            variant='wan22_video_lora400', checkpoint=str(run / 'latest.pt'))
    _log(run / 'train.jsonl', kind='complete', update=final,
         checkpoint=str(run / 'latest.pt'))


if __name__ == '__main__':
    main()
