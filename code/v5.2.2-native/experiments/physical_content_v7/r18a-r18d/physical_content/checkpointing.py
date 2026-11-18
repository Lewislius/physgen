"""Fresh-only initialization and same-run resume, with no old project weights."""
from pathlib import Path
import os
import random

import torch

from physical_scaling.common import atomic_torch, cpu_tree
from .common import canonical_hash, design_hash, sha256_file


def save_checkpoint(model, optimizer, cfg, run, update, cursor, provenance=None):
    if not 1 <= update <= cfg['train']['max_updates']:
        raise ValueError('Checkpoint update is outside the configured run')
    named = dict(model.named_parameters())
    if set(model.learned_names) - set(named):
        raise KeyError('Checkpoint parameter ownership changed')
    state = dict(
        revision=cfg['revision'],
        variant=cfg['variant'],
        variant_id=cfg['variant_id'],
        config=cfg,
        design_hash=design_hash(cfg),
        update=update,
        step=update,
        cursor=cursor,
        model={name: named[name].detach().cpu() for name in model.learned_names},
        constants=model.constant_state(),
        optimizer=cpu_tree(optimizer.state_dict()),
        rng=dict(
            python=random.getstate(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        ),
        video_lora_targets=list(model.video_lora_targets),
        parameter_counts=model.trainable_count(),
        provenance=provenance or {},
        base_provenance=model.base_bank.checkpoint_provenance(cfg['paths']['wan_checkpoint']),
    )
    latest = Path(run) / 'latest.pt'
    atomic_torch(state, latest)
    if update in cfg['train']['keep_updates'] or update == cfg['train']['max_updates']:
        target = Path(run) / f'step{update:04d}.pt'
        if target.exists():
            target.unlink()
        os.link(latest, target)
    return latest


def restore_checkpoint(model, optimizer, path, cfg):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('revision') != cfg['revision']:
        raise ValueError('Resume is not an R18 checkpoint for this run')
    if state.get('variant') != cfg['variant'] or state.get('design_hash') != design_hash(cfg):
        raise ValueError('Resume requires the same R18 variant and exact design/config')
    if int(state['update']) > cfg['train']['max_updates']:
        raise ValueError('A run reaching the configured maximum cannot be resumed further')
    named = dict(model.named_parameters())
    if set(state['model']) != set(model.learned_names):
        raise ValueError('Checkpoint parameter keys/ownership differ from this model')
    with torch.no_grad():
        for name, value in state['model'].items():
            if named[name].shape != value.shape:
                raise ValueError(f'Checkpoint shape differs: {name}')
            named[name].copy_(value)
        buffers = dict(model.named_buffers())
        if set(state['constants']) != {name for name in buffers if name.startswith('enhancement.')}:
            raise ValueError('Checkpoint constants differ from fresh enhancement buffers')
        for name, value in state['constants'].items():
            buffers[name].copy_(value)
    optimizer.load_state_dict(state['optimizer'])
    if torch.cuda.is_available() and state['rng'].get('cuda') is not None:
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
    random.setstate(state['rng']['python'])
    torch.set_rng_state(state['rng']['torch'])
    if state.get('video_lora_targets') != list(model.video_lora_targets):
        raise ValueError('Video LoRA target map changed across resume')
    return int(state['update']), int(state['cursor']), state.get('provenance', {})


def provenance(cfg, audit_report):
    return dict(
        initialization='official Wan2.2-TI2V-5B fresh; no project checkpoint',
        interface_hash=audit_report['interface_hash'],
        train_manifest_hash=audit_report['train_manifest_hash'],
        wan_index_hash=audit_report['wan_index_hash'],
        qwen_index_hash=audit_report['qwen_index_hash'],
        data_revision=cfg['train'].get('data_revision'),
        config_hash=canonical_hash(cfg),
        no_project_checkpoint=True,
    )
