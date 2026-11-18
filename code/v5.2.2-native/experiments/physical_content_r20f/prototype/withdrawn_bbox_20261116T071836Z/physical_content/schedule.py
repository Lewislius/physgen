"""Compressed 400-step optimizer and raw-route schedules for R18E-H.

The v10 1000-step phase boundaries are scaled by 400/1000 = 0.4 and rounded to
integer optimizer updates:

- video-LoRA warmup: 51
- physics warmup: 13
- peak phase: through update 240
- cosine LR decay: 241..400, to 0.2x
- route warmup: 13; hold to 240; linear decay to 0.015 at 400
"""
import math

from .common import ROUTE_GROUPS


def _check(update, cfg):
    maximum = int(cfg['train']['max_updates'])
    if not 1 <= update <= maximum:
        raise ValueError(f'Optimizer update must be in 1..{maximum}')


def lr_multiplier(role, update, cfg):
    _check(update, cfg)
    train = cfg['train']
    if role == 'wan_video_lora':
        offset = int(train.get('interface_updates', 0))
        if update <= offset:
            return 0.0
        local_update = update - offset
    else:
        local_update = update
    warmup = (train['video_lora_warmup'] if role == 'wan_video_lora'
              else train['physics_warmup'])
    if local_update <= warmup:
        return local_update / warmup
    peak_end = int(train['peak_end_update'])
    total = int(train['max_updates'])
    if update <= peak_end or total <= peak_end:
        return 1.0
    minimum = float(train['minimum_lr_ratio'])
    progress = min(1.0, (update - peak_end) / (total - peak_end))
    return minimum + (1.0 - minimum) * .5 * (1.0 + math.cos(math.pi * progress))


def route_coefficient(update, cfg):
    _check(update, cfg)
    loss = cfg['loss']
    peak = float(loss['route_peak'])
    warmup = int(loss['route_warmup'])
    decay_start = int(loss['route_decay_start'])
    final = float(loss['route_final'])
    total = int(cfg['train']['max_updates'])
    if update <= warmup:
        return peak * update / warmup
    if update <= decay_start:
        return peak
    if update >= total:
        return final
    return peak - (peak - final) * (update - decay_start) / (total - decay_start)


def region_coefficient(update, cfg):
    _check(update, cfg)
    return float(cfg['loss']['region_peak']) * min(1., update / cfg['loss']['region_warmup'])


def set_lrs(optimizer, cfg, update):
    for group in optimizer.param_groups:
        role = group['role']
        if role not in ROUTE_GROUPS:
            raise ValueError(f'Unknown optimizer role {role}')
        group['lr'] = cfg['train']['peak_lrs'][role] * lr_multiplier(role, update, cfg)
    return {role: lr_multiplier(role, update, cfg) for role in ROUTE_GROUPS}


def learning_rates(cfg, update):
    return {role: cfg['train']['peak_lrs'][role] * lr_multiplier(role, update, cfg)
            for role in ROUTE_GROUPS}
