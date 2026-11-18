"""The canonical optimizer and raw-route schedules for the 400/1000 runs.

400-update run (1-based optimizer updates):
S1 1-32     LR warmup; route rises linearly to 0.03
S2 33-128   peak LR; full route 0.03
S3 129-320  cosine LR decay to 0.2x; route decays 0.03 -> 0.01
S4 321-400  constant 0.2x LR; route 0.01

1000-update run (the v9 design):
S1 1-64     LR warmup; route rises then holds
S2 65-200   peak LR; route decays toward zero
S3 201-320  cosine LR decay starts; route reaches zero at 320
S4 321-1000 cosine LR continues to 0.2x; no route auxiliary
"""

import math

from .common import ROUTE_GROUPS


def _check(update, max_updates):
    if max_updates not in (400, 1000):
        raise ValueError('Only the 400- and 1000-update schedules exist')
    if not 1 <= update <= max_updates:
        raise ValueError(f'Optimizer update must be in 1..{max_updates}')


def lr_multiplier(update, max_updates=400):
    _check(update, max_updates)
    if max_updates == 400:
        if update <= 32:
            return update / 32
        if update <= 128:
            return 1.0
        if update <= 320:
            return 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * (update - 128) / 192))
        return 0.2
    if update <= 64:
        return update / 64
    if update <= 200:
        return 1.0
    return 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * (update - 200) / 800))


def route_coefficient(update, max_updates=400):
    """Raw-C route auxiliary; it never switches to a different loss term."""
    _check(update, max_updates)
    if max_updates == 400:
        if update <= 32:
            return 0.03 * update / 32
        if update <= 128:
            return 0.03
        if update <= 320:
            return 0.03 - 0.02 * (update - 128) / 192
        return 0.01
    if update <= 32:
        return 0.03 * min(update / 32, 1.0)
    if update <= 64:
        return 0.03
    if update < 320:
        return 0.03 * (320 - update) / (320 - 64)
    return 0.0


def set_lrs(optimizer, cfg, update):
    max_updates = int(cfg['train']['max_updates'])
    multiplier = lr_multiplier(update, max_updates)
    peak = cfg['train']['peak_lrs']
    for group in optimizer.param_groups:
        role = group['role']
        if role not in ROUTE_GROUPS:
            raise ValueError(f'Unknown optimizer role {role}')
        group['lr'] = peak[role] * multiplier
    return multiplier


def learning_rates(cfg, update):
    max_updates = int(cfg['train']['max_updates'])
    multiplier = lr_multiplier(update, max_updates)
    return {role: cfg['train']['peak_lrs'][role] * multiplier for role in ROUTE_GROUPS}
