"""Task-parameterized flow matching and the four-scale top-25% tail."""
import math

import torch
from torch.nn import functional as F


def _error(prediction, noise, clean, task):
    target = noise - clean
    if task == 'i2v':
        prediction, target = prediction[:, :, 1:], target[:, :, 1:]
    elif task != 't2v':
        raise ValueError(task)
    return (prediction.float() - target.float()).square().mean(1)


def generation_loss(prediction, noise, clean, valid, task,
                    tail_weight=0.2, tail_fraction=0.25):
    error = _error(prediction, noise, clean, task)
    mask = valid.float()[None, None].expand_as(error)
    support_all = mask.sum()
    if not bool(support_all > 0):
        raise ValueError('Valid unknown-video region is empty')
    mean = (error * mask).sum() / support_all.clamp_min(1)
    if tail_weight == 0:
        return mean, dict(fm=mean, tail=mean)
    t, h, w = error.shape[-3:]
    windows = [
        (1, h, w),
        (min(4, t), h, w),
        (min(4, t), max(1, h // 5), max(1, w // 4)),
        (t, max(1, h // 5), max(1, w // 4)),
    ]
    terms = []
    for window in windows:
        weighted = F.avg_pool3d((error * mask)[:, None], window,
                                stride=window, ceil_mode=True)
        support = F.avg_pool3d(mask[:, None], window, stride=window, ceil_mode=True)
        values = (weighted / support.clamp_min(1e-8))[support > 0]
        if values.numel() == 0:
            raise ValueError(f'Empty valid support in tail window {window}')
        count = max(1, math.ceil(values.numel() * tail_fraction))
        terms.append(values.topk(count).values.mean())
    tail = torch.stack(terms).mean()
    return (1 - tail_weight) * mean + tail_weight * tail, dict(fm=mean, tail=tail)


def flow_matching(prediction, noise, clean, valid, task):
    error = _error(prediction, noise, clean, task)
    mask = valid.float()[None, None].expand_as(error)
    return (error * mask).sum() / mask.sum().clamp_min(1)
