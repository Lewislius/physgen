"""Flow-matching objective with an explicit known-frame boundary.

R15Bt is T2V: ``known_frames`` is always zero and every latent frame is trained.
The function still accepts a nonzero boundary so the I2V R15B behavior can be
checked in CPU tests without maintaining a second copy of the loss.
"""
import math
import torch
from torch.nn import functional as F


def generation_loss(prediction, noise, clean, valid, tail_weight=.2,
                    tail_fraction=.25, known_frames=0):
    error = (prediction[:, :, known_frames:].float()
             - (noise - clean)[:, :, known_frames:].float()).square().mean(1)
    mask = valid.float()[None, None].expand_as(error)
    mean = (error * mask).sum() / mask.sum().clamp_min(1)
    if tail_weight == 0:
        return mean, dict(fm=mean, tail=mean)
    t, h, w = error.shape[-3:]
    windows = [(1, h, w), (min(4, t), h, w),
               (min(4, t), max(1, h // 5), max(1, w // 4)),
               (t, max(1, h // 5), max(1, w // 4))]
    terms = []
    for window in windows:
        weighted = F.avg_pool3d((error * mask)[:, None], window, stride=window,
                                ceil_mode=True)
        support = F.avg_pool3d(mask[:, None], window, stride=window, ceil_mode=True)
        values = (weighted / support.clamp_min(1e-8))[support > 0]
        count = max(1, math.ceil(len(values) * tail_fraction))
        terms.append(values.topk(count).values.mean())
    tail = torch.stack(terms).mean()
    return (1 - tail_weight) * mean + tail_weight * tail, dict(fm=mean, tail=tail)


def flow_matching(prediction, noise, clean, valid, known_frames=0):
    error = (prediction[:, :, known_frames:].float()
             - (noise - clean)[:, :, known_frames:].float()).square().mean(1)
    return ((error * valid.float()[None]).sum()
            / (valid.float().sum() * error.shape[0] * error.shape[1]).clamp_min(1))
