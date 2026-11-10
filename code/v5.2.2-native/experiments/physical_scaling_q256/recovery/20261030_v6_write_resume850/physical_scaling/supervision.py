"""The three objectives retain distinct target meanings and gradient paths."""
import torch
from torch.nn import functional as F


def slot_groups(count, relations, device=None):
    if not 1 <= relations <= min(4, count):
        raise ValueError('Need one to four input-supported relations')
    return torch.arange(count, device=device) * relations // count


def group_log_prior(groups, dtype=torch.float32):
    # Correct unequal slot cardinalities: each relation and reference have equal
    # total prior mass at equal logits; the reference has exactly zero value.
    counts = torch.bincount(groups).float()
    return torch.cat((-counts[groups].log(), counts.new_zeros(1))).to(dtype)


def association_target(relation, background, temperature=1., eps=1e-8):
    """Inputs [...,R], [...]; output [...,1+R] with reference category first."""
    delta = (relation.float().clamp_min(eps).log()
             - background.float().clamp_min(eps).log()[...,None])
    return torch.cat((torch.zeros_like(delta[...,:1]), delta), -1).div(temperature).softmax(-1)


def write_support(relation, background):
    """V6 draft: positive background contrast, per-relation max, overlap sum cap."""
    value = (relation.float() - background.float()[...,None]).clamp_min(0)
    maxima = value.amax(dim=-2, keepdim=True)
    value = value / maxima.clamp_min(1e-12)
    return value / value.sum(-1, keepdim=True).clamp_min(1)


def grouped_probabilities(attention, groups):
    """Actual attention [B,heads,S,N+1] -> [B,S,1+R], independent of Q count."""
    mean = attention.float().mean(1)
    parts = [mean[...,-1:]]
    for r in range(int(groups.max()) + 1):
        parts.append(mean[...,:-1][...,groups == r].sum(-1, keepdim=True))
    return torch.cat(parts, -1)


def association_kl(student, target, valid):
    target = target.detach().float()
    terms = torch.where(target > 0, target*(target.clamp_min(1e-8).log()-student.float().clamp_min(1e-8).log()), 0).sum(-1)
    return (terms * valid).sum() / valid.sum().clamp_min(1)


def write_regression(delta, target, valid):
    # Deliberately reproduces V6 draft, including teacher-determined amplitude.
    # This is an experimental synthetic residual, not physical ground truth.
    target = target.detach().float()
    denom = valid.sum().clamp_min(1)*target.shape[-1]
    scale = ((target.square()*valid[...,None]).sum()/denom).sqrt().clamp_min(1e-3)
    error = F.smooth_l1_loss((delta.float()-target)/scale, torch.zeros_like(target), reduction='none')
    return (error * valid[...,None]).sum()/denom


def flow_matching(prediction, noise, clean, valid):
    error = (prediction[:,:,1:].float()-(noise-clean)[:,:,1:].float()).square().mean(1)
    return (error*valid).sum()/(valid.sum()*error.shape[0]*error.shape[1]).clamp_min(1)


def resample_maps(maps, observed_times, target_times, height, width):
    """[R,T,H,W] weak evidence, interpolated in actual presentation seconds."""
    r,t,h,w = maps.shape
    spatial = F.interpolate(maps.reshape(r*t,1,h,w).float(), size=(height,width), mode='bilinear', align_corners=False).reshape(r,t,height,width)
    observed_times, target_times = observed_times.float(), target_times.float()
    right = torch.searchsorted(observed_times, target_times).clamp(1,t-1)
    left = right-1
    alpha = ((target_times-observed_times[left])/(observed_times[right]-observed_times[left]).clamp_min(1e-8)).clamp(0,1)
    value = spatial[:,left]*(1-alpha[None,:,None,None])+spatial[:,right]*alpha[None,:,None,None]
    return value.permute(1,2,3,0).reshape(-1,r)
