import math

import torch

from .geometry import area_mask, content_rect


def weighted_mean(error, weights):
    weights = weights.to(error).expand_as(error)
    return (error * weights).sum() / weights.sum().clamp_min(1e-12)


def fm_loss(prediction, target, first_known):
    if first_known:
        prediction, target = prediction[:, :, 1:], target[:, :, 1:]
    return (prediction.float() - target.detach().float()).square().mean()


@torch.no_grad()
def correction_benefit(prediction, base, target, first_known):
    """Exact MSE decomposition; positive gain means better velocity, not physics."""
    if first_known:
        prediction, base, target = prediction[:, :, 1:], base[:, :, 1:], target[:, :, 1:]
    delta, residual = prediction.float()-base.float(), target.float()-base.float()
    energy = delta.square().mean()
    dot = (delta*residual).mean()
    denom = energy.sqrt()*residual.square().mean().sqrt()
    return dict(delta_energy=float(energy), twice_residual_dot_delta=float(2*dot),
                predicted_fm_gain=float(2*dot-energy),
                residual_cosine=float(dot/denom) if float(denom)>0 else None)


@torch.no_grad()
def structure_components(prediction, target, scale, weight):
    """Decompose global feature MSE without adding another training objective."""
    error = prediction.float()-target.float()
    static = error.mean(1, keepdim=True)
    dynamic = error-static
    def loss(value):
        return float(weighted_mean(value.square().mean(-1)[0], weight[None])/(scale.square()+1e-8))
    return dict(static_error=loss(static), dynamic_error=loss(dynamic))


def struct_loss(p15, target, scale, weight, instances=None):
    error = (p15.float() - target.detach().float()).square().mean(-1)[0]
    global_error = weighted_mean(error, weight.reshape(1, -1))
    local = []
    if instances is not None:
        for instance in instances:
            w = instance.float().reshape_as(error) * weight.reshape(1, -1)
            if bool(w.sum() > 0):
                local.append(weighted_mean(error, w))
    value = .5 * (global_error + torch.stack(local).mean()) if local else global_error
    return value / (scale.square() + 1e-8)


def struct_noise_weight(sigma):
    return ((1 - sigma) / .5).clamp(0, 1).square()


def generation_objective(loss_fm, loss_temp, accumulation=8, fm_weight=1., temp_weight=.05):
    """One temporal exposure per batch; only FM is averaged over microbatches."""
    return fm_weight * loss_fm / accumulation + temp_weight * loss_temp


def auxiliary_objective(loss_struct, sigma, accumulation=8, weight=.1):
    return weight * struct_noise_weight(sigma) * loss_struct / accumulation


def local_pair_specs(record, exposure):
    if not record.get("objects_reliable", False):
        return []
    interactions = record.get("interactions", []) or [record["motion_peak"]]
    centre = interactions[exposure % len(interactions)]
    key = set(round(i * 119 / 7) for i in range(8))
    begin = min(104, max(0, int(centre) - 8))
    key.update(range(begin, begin + 16))
    h, w = record["geometry"]["h"], record["geometry"]["w"]
    specs = []
    for obj in record.get("objects", []):
        if not obj.get("reliable", False):
            continue
        boxes = obj["boxes"]
        for pair in sorted(key):
            a, b = boxes[pair], boxes[pair + 1]
            # A missing/occluded frame has no reliable local target; global differences remain.
            if a is None or b is None:
                continue
            x0, y0, x1, y1 = min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
            dx, dy = .25 * (x1 - x0), .25 * (y1 - y0)
            box = [max(0, math.floor(x0 - dx)), max(0, math.floor(y0 - dy)),
                   min(w, math.ceil(x1 + dx)), min(h, math.ceil(y1 + dy))]
            if box[2] > box[0] and box[3] > box[1]:
                specs.append(dict(object_id=obj["id"], pair=pair, box=box))
    return specs


def temporal_loss(prediction, reference, geo, specs):
    p, r = prediction["global_rgb"], reference["global_rgb"].detach()
    if p.shape[0] != geo.get("frames", 121) or r.shape != p.shape:
        raise ValueError("Temporal objective must cover the complete decoded source window")
    h, w = geo["h"], geo["w"]
    mask = area_mask(content_rect(geo), h, w, p.shape[-2], p.shape[-1], p.device)
    error = (p.diff(dim=0) - r.diff(dim=0)).abs()
    global_error = weighted_mean(error, mask[None, None])
    by_object = {}
    rect = content_rect(geo)
    for i, spec in enumerate(specs):
        pa, pb = prediction["crops"][2 * i], prediction["crops"][2 * i + 1]
        ra, rb = reference["crops"][2 * i].detach(), reference["crops"][2 * i + 1].detach()
        x0, y0, x1, y1 = spec["box"]
        local_rect = [rect[0] - x0, rect[1] - y0, rect[2] - x0, rect[3] - y0]
        local_mask = area_mask(local_rect, y1 - y0, x1 - x0, pa.shape[-2], pa.shape[-1], pa.device)
        by_object.setdefault(spec["object_id"], []).append(weighted_mean(((pb - pa) - (rb - ra)).abs(), local_mask[None]))
    if by_object:
        local = torch.stack([torch.stack(errors).mean() for errors in by_object.values()]).mean()
        return .5 * (global_error + local)
    return global_error


def v4_struct_coefficient(sigma, step, weights):
    """Limit the coefficient, not the raw loss; step is one-based here."""
    warmup = min(1., step / weights["struct_warmup_steps"])
    noise = ((1 - sigma) / weights["struct_noise_width"]).clamp(0, 1)
    return weights["struct"] * warmup * noise
