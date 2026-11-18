"""Matched R16G generation plus routing alignment, with T2V full-frame loss."""
import torch

from .supervision import generation_loss


def backward_sample(model, evidence, sample, text, sigma, noise, coefficient,
                    accumulation):
    noisy = (1 - sigma) * sample['latent'] + sigma * noise
    # T2V: no first latent and all frames are unknown.
    content, groups = model.enhancement(evidence)
    prediction, _ = model.velocity(noisy, None, text, sigma, content,
                                   sample['times'], groups)
    generation, parts = generation_loss(
        prediction, noise, sample['latent'], sample['valid'],
        known_frames=0, **model.cfg['loss']['generation'])
    auxiliary = model.enhancement.content_loss(content, evidence)
    objective = generation + coefficient * auxiliary
    if not torch.isfinite(objective):
        raise FloatingPointError('Nonfinite full generation objective')
    (objective / accumulation).backward()
    value = content['execution'].detach().float()[0]
    normed = torch.nn.functional.normalize(value, dim=-1)
    n = len(value)
    return dict(fm=float(parts['fm'].detach()), tail=float(parts['tail'].detach()),
                generation=float(generation.detach()), aux=float(auxiliary.detach()),
                weighted_aux=float(coefficient) * float(auxiliary.detach()),
                route_aux=float(auxiliary.detach()),
                weighted_route_aux=float(coefficient) * float(auxiliary.detach()),
                c_pair_cosine=float((normed.sum(0).square().sum() - n) / (n * (n - 1))),
                c_centered_energy=float(
                    (value - value.mean(0)).square().sum()
                    / value.square().sum().clamp_min(1e-20)))
