"""One real-video generation objective; source statistics add no training loss."""
import torch

from .supervision import generation_loss


def backward_micro(model, evidence, sample, text, sigma, noise,
                   route_weight, accumulation):
    noisy = (1 - sigma) * sample['latent'] + sigma * noise
    content, groups = model.prepare_content(evidence, text)
    prediction, _ = model.velocity(
        noisy, sample.get('first'), text, sigma, content, groups, sample['times'])
    generation, parts = generation_loss(
        prediction, noise, sample['latent'], sample['valid'], model.task,
        **model.cfg['loss']['generation'])
    if route_weight != 0:
        raise ValueError('DT has no auxiliary route-prediction loss')
    route = generation.new_zeros(())
    objective = generation
    if not torch.isfinite(objective):
        raise FloatingPointError('Nonfinite NoChunks generation/route objective')
    (objective / accumulation).backward()
    value = content['raw']['execution'].detach().float()[0]
    normed = torch.nn.functional.normalize(value, dim=-1)
    n = len(value)
    source = content['raw']['source_weights'].detach().float()
    return dict(
        fm=float(parts['fm'].detach()),
        tail=float(parts['tail'].detach()),
        generation=float(generation.detach()),
        route=float(route.detach()),
        weighted_route=float(route_weight) * float(route.detach()),
        c_pair_cosine=float((normed.sum(0).square().sum() - n) / (n * (n - 1))),
        c_centered_energy=float(
            (value - value.mean(0)).square().sum() / value.square().sum().clamp_min(1e-20)),
        source_max_probability=float(source.amax(-1).mean()),
        source_entropy_nats=float(-(source * source.clamp_min(1e-20).log()).sum(-1).mean()),
        first_frame_fm=float(
            ((prediction[:, :, :1] - (noise - sample['latent'])[:, :, :1])
             .square().mean()).detach()) if model.task == 't2v' else None,
    )
