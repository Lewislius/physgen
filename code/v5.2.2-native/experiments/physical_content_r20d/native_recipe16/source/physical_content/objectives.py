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
    route = model.enhancement.routing_fidelity(content['raw'])
    objective = generation + route_weight * route
    if not torch.isfinite(objective):
        raise FloatingPointError('Nonfinite NoChunks generation/route objective')
    (objective / accumulation).backward()
    value = content['raw']['execution'].detach().float()[0]
    normed = torch.nn.functional.normalize(value, dim=-1)
    n = len(value)
    native = content['raw']['native'].detach().float()[0]
    difference = value-native
    write = torch.stack(list(model.write_statistics.values()))
    return dict(
        fm=float(parts['fm'].detach()),
        tail=float(parts['tail'].detach()),
        generation=float(generation.detach()),
        route=float(route.detach()),
        weighted_route=float(route_weight) * float(route.detach()),
        c_pair_cosine=float((normed.sum(0).square().sum() - n) / (n * (n - 1))),
        c_centered_energy=float(
            (value - value.mean(0)).square().sum() / value.square().sum().clamp_min(1e-20)),
        c_update_over_train_source_rms=float(difference.square().mean().sqrt()
                                            / model.enhancement.source_scale.clamp_min(1e-8)),
        physical_qk_relative_rms_mean=float(write.mean()),
        physical_qk_relative_rms_max=float(write.max()),
        first_frame_fm=float(
            ((prediction[:, :, :1] - (noise - sample['latent'])[:, :, :1])
             .square().mean()).detach()) if model.task == 't2v' else None,
    )
