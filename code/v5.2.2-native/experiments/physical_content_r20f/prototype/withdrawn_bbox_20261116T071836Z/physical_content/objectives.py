"""Real-video generation plus supervised relevance of the executed C read."""
import torch

from .supervision import generation_loss


def backward_micro(model, evidence, sample, text, sigma, noise,
                   region_weight, accumulation):
    noisy = (1 - sigma) * sample['latent'] + sigma * noise
    content, groups = model.prepare_content(evidence, text)
    target = sample.get('physical_target') if evidence.get('caption', 'positive') else None
    prediction, region = model.velocity(
        noisy, sample.get('first'), text, sigma, content, groups, sample['times'],
        physical_target=target)
    generation, parts = generation_loss(
        prediction, noise, sample['latent'], sample['valid'], model.task,
        **model.cfg['loss']['generation'])
    # At high noise the actual spatial placement is unobservable. Do not ask
    # the reader to reconstruct a clean-video layout from pure random noise.
    weighted_region = region_weight * (1-sigma).square() * region
    objective = generation + weighted_region
    if not torch.isfinite(objective):
        raise FloatingPointError('Nonfinite whole-video generation/localization objective')
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
        region=float(region.detach()),
        weighted_region=float(weighted_region.detach()),
        c_pair_cosine=float((normed.sum(0).square().sum() - n) / (n * (n - 1))),
        c_centered_energy=float(
            (value - value.mean(0)).square().sum() / value.square().sum().clamp_min(1e-20)),
        c_update_over_train_source_rms=float(difference.square().mean().sqrt()
                                            / model.enhancement.source_scale.clamp_min(1e-8)),
        write_relative_rms_mean=float(write.mean()),
        write_relative_rms_max=float(write.max()),
        first_frame_fm=float(
            ((prediction[:, :, :1] - (noise - sample['latent'])[:, :, :1])
             .square().mean()).detach()) if model.task == 't2v' else None,
    )
