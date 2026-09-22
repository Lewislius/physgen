import torch
import torch.nn.functional as F


def noisy_view(clean, first, noise, sigma):
    noisy = (1 - sigma) * clean + sigma * noise
    return torch.cat((first, noisy[:, :, 1:]), dim=2)


def endpoint(noisy, prediction, first, sigma):
    clean = noisy - sigma * prediction
    return torch.cat((first, clean[:, :, 1:]), dim=2)


def weighted_mean(values, weight):
    return (values * weight).sum() / weight.sum()


def feature_distance(predicted, target, weight):
    return weighted_mean(1 - F.cosine_similarity(predicted.float(), target.float(), dim=-1), weight)


def prior_supervision_enabled(config):
    return (config["corrector"].get("initialization") in ("image_text", "image_text_gt")
            and config["loss"].get("prior_weight", 0) > 0)


def prior_distance(prior, target, weight, time_tokens):
    """Fixed spatial pooling per teacher time slot; no learned target/readout projection."""
    batch, _, width = prior.shape
    weights = weight.float().reshape(batch, time_tokens, -1)
    mass = weights.sum(2)
    def pool(tokens):
        values = tokens.float().reshape(batch, time_tokens, -1, width)
        return (values * weights.unsqueeze(-1)).sum(2) / mass.clamp_min(1e-8).unsqueeze(-1)
    distance = (pool(prior) - pool(target.detach())).square().mean(-1)
    return (distance * mass).sum() / mass.sum().clamp_min(1e-8)


def process_losses(info, target, weight):
    losses = {}
    metrics = dict(info["metrics"])
    target = target.detach().float()
    metrics["P/target_rms"] = weighted_mean(target.square().mean(-1), weight).sqrt()
    metrics["P/content_fraction"] = weight.mean()
    metrics["P/initial_mse"] = weighted_mean((info["initial"].float() - target).square().mean(-1), weight)
    # Per-video constant target baseline: diagnostic only, never a student input.
    average_target = (target * weight.unsqueeze(-1)).sum(1, keepdim=True) / weight.sum(1, keepdim=True).unsqueeze(-1)
    metrics["P/constant_target_mse"] = weighted_mean((target - average_target).square().mean(-1), weight)
    for name, state in info["states"].items():
        current = weighted_mean((state.float() - target).square().mean(-1), weight)
        previous = weighted_mean((info["before"][name].float() - target).square().mean(-1), weight)
        family = name.split("@", 1)[0]
        losses.setdefault(family, []).append(current)
        metrics[f"{name}/struct_mse"] = current.detach()
        metrics[f"{name}/struct_mse_before"] = previous.detach()
        metrics[f"{name}/struct_gain"] = previous.detach() - current.detach()
        metrics[f"{name}/target_cosine"] = 1 - feature_distance(state.detach(), target, weight)
        metrics[f"{name}/remaining_target_rms"] = current.detach().sqrt()
    # Increasing A's invocation count must not divide B's loss contribution by that count.
    family_losses = [torch.stack(values).mean() for values in losses.values()]
    return torch.stack(family_losses).mean(), metrics


def training_loss(prediction, noise, clean, first, noisy, sigma, info, target, target_weight,
                  step, config, output_active, decoder=None, teacher=None, *, write_loss=None,
                  return_terms=False):
    cfg = config["loss"]
    fm = F.mse_loss(prediction[:, :, 1:].float(), (noise - clean)[:, :, 1:].float())
    struct, metrics = process_losses(info, target, target_weight)
    noise_weight = ((1 - sigma) / cfg["struct_noise_width"]).clamp(0, 1)
    warmup = min(1.0, (step + 1) / cfg["struct_warmup_steps"])
    weighted_fm = cfg["fm_weight"] * fm
    weighted_struct = cfg["struct_weight"] * warmup * noise_weight * struct
    prior = prediction.new_zeros(())
    prior_active = prior_supervision_enabled(config) and info.get("prior") is not None
    if prior_active:
        prior = prior_distance(info["prior"], target, target_weight, info["prior_time_tokens"])
    # P0 has no sigma input: its auxiliary target must not vanish at the first sampling step.
    weighted_prior = cfg.get("prior_weight", 0.0) * warmup * prior
    out = prediction.new_zeros(())
    if output_active:
        decoded = decoder(endpoint(noisy, prediction, first, sigma))
        predicted_target = teacher(decoded, config["data"]["teacher_frames"])
        out = feature_distance(predicted_target, target, target_weight)
        metrics["output/pixel_rms"] = decoded.detach().float().square().mean().sqrt()
        metrics["output/clamped_fraction"] = (decoded.detach().abs() >= 1).float().mean()
    weighted_out = cfg["out_weight"] * out
    write = prediction.new_zeros(()) if write_loss is None else write_loss
    weighted_write = cfg.get("write_weight", 0.0) * warmup * noise_weight * write
    total = weighted_fm + weighted_struct + weighted_prior + weighted_write + weighted_out
    metrics.update({"loss/fm": fm.detach(), "loss/struct": struct.detach(), "loss/out": out.detach(),
                    "loss/weighted_fm": weighted_fm.detach(), "loss/weighted_struct": weighted_struct.detach(),
                    "loss/weighted_out": weighted_out.detach(), "loss/total": total.detach(),
                    "loss/prior": prior.detach(), "loss/weighted_prior": weighted_prior.detach(),
                    "prior/active": float(prior_active),
                    "prior/gt_conditioned": float(info.get("prior_gt_conditioned", False)),
                    "loss/struct_noise_weight": noise_weight.detach(), "loss/struct_warmup": warmup,
                    "output/active": float(output_active), "sigma": sigma.detach()})
    metrics.update({"loss/write": write.detach(), "loss/weighted_write": weighted_write.detach(),
                    "write/active": float(write_loss is not None),
                    "loss/write_coefficient": cfg.get("write_weight", 0.0) * warmup * noise_weight.detach()})
    if return_terms:
        return total, metrics, dict(fm=weighted_fm, struct=weighted_struct, prior=weighted_prior,
                                   write=weighted_write)
    return total, metrics
