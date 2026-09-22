"""Exactly FM + STRUCT + Flow-Oracle. No PRIOR or perturbed-input WRITE."""
import torch
import torch.nn.functional as F


def noisy_view(clean, first, noise, sigma):
    noisy = (1 - sigma) * clean.float() + sigma * noise.float()
    return torch.cat((first.float(), noisy[:, :, 1:]), dim=2) if first is not None else noisy


def flow_target(clean, noise):
    return noise.detach().float() - clean.detach().float()


def flow_mse(prediction, target, first_known=True):
    start = 1 if first_known else 0
    if prediction.shape != target.shape or prediction.shape[2] <= start:
        raise ValueError("FM needs matching latent shapes and at least one unknown temporal slice")
    return F.mse_loss(prediction[:, :, start:].float(), target[:, :, start:].float())


def masked_mse(value, mask):
    if mask.shape != (*value.shape[:-1], 1):
        raise ValueError("Hidden mask must be [batch, tokens, 1]")
    return (value.float().square() * mask).sum() / (mask.sum() * value.shape[-1]).clamp_min(1)


def structure_loss(states, target, weight):
    target, weight = target.detach().float(), weight.detach().float()
    if not states or weight.shape != target.shape[:-1] or not bool(weight.sum() > 0):
        raise ValueError("STRUCT needs actual decoded states and positive valid-token weights")
    losses, metrics = [], {}
    for label, state in states.items():
        if state.shape != target.shape:
            raise ValueError("JEPA decoder output and GT must match in token coordinates and width")
        value = ((state.float() - target).square().mean(-1) * weight).sum() / weight.sum()
        losses.append(value)
        metrics[f"{label}/struct_mse"] = value.detach()
        metrics[f"{label}/struct_cosine"] = (F.cosine_similarity(state.float(), target, dim=-1) * weight).sum().detach() / weight.sum()
    return torch.stack(losses).mean(), metrics


def oracle_regression(deltas, targets, mask):
    # Divide by selected-site count, not accepted count: rejection must not amplify others.
    zero = next(iter(deltas.values())).float().sum() * 0
    if not targets:
        return zero
    terms = []
    for label, target in targets.items():
        if target.accepted:
            terms.append(masked_mse((deltas[label].float() - target.delta.detach()) / target.scale.detach(), mask))
        else:
            terms.append(deltas[label].float().sum() * 0)
    return torch.stack(terms).mean()


def training_loss(prediction, clean, noise, first, sigma, info, target, weight, oracle_targets, step, config):
    cfg = config["loss"]
    fm = flow_mse(prediction, flow_target(clean, noise), first is not None)
    struct, metrics = structure_loss(info["states"], target, weight)
    oracle = oracle_regression(info["deltas"], oracle_targets, info["write_mask"])
    warmup = min(1.0, (step + 1) / cfg["warmup_steps"])
    noise_weight = ((1 - sigma) / cfg["struct_noise_width"]).clamp(0, 1)
    terms = dict(fm=cfg["fm_weight"] * fm,
                 struct=cfg["struct_weight"] * warmup * noise_weight * struct,
                 oracle=cfg["oracle_weight"] * warmup * oracle)
    total = sum(terms.values())
    metrics.update(info["metrics"])
    metrics.update({"loss/fm": fm.detach(), "loss/struct": struct.detach(), "loss/oracle": oracle.detach(),
                    "loss/total": total.detach(), "loss/warmup": warmup, "loss/struct_noise_weight": noise_weight.detach()})
    metrics.update({f"loss/weighted_{name}": value.detach() for name, value in terms.items()})
    return total, metrics, terms
