"""Finite-step, verified same-layer FM correction targets; no higher-order graph.

This is model-dependent local supervision, not unique hidden ground truth.
With one infinitesimal step it can reduce to rescaled FM gradients. Multi-step
suffix optimization/backtracking supplies a finite target, not a new data label.
"""
from dataclasses import dataclass
import math

import torch
import torch.distributed as dist

from .losses import flow_mse, masked_mse


@dataclass
class OracleTarget:
    delta: torch.Tensor
    scale: torch.Tensor
    accepted: bool
    metrics: dict


def all_ranks(condition, device):
    """FSDP suffixes require identical forward/backward counts on every rank.

    A distributed proposal is accepted only if it improves every local example.
    Single-GPU behavior is unchanged. Never rank-locally break a suffix search.
    """
    value = torch.tensor(int(bool(condition)), device=device, dtype=torch.int32)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return bool(value.item())


def validate_oracle(cfg):
    for key in ("inner_steps", "max_backtracks", "sites_per_micro"):
        if type(cfg.get(key)) is not int or cfg[key] < (0 if key == "sites_per_micro" else 1):
            raise ValueError(f"oracle.{key} has an invalid count")
    for key in ("relative_step", "regularization", "min_relative_gain", "min_absolute_gain", "scale_floor", "gradient_floor", "armijo"):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"oracle.{key} must be finite and nonnegative")
    if min(cfg["relative_step"], cfg["scale_floor"], cfg["gradient_floor"]) <= 0:
        raise ValueError("Oracle step and numerical floors must be positive")
    if not 0 < cfg["backtrack_factor"] < 1 or not 0 < cfg["armijo"] < 1:
        raise ValueError("Oracle backtracking factor and Armijo coefficient must lie in (0, 1)")


def select_sites(labels, step, micro, accumulation, cfg):
    count = cfg["sites_per_micro"] or len(labels)
    if count > len(labels):
        raise ValueError("sites_per_micro exceeds available correctors")
    start = ((step * accumulation + micro) * count) % len(labels)
    return [labels[(start + i) % len(labels)] for i in range(count)]


def solve_oracle(suffix, before, initial_delta, target, mask, cfg, first_known=True):
    """Optimize TOTAL residual, starting at the actual current writer output.

    suffix receives already-corrected H and must replay the deployed downstream
    computation with fixed parameters and the captured current-site P. autograd.grad
    never populates parameter .grad. All candidate searches use the SAME FP32
    residual addition/autocast as training. A rejected target contributes no loss.
    """
    validate_oracle(cfg)
    before, initial_delta, target, mask = (x.detach().float() for x in (before, initial_delta, target, mask))
    if before.shape != initial_delta.shape or not bool(mask.sum() > 0):
        raise ValueError("Oracle requires matching hidden/residual shapes and writable tokens")
    scale = masked_mse(before, mask).sqrt().clamp_min(cfg["scale_floor"]).detach()
    current = initial_delta * mask
    accepted_steps, evaluations, backward_calls = 0, 0, 0
    initial_fm = final_fm = initial_objective = final_objective = None
    last_step, grad_rms = 0.0, 0.0

    def evaluate(delta):
        fm = flow_mse(suffix(before + delta), target, first_known)
        return fm, fm + cfg["regularization"] * masked_mse(delta / scale, mask)

    for iteration in range(cfg["inner_steps"]):
        with torch.enable_grad():
            variable = current.detach().requires_grad_(True)
            fm, objective = evaluate(variable)
            evaluations += 1
            if not all_ranks(torch.isfinite(objective), before.device):
                raise FloatingPointError("Non-finite oracle baseline objective")
            gradient, = torch.autograd.grad(objective, variable, create_graph=False)
            backward_calls += 1
        fm_value, objective_value = float(fm.detach()), float(objective.detach())
        del fm, objective, variable
        if iteration == 0:
            initial_fm = final_fm = fm_value
            initial_objective = final_objective = objective_value
        gradient = gradient.detach().float() * mask
        if not all_ranks(torch.isfinite(gradient).all(), before.device):
            raise FloatingPointError("Non-finite oracle hidden gradient")
        gradient_scale = masked_mse(gradient, mask).sqrt()
        grad_rms = float(gradient_scale)
        if not all_ranks(grad_rms > cfg["gradient_floor"], before.device):
            break
        direction = -gradient / gradient_scale
        radius = cfg["relative_step"] * scale
        # L uses a mean but its gradient dot displacement uses a SUM (Taylor derivative).
        slope = float((gradient * direction).sum())
        found = False
        with torch.no_grad():
            for backtrack in range(cfg["max_backtracks"]):
                length = radius * cfg["backtrack_factor"] ** backtrack
                candidate = (current + length * direction) * mask
                candidate_fm, candidate_objective = evaluate(candidate)
                evaluations += 1
                cf, co = float(candidate_fm), float(candidate_objective)
                minimum_gain = max(cfg["min_absolute_gain"], cfg["min_relative_gain"] * abs(fm_value))
                improves = (math.isfinite(co) and cf < fm_value - minimum_gain
                            and co <= objective_value + cfg["armijo"] * float(length) * slope)
                if all_ranks(improves, before.device):
                    current = candidate.detach()
                    final_fm, final_objective, last_step = cf, co, float(length / scale)
                    accepted_steps += 1
                    found = True
                    break
        if not found:
            break
    metrics = dict(initial_fm=initial_fm, final_fm=final_fm, fm_gain=initial_fm - final_fm,
                   relative_fm_gain=(initial_fm - final_fm) / max(abs(initial_fm), 1e-12),
                   initial_objective=initial_objective, final_objective=final_objective,
                   accepted=float(accepted_steps > 0), accepted_steps=accepted_steps,
                   suffix_forwards=evaluations, suffix_backwards=backward_calls,
                   gradient_rms=grad_rms, final_relative_step=last_step,
                   target_rms=float(masked_mse(current, mask).sqrt()),
                   target_change_rms=float(masked_mse(current - initial_delta, mask).sqrt()),
                   hidden_scale=float(scale))
    return OracleTarget(current.detach(), scale.detach(), accepted_steps > 0, metrics)


def build_oracle_targets(model, snapshots, labels, text, sigma, coords, target, cfg, first_known=True):
    targets, metrics = {}, {}
    for label in labels:
        snapshot = snapshots[label]
        def suffix(hidden):
            return model.suffix(snapshot, hidden, text.detach(), sigma.detach(), coords)
        result = solve_oracle(suffix, snapshot.before, snapshot.delta, target, snapshot.write_mask, cfg, first_known)
        targets[label] = result
        metrics.update({f"oracle/{label}/{name}": value for name, value in result.metrics.items()})
    metrics["oracle/selected_sites"] = len(labels)
    metrics["oracle/acceptance"] = sum(float(t.accepted) for t in targets.values()) / max(len(targets), 1)
    return targets, metrics
