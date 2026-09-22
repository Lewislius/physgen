import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from . import VERSION
from .runtime import (checkpoint_identity_compatible, digest, file_identity, read_json,
                      save_json, save_tensor, verify_files)

TRAINING_RECIPE = "fm_joint_independent_cores_struct_global_share20_v2"
GATED_RECIPE = "fm_struct_v4_limits_gated_uniform_v1"
REPAIR_RECIPE = "fm_struct_write_repair_trajectory_v1"
SCHEDULE_FORMULA = "all_groups_lr_warmup50_cosine1200_v1"


def ramp(step):
    # Zero-initialized writers preserve the base output initially. Their residual
    # budget must be nonzero from step 1 so FM can actually train the writers.
    return 1.


def lr_factor(step, group):
    # Only the learning rate warms up; losses and trainable groups never switch.
    if step <= 50:
        return max(0., step / 50)
    return .1 + .9 * (1 + math.cos(math.pi * (step - 50) / 1150)) / 2


def optimizer_for(adapter, cfg):
    groups = []
    for label, module in adapter.groups().items():
        decay, no_decay = [], []
        for name, param in module.named_parameters():
            exempt = param.ndim < 2 or any(key in name for key in ("norm", "embedding", "queries"))
            (no_decay if exempt else decay).append(param)
        for params, weight_decay in ((decay, .01), (no_decay, 0.)):
            if params:
                groups.append(dict(params=params, name=label, lr=cfg["train"]["peak_lr"][label], weight_decay=weight_decay))
    return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)


def set_schedule(adapter, optimizer, cfg, step):
    for module in adapter.groups().values():
        module.requires_grad_(True)
    for group in optimizer.param_groups:
        name = group["name"]
        group["lr"] = cfg["train"]["peak_lr"][name] * lr_factor(step, name)


class GradientMixer:
    def __init__(self, core, generation_parameters=None, diagnostic_parameters=None):
        self.params = list(core.parameters() if hasattr(core, "parameters") else core)
        self.generation_params = list(generation_parameters) if generation_parameters is not None else self.params
        if not self.params or not set(map(id, self.params)) <= set(map(id, self.generation_params)):
            raise ValueError("STRUCT parameters must be a nonempty subset of the generation parameters")
        self.aux = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.used = [False] * len(self.params)
        diagnostic_ids = set(map(id, diagnostic_parameters)) if diagnostic_parameters is not None else set(map(id, self.params))
        self.diagnostic = [i for i, p in enumerate(self.params) if id(p) in diagnostic_ids]
        if not self.diagnostic or not diagnostic_ids <= set(map(id, self.params)):
            raise ValueError("Core diagnostics must be a nonempty subset of auxiliary parameters")

    def accumulate_aux(self, loss, parameters=None):
        params = self.params if parameters is None else list(parameters)
        lookup = {id(p):i for i,p in enumerate(self.params)}
        if any(id(p) not in lookup for p in params):
            raise ValueError('Auxiliary gradient route is outside the mixer parameters')
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for param, grad in zip(params, grads):
            index = lookup[id(param)]
            if grad is not None:
                self.used[index] = True
                self.aux[index].add_(grad.detach().float())

    @torch.no_grad()
    def align_to_generation(self, groups):
        """Remove only the auxiliary component opposing FM, separately per group.

        For SGD this makes <g_FM, g_aux> nonnegative before clipping. This is
        NOT a guarantee about finite AdamW steps, held-out loss, or video physics.
        Zero FM leaves auxiliary learning intact (essential at zero-writer init).
        """
        lookup = {id(p): i for i, p in enumerate(self.params)}
        assigned, report = set(), {}
        for name, module in groups.items():
            ids = [lookup[id(p)] for p in module.parameters() if id(p) in lookup]
            if not ids or assigned.intersection(ids):
                raise ValueError("Auxiliary alignment groups must partition auxiliary parameters")
            assigned.update(ids)
            zero = self.aux[ids[0]].new_zeros(())
            pairs = [(self.params[i].grad, self.aux[i]) for i in ids]
            fm2 = sum((g.float().square().sum() for g, _ in pairs if g is not None), zero)
            aux2 = sum((a.square().sum() for _, a in pairs), zero)
            dot = sum(((g.float()*a).sum() for g, a in pairs if g is not None), zero)
            if not bool(torch.isfinite(fm2) & torch.isfinite(aux2) & torch.isfinite(dot)):
                raise FloatingPointError(f"Nonfinite FM/STRUCT gradients in {name}")
            conflict = bool(dot < 0) and bool(fm2 > 0)
            projection = dot/fm2 if conflict else zero
            if conflict:
                for g, a in pairs:
                    if g is not None:
                        a.sub_(g.float()*projection)
            after2 = sum((a.square().sum() for _, a in pairs), zero)
            after_dot = sum(((g.float()*a).sum() for g, a in pairs if g is not None), zero)
            denom = (fm2*aux2).sqrt()
            report[name] = dict(fm_norm=float(fm2.sqrt()), auxiliary_norm_before=float(aux2.sqrt()),
                                auxiliary_norm_after=float(after2.sqrt()), dot_before=float(dot),
                                dot_after=float(after_dot), conflict=conflict,
                                cosine_before=float(dot/denom) if float(denom)>0 else None,
                                removed_norm=float(projection.abs()*fm2.sqrt()))
        if assigned != set(range(len(self.params))):
            raise ValueError("Auxiliary alignment groups did not cover all auxiliary parameters")
        return report

    def combine(self, max_share=.2):
        """Merge gradients and measure the same core parameters for both objectives.

        max_share=None adds the already weighted STRUCT gradient unchanged.
        A numeric share retains the historical sum-of-norms cap for legacy runs.
        The corrected recipe also includes TextInit in the auxiliary parameters;
        diagnostic_parameters limits the core-specific measurements to the cores.
        """
        if max_share is not None and not 0 <= max_share < 1:
            raise ValueError("STRUCT gradient share must be in [0, 1)")
        device = self.params[0].device
        core = [self.params[i] for i in self.diagnostic]
        core_aux = [self.aux[i] for i in self.diagnostic]
        core_norm = torch.stack([p.grad.float().square().sum() if p.grad is not None else
                                       torch.zeros((), device=device) for p in core]).sum().sqrt()
        generation_norm = torch.stack([p.grad.float().square().sum() if p.grad is not None else
                                       torch.zeros((), device=device) for p in self.generation_params]).sum().sqrt()
        auxiliary_norm = torch.stack([g.square().sum() for g in self.aux]).sum().sqrt()
        core_auxiliary_norm = torch.stack([g.square().sum() for g in core_aux]).sum().sqrt()
        if not bool(torch.isfinite(generation_norm) & torch.isfinite(auxiliary_norm)):
            raise FloatingPointError("Nonfinite generation/STRUCT gradient")
        dot = torch.stack([(p.grad.float() * g).sum() if p.grad is not None else
                           torch.zeros((), device=device) for p, g in zip(core, core_aux)]).sum()
        cosine = float(dot / (core_norm * core_auxiliary_norm)) if float(core_norm * core_auxiliary_norm) > 0 else None
        budget = None if max_share is None else (max_share / (1 - max_share)) * generation_norm
        alpha = torch.ones_like(auxiliary_norm) if budget is None else torch.minimum(
            torch.ones_like(auxiliary_norm), budget / (auxiliary_norm + 1e-12))
        for index, (param, grad) in enumerate(zip(self.params, self.aux)):
            if param.grad is None and not self.used[index]:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad.add_(grad * alpha)
        effective = alpha * auxiliary_norm
        return dict(core_generation_grad=float(core_norm), generation_grad_all=float(generation_norm),
                    core_auxiliary_grad=float(core_auxiliary_norm), auxiliary_grad_all=float(auxiliary_norm), aux_alpha=float(alpha),
                    effective_aux_grad=float(effective), aux_grad_budget=float(budget) if budget is not None else None,
                    aux_grad_share=float(effective / (generation_norm + effective + 1e-12)),
                    aux_max_share=max_share, core_generation_struct_cosine=cosine,
                    core_struct_to_generation=float(alpha * core_auxiliary_norm / core_norm) if float(core_norm) > 0 else None)


class EMA:
    def __init__(self, adapter, decay=.995):
        self.decay = decay
        self.values = {n: p.detach().float().clone() for n, p in adapter.named_parameters()}
        self.updates = 0

    @torch.no_grad()
    def update(self, adapter):
        for name, param in adapter.named_parameters():
            self.values[name].lerp_(param.float(), 1 - self.decay)
        self.updates += 1

    def state_dict(self):
        return dict(decay=self.decay, updates=self.updates, values=self.values)

    @contextmanager
    def average_parameters(self, adapter):
        original = {n: p.detach().clone() for n, p in adapter.named_parameters()}
        try:
            with torch.no_grad():
                for name, param in adapter.named_parameters():
                    param.copy_(self.values[name])
            yield
        finally:
            with torch.no_grad():
                for name, param in adapter.named_parameters():
                    param.copy_(original[name])

    @torch.no_grad()
    def load_state_dict(self, state):
        self.decay, self.updates = state["decay"], state["updates"]
        for name in self.values:
            self.values[name].copy_(state["values"][name])


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(root, step, adapter, optimizer, ema, plan, cfg, identity):
    recipe = cfg["train"]["recipe"]
    if ema is None or ema.updates != step:
        raise ValueError("Joint training requires one EMA update per completed step")
    path = Path(root) / f"step{step:04d}"
    path.mkdir(parents=True, exist_ok=True)
    if (path / "complete.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed checkpoint: {path}")
    save_json(cfg, path / "config.json")
    save_tensor(adapter.state_dict(), path / "adapter.pt")
    save_tensor(dict(step=step, optimizer=optimizer.state_dict(), ema=ema.state_dict(),
                     plan=plan.state_dict(), rng=rng_state(), training_recipe=recipe,
                     identity=identity, version=VERSION,
                     scheduler=dict(completed_step=step, formula=SCHEDULE_FORMULA,
                                    group_lr=[g["lr"] for g in optimizer.param_groups])), path / "training.pt")
    values = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
    values.update({k: v.detach().cpu() for k, v in ema.values.items()})
    save_tensor(values, path / "ema.pt")
    names = ["config.json", "adapter.pt", "training.pt", "ema.pt"]
    save_json(dict(step=step, version=VERSION, identity=identity, training_recipe=recipe,
                   ema_updates=ema.updates,
                   files={name: file_identity(path / name) for name in names}), path / "complete.json")
    if step == 1200:
        save_json(dict(checkpoint=str(path.resolve()), step=step), Path(root) / "final.json")
    return path


def load_training(path, adapter, optimizer, plan, cfg, identity):
    path = Path(path)
    complete = read_json(path / "complete.json")
    if complete.get("training_recipe") != cfg["train"]["recipe"]:
        raise ValueError("Checkpoint uses a different training recipe; corrected gated training requires a fresh run")
    if complete["version"] != VERSION or not checkpoint_identity_compatible(complete["identity"], identity):
        raise ValueError("Checkpoint architecture/data/assets/code differ; refusing resume")
    verify_files(path, complete["files"], full=True)
    if digest(read_json(path / "config.json")) != digest(cfg):
        raise ValueError("Cannot resume with a changed training recipe")
    adapter.load_state_dict(torch.load(path / "adapter.pt", map_location="cpu", weights_only=True), strict=True)
    # This is a locally produced trusted checkpoint; Python/NumPy RNG states need pickle.
    state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    if (state["step"] != complete["step"] or state["version"] != VERSION or state["identity"] != complete["identity"] or
            state.get("training_recipe") != cfg["train"]["recipe"] or not 0 < state["step"] <= 1200):
        raise ValueError("Checkpoint completion marker and training state disagree")
    optimizer.load_state_dict(state["optimizer"])
    plan.load_state_dict(state["plan"])
    counts = state["plan"]["counts"]
    if cfg["train"]["recipe"] in (GATED_RECIPE, REPAIR_RECIPE):
        valid_coverage = (len(counts) == len(plan.records) and
                          state["plan"]["cursor"] == 8 * state["step"] and
                          sum(c["i2v"] + c["t2v"] for c in counts) == 8 * state["step"] and
                          all(c["temp"] == 0 and min(c.values()) >= 0 for c in counts))
    else:
        valid_coverage = not (len(counts) != len(plan.records) or
            state["plan"]["ordinary"] + state["plan"]["temporal"] != 8 * state["step"] or
            state["plan"]["temporal"] != state["step"] or
            sum(c["i2v"] for c in counts) != 6 * state["step"] or
            sum(c["t2v"] for c in counts) != 2 * state["step"] or
            sum(c["temp"] for c in counts) != state["step"])
    if not valid_coverage:
        raise ValueError("Exposure queue coverage differs from the completed step")
    if state["ema"] is None:
        raise ValueError("Missing EMA for joint training from step 1")
    ema = EMA(adapter)
    ema.load_state_dict(state["ema"])
    expected_updates = state["step"]
    if complete["ema_updates"] != expected_updates or ema.updates != expected_updates:
        raise ValueError("EMA update count differs from the completed schedule")
    scheduler = state["scheduler"]
    expected_lr = [cfg["train"]["peak_lr"][g["name"]] * lr_factor(state["step"], g["name"])
                   for g in optimizer.param_groups]
    if (scheduler["completed_step"] != state["step"] or
            scheduler["formula"] != SCHEDULE_FORMULA or
            len(scheduler["group_lr"]) != len(expected_lr) or
            any(not math.isclose(saved, expected, rel_tol=1e-12) for saved, expected in
                zip(scheduler["group_lr"], expected_lr))):
        raise ValueError("Checkpoint LR schedule differs from its completed step")
    for param_state in optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            if name in param_state and param_state[name].dtype != torch.float32:
                raise ValueError("Adam moments must remain FP32")
    # Restore only AFTER loading frozen models; constructors can consume torch RNG state.
    return state["step"], ema, state["rng"]
