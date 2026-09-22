"""Atomic LoRA-only model snapshots plus optimizer/EMA/RNG/exposure resume state."""
import math
from pathlib import Path

import torch

from . import ROOT, TRAINING_RECIPE, VERSION
from .lora import load_lora_state_dict, lora_state_dict
from .optim import EMA, SCHEDULE_FORMULA, lr_factor, rng_state
from .runtime import (checkpoint_code_compatible, checkpoint_identity_compatible,
                      code_fingerprint, digest, file_identity, lock_assets, output_path,
                      read_json, save_json, save_tensor, verify_files)


def save_checkpoint(root, step, model, optimizer, ema, plan, cfg, identity):
    if (not 0 < step <= cfg["train"]["steps"] or step % cfg["train"]["save_every"] or
            ema.updates != step or cfg["train"]["recipe"] != TRAINING_RECIPE):
        raise ValueError("Checkpoint does not match the completed LoRA schedule")
    plan.validate_state(plan.state_dict(), step)
    path = output_path(Path(root) / f"step{step:04d}")
    path.mkdir(parents=True, exist_ok=True)
    if (path / "complete.json").exists():
        raise FileExistsError(f"Completed checkpoint already exists: {path}")
    save_json(cfg, path / "config.json")
    save_tensor(lora_state_dict(model), path / "lora.pt")
    save_tensor(dict(step=step, optimizer=optimizer.state_dict(), ema=ema.state_dict(),
                     plan=plan.state_dict(), rng=rng_state(), training_recipe=TRAINING_RECIPE,
                     identity=identity, version=VERSION,
                     scheduler=dict(completed_step=step, formula=SCHEDULE_FORMULA,
                                    group_lr=[g["lr"] for g in optimizer.param_groups])), path / "training.pt")
    save_tensor(ema.state_dict()["values"], path / "ema.pt")
    names = ("config.json", "lora.pt", "training.pt", "ema.pt")
    save_json(dict(step=step, version=VERSION, identity=identity, training_recipe=TRAINING_RECIPE,
                   ema_updates=ema.updates, batch_size=8,
                   files={name: file_identity(path / name) for name in names}), path / "complete.json")
    if step == cfg["train"]["steps"]:
        save_json(dict(checkpoint=str(path), step=step), Path(root) / "final.json")
    return path


def inspect_checkpoint(path, cfg, identity=None, *, inference=False):
    path = Path(path).expanduser().resolve()
    complete = read_json(path / "complete.json")
    step = complete.get("step")
    if (complete.get("training_recipe") != TRAINING_RECIPE or complete.get("version") != VERSION or
            complete.get("batch_size") != 8 or type(step) is not int or
            not 0 < step <= cfg["train"]["steps"] or step % cfg["train"]["save_every"] or
            complete.get("ema_updates") != step):
        raise ValueError("Expected a completed static-LoRA checkpoint, not a v4.2 adapter/full-model checkpoint")
    if identity is not None and not checkpoint_identity_compatible(complete["identity"], identity):
        raise ValueError("LoRA source/config/assets/code identity differs")
    files = complete["files"]
    if inference:
        # Generation requires only the saved configuration and EMA. Optimizer,
        # RNG and raw training weights are deliberately not read or required.
        names = ("config.json", "ema.pt")
        if not all(name in files for name in names):
            raise ValueError("Completed checkpoint must describe config.json and ema.pt")
        files = {name: files[name] for name in names}
    verify_files(path, files, full=True)
    if digest(read_json(path / "config.json")) != digest(cfg):
        raise ValueError("LoRA checkpoint configuration differs")
    return complete


def load_training(path, model, optimizer, plan, cfg, identity):
    path = Path(path)
    complete = inspect_checkpoint(path, cfg, identity)
    # Only use locally produced trusted training states (Python/NumPy RNG objects).
    state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    step = complete["step"]
    if (state["step"] != step or state["version"] != VERSION or state["identity"] != complete["identity"] or
            state["training_recipe"] != TRAINING_RECIPE or state["ema"]["updates"] != step):
        raise ValueError("Training state differs from completion marker")
    plan.validate_state(state["plan"], step)
    scheduler = state["scheduler"]
    expected_lr = cfg["train"]["peak_lr"] * lr_factor(step, "lora")
    groups = state["optimizer"]["param_groups"]
    if (scheduler["formula"] != SCHEDULE_FORMULA or scheduler["completed_step"] != step or
            len(groups) != 1 or groups[0]["name"] != "lora" or len(scheduler["group_lr"]) != 1 or
            not math.isclose(scheduler["group_lr"][0], expected_lr, rel_tol=1e-12) or
            not math.isclose(groups[0]["lr"], expected_lr, rel_tol=1e-12)):
        raise ValueError("LoRA optimizer/scheduler differs from completed step")
    load_lora_state_dict(model, torch.load(path / "lora.pt", map_location="cpu", weights_only=True))
    optimizer.load_state_dict(state["optimizer"])
    for values in optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            if values[name].dtype != torch.float32:
                raise ValueError("LoRA optimizer moments must remain FP32")
    plan.load_state_dict(state["plan"])
    ema = EMA(model, cfg["train"]["ema_decay"])
    ema.load_state_dict(state["ema"])
    return step, ema, state["rng"]


def inference_code_compatible(saved, current):
    if checkpoint_code_compatible(saved, current):
        return True
    # Inference-only, exact-digest transitions live here, not in the shared
    # v4.2 ledger. They do not authorize resuming training after source changes.
    ledger = ROOT / "inference/checkpoint_compatibility.json"
    if not ledger.is_file():
        return False
    return any(entry["before"] == saved and entry["after"] == current
               for entry in read_json(ledger)["transitions"])


def checkpoint_path(cfg, value=None):
    path = Path(value).expanduser().resolve() if value else Path(
        read_json(Path(cfg["paths"]["checkpoints"]) / "latest_final.json")["checkpoint"])
    complete = inspect_checkpoint(path, cfg, inference=True)
    if not inference_code_compatible(complete["identity"]["code"], digest(code_fingerprint())):
        raise ValueError("Inference source differs from the frozen training source")
    if complete["identity"]["assets"] != digest(lock_assets(cfg)):
        raise ValueError("Base model asset inventory differs from training")
    return path
