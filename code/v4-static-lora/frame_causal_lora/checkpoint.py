"""Reuse atomic save/resume with distinct architecture versions and source hashes."""
from pathlib import Path

from static_lora import ROOT
from static_lora.runtime import digest, lock_assets, read_json
from four_gpu.runtime import isolated_module
from .config import code_fingerprint


def _implementation(cfg):
    module = isolated_module("static_lora._frame_causal_checkpoint", ROOT / "static_lora/checkpoint.py")
    module.VERSION = cfg["model_version"]
    return module


def save_checkpoint(root, step, model, optimizer, ema, plan, cfg, identity):
    return _implementation(cfg).save_checkpoint(root, step, model, optimizer, ema, plan, cfg, identity)


def inspect_checkpoint(path, cfg, identity=None, *, inference=False):
    return _implementation(cfg).inspect_checkpoint(path, cfg, identity, inference=inference)


def load_training(path, model, optimizer, plan, cfg, identity):
    return _implementation(cfg).load_training(path, model, optimizer, plan, cfg, identity)


def checkpoint_path(cfg, value=None):
    path = Path(value).expanduser().resolve() if value else Path(
        read_json(Path(cfg["paths"]["checkpoints"]) / "latest_final.json")["checkpoint"])
    complete = inspect_checkpoint(path, cfg, inference=True)
    if complete["identity"]["code"] != digest(code_fingerprint()):
        raise ValueError("Frame-causal inference source differs from the training source")
    if complete["identity"]["assets"] != digest(lock_assets(cfg, inference=True)):
        raise ValueError("Base model asset inventory differs from training")
    return path

