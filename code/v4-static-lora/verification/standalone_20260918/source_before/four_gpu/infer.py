"""Infer from a four-GPU training checkpoint with the same four-device placement."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from static_lora import ROOT
from static_lora.checkpoint import inspect_checkpoint
from static_lora.runtime import checkpoint_code_compatible, digest, lock_assets, read_json
from four_gpu.backbone import load_model
from four_gpu.config import DEFAULT_CONFIG, code_fingerprint, configuration
from four_gpu.runtime import isolated_module, log, require_four_gpus


def checkpoint_path(cfg, value=None):
    path = Path(value).expanduser().resolve() if value else Path(
        read_json(Path(cfg["paths"]["checkpoints"]) / "latest_final.json")["checkpoint"])
    complete = inspect_checkpoint(path, cfg)
    if not checkpoint_code_compatible(complete["identity"]["code"], digest(code_fingerprint())):
        raise ValueError("Four-GPU inference source differs from the training source")
    if complete["identity"]["assets"] != digest(lock_assets(cfg)):
        raise ValueError("Base asset inventory differs from training")
    return path


def implementation():
    module = isolated_module("four_gpu._shared_inference", ROOT / "inference/infer.py")
    module.DEFAULT_CONFIG = DEFAULT_CONFIG
    module.configuration = configuration
    module.checkpoint_path = checkpoint_path
    module.require_single_gpu = require_four_gpus
    module.load_model = load_model
    module.log = log
    return module


if __name__ == "__main__":
    implementation().main()
