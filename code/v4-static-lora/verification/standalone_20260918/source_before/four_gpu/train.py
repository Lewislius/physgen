"""Reuse the single-process FM loop with one Wan partitioned over four GPUs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from static_lora import ROOT
from four_gpu.backbone import load_model
from four_gpu.config import DEFAULT_CONFIG, code_fingerprint, configuration
from four_gpu.runtime import isolated_module, log, require_four_gpus, save_json


def implementation():
    module = isolated_module("four_gpu._shared_training", ROOT / "train/train.py")
    module.DEFAULT_CONFIG = DEFAULT_CONFIG
    module.configuration = configuration
    module.code_fingerprint = code_fingerprint
    module.require_single_gpu = require_four_gpus
    module.load_model = load_model
    module.log = log
    module.save_json = save_json
    return module


if __name__ == "__main__":
    implementation().main()
