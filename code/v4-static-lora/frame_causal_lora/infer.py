"""Inference entry preserving the saved frame mask and device profile."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from static_lora import ROOT
from static_lora.runtime import read_json
from four_gpu.infer import implementation as four_implementation
from four_gpu.runtime import isolated_module
from frame_causal_lora.backbone import load_model
from frame_causal_lora.checkpoint import checkpoint_path
from frame_causal_lora.config import DEFAULT_CONFIG, configuration


def implementation(config_path=DEFAULT_CONFIG):
    cfg = configuration(config_path)
    module = (four_implementation() if cfg["comparison"]["profile"] == "4x48g" else
              isolated_module("frame_causal_lora._shared_inference", ROOT / "inference/infer.py"))
    module.DEFAULT_CONFIG = config_path
    module.VERSION = cfg["model_version"]
    module.configuration = configuration
    module.load_model = load_model
    module.checkpoint_path = checkpoint_path
    return module


def main():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config")
    selector.add_argument("--checkpoint")
    selector.add_argument("--phase")
    selector.add_argument("--output")
    selected, _ = selector.parse_known_args()
    config_path = selected.config
    if selected.phase in ("sample", "report") and selected.output:
        plan = read_json(Path(selected.output).expanduser() / "cases.json")
        config_path = Path(plan["checkpoint"]) / "config.json"
    elif config_path is None and selected.checkpoint:
        config_path = Path(selected.checkpoint).expanduser() / "config.json"
    implementation(config_path or DEFAULT_CONFIG).main()


if __name__ == "__main__":
    main()
