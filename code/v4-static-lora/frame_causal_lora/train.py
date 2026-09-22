"""Train either frame-causal LoRA comparison on one GPU or four model stages."""
import argparse
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from static_lora import ROOT
from four_gpu.runtime import isolated_module
from four_gpu.train import implementation as four_implementation
from frame_causal_lora.backbone import load_model
from frame_causal_lora.checkpoint import load_training, save_checkpoint
from frame_causal_lora.config import DEFAULT_CONFIG, code_fingerprint, configuration, variant_name


def run_name(cfg, label=None):
    if label is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", label):
        raise ValueError("run-name is an optional label of 1..64 letters/digits/._-; no paths")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f'{variant_name(cfg["attention"])}_{cfg["comparison"]["profile"]}_{stamp}_{uuid.uuid4().hex[:6]}'
    return name + ("_" + label if label else "")


def implementation(config_path=DEFAULT_CONFIG):
    cfg = configuration(config_path)
    module = (four_implementation() if cfg["comparison"]["profile"] == "4x48g" else
              isolated_module("frame_causal_lora._shared_training", ROOT / "train/train.py"))
    module.DEFAULT_CONFIG = config_path
    module.configuration = configuration
    module.code_fingerprint = code_fingerprint
    module.load_model = load_model
    module.save_checkpoint = save_checkpoint
    module.load_training = load_training
    shared_train = module.train

    def train(resolved, args, device):
        args = copy(args)
        if args.resume and args.run_name:
            raise ValueError("Resume preserves the original run name; omit --run-name")
        if not args.resume:
            # Even a user-provided label cannot remove the version or timestamp.
            args.run_name = run_name(resolved, args.run_name)
        return shared_train(resolved, args, device)

    module.train = train
    return module


def main():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", default=str(DEFAULT_CONFIG))
    selected, _ = selector.parse_known_args()
    implementation(selected.config).main()


if __name__ == "__main__":
    main()

