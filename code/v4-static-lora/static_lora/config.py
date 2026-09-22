"""A frozen v4.2 data/schedule reference plus explicit LoRA-only choices."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import yaml

from . import ROOT, TRAINING_RECIPE, VERSION
from .runtime import digest, output_path, read_json

DEFAULT_CONFIG = ROOT / "configs/train.yaml"
TARGETS = [f"{attention}.{projection}" for attention in ("self_attn", "cross_attn")
           for projection in ("q", "k", "v", "o")]


def configuration(path=DEFAULT_CONFIG):
    path = Path(path).expanduser().resolve()
    source = path.read_text().replace("${LORA_ROOT}", str(ROOT))
    raw = json.loads(source) if path.suffix == ".json" else yaml.safe_load(source)
    resolved = isinstance(raw, dict) and "model_version" in raw
    settings = deepcopy(raw.get("baseline", {}) if resolved else raw)
    if not isinstance(settings, dict) or set(settings) != {"reference_config", "lora"}:
        raise ValueError("Use the static LoRA config or its saved config.json")
    reference = Path(settings["reference_config"]).expanduser()
    reference = (path.parent / reference).resolve() if not reference.is_absolute() else reference.resolve()
    settings["reference_config"] = str(reference)
    base = read_json(reference)
    if (base["train"]["accumulation"], base["train"]["steps"], base["train"]["save_every"]) != (8, 1200, 100):
        raise ValueError("Reference must be formal v4.2 BS8 / 1200 updates / save every 100")
    spec = settings["lora"]
    if (set(spec) != {"rank", "alpha", "dropout", "targets"} or
            type(spec["rank"]) is not int or not 1 <= spec["rank"] <= 64 or
            not math.isfinite(spec["alpha"]) or spec["alpha"] <= 0 or
            spec["dropout"] != 0. or spec["targets"] != TARGETS):
        raise ValueError("Expected attention q/k/v/o LoRA, positive rank/alpha, dropout=0")
    cfg = deepcopy(base)
    cfg.pop("state")
    cfg.pop("preparation", None)
    cfg["model_version"] = VERSION
    cfg["baseline"] = settings
    cfg["reference_sha256"] = digest(base)
    cfg["lora"] = deepcopy(spec)
    cfg["loss"] = dict(fm=1.)
    common = ("seed", "steps", "world_size", "accumulation", "save_every", "ema_decay", "grad_clip")
    cfg["train"] = {key: base["train"][key] for key in common}
    cfg["train"].update(recipe=TRAINING_RECIPE, peak_lr=base["train"]["peak_lr"]["core5"],
                        warmup_steps=50, min_lr_factor=.1, weight_decay=.01,
                        betas=[.9, .999], eps=1e-8, gradient_checkpointing=True)
    cfg["noise"] = dict(ordinary=[.02, .999], low=[.05, .15], low_per_update=1, repair=False)
    for key, suffix in dict(checkpoints="checkpoints", logs="train/train_log", outputs="inference_outputs").items():
        cfg["paths"][key] = str(output_path(ROOT / suffix))
    if resolved and digest(raw) != digest(cfg):
        raise ValueError("Saved config differs from its frozen reference/LoRA recipe")
    sys.path.insert(0, cfg["paths"]["wan_code"])
    return cfg
