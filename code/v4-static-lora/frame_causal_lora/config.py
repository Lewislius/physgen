"""Independent comparison identities, sharing the original LoRA/FM recipe."""
from copy import deepcopy
import json
from pathlib import Path

import yaml

from static_lora import ROOT
from static_lora.config import configuration as single_configuration
from static_lora.runtime import digest, output_path, sha256
from four_gpu.config import configuration as four_configuration, code_fingerprint as base_fingerprint
from .attention import validate_attention

DEFAULT_CONFIG = ROOT / "configs/train_frame_causal_full_1x96g.yaml"


def variant_name(attention):
    spec = validate_attention(attention)
    return (spec["mode"] if spec["previous_frames"] is None
            else f'{spec["mode"]}_k{spec["previous_frames"]}')


def configuration(path=DEFAULT_CONFIG):
    path = Path(path).expanduser().resolve()
    source = path.read_text().replace("${LORA_ROOT}", str(ROOT))
    raw = json.loads(source) if path.suffix == ".json" else yaml.safe_load(source)
    if not isinstance(raw, dict):
        raise ValueError("Expected a frame-causal comparison configuration")
    resolved = "model_version" in raw
    settings = deepcopy(raw.get("comparison", {}) if resolved else raw)
    if set(settings) != {"base_config", "profile", "attention"}:
        raise ValueError("Use a frame-causal config or its saved config.json")
    profile = settings["profile"]
    if profile not in ("1x96g", "4x48g"):
        raise ValueError("profile must be 1x96g or 4x48g")
    settings["attention"] = validate_attention(settings["attention"])
    base = Path(settings["base_config"]).expanduser()
    base = (path.parent / base).resolve() if not base.is_absolute() else base.resolve()
    settings["base_config"] = str(base)
    cfg = (four_configuration if profile == "4x48g" else single_configuration)(base)
    variant = variant_name(settings["attention"])
    cfg["model_version"] = f"wan22_ti2v5b_lora_{variant}_v1"
    cfg["comparison"] = settings
    cfg["attention"] = deepcopy(settings["attention"])
    for key, parent in dict(checkpoints="checkpoints", logs="train/train_log", outputs="inference_outputs").items():
        cfg["paths"][key] = str(output_path(ROOT / parent / variant / profile))
    if resolved and digest(raw) != digest(cfg):
        raise ValueError("Saved frame-causal config differs from its frozen recipe/window/profile")
    return cfg


def code_fingerprint():
    files = base_fingerprint()
    files.update({str(p.relative_to(ROOT)): sha256(p)
                  for p in sorted((ROOT / "frame_causal_lora").glob("*.py"))})
    return files

