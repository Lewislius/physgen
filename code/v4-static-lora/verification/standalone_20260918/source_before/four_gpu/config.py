"""Keep the single-GPU recipe intact; isolate the four-GPU execution profile."""
import json
from pathlib import Path

import yaml

from static_lora import ROOT
from static_lora.config import configuration as single_configuration
from static_lora.runtime import code_fingerprint as single_fingerprint, digest, sha256

DEFAULT_CONFIG = ROOT / "configs/train_4x48g.yaml"
EXPECTED = dict(strategy="model_parallel", devices=4, block_counts=[8, 8, 7, 7], activation_offload=True)


def configuration(path=DEFAULT_CONFIG):
    path = Path(path).expanduser().resolve()
    source = path.read_text().replace("${LORA_ROOT}", str(ROOT))
    raw = json.loads(source) if path.suffix == ".json" else yaml.safe_load(source)
    resolved = isinstance(raw, dict) and "model_version" in raw
    settings = dict(raw.get("execution", {}) if resolved else raw)
    if set(settings) != set(EXPECTED) | {"base_config"}:
        raise ValueError("Use the four-GPU config or its saved config.json")
    for key, value in EXPECTED.items():
        if settings[key] != value:
            raise ValueError(f"Four-GPU profile requires {key}={value}")
    base = Path(settings["base_config"]).expanduser()
    settings["base_config"] = str((path.parent / base).resolve() if not base.is_absolute() else base.resolve())
    cfg = single_configuration(settings["base_config"])
    cfg["execution"] = settings
    # One process owns four CUDA devices. There is NO four-way data replication.
    # world_size=1, microbatch=1 and accumulation=8 therefore retain global BS8.
    for key, directory in dict(checkpoints="checkpoints/four_gpu", logs="train/train_log/four_gpu",
                               outputs="inference_outputs/four_gpu").items():
        cfg["paths"][key] = str(ROOT / directory)
    if resolved and digest(raw) != digest(cfg):
        raise ValueError("Saved four-GPU config differs from its training recipe")
    return cfg


def code_fingerprint():
    files = single_fingerprint()
    files.update({str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "four_gpu").glob("*.py"))})
    return files
