"""Pure Wan2.2 LoRA baseline; the sibling v4.2 is a read-only data/runtime dependency."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
V42_ROOT = ROOT.parent / "v4.2"
if not (V42_ROOT / "physgen_v42").is_dir():
    raise RuntimeError(f"Missing read-only v4.2 dependency: {V42_ROOT}")
sys.path.append(str(V42_ROOT))

VERSION = "wan22_ti2v5b_static_lora_fm_v1"
TRAINING_RECIPE = "wan_lora_fm_bs8_v42_exposures_no_repair_v1"
