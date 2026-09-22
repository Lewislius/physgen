"""Standalone Wan2.2 LoRA experiment; all Python helpers are owned locally."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VERSION = "wan22_ti2v5b_static_lora_fm_v1"
TRAINING_RECIPE = "wan_lora_fm_bs8_v42_exposures_no_repair_v1"
