"""Validate the training contract before allocating models or launching workers."""
import math

from .corrector import ARCHITECTURE
from .oracle import validate_oracle


def validate_config(config):
    cfg, loss, train = config["corrector"], config["loss"], config["train"]
    if cfg.get("architecture") != ARCHITECTURE or cfg.get("initialization") != "image_text":
        raise ValueError("V4.3 uses native joint correctors and deployment-available initialization")
    if config.get("stage") != "A" or cfg.get("a_blocks") != [5, 15, 25]:
        raise ValueError("The production V4.3 contract has exactly the three A sites at 5/15/25")
    native = dict(hidden_width=3072, jepa_width=1664, position_width=1664, text_width=4096, latent_channels=48)
    for key, value in native.items():
        if cfg.get(key) != value:
            raise ValueError(f"Native width contract requires {key}={value}; no silent width reduction is allowed")
    if cfg["mapping_width"] < 3072 or cfg["initializer_width"] < 1664:
        raise ValueError("Mapping/initialization must not introduce a smaller hidden bottleneck")
    expected_loss = {"fm_weight", "struct_weight", "oracle_weight", "warmup_steps", "struct_noise_width"}
    if set(loss) != expected_loss:
        raise ValueError("V4.3 has exactly FM + STRUCT + Flow-Oracle; PRIOR/WRITE/OUT are unsupported")
    if config["data"]["training_mode"] != "i2v" or train.get("state_warmup_every") != 0:
        raise ValueError("V4.3 trains I2V with P reset at each model invocation")
    if "gt_condition_dropout" in train:
        raise ValueError("V4.3 does not expose GT videos to the initializer")
    for key in ("fm_weight", "struct_weight", "oracle_weight"):
        if not math.isfinite(loss[key]) or loss[key] < 0:
            raise ValueError(f"loss.{key} must be finite and nonnegative")
    if loss["fm_weight"] <= 0 or loss["struct_weight"] <= 0:
        raise ValueError("FM and STRUCT are required; only Oracle can be disabled for ablation")
    if loss["warmup_steps"] < 1 or loss["struct_noise_width"] <= 0:
        raise ValueError("Loss warmup and noise width must be positive")
    for key in ("steps", "warmup_steps", "global_batch_size", "save_every"):
        if type(train[key]) is not int or train[key] < 1:
            raise ValueError(f"train.{key} must be positive")
    if not 0 <= train["text_dropout"] <= 1 or not 0 <= train["sigma_min"] < train["sigma_max"] <= 1:
        raise ValueError("Invalid text dropout or sigma range")
    if min(train["learning_rate"], train["grad_clip"]) <= 0:
        raise ValueError("Learning rate and gradient clip must be positive")
    if train["workers"] < 0 or train["loader_timeout_seconds"] < 0 or train["sample_limit"] < 0:
        raise ValueError("Worker/timeout values must be nonnegative")
    validate_oracle(config["oracle"])
    if config["oracle"]["sites_per_micro"] > len(cfg["a_blocks"]):
        raise ValueError("Oracle requests too many sites")
    return config
