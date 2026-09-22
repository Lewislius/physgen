"""Use the same attention policy for training, validation, and inference."""
from static_lora.backbone import load_model as single_load_model
from static_lora.runtime import log
from four_gpu.backbone import load_model as four_load_model
from .attention import install_frame_attention
from .config import variant_name


def load_model(cfg, device, training=True):
    loader = four_load_model if cfg["comparison"]["profile"] == "4x48g" else single_load_model
    model = loader(cfg, device, training=training)
    install_frame_attention(model.wan, cfg["attention"])
    log(event="frame_attention_installed", variant=variant_name(cfg["attention"]),
        attention=cfg["attention"], frame_unit="vae_latent_frame", intra_frame="global",
        current_frame_included=True, self_attention_blocks=len(model.wan.blocks))
    return model

