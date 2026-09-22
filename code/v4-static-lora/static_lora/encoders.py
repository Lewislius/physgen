"""Native T5 and FP32 VAE for LoRA inference; no teacher model."""
from pathlib import Path

import torch


def load_vae(paths, device):
    from wan.modules.vae2_2 import Wan2_2_VAE
    # FP32 means FP32 convolution/matmul as well as FP32 storage, not TF32 internals.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    # The wrapper's encode/decode autocast and decode clamp are deliberately bypassed.
    vae = Wan2_2_VAE(vae_pth=str(Path(paths["wan_checkpoint"]) / "Wan2.2_VAE.pth"),
                    dtype=torch.float32, device=device)
    if any(p.dtype != torch.float32 for p in vae.model.parameters()):
        raise ValueError("Original VAE checkpoint must contain FP32 weights; do not widen a BF16 cache")
    vae.model.eval().requires_grad_(False)
    if any(s.dtype != torch.float32 for s in vae.scale):
        raise ValueError("VAE scale must be original FP32")
    return vae


@torch.no_grad()
def encode_vae(vae, video):
    with torch.autocast(video.device.type, enabled=False):
        return vae.model.encode(video.float(), vae.scale).float()


def load_text(paths, device):
    from wan.modules.t5 import T5EncoderModel
    root = Path(paths["wan_checkpoint"])
    return T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=device,
                          checkpoint_path=str(root / "models_t5_umt5-xxl-enc-bf16.pth"),
                          tokenizer_path=str(root / "google/umt5-xxl"))


class Decoder:
    """FP32 causal decoder with streaming frame export."""
    def __init__(self, vae, recompute=True, save_on_cpu=True):
        self.vae, self.recompute, self.save_on_cpu = vae, recompute, save_on_cpu

    def _initial(self, latent):
        from wan.modules.vae2_2 import count_conv3d
        z = latent.float() / self.vae.scale[1].view(1, 48, 1, 1, 1) + self.vae.scale[0].view(1, 48, 1, 1, 1)
        return self.vae.model.conv2(z), tuple([None] * count_conv3d(self.vae.model.decoder))


    @torch.no_grad()
    def frames(self, latent):
        from wan.modules.vae2_2 import unpatchify
        with torch.autocast(latent.device.type, enabled=False):
            z, cache = self._initial(latent)
            for i in range(z.shape[2]):
                current = list(cache)
                out = self.vae.model.decoder(z[:, :, i:i + 1], feat_cache=current, feat_idx=[0], first_chunk=(i == 0))
                cache = tuple(current)
                rgb = (unpatchify(out, patch_size=2).float() + 1) * .5
                for frame in rgb[0].transpose(0, 1):
                    yield frame
