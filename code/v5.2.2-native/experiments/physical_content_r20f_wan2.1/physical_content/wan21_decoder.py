"""Official Wan2.1 VAE decoder for [1, 16, 21, 60, 104] latents.

Wan2.1 uses a 16-channel, 4x8x8 VAE.  The Wan2.2 ``VideoDecoder`` used by the
5B candidate loads ``vae2_2`` and a 48-channel checkpoint; it is deliberately
not imported here.
"""
from pathlib import Path

import torch
from torch import nn

from .wan21_runtime import assert_wan21_source, install_wan21_namespace


class Wan21Decoder(nn.Module):
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        install_wan21_namespace(cfg['paths']['wan_code'])
        assert_wan21_source(cfg['paths']['wan_code'])
        from wan.modules.vae import WanVAE

        checkpoint = Path(cfg['paths']['wan_checkpoint']) / 'Wan2.1_VAE.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if int(cfg['data']['latent_shape'][0]) != 16:
            raise ValueError('Wan2.1 decoder requires 16 latent channels')
        self.vae = WanVAE(z_dim=16, vae_pth=str(checkpoint),
                          dtype=torch.bfloat16, device=device)

    @torch.no_grad()
    def forward(self, latent):
        """Decode one latent into [T, 3, H, W] RGB in [-1, 1]."""
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 16:
            raise ValueError(f'Expected [1,16,T,H,W] Wan2.1 latent, got {tuple(latent.shape)}')
        video = self.vae.decode([latent[0].float()])[0]
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(f'Unexpected Wan2.1 VAE output: {tuple(video.shape)}')
        return video.permute(1, 0, 2, 3).float().clamp(-1, 1)
