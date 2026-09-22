from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .backbone import shard_blocks
from .coordinates import teacher_indices
from .views import resize_video
from .environments import require_environment


def load_vae(paths, device):
    require_environment('runtime')
    from wan.modules.vae2_2 import Wan2_2_VAE
    vae = Wan2_2_VAE(vae_pth=str(Path(paths["wan_checkpoint"]) / "Wan2.2_VAE.pth"),
                    dtype=torch.bfloat16, device=device)
    vae.model.to(dtype=torch.bfloat16)
    return vae


def load_text_encoder(paths, device):
    require_environment('runtime')
    from wan.modules.t5 import T5EncoderModel
    root = Path(paths["wan_checkpoint"])
    return T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=device,
                          checkpoint_path=str(root / "models_t5_umt5-xxl-enc-bf16.pth"),
                          tokenizer_path=str(root / "google/umt5-xxl"))


class VideoTeacher(nn.Module):
    def __init__(self, paths, device, sharded=False, recompute=False):
        super().__init__()
        require_environment('teacher')
        from app.vjepa_2_1.models.vision_transformer import vit_gigantic_xformers
        # Construct on CPU: the official constructor reads a CPU drop-path schedule.
        encoder = vit_gigantic_xformers(
            img_size=(384, 384), patch_size=16, num_frames=64, tubelet_size=2,
            use_sdpa=True, use_silu=False, wide_silu=True, uniform_power=False,
            use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
            use_activation_checkpointing=recompute and not sharded, n_output_distillation=4,
        )
        state = torch.load(paths["teacher_checkpoint"], map_location="cpu", mmap=True, weights_only=True)["target_encoder"]
        state = {key.replace("module.", "").replace("backbone.", ""): value for key, value in state.items()}
        encoder.load_state_dict(state, strict=True, assign=True)
        encoder.eval().requires_grad_(False).to(dtype=torch.bfloat16)
        if sharded:
            # Checkpoint inside FSDP so recomputation does not repeat its outer hooks.
            shard_blocks(encoder.blocks, device, recompute=recompute)
            for name, module in encoder.named_children():
                if name != "blocks":
                    module.to(device)
            encoder.img_mod_embed.data = encoder.img_mod_embed.data.to(device)
            encoder.video_mod_embed.data = encoder.video_mod_embed.data.to(device)
        else:
            encoder.to(device)
        self.encoder = encoder
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1, 1))

    def forward(self, video, frame_count):
        indices = teacher_indices(video.shape[2], frame_count).to(video.device)
        selected = video[:, :, indices].float()
        selected, _ = resize_video(selected, 384, 384, upscale=True)
        normalized = ((selected + 1) * 0.5 - self.mean) / self.std
        return self.encoder(normalized, training=False)


class DifferentiableDecoder:
    """Wan causal decoding with explicit cache arguments and chunk recomputation."""

    def __init__(self, vae, recompute):
        self.vae = vae
        self.recompute = recompute

    def __call__(self, latent):
        from wan.modules.vae2_2 import count_conv3d, unpatchify
        model = self.vae.model
        z = latent / self.vae.scale[1].view(1, 48, 1, 1, 1) + self.vae.scale[0].view(1, 48, 1, 1, 1)
        z = model.conv2(z)
        cache = tuple([None] * count_conv3d(model.decoder))
        outputs = []
        for index in range(z.shape[2]):
            def decode_chunk(chunk, previous, first_chunk=(index == 0)):
                current = list(previous)
                out = model.decoder(chunk, feat_cache=current, feat_idx=[0], first_chunk=first_chunk)
                return out, tuple(current)
            if self.recompute and torch.is_grad_enabled():
                out, cache = checkpoint(decode_chunk, z[:, :, index:index + 1], cache, use_reentrant=False)
            else:
                out, cache = decode_chunk(z[:, :, index:index + 1], cache)
            outputs.append(out)
        return unpatchify(torch.cat(outputs, dim=2), patch_size=2).float().clamp(-1, 1)
