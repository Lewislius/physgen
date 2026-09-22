from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import fit_geometry, letterbox, teacher_indices


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


class VideoTeacher(nn.Module):
    def __init__(self, paths, device):
        super().__init__()
        sys.path.insert(0, paths["teacher_code"])
        from app.vjepa_2_1.models.vision_transformer import vit_gigantic_xformers
        encoder = vit_gigantic_xformers(
            img_size=(384, 384), patch_size=16, num_frames=64, tubelet_size=2,
            use_sdpa=True, use_silu=False, wide_silu=True, uniform_power=False,
            use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
            use_activation_checkpointing=False, n_output_distillation=4)
        state = torch.load(paths["teacher_checkpoint"], map_location="cpu", mmap=True, weights_only=True)["target_encoder"]
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        encoder.load_state_dict(state, strict=True, assign=True)
        self.encoder = encoder.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
        self.register_buffer("mean", torch.tensor([.485, .456, .406], device=device).reshape(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225], device=device).reshape(1, 3, 1, 1, 1))

    @torch.no_grad()
    def forward(self, video, first_image=False):
        selected = video[:, :, :1].expand(-1, -1, 32, -1, -1) if first_image else video[:, :, teacher_indices(video.shape[2])]
        geo = fit_geometry(selected.shape[-2], selected.shape[-1], 384, 384)
        selected = letterbox(selected, geo)
        with torch.autocast(video.device.type, dtype=torch.bfloat16, cache_enabled=False):
            encoded = self.encoder(((selected.float() + 1) * .5 - self.mean) / self.std, training=False)
        if encoded.shape != (1, 16 * 24 * 24, 1664):
            raise ValueError(f"JEPA output must be final [1,9216,1664]; got {encoded.shape}")
        encoded = encoded.reshape(1, 16, 576, 1664)
        return encoded.float().mean(dim=1) if first_image else encoded


def small_frame(x, longest):
    h, w = x.shape[-2:]
    size = (max(1, round(h * min(1, longest / max(h, w)))), max(1, round(w * min(1, longest / max(h, w)))))
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)


class Decoder:
    """FP32 causal decoder. Explicit cache is differentiable across all 31 chunks."""
    def __init__(self, vae, recompute=True, save_on_cpu=True):
        self.vae, self.recompute, self.save_on_cpu = vae, recompute, save_on_cpu

    def _initial(self, latent):
        from wan.modules.vae2_2 import count_conv3d
        z = latent.float() / self.vae.scale[1].view(1, 48, 1, 1, 1) + self.vae.scale[0].view(1, 48, 1, 1, 1)
        return self.vae.model.conv2(z), tuple([None] * count_conv3d(self.vae.model.decoder))

    def views(self, latent, local_pairs):
        """Return only downsampled global RGB and requested local crops, never full RGB history."""
        from wan.modules.vae2_2 import unpatchify
        frame_count = 4 * (latent.shape[2] - 1) + 1
        by_frame = {}
        for index, spec in enumerate(local_pairs):
            if not 0 <= spec["pair"] < frame_count - 1:
                raise ValueError("Local temporal pair is outside the decoded clip")
            for side, frame in enumerate((spec["pair"], spec["pair"] + 1)):
                by_frame.setdefault(frame, []).append((index * 2 + side, spec["box"]))
        scope = torch.autograd.graph.save_on_cpu(pin_memory=True) if (
            self.save_on_cpu and latent.is_cuda and torch.is_grad_enabled()) else nullcontext()
        with scope, torch.autocast(latent.device.type, enabled=False):
            z, cache = self._initial(latent)
            globals_, crops, out_of_range = [], {}, []
            for i in range(z.shape[2]):
                start = 0 if i == 0 else 1 + 4 * (i - 1)
                def chunk(value, previous, first=(i == 0), offset=start):
                    current = list(previous)
                    with torch.autocast(value.device.type, enabled=False):
                        out = self.vae.model.decoder(value, feat_cache=current, feat_idx=[0], first_chunk=first)
                        rgb = (unpatchify(out, patch_size=2).float() + 1) * .5
                        flat = rgb[0].transpose(0, 1)
                        global_rgb = small_frame(flat, 384)
                        local = {}
                        for j in range(flat.shape[0]):
                            for key, box in by_frame.get(offset + j, []):
                                x0, y0, x1, y1 = box
                                local[key] = small_frame(flat[j:j + 1, :, y0:y1, x0:x1], 192)[0]
                        outside = ((rgb.detach() < 0) | (rgb.detach() > 1)).float().sum()
                        return global_rgb, local, tuple(current), outside
                if self.recompute and torch.is_grad_enabled():
                    g, local, cache, outside = checkpoint(chunk, z[:, :, i:i + 1], cache, use_reentrant=False)
                else:
                    g, local, cache, outside = chunk(z[:, :, i:i + 1], cache)
                globals_.append(g)
                crops.update(local)
                out_of_range.append(outside)
            global_rgb = torch.cat(globals_)
            if global_rgb.shape[0] != frame_count or len(crops) != 2 * len(local_pairs):
                raise RuntimeError("Causal VAE chunk length/local crop coverage mismatch")
            return dict(global_rgb=global_rgb, crops=crops,
                        out_of_range=torch.stack(out_of_range).sum() /
                        (frame_count * 3 * latent.shape[-2] * 16 * latent.shape[-1] * 16))

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
