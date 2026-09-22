"""Exact frame-block attention using rectangular, noncausal FlashAttention calls.

Wan flattens [F, H, W] in that order. Every query in frame i attends to ALL
spatial tokens in [0, i], or [max(0, i-K), i]. K counts previous VAE latent
frames; the entire current frame is always included. No token-triangular mask
or token-distance window is used, and no full [F*H*W, F*H*W] mask is allocated.
"""
import sys
from types import MethodType

import torch


def validate_attention(spec):
    if not isinstance(spec, dict) or set(spec) != {"mode", "previous_frames"}:
        raise ValueError("Attention requires mode and previous_frames")
    mode, previous = spec["mode"], spec["previous_frames"]
    if mode == "frame_causal_full":
        if previous is not None:
            raise ValueError("frame_causal_full requires previous_frames: null")
    elif mode == "frame_causal_window":
        if type(previous) is not int or previous < 0:
            raise ValueError("frame_causal_window requires integer previous_frames >= 0")
    else:
        raise ValueError(f"Unknown frame attention mode: {mode}")
    return dict(spec)


def frame_attention(q, k, v, grid_sizes, seq_lens, previous_frames, kernel):
    """Differentiable attention for the existing unpadded microbatch=1 recipe.

RoPE must already have been applied at absolute video positions. The kernel
has Wan's flash_attention signature, and receives physically sliced K/V, so
even a FlashAttention implementation without local windows respects the mask.
    """
    if previous_frames is not None and (type(previous_frames) is not int or previous_frames < 0):
        raise ValueError("previous_frames must be None or a nonnegative integer")
    if (q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1
            or tuple(grid_sizes.shape) != (1, 3) or tuple(seq_lens.shape) != (1,)):
        raise ValueError("Frame attention requires one unpadded video and matching Q/K/V")
    frames, height, width = grid_sizes.tolist()[0]
    if any(type(n) is not int or n <= 0 for n in (frames, height, width)):
        raise ValueError("Frame grid dimensions must be positive integers")
    spatial = height * width
    if frames * spatial != q.shape[1] or seq_lens.item() != q.shape[1]:
        raise ValueError("Sequence length must exactly equal F*H*W; padding is unsupported")

    # Native Wan RoPE returns FP32; its FlashAttention casts to V's half dtype.
    # Cast once before slicing to avoid repeatedly converting overlapping K/V.
    # Restore the native FP32 output before the LoRA output projection.
    output_dtype = q.dtype
    if q.is_cuda:
        kernel_dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        q, k, v = (value.to(kernel_dtype).contiguous() for value in (q, k, v))
    outputs = []
    for frame in range(frames):
        first = 0 if previous_frames is None else max(0, frame - previous_frames)
        stop = (frame + 1) * spatial
        outputs.append(kernel(q=q[:, frame * spatial:stop],
                              k=k[:, first * spatial:stop], v=v[:, first * spatial:stop],
                              causal=False, window_size=(-1, -1)))
    return torch.cat(outputs, dim=1).to(output_dtype)


def _forward(self, x, seq_lens, grid_sizes, freqs):
    batch, length = x.shape[:2]
    shape = (batch, length, self.num_heads, self.head_dim)
    q = self.norm_q(self.q(x)).view(shape)
    k = self.norm_k(self.k(x)).view(shape)
    v = self.v(x).view(shape)
    q = self._frame_rope_apply(q, grid_sizes, freqs)
    k = self._frame_rope_apply(k, grid_sizes, freqs)
    attended = frame_attention(q, k, v, grid_sizes, seq_lens,
                               self._frame_attention_spec["previous_frames"], self._frame_kernel)
    return self.o(attended.flatten(2))


def install_frame_attention(wan, spec):
    """Change only each self-attention forward; retain all modules/LoRA names."""
    spec = validate_attention(spec)
    if tuple(wan.patch_size)[0] != 1:
        raise ValueError("One temporal patch must equal one compressed VAE frame")
    for block in wan.blocks:
        attention = block.self_attn
        if hasattr(attention, "_frame_attention_spec"):
            raise ValueError("Frame attention is already installed")
        if tuple(attention.window_size) != (-1, -1):
            raise ValueError("Expected original global Wan self-attention")
    for block in wan.blocks:
        attention = block.self_attn
        native = sys.modules[type(attention).__module__]
        attention._frame_rope_apply = native.rope_apply
        attention._frame_kernel = native.flash_attention
        attention._frame_attention_spec = dict(spec)
        attention.forward = MethodType(_forward, attention)
    return wan

