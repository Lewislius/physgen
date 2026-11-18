"""Non-cyclic local space-time attention on the actual Wan video grid."""
import math
import torch
from torch.nn import functional as F


def partition(value, geometry, window=(3, 3, 3), shift=0, pad_value=0):
    """[B,T*H*W,heads,dim] -> [B*windows,heads,window_tokens,dim]."""
    batch, length, heads, dim = value.shape
    if length != math.prod(geometry) or any(not 0 <= shift < size for size in window):
        raise ValueError('Window partition requires the true grid and a bounded offset')
    padded = tuple(math.ceil((size + shift) / win) * win for size, win in zip(geometry, window))
    right = tuple(pad - size - shift for pad, size in zip(padded, geometry))
    grid = F.pad(value.reshape(batch, *geometry, heads, dim),
                 (0, 0, 0, 0, shift, right[2], shift, right[1], shift, right[0]), value=pad_value)
    blocks = tuple(size // win for size, win in zip(padded, window))
    wt, wh, ww = window
    windows = grid.reshape(batch, blocks[0], wt, blocks[1], wh, blocks[2], ww, heads, dim)
    windows = windows.permute(0, 1, 3, 5, 7, 2, 4, 6, 8).reshape(-1, heads, wt * wh * ww, dim)
    return windows, (batch, tuple(geometry), padded, tuple(window), shift)


def unpartition(windows, layout):
    batch, geometry, padded, window, shift = layout
    heads, dim = windows.shape[1], windows.shape[-1]
    blocks = tuple(size // win for size, win in zip(padded, window))
    wt, wh, ww = window
    grid = windows.reshape(batch, *blocks, heads, wt, wh, ww, dim)
    grid = grid.permute(0, 1, 5, 2, 6, 3, 7, 4, 8).reshape(batch, *padded, heads, dim)
    grid = grid[:, shift:shift + geometry[0], shift:shift + geometry[1], shift:shift + geometry[2]]
    return grid.reshape(batch, math.prod(geometry), heads, dim)


def local_attention(q, k, v, geometry, *, window=(3, 3, 3), shift=0, temporal_enabled=True):
    """Padded tokens never supply values; a shift never wraps grid boundaries.

    Temporal-off preserves each window's spatial interaction but removes all
    cross-frame edges. Absolute grid RoPE is applied by the caller before this.
    """
    qw, layout = partition(q.to(v.dtype), geometry, window, shift)
    kw, _ = partition(k.to(v.dtype), geometry, window, shift)
    vw, _ = partition(v, geometry, window, shift)
    batch = v.shape[0]
    frames = torch.arange(geometry[0], device=v.device).reshape(1, -1, 1, 1)
    frames = frames.expand(batch, *geometry).reshape(batch, -1, 1, 1)
    frame_windows, _ = partition(frames, geometry, window, shift, pad_value=-1)
    frame_windows = frame_windows[..., 0]
    allowed = frame_windows[:, :, None, :] >= 0
    if not temporal_enabled:
        allowed = allowed & (frame_windows[:, :, :, None] == frame_windows[:, :, None, :])
    result = F.scaled_dot_product_attention(qw, kw, vw, attn_mask=allowed, dropout_p=0.)
    return unpartition(result, layout)
