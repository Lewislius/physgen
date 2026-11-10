"""Lossless state patches, plus validity-aware reductions for losses and summaries."""
import torch.nn.functional as F


def pack_grid(value, size=2):
    """B,C,...,H,W -> B,C*size**2,...,ceil(H/size),ceil(W/size)."""
    h, w = value.shape[-2:]
    value = F.pad(value, (0, (-w) % size, 0, (-h) % size))
    return F.pixel_unshuffle(value.movedim(1, -3), size).movedim(-3, 1)


def unpack_grid(value, shape, size=2):
    """Invert packing and remove only padding; do not average or interpolate."""
    return F.pixel_shuffle(value.movedim(1, -3), size).movedim(-3, 1)[..., :shape[0], :shape[1]]


def pool_grid(value, valid, shape):
    """Pool arbitrary leading dimensions without mixing time or channels."""
    h, w = value.shape[-2:]
    shape = tuple(shape)
    if tuple(valid.shape) != (h, w):
        raise ValueError("Spatial validity does not match the source grid")
    weight = valid.to(value)[None, None]
    denominator = F.adaptive_avg_pool2d(weight, shape)
    numerator = F.adaptive_avg_pool2d(value.reshape(-1, 1, h, w)*weight, shape)
    pooled = numerator/denominator.clamp_min(1e-8)
    return pooled.reshape(*value.shape[:-2], *shape), denominator[0, 0]

