"""Canonical image bucketing and letterbox; no other experiment imports."""
import torch
import torch.nn.functional as F


def fit_geometry(h, w, target_h, target_w, upscale=True):
    if min(h, w, target_h, target_w) <= 0:
        raise ValueError("Source and target image dimensions must be positive")
    ratio = min(target_h / h, target_w / w)
    if not upscale:
        ratio = min(1., ratio)
    rh, rw = max(1, min(target_h, round(h * ratio))), max(1, min(target_w, round(w * ratio)))
    return dict(source_h=h, source_w=w, h=target_h, w=target_w, rh=rh, rw=rw,
                top=(target_h - rh) // 2, left=(target_w - rw) // 2)


def canvas(h, w, max_long_side=512, max_area=147456):
    """v4's exact 32-pixel bucket selection, without importing or editing v4."""
    buckets = [(a, b) for a in range(32, max_long_side + 1, 32)
               for b in range(32, max_long_side + 1, 32) if a * b <= max_area]
    return max(buckets, key=lambda shape: (min(1., shape[0] / h, shape[1] / w), -shape[0] * shape[1]))


def letterbox(video, geo, fill=-1.):
    """CTHW or BCTHW, with the same integer geometry for images and videos."""
    batched = video.ndim == 5
    if not batched:
        video = video.unsqueeze(0)
    b, c, t, h, w = video.shape
    x = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    x = F.interpolate(x, size=(geo["rh"], geo["rw"]), mode="bilinear", align_corners=False, antialias=True)
    x = F.pad(x, (geo["left"], geo["w"] - geo["rw"] - geo["left"],
                  geo["top"], geo["h"] - geo["rh"] - geo["top"]), value=fill)
    x = x.reshape(b, t, c, geo["h"], geo["w"]).permute(0, 2, 1, 3, 4)
    return x if batched else x.squeeze(0)
