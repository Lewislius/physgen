from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def teacher_indices(frames=121):
    if frames < 2:
        raise ValueError("Video teacher requires at least two source frames")
    starts = torch.linspace(0, frames - 2, 16).round().long()
    return torch.stack((starts, starts + 1), dim=1).flatten()


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


def content_rect(geo, teacher=False):
    rect = [geo["left"], geo["top"], geo["left"] + geo["rw"], geo["top"] + geo["rh"]]
    if teacher:
        g = fit_geometry(geo["h"], geo["w"], 384, 384)
        rect = [rect[0] * g["rw"] / geo["w"] + g["left"],
                rect[1] * g["rh"] / geo["h"] + g["top"],
                rect[2] * g["rw"] / geo["w"] + g["left"],
                rect[3] * g["rh"] / geo["h"] + g["top"]]
    return rect


def area_mask(rect, height, width, rows, cols, device=None):
    x0, y0, x1, y1 = rect
    ys = torch.arange(rows, device=device).float() * height / rows
    xs = torch.arange(cols, device=device).float() * width / cols
    wy = (torch.minimum(ys + height / rows, ys.new_tensor(y1)) - ys.clamp(min=y0)).clamp(min=0) / (height / rows)
    wx = (torch.minimum(xs + width / cols, xs.new_tensor(x1)) - xs.clamp(min=x0)).clamp(min=0) / (width / cols)
    return wy[:, None] * wx[None, :]


def sinusoid(values, width):
    frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2, device=values.device).float() / (width // 2))
    x = values.float()[..., None] * frequencies
    return torch.cat((x.sin(), x.cos()), dim=-1)


def interpolate_time(tokens, source, target):
    """[B,T,S,C] -> [B,T',S,C], clamped at the two temporal boundaries."""
    target = target.clamp(source[0], source[-1])
    right = torch.searchsorted(source.contiguous(), target.contiguous()).clamp(1, len(source) - 1)
    left = right - 1
    weight = ((target - source[left]) / (source[right] - source[left]).clamp_min(1e-12)).reshape(1, -1, 1, 1)
    return tokens[:, left] * (1 - weight) + tokens[:, right] * weight


@dataclass
class Geometry:
    geo: dict
    p_phase: torch.Tensor
    h_phase: torch.Tensor
    p_xy: torch.Tensor
    h_xy: torch.Tensor
    p_weight: torch.Tensor
    h_weight: torch.Tensor
    v_weight: torch.Tensor

    @classmethod
    def build(cls, geo, device):
        h, w = geo["h"], geo["w"]
        frames = geo.get("frames", 121)
        latent_frames = 1 + (frames - 1) // 4
        pp = teacher_indices(frames).float().reshape(16, 2).mean(1).to(device) / (frames - 1)
        hp = torch.cat((torch.zeros(1), (torch.arange(1, latent_frames).float() * 4 - 1.5) / (frames - 1))).to(device)
        tg = fit_geometry(h, w, 384, 384)
        py = ((torch.arange(24, device=device) + .5) * 16 - tg["top"]) / tg["rh"]
        px = ((torch.arange(24, device=device) + .5) * 16 - tg["left"]) / tg["rw"]
        hy = (torch.arange(h // 32, device=device) + .5) / (h // 32)
        hx = (torch.arange(w // 32, device=device) + .5) / (w // 32)
        def xy(y, x):
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            return torch.stack((xx, yy), dim=-1).flatten(0, 1)
        pweight = area_mask(content_rect(geo, True), 384, 384, 24, 24, device).flatten()
        hweight = area_mask(content_rect(geo), h, w, h // 32, w // 32, device).flatten()
        vweight = area_mask(content_rect(geo), h, w, h // 16, w // 16, device).flatten()
        if not bool((pweight > 0).any() and (hweight > 0).any()):
            raise ValueError("Empty image content")
        return cls(geo, pp, hp, xy(py, px), xy(hy, hx), pweight, hweight, vweight)

    def position(self, phase, xy, width):
        # Each axis has an even sinusoidal width; trim/pad to the requested channel count.
        n = 2 * math.ceil(width / 6)
        t = sinusoid(phase * 16, n)[:, None].expand(-1, len(xy), -1)
        x = sinusoid(xy[:, 0] * 24, n)[None].expand(len(phase), -1, -1)
        y = sinusoid(xy[:, 1] * 24, n)[None].expand(len(phase), -1, -1)
        return torch.cat((t, y, x), dim=-1)[..., :width].unsqueeze(0)
