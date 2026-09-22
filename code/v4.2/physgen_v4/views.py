import torch.nn.functional as F


def choose_canvas(height, width, max_long_side, max_area):
    buckets = [(h, w) for h in range(32, max_long_side + 1, 32)
               for w in range(32, max_long_side + 1, 32) if h * w <= max_area]
    return max(buckets, key=lambda shape: (min(1.0, shape[0] / height, shape[1] / width),
                                           -shape[0] * shape[1]))


def fit_geometry(height, width, canvas_height, canvas_width, upscale):
    scale = min(canvas_height / height, canvas_width / width)
    if not upscale:
        scale = min(1.0, scale)
    resized_h, resized_w = round(height * scale), round(width * scale)
    return dict(height=height, width=width, resized_height=resized_h, resized_width=resized_w,
                canvas_height=canvas_height, canvas_width=canvas_width,
                top=(canvas_height - resized_h) // 2, left=(canvas_width - resized_w) // 2)


def resize_video(video, canvas_height, canvas_width, upscale):
    """Differentiable BCTHW resize with preserved aspect ratio and centered constant padding."""
    batch, channels, frames, height, width = video.shape
    geo = fit_geometry(height, width, canvas_height, canvas_width, upscale)
    flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).float()
    flat = F.interpolate(flat, size=(geo["resized_height"], geo["resized_width"]),
                         mode="bilinear", align_corners=False, antialias=True)
    flat = F.pad(flat, (geo["left"], canvas_width - geo["resized_width"] - geo["left"],
                       geo["top"], canvas_height - geo["resized_height"] - geo["top"]), value=-1)
    return flat.reshape(batch, frames, channels, canvas_height, canvas_width).permute(0, 2, 1, 3, 4), geo
