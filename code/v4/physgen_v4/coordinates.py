from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from .views import fit_geometry


def sinusoid(values, width):
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(width // 2, device=values.device).float() / (width // 2)
    )
    angles = values.float().unsqueeze(-1) * frequencies
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


def encode_position(time, y, x, seconds_scale):
    tt, yy, xx = torch.meshgrid(time, y, x, indexing="ij")
    return torch.cat((sinusoid(tt / seconds_scale, 512), sinusoid(yy * 24, 576),
                      sinusoid(xx * 24, 576)), dim=-1).reshape(1, -1, 1664)


def teacher_indices(frames, teacher_frames):
    count = 2 * (min(frames, teacher_frames) // 2)
    return torch.linspace(0, frames - 1, count).round().long()


def fractional_index(source, target):
    if source.numel() == 1:
        return torch.zeros_like(target)
    upper = torch.searchsorted(source.contiguous(), target.contiguous()).clamp(1, source.numel() - 1)
    lower = upper - 1
    weight = (target - source[lower]) / (source[upper] - source[lower])
    return (lower + weight) * (2.0 / (source.numel() - 1)) - 1.0


def resample_tokens(tokens, source, target):
    """Fixed trilinear sampling; axes are seconds and normalized canvas coordinates."""
    time, y, x = [fractional_index(a, b) for a, b in zip(source, target)]
    tt, yy, xx = torch.meshgrid(time, y, x, indexing="ij")
    grid = torch.stack((xx, yy, tt), dim=-1).unsqueeze(0)
    volume = tokens.reshape(tokens.shape[0], *[a.numel() for a in source], tokens.shape[-1])
    volume = volume.permute(0, 4, 1, 2, 3)
    result = F.grid_sample(volume.float(), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return result.flatten(2).transpose(1, 2).to(tokens.dtype)


@dataclass
class Coordinates:
    latent: tuple
    hidden: tuple
    process: tuple
    position: torch.Tensor
    first_position: torch.Tensor
    hidden_position: torch.Tensor
    process_valid_indices: torch.Tensor
    latent_canvas: tuple
    seconds_scale: float

    def pooled_video_position(self, height, width):
        """Use the same spatial averaging bins as the GT latent, retaining every time slot."""
        time, y, x = self.latent_canvas
        y = F.adaptive_avg_pool1d(y[None, None], height).flatten()
        x = F.adaptive_avg_pool1d(x[None, None], width).flatten()
        return encode_position(time, y, x, self.seconds_scale)

    @classmethod
    def build(cls, times, height, width, teacher_frames, seconds_scale, teacher_size=384):
        device = times.device
        latent_t = times[::4]
        indices = teacher_indices(times.numel(), teacher_frames).to(device)
        process_t = times[indices].reshape(-1, 2).mean(-1)

        def centers(size, stride):
            return (torch.arange(size // stride, device=device).float() + 0.5) * stride / size

        latent = (latent_t, centers(height, 16), centers(width, 16))
        hidden = (latent_t, centers(height, 32), centers(width, 32))
        teacher_y = centers(teacher_size, 16)
        teacher_x = centers(teacher_size, 16)
        geo = fit_geometry(height, width, teacher_size, teacher_size, upscale=True)
        process_y = (teacher_y * teacher_size - geo["top"]) / geo["resized_height"]
        process_x = (teacher_x * teacher_size - geo["left"]) / geo["resized_width"]
        process = (process_t, process_y, process_x)
        position = encode_position(process_t, teacher_y, teacher_x, seconds_scale)
        # Map first-image tokens into the same letterboxed coordinate system as P queries.
        first_y = (latent[1] * geo["resized_height"] + geo["top"]) / teacher_size
        first_x = (latent[2] * geo["resized_width"] + geo["left"]) / teacher_size
        first_position = encode_position(latent_t[:1], first_y, first_x, seconds_scale)
        hidden_y = (hidden[1] * geo["resized_height"] + geo["top"]) / teacher_size
        hidden_x = (hidden[2] * geo["resized_width"] + geo["left"]) / teacher_size
        hidden_position = encode_position(latent_t, hidden_y, hidden_x, seconds_scale)
        # Do not let H read unsupervised tokens entirely inside the teacher's letterbox.
        # Construct indices from geometry, without a GPU boolean-index shape synchronization.
        side = teacher_size // 16
        valid = [t * side * side + y * side + x
                 for t in range(process_t.numel()) for y in range(side) for x in range(side)
                 if y * 16 < geo["top"] + geo["resized_height"] and (y + 1) * 16 > geo["top"]
                 and x * 16 < geo["left"] + geo["resized_width"] and (x + 1) * 16 > geo["left"]]
        indices = torch.tensor(valid, device=device, dtype=torch.long)
        return cls(latent, hidden, process, position, first_position, hidden_position, indices,
                   (latent_t, first_y, first_x), seconds_scale)
