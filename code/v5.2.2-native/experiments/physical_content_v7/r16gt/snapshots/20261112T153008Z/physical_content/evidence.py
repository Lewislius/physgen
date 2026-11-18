"""Shared lexical resampling for the R10A-shaped enhancement."""
import torch
from torch.nn import functional as F


def resample_words(words, word_groups, count=256):
    """Use the same lexical sequence for enhancement anchors and semantic targets."""
    if words.ndim == 2:
        words = words[None]
    groups = torch.arange(count, device=words.device) * int(word_groups.max() + 1) // count
    chunks, positions = [], []
    for group in range(int(word_groups.max()) + 1):
        x = words[:, word_groups == group]
        slots = int((groups == group).sum())
        chunks.append(F.interpolate(
            x.transpose(1, 2).float(), size=slots, mode='linear',
            align_corners=True).transpose(1, 2))
        positions.append(torch.linspace(0, len(x[0]) - 1, slots, device=words.device))
    return torch.cat(chunks, 1), groups, torch.cat(positions)
