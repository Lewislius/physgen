"""Full-width C alignment, initialized from the context the native reader sees.

Statistics use only cached training captions and the frozen Wan text embedding.
They initialize a representation, not a T5 imitation objective. All learned
alignment parameters are saved with C; resume and inference need no calibration.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def native_text_context(wan, text):
    text = text[:, :wan.text_len]
    return wan.text_embedding(F.pad(text, (0, 0, 0, wan.text_len-text.shape[1])))


class FullWidthAlignment(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=eps, elementwise_affine=False)
        self.gain = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, content):
        # Normalize in float32, then return the model's compute dtype. This is
        # always width -> width; gain and bias each have one value per channel.
        value = self.norm(content.float())*self.gain.float()+self.bias.float()
        return value.to(content.dtype)

    @torch.no_grad()
    def initialize(self, mean, std):
        if mean.shape != self.bias.shape or std.shape != self.gain.shape:
            raise ValueError('Native context statistics must cover every C channel')
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()) or bool((std < 0).any()):
            raise ValueError('Native context statistics are not finite nonnegative scales')
        self.gain.copy_(std.to(self.gain))
        self.bias.copy_(mean.to(self.bias))


@torch.no_grad()
def context_statistics(wan, cfg):
    from .data import manifest, load_tensor
    rows = [row for row in manifest(cfg) if row['split'] == 'train']
    selected = cfg['data'].get('train_ids')
    if selected is not None:
        by_id = {row['id']: row for row in rows}
        rows = [by_id[name] for name in selected]
    count = min(cfg['alignment']['samples'], len(rows))
    if count < 1: raise ValueError('Context initialization requires training captions')
    indices = torch.linspace(0, len(rows)-1, count).long().tolist()
    parameter = next(wan.text_embedding.parameters())
    total = total_square = None
    tokens = 0; valid_tokens = 0; ids = []
    for index in indices:
        row = rows[index]; ids.append(row['id'])
        text = load_tensor(Path(cfg['paths']['cache'])/'base/text'/f'{row["id"]}.pt')
        valid_tokens += min(text.shape[0], wan.text_len)
        # The live model is BF16 with BF16 autocast in training. Supplying its
        # parameter dtype reproduces that text path without loading Qwen/video.
        with torch.autocast(parameter.device.type, enabled=False):
            context = native_text_context(wan, text[None].to(parameter)).detach()
        flat = context.reshape(-1, context.shape[-1]).double()
        summed, squared = flat.sum(0), flat.square().sum(0)
        total = summed if total is None else total+summed
        total_square = squared if total_square is None else total_square+squared
        tokens += flat.shape[0]
    mean = total/tokens
    std = (total_square/tokens-mean.square()).clamp_min(0).sqrt()
    mean, std = mean.float().cpu(), std.float().cpu()
    report = dict(source='cached train captions -> native text_embedding, including native padding',
                  train_ids=ids, samples=count, tokens=tokens, unpadded_tokens=valid_tokens,
                  width=mean.numel(), mean_rms=float(mean.square().mean().sqrt()),
                  std_rms=float(std.square().mean().sqrt()), std_min=float(std.min()),
                  std_max=float(std.max()), context_rms=float((mean.square()+std.square()).mean().sqrt()),
                  compute_dtype=str(parameter.dtype), initialization_only=True)
    return dict(mean=mean, std=std, report=report)


def initialize_condition_alignment(model, cfg):
    if not hasattr(model.conditioner, 'alignment'): return None
    statistics = context_statistics(model.generator.wan, cfg)
    model.conditioner.alignment.initialize(statistics['mean'], statistics['std'])
    return statistics
