"""V4.3: JEPA state lifted into Wan width, bidirectional native-width interaction.

Persistent state lives in hidden_width after initialization. Independent JEPA
readouts supervise that same state; H is never compressed to JEPA width.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .coordinates import sinusoid

ARCHITECTURE = "v43_native_joint_flow_oracle_v1"
CONDITION_INITIALIZATIONS = ("image_text",)


def rms(x):
    return x.detach().float().square().mean().sqrt()


def recompute(enabled, function, *args):
    if enabled and torch.is_grad_enabled():
        return checkpoint(function, *args, use_reentrant=False)
    return function(*args)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, query, context):
        def split(x):
            return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
        q, k, v = split(self.query(query)), split(self.key(context)), split(self.value(context))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.output(out.transpose(1, 2).reshape_as(query))


class ResidualMap(nn.Module):
    """Untied linear skip plus nonlinear residual; output retains absolute scale."""
    def __init__(self, source, target, expansion):
        super().__init__()
        self.skip = nn.Linear(source, target)
        self.norm = nn.LayerNorm(source)
        self.hidden = nn.Linear(source, expansion)
        self.output = nn.Linear(expansion, target)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return (self.skip(x) + self.output(F.silu(self.hidden(self.norm(x))))).float()


class PriorBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = Attention(width, heads)
        self.cross_norm = nn.LayerNorm(width)
        self.cross_attention = Attention(width, heads)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x, context):
        z = self.self_norm(x)
        x = x + self.self_attention(z, z)
        x = x + self.cross_attention(self.cross_norm(x), context)
        return x + self.ff(self.ff_norm(x))


class ConditionInitializer(nn.Module):
    """First latent + full text + time/space queries; never sees future GT."""
    def __init__(self, cfg):
        super().__init__()
        width = cfg["initializer_width"]
        self.checkpoint = cfg["checkpoint_cell"]
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Linear(cfg["position_width"], width, bias=False)
        self.image = nn.Sequential(nn.LayerNorm(cfg["latent_channels"]), nn.Linear(cfg["latent_channels"], width))
        self.text = nn.Sequential(nn.LayerNorm(cfg["text_width"]), nn.Linear(cfg["text_width"], width))
        self.modality = nn.Parameter(torch.zeros(2, width))
        self.blocks = nn.ModuleList([PriorBlock(width, cfg["initializer_heads"])
                                    for _ in range(cfg["initializer_depth"])])
        self.output = ResidualMap(width, cfg["jepa_width"], max(width, cfg["jepa_width"]))

    def forward(self, first, text, coords):
        context = self.text(text) + self.modality[0]
        if first is not None:
            image = first.squeeze(2).flatten(2).transpose(1, 2)
            image = self.image(image) + self.position(coords.first_position) + self.modality[1]
            context = torch.cat((context, image), dim=1)
        queries = (self.query + self.position(coords.position)).expand(text.shape[0], -1, -1)
        for block in self.blocks:
            queries = recompute(self.checkpoint, block, queries, context)
        return self.output(queries)


class Stream(nn.Module):
    """Separate feature calibration for P and H, with identical working width."""
    def __init__(self, width, heads, ratio):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm_attention = nn.LayerNorm(width, elementwise_affine=False)
        self.norm_ff = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.qkv = nn.Linear(width, 3 * width)
        self.q_norm = nn.LayerNorm(width // heads)
        self.k_norm = nn.LayerNorm(width // heads)
        self.output = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, ratio * width), nn.GELU(), nn.Linear(ratio * width, width))
        # Active internal interaction from the start; only the final H writer is zero-init.
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        with torch.no_grad():
            self.modulation[-1].bias[2 * width:3 * width].fill_(0.1)
            self.modulation[-1].bias[5 * width:6 * width].fill_(0.1)

    def prepare(self, x, position, condition):
        shift, scale, gate, ff_shift, ff_scale, ff_gate = self.modulation(condition).chunk(6, -1)
        content = self.norm_attention(x) * (1 + scale) + shift
        q, k, v = self.qkv(content).chunk(3, -1)
        # Coordinates address Q/K only, leaving values in the learned content space.
        def split(z):
            return z.reshape(z.shape[0], z.shape[1], self.heads, -1).transpose(1, 2)
        q, k, v = self.q_norm(split(q + position)), self.k_norm(split(k + position)), split(v)
        return q, k, v, (gate, ff_shift, ff_scale, ff_gate)

    def finish(self, x, attention, modulation):
        gate, shift, scale, ff_gate = modulation
        attention = attention.transpose(1, 2).reshape_as(x)
        x = x.float() + gate.float() * self.output(attention).float()
        z = self.norm_ff(x) * (1 + scale) + shift
        return x + ff_gate.float() * self.ff(z).float()


class JointPHBlock(nn.Module):
    """One joint softmax over P/H keys: P↔H and within-stream interaction."""
    def __init__(self, cfg):
        super().__init__()
        args = (cfg["hidden_width"], cfg["heads"], cfg["mlp_ratio"])
        self.p = Stream(*args)
        self.h = Stream(*args)

    def forward(self, p, h, p_position, h_position, condition):
        pq, pk, pv, pm = self.p.prepare(p, p_position, condition)
        hq, hk, hv, hm = self.h.prepare(h, h_position, condition)
        query, key, value = (torch.cat(parts, dim=2) for parts in ((pq, hq), (pk, hk), (pv, hv)))
        # SDPA selects the memory-efficient CUDA kernel; no dense attention matrix is stored.
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        pa, ha = attended.split((p.shape[1], h.shape[1]), dim=2)
        return self.p.finish(p, pa, pm), self.h.finish(h, ha, hm)


class NativeSite(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["hidden_width"]
        self.checkpoint = cfg["checkpoint_cell"]
        self.position = nn.Linear(cfg["position_width"], width, bias=False)
        self.sigma = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.text = nn.Linear(cfg["text_width"], width)
        self.text_norm = nn.LayerNorm(width)
        self.p_norm = nn.LayerNorm(width)
        self.text_attention = Attention(width, cfg["heads"])
        self.types = nn.Parameter(torch.zeros(2, width))
        self.blocks = nn.ModuleList([JointPHBlock(cfg) for _ in range(cfg["depth"])])
        self.to_jepa = ResidualMap(width, cfg["jepa_width"], cfg["mapping_width"])
        self.write_norm = nn.LayerNorm(width)
        self.writer = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.writer.weight)

    def forward(self, hidden, process, text, sigma, coords, write_mask):
        indices = coords.process_valid_indices
        # Exclude pure teacher-letterbox tokens from every joint attention, not just the writer.
        p = process.index_select(1, indices)
        p_position = self.position(coords.position.index_select(1, indices)) + self.types[0]
        h_position = self.position(coords.hidden_position) + self.types[1]
        context = self.text_norm(self.text(text))
        condition = self.sigma(sinusoid(sigma.reshape(-1), hidden.shape[-1])).unsqueeze(1)
        condition = condition + context.mean(dim=1, keepdim=True)
        p = p.float() + recompute(self.checkpoint, self.text_attention, self.p_norm(p), context).float()
        h = hidden
        for block in self.blocks:
            p, h = recompute(self.checkpoint, block, p, h, p_position, h_position, condition)
        state = process.float().index_copy(1, indices, p.float())
        jepa = recompute(self.checkpoint, self.to_jepa, state)
        delta = self.writer(self.write_norm(h)).float() * write_mask
        # FP32 addition preserves a small oracle-supervised residual against a large BF16 H.
        corrected = hidden.float() + delta
        metrics = dict(hidden_rms=rms(hidden), write_rms=rms(delta),
                       write_relative_rms=rms(delta) / rms(hidden).clamp_min(1e-6),
                       state_rms=rms(state), state_change_rms=rms(state - process), jepa_rms=rms(jepa))
        return corrected, state, jepa, delta, metrics


class Corrector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        if config.get("architecture") != ARCHITECTURE:
            raise ValueError("V4.3 requires its own architecture/checkpoint; V4.2 weights are incompatible")
        for key in ("hidden_width", "jepa_width", "position_width", "text_width", "latent_channels",
                    "heads", "depth", "mlp_ratio", "mapping_width", "initializer_width", "initializer_heads", "initializer_depth"):
            if type(config.get(key)) is not int or config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if config["hidden_width"] % config["heads"] or config["hidden_width"] % 2:
            raise ValueError("hidden_width must be even and divisible by heads")
        if config["initializer_width"] % config["initializer_heads"]:
            raise ValueError("initializer_width must be divisible by initializer_heads")
        self.a_blocks = tuple(config["a_blocks"])
        if not self.a_blocks or list(self.a_blocks) != sorted(set(self.a_blocks)) or any(type(n) is not int or n < 1 for n in self.a_blocks):
            raise ValueError("a_blocks must contain sorted, unique, positive layer numbers")
        self.initialization, self.fusion, self.parameter_sharing = "image_text", "native_joint", "per_block"
        self.include_b = False
        self.initializer = ConditionInitializer(config)
        self.lift = ResidualMap(config["jepa_width"], config["hidden_width"], config["mapping_width"])
        self.A = nn.ModuleDict({str(n): NativeSite(config) for n in self.a_blocks})

    def initialize(self, first, text, coords, *, gt_video=None, **kwargs):
        if gt_video is not None:
            raise ValueError("GT is supervision only; V4.3 initialization cannot consume a future video")
        return recompute(self.config["checkpoint_cell"], self.lift,
                         self.initializer(first, text, coords))

    def forward(self, hidden, process, text, sigma, coords, block, write_mask):
        return self.A[str(block)](hidden, process, text, sigma, coords, write_mask)

    def set_stage(self, stage):
        if stage != "A":
            raise ValueError("V4.3 has only its native joint A correctors")
        self.requires_grad_(True)

    def parameter_groups(self):
        return {"initialization": [*self.initializer.parameters(), *self.lift.parameters()],
                **{f"A@{n}": list(site.parameters()) for n, site in self.A.items()}}

    def load_checkpoint_weights(self, weights, source_config, **kwargs):
        if source_config != self.config:
            raise ValueError("V4.3 checkpoint architecture/configuration must match exactly")
        self.load_state_dict(weights, strict=True)
        return dict(architecture=ARCHITECTURE, restored=True)
