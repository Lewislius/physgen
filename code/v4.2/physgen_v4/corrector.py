import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .coordinates import resample_tokens, sinusoid

CONDITION_INITIALIZATIONS = ("image_text", "image_text_gt")


def rms(x):
    return x.detach().float().square().mean().sqrt()


class Attention(nn.Module):
    def __init__(self, width=1664, heads=26):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, x, context, values=None):
        def split(t):
            return t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2)
        values = context if values is None else values
        q, k, v = split(self.query(x)), split(self.key(context)), split(self.value(values))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return self.output(y.transpose(1, 2).reshape_as(x))


class PositionedCrossAttention(nn.Module):
    """Global content retrieval: coordinates/conditions address Q/K, values carry content."""

    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(1664)
        self.context_norm = nn.LayerNorm(1664)
        self.attention = Attention()

    def forward(self, query, context, query_position, context_position, condition):
        q = self.query_norm(query) + query_position + condition
        content = self.context_norm(context)
        return self.attention(q, content + context_position, content)


class CellBlock(nn.Module):
    def __init__(self, width=1664, heads=26):
        super().__init__()
        self.norm_self = nn.LayerNorm(width)
        self.norm_cross = nn.LayerNorm(width)
        self.norm_mlp = nn.LayerNorm(width)
        self.self_attention = Attention(width, heads)
        self.cross_attention = Attention(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x, context):
        z = self.norm_self(x)
        x = x + self.self_attention(z, z)
        x = x + self.cross_attention(self.norm_cross(x), context)
        return x + self.mlp(self.norm_mlp(x))


class ConditionPrior(nn.Module):
    """One P0 initializer, optionally conditioned on a clean training video."""

    def __init__(self, config):
        super().__init__()
        width, heads = config.get("prior_width", 512), config.get("prior_heads", 8)
        depth = config.get("prior_depth", 2)
        if width <= 0 or heads <= 0 or width % heads or depth < 1:
            raise ValueError("P prior needs positive width/depth and width divisible by heads")
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Linear(1664, width, bias=False)
        self.image_norm = nn.LayerNorm(48)
        self.image = nn.Linear(48, width)
        self.text_norm = nn.LayerNorm(1664)
        self.text = nn.Linear(1664, width)
        self.modality = nn.Parameter(torch.zeros(2, width))
        self.use_gt = config.get("initialization") == "image_text_gt"
        if self.use_gt:
            self.video_pool = config.get("prior_video_pool", 8)
            if type(self.video_pool) is not int or self.video_pool < 1:
                raise ValueError("prior_video_pool must be a positive integer")
            self.video_norm = nn.LayerNorm(48)
            self.video = nn.Linear(48, width)
            self.video_modality = nn.Parameter(torch.zeros(width))
        self.blocks = nn.ModuleList([CellBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        # The nonlinear native-width head avoids a fixed (width+1)-dimensional output subspace.
        self.output = nn.Sequential(nn.Linear(width, 1664), nn.GELU(), nn.Linear(1664, 1664))

    def forward(self, first, text, coords, recompute, gt_video=None):
        text = self.text(self.text_norm(text))
        context = text + self.modality[0]
        if first is not None:
            image = first.squeeze(2).flatten(2).transpose(1, 2)
            image = self.image(self.image_norm(image)) + self.position(coords.first_position)
            context = torch.cat((context, image + self.modality[1]), dim=1)
        if gt_video is not None:
            if first is None or not self.use_gt or not self.training:
                raise ValueError("GT video conditioning is only supported by image_text_gt in training mode")
            expected = (first.shape[0], 48, *[a.numel() for a in coords.latent])
            if tuple(gt_video.shape) != expected or gt_video.device != first.device:
                raise ValueError(f"GT video must match the cached clean latent geometry/device: {expected}")
            height, width = (min(size, self.video_pool) for size in gt_video.shape[-2:])
            # All temporal slots and spatial pixels participate; only space is compressed.
            video = F.adaptive_avg_pool3d(gt_video.detach().float(), (gt_video.shape[2], height, width))
            video = video.flatten(2).transpose(1, 2)
            video = (self.video(self.video_norm(video))
                     + self.position(coords.pooled_video_position(height, width)) + self.video_modality)
            context = torch.cat((context, video), dim=1)
        queries = (self.query + self.position(coords.position)).expand(text.shape[0], -1, -1)
        for block in self.blocks:
            if recompute and torch.is_grad_enabled():
                queries = checkpoint(block, queries, context, use_reentrant=False)
            else:
                queries = block(queries, context)
        return self.output(self.norm(queries)).float()


class StateCore(nn.Module):
    """One corrector's complete state update and P/H attention parameters."""

    def __init__(self, config, include_initializer=False):
        super().__init__()
        if include_initializer:
            # Preserve legacy state_dict keys AND optimizer parameter ordering.
            self.initialize = (ConditionPrior(config) if config.get("initialization") in CONDITION_INITIALIZATIONS
                               else nn.Linear(97, 1664))
        self.text = nn.Linear(4096, 1664)
        self.sigma = nn.Sequential(nn.Linear(1664, 1664), nn.SiLU(), nn.Linear(1664, 1664))
        self.site = nn.Embedding(2, 1664)
        self.blocks = nn.ModuleList([CellBlock(), CellBlock()])
        self.norm = nn.LayerNorm(1664)
        self.delta = nn.Linear(1664, 1664)
        self.state_gate = nn.Linear(1664, 1)
        if config.get("fusion", "interpolate") == "cross_attention":
            self.read_attention = PositionedCrossAttention()
            self.write_attention = PositionedCrossAttention()
            self.read_gate = nn.Sequential(nn.LayerNorm(1664), nn.Linear(1664, 1))
            nn.init.zeros_(self.read_gate[-1].weight)
            nn.init.constant_(self.read_gate[-1].bias, -2.0)

    def observe(self, process, hidden, process_position, hidden_position, condition):
        gate = self.read_gate(process).sigmoid()
        observation = self.read_attention(process, hidden, process_position, hidden_position, condition)
        return gate * observation, gate

    def forward(self, z, context, recompute):
        for block in self.blocks:
            if recompute and torch.is_grad_enabled():
                z = checkpoint(block, z, context, use_reentrant=False)
            else:
                z = block(z, context)
        z = self.norm(z)
        return self.delta(z), self.state_gate(z).sigmoid()


class SharedCore(StateCore):
    """Compatibility only for checkpoints predating per-block correctors."""

    def __init__(self, config):
        super().__init__(config, include_initializer=True)


class ProcessInitializer(nn.Module):
    """P0 has its own text projection, separate from every block corrector."""

    def __init__(self, config):
        super().__init__()
        self.initialize = (ConditionPrior(config) if config.get("initialization") in CONDITION_INITIALIZATIONS
                           else nn.Linear(97, 1664))
        if config.get("initialization", "latent") != "latent":
            self.text = nn.Linear(4096, 1664)


class Site(nn.Module):
    def __init__(self, is_b):
        super().__init__()
        self.read_norm = nn.LayerNorm(3072)
        self.read = nn.Linear(3072, 1664)
        self.write_norm = nn.LayerNorm(1664)
        self.writer = nn.Linear(1664, 3072, bias=False)
        self.write_gate = nn.Linear(1664, 1)
        nn.init.zeros_(self.writer.weight)
        if is_b:
            self.b = nn.Parameter(torch.zeros(()))


class BlockCorrector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.core = StateCore(config)
        self.site = Site(False)


class Corrector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.include_b = config.get("include_b", True)
        if config.get("architecture") == "v4_fullwidth3_write_v1":
            if (config.get("a_blocks") != [5, 15, 25] or config.get("a_every") != 0
                    or config.get("parameter_sharing") != "per_block"
                    or config.get("fusion") != "cross_attention" or self.include_b):
                raise ValueError("v4_fullwidth3_write_v1 requires exactly three independent V4 correctors at 5/15/25, without B")
        # Missing in older checkpoint configs: preserve their trained initialization path.
        self.initialization = config.get("initialization", "latent")
        self.fusion = config.get("fusion", "interpolate")
        if self.initialization not in ("latent", "latent_text", *CONDITION_INITIALIZATIONS):
            raise ValueError(f"Unknown P initialization: {self.initialization}")
        if self.fusion not in ("interpolate", "cross_attention"):
            raise ValueError(f"Unknown P/H fusion: {self.fusion}")
        if config.get("a_every", 0) < 0 or config.get("early_write_width", 0) < 0:
            raise ValueError("a_every and early_write_width must be nonnegative")
        if not 0 <= config.get("early_write_floor", 0.1) <= 1:
            raise ValueError("early_write_floor must be between zero and one")
        # Missing only in legacy checkpoint configs. New training declares per_block.
        self.parameter_sharing = config.get("parameter_sharing", "shared")
        if self.parameter_sharing not in ("shared", "per_block"):
            raise ValueError(f"Unknown corrector parameter_sharing: {self.parameter_sharing}")
        if type(config.get("a_every", 0)) is not int:
            raise ValueError("a_every must be an integer")
        if type(config.get("iterations")) is not int or config["iterations"] < 1:
            raise ValueError("iterations must be a positive integer")
        if self.parameter_sharing == "per_block":
            blocks = config.get("a_blocks")
            if (not isinstance(blocks, (list, tuple)) or not blocks
                    or any(type(n) is not int or n < 1 for n in blocks)
                    or list(blocks) != sorted(set(blocks))):
                raise ValueError("per_block requires explicit, strictly increasing positive integer a_blocks")
            if type(config.get("block_b")) is not int or config["block_b"] < 1:
                raise ValueError("block_b must be a positive integer")
            interval = config.get("a_every", 0)
            if interval and list(blocks) != list(range(interval, blocks[-1] + 1, interval)):
                raise ValueError("a_blocks must agree with a_every (use a_every=0 for an explicit custom layout)")
            self.a_blocks = tuple(blocks)
            self.prior = ProcessInitializer(config)
            # Construct each network separately. Repeating a module reference would tie weights.
            self.A = nn.ModuleDict({str(n): BlockCorrector(config) for n in self.a_blocks})
            if self.include_b:
                self.B_core = StateCore(config)
        else:
            self.shared = SharedCore(config)
            self.A = Site(False)
        if self.include_b:
            self.B = Site(True)

    def set_stage(self, stage):
        if stage not in ("A", "B", "AB"):
            raise ValueError(f"Unknown corrector stage: {stage}")
        if stage != "A" and not self.include_b:
            raise ValueError("This architecture contains only the three A correctors; B/AB is unavailable")
        self.requires_grad_(False)
        if stage in ("A", "AB"):
            (self.prior if self.parameter_sharing == "per_block" else self.shared).requires_grad_(True)
            self.A.requires_grad_(True)
        if stage in ("B", "AB"):
            self.B.requires_grad_(True)
            if self.parameter_sharing == "per_block":
                self.B_core.requires_grad_(True)

    def parameter_groups(self):
        if self.parameter_sharing == "per_block":
            groups = {"initialization": list(self.prior.parameters())}
            groups.update({f"A@{n}": list(self.A[str(n)].parameters()) for n in self.a_blocks})
            if self.include_b:
                groups["B"] = list(self.B_core.parameters()) + list(self.B.parameters())
            return {name: active for name, parameters in groups.items()
                    if (active := [p for p in parameters if p.requires_grad])}
        groups = {
            "initialization": [p for p in self.shared.initialize.parameters() if p.requires_grad],
            "shared": [p for name, p in self.shared.named_parameters()
                       if p.requires_grad and not name.startswith("initialize.")],
            "A": [p for p in self.A.parameters() if p.requires_grad],
            "B": [p for p in self.B.parameters() if p.requires_grad],
        }
        return {name: parameters for name, parameters in groups.items() if parameters}

    def modules_for(self, name, block=None):
        """Resolve by physical Wan block; never fall back to another A's parameters."""
        if name not in ("A", "B"):
            raise ValueError(f"Unknown corrector family: {name}")
        if name == "B" and not self.include_b:
            raise ValueError("No B corrector exists in this architecture")
        if self.parameter_sharing == "shared":
            return self.shared, self.A if name == "A" else self.B
        if name == "B":
            if block is not None and block != self.config["block_b"]:
                raise ValueError(f"B block {block} does not match block_b={self.config['block_b']}")
            return self.B_core, self.B
        if block is None and len(self.a_blocks) == 1:
            block = self.a_blocks[0]
        if type(block) is not int or block not in self.a_blocks:
            raise ValueError(f"A requires an explicit registered block from {self.a_blocks}; got {block}")
        unit = self.A[str(block)]
        return unit.core, unit.site

    def gt_condition_parameters(self):
        """Only these parameters are unused when the GT modality is absent."""
        if self.initialization != "image_text_gt":
            return []
        prior = (self.prior if self.parameter_sharing == "per_block" else self.shared).initialize
        return [prior.video_modality, *prior.video_norm.parameters(), *prior.video.parameters()]

    def load_checkpoint_weights(self, weights, source_config, allow_new_initializer=False, allow_new_fusion=False,
                                allow_new_parameter_sharing=False):
        source = source_config.get("initialization", "latent")
        replace = self.initialization in CONDITION_INITIALIZATIONS and source in ("latent", "latent_text")
        add_gt = self.initialization == "image_text_gt" and source == "image_text"
        if source != self.initialization and not (replace or add_gt):
            raise RuntimeError(f"Cannot change P initialization from {source} to {self.initialization}")
        source_fusion = source_config.get("fusion", "interpolate")
        replace_fusion = self.fusion == "cross_attention" and source_fusion == "interpolate"
        source_sharing = source_config.get("parameter_sharing", "shared")
        split = self.parameter_sharing == "per_block" and source_sharing == "shared"
        if source_sharing != self.parameter_sharing and not split:
            raise RuntimeError("Cannot merge per-block correctors into a shared checkpoint architecture")
        if split and not allow_new_parameter_sharing:
            raise RuntimeError("Splitting shared correctors requires INIT_FROM in a trainable A/AB stage; "
                               "RESUME must retain the saved architecture and optimizer groups")
        if source_sharing == self.parameter_sharing == "per_block":
            if (tuple(source_config.get("a_blocks", ())) != self.a_blocks
                    or source_config.get("block_b") != self.config.get("block_b")):
                raise RuntimeError("Checkpoint block layout mismatch; cannot silently relabel trained correctors")
        if (replace or add_gt) and not allow_new_initializer:
            raise RuntimeError("A new image_text prior requires INIT_FROM in a stage that trains the initializer (A/AB); "
                               "do not freeze an untrained prior in B1. RESUME must use the saved architecture.")
        if replace_fusion and not allow_new_fusion:
            raise RuntimeError("New cross_attention fusion requires INIT_FROM in a trainable A/AB stage; "
                               "RESUME must keep the saved fusion and B1 cannot freeze new attention weights.")
        if split:
            transferred = {}
            for name, value in weights.items():
                if name.startswith("shared.initialize."):
                    transferred["prior.initialize." + name[len("shared.initialize."):]] = value
                elif name.startswith("shared."):
                    suffix = name[len("shared."):]
                    for n in self.a_blocks:
                        transferred[f"A.{n}.core.{suffix}"] = value
                    transferred["B_core." + suffix] = value
                    if suffix.startswith("text.") and hasattr(self.prior, "text"):
                        transferred["prior." + suffix] = value
                elif name.startswith("A."):
                    for n in self.a_blocks:
                        transferred[f"A.{n}.site." + name[2:]] = value
                else:
                    transferred[name] = value
            # load_state_dict copies values into independently allocated Parameters (assign=False).
            weights = transferred
        initializer_prefix = "prior.initialize." if self.parameter_sharing == "per_block" else "shared.initialize."
        prefixes = (initializer_prefix,) if replace else ()
        if add_gt:
            prefixes += tuple(initializer_prefix + suffix for suffix in ("video_norm.", "video.", "video_modality"))
        if replace and source_sharing == self.parameter_sharing == "per_block" and source == "latent":
            # A latent-only independent prior has no text projection to transfer.
            prefixes += ("prior.text.",)
        if replace_fusion:
            cores = ([f"A.{n}.core." for n in self.a_blocks] + ["B_core."]
                     if self.parameter_sharing == "per_block" else ["shared."])
            prefixes += tuple(core + suffix for core in cores
                              for suffix in ("read_attention.", "write_attention.", "read_gate."))
        transferred = {name: value for name, value in weights.items()
                       if not (replace and name.startswith(initializer_prefix))}
        expected_state = self.state_dict()
        expected_missing = {name for name in expected_state if name.startswith(prefixes)}
        missing = set(expected_state) - set(transferred)
        unexpected = set(transferred) - set(expected_state)
        wrong_shapes = [name for name in set(expected_state) & set(transferred)
                        if expected_state[name].shape != transferred[name].shape]
        # Validate everything before mutating the model; only declared new modules may be absent.
        if missing != expected_missing or unexpected or wrong_shapes:
            raise RuntimeError(f"Unexpected checkpoint mismatch: missing={sorted(missing - expected_missing)}, "
                               f"unexpected={sorted(unexpected)}, shapes={sorted(wrong_shapes)}, "
                               f"supplied_new_keys={sorted(expected_missing - missing)}")
        self.load_state_dict(transferred, strict=not bool(prefixes))
        if replace_fusion:
            # A writer trained on interpolated P must not consume the new retrieval features.
            sites = [unit.site for unit in self.A.values()] if self.parameter_sharing == "per_block" else [self.A]
            with torch.no_grad():
                for site in sites + [self.B]:
                    site.writer.weight.zero_()
                self.B.b.zero_()
        return dict(initializer=(f"new {self.initialization} prior" if replace else
                                 "added GT video condition" if add_gt else "restored"),
                    source_initialization=source, source_fusion=source_fusion,
                    fusion="new cross_attention" if replace_fusion else "restored",
                    parameter_sharing="split into independent copies" if split else self.parameter_sharing,
                    source_parameter_sharing=source_sharing,
                    writers="zero restarted; B state scale zero" if replace_fusion else "restored")

    def recomputed(self, function, *args):
        if self.config["checkpoint_cell"] and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def initialize(self, first, text, coords, *, noisy=None, gt_video=None):
        initializer = self.prior if self.parameter_sharing == "per_block" else self.shared
        if gt_video is not None and self.initialization != "image_text_gt":
            raise ValueError("GT conditioning requires initialization=image_text_gt")
        if self.initialization in CONDITION_INITIALIZATIONS:
            return initializer.initialize(first, initializer.text(text), coords, self.config["checkpoint_cell"], gt_video)
        if first is None:
            raise ValueError("Text-only inference requires an image_text or image_text_gt condition prior")
        # Compatibility only: deployed old checkpoints retain their original latent initializer.
        if noisy is None:
            raise ValueError("Legacy P initialization requires the current noisy latent")
        batch, _, length, height, width = noisy.shape
        first = first.expand(batch, 48, length, height, width)
        mask = torch.cat((noisy.new_ones(batch, 1, 1, height, width),
                          noisy.new_zeros(batch, 1, length - 1, height, width)), dim=2)
        inputs = torch.cat((noisy, first, mask), dim=1).flatten(2).transpose(1, 2)
        # Small B updates accumulate in FP32 instead of rounding away against a BF16 state.
        process = initializer.initialize(resample_tokens(inputs, coords.latent, coords.process)).float()
        if self.initialization == "latent_text":
            # Callers supply valid, unpadded UMT5 tokens, including empty/negative prompts.
            # Pool before projection so initialization adds only one text vector.
            prompt = initializer.text(text.float().mean(dim=1, keepdim=True)).float()
            process = process + prompt
        return process

    def forward(self, name, hidden, process, text, sigma, coords, block=None):
        core, site = self.modules_for(name, block)
        cfg = self.config
        q = torch.ones_like(sigma) if name == "A" else ((cfg["handoff_sigma"] - sigma) / cfg["handoff_width"]).clamp(0, 1)
        state_scale = torch.ones_like(sigma) if name == "A" else site.b.tanh()
        # The low-noise taper changes update amplitude, independently of loss weighting.
        taper = cfg["late_floor"] + (1 - cfg["late_floor"]) * (sigma / cfg["late_sigma"]).clamp(0, 1)
        eta = cfg["state_strength"] * q * taper * state_scale
        gamma = cfg["write_strength"] * q * taper
        early_width = cfg.get("early_write_width", 0.0)
        early = torch.ones_like(sigma)
        if early_width > 0:
            floor = cfg.get("early_write_floor", 0.1)
            early = floor + (1 - floor) * ((1 - sigma) / early_width).clamp(0, 1)
        gamma = gamma * early
        hidden_tokens = site.read(site.read_norm(hidden))
        cross = self.fusion == "cross_attention"
        if not cross:
            observation = resample_tokens(hidden_tokens, coords.hidden, coords.process)
        context = core.text(text)
        condition = core.sigma(sinusoid(sigma, 1664)).reshape(1, 1, 1664)
        condition = condition + core.site.weight[0 if name == "A" else 1]
        if block is not None:
            condition = condition + sinusoid(sigma.new_tensor(float(block)), 1664).reshape(1, 1, 1664)
        position = coords.position.to(process.dtype)
        before = process
        for _ in range(cfg["iterations"]):
            if cross:
                # Re-query the same H using the current P at each internal iteration.
                observation, read_gate = self.recomputed(core.observe, process, hidden_tokens,
                    position, coords.hidden_position, condition)
            delta, gate = core(process + observation + position + condition, context, cfg["checkpoint_cell"])
            increment = (eta * gate * delta).to(process.dtype)
            process = process + increment
        residual, write_gate, retrieved = self.write_residual(name, hidden, process, sigma, coords, block,
            hidden_tokens=hidden_tokens, condition=condition, gamma=gamma)
        metrics = {
            "p_before_rms": rms(before), "p_after_rms": rms(process),
            "candidate_rms": rms(delta), "effective_update_rms": rms(process - before),
            "last_increment_rms": rms(increment), "hidden_rms": rms(hidden), "write_rms": rms(residual),
            "write_relative_rms": rms(residual) / (rms(hidden) + 1e-8),
            "state_gate_mean": gate.detach().float().mean(),
            "state_gate_low_fraction": (gate.detach() < 0.05).float().mean(),
            "state_gate_high_fraction": (gate.detach() > 0.95).float().mean(),
            "write_gate_mean": write_gate.detach().float().mean(),
            "write_gate_low_fraction": (write_gate.detach() < 0.05).float().mean(),
            "write_gate_high_fraction": (write_gate.detach() > 0.95).float().mean(),
            "q": q.detach(), "state_scale": state_scale.detach(), "early_write_scale": early.detach(),
        }
        if cross:
            metrics.update(read_gate_mean=read_gate.detach().float().mean(),
                           observation_rms=rms(observation), retrieval_rms=rms(retrieved))
        return hidden + residual, process, before.detach(), metrics

    def write_residual(self, name, hidden, process, sigma, coords, block=None, *,
                       hidden_tokens=None, condition=None, gamma=None):
        """The original V4 write path, also callable with detached paired-repair inputs.

        There is no 512-dimensional bottleneck, per-token/frame RMS bound, or
        post-Wan velocity bound. Normal FM retains the complete computation graph.
        """
        core, site = self.modules_for(name, block)
        cfg = self.config
        if hidden_tokens is None:
            hidden_tokens = site.read(site.read_norm(hidden))
        if condition is None:
            condition = core.sigma(sinusoid(sigma, 1664)).reshape(1, 1, 1664)
            condition = condition + core.site.weight[0 if name == "A" else 1]
            if block is not None:
                condition = condition + sinusoid(sigma.new_tensor(float(block)), 1664).reshape(1, 1, 1664)
        if gamma is None:
            q = torch.ones_like(sigma) if name == "A" else ((cfg["handoff_sigma"] - sigma) / cfg["handoff_width"]).clamp(0, 1)
            taper = cfg["late_floor"] + (1 - cfg["late_floor"]) * (sigma / cfg["late_sigma"]).clamp(0, 1)
            gamma = cfg["write_strength"] * q * taper
            if cfg.get("early_write_width", 0.0) > 0:
                floor = cfg.get("early_write_floor", 0.1)
                gamma = gamma * (floor + (1 - floor) * ((1 - sigma) / cfg["early_write_width"]).clamp(0, 1))
        if self.fusion == "cross_attention":
            indices = coords.process_valid_indices
            retrieved = self.recomputed(core.write_attention, hidden_tokens,
                process.index_select(1, indices), coords.hidden_position,
                coords.position.to(process.dtype).index_select(1, indices), condition)
        else:
            retrieved = process
        normalized = site.write_norm(retrieved)
        gate = site.write_gate(normalized).sigmoid()
        residual = gate * site.writer(normalized)
        if self.fusion != "cross_attention":
            residual = resample_tokens(residual, coords.process, coords.hidden)
        return (gamma * residual).to(hidden.dtype), gate, retrieved
