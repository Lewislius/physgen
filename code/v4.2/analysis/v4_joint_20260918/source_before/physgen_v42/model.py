import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import interpolate_time, sinusoid
from .monitoring import residual_profile


def rms(x, dims=-1, weight=None):
    square = x.float().square()
    if weight is None:
        return square.mean(dim=dims, keepdim=True).sqrt()
    weight = weight.to(square).expand_as(square)
    return ((square * weight).sum(dim=dims, keepdim=True) /
            weight.sum(dim=dims, keepdim=True).clamp_min(1e-12)).sqrt()


def soft_bound(u, scale, budget, dims=-1, weight=None):
    """Only the reference is detached. The candidate norm MUST retain its gradient."""
    u = u.float()
    a = torch.as_tensor(budget, device=u.device, dtype=torch.float32) * torch.as_tensor(
        scale, device=u.device, dtype=torch.float32).detach()
    if weight is None:
        mean_square = u.square().mean(dim=dims, keepdim=True)
    else:
        weight = weight.to(u).expand_as(u)
        mean_square = (u.square() * weight).sum(dim=dims, keepdim=True) / weight.sum(
            dim=dims, keepdim=True).clamp_min(1e-12)
    return a * u / torch.sqrt(a.square() + mean_square + 1e-12)


def noise_window(sigma):
    def smooth(x):
        x = x.clamp(0, 1)
        return x.square() * (3 - 2 * x)
    return smooth((1 - sigma) / .20) * smooth(sigma / .15)


@torch.no_grad()
def state_update_metrics(candidate, before, after, weight, scale):
    """Measure actual FP32 additions, including suppression by the differentiable bound."""
    weight = weight.reshape(1, 1, -1, 1)
    actual = after.float() - before.float()
    proposed = rms(candidate, (-2, -1), weight)
    effective = rms(actual, (-2, -1), weight)
    reference = rms(before, (-2, -1), weight).clamp_min(1e-12)
    tokens = rms(actual)[..., weight.flatten() > 0, :]
    return dict(candidate_rms=float(proposed.mean()), actual_rms=float(effective.mean()),
                compression=float((effective / proposed.clamp_min(1e-12)).mean()),
                delta_over_before=float((effective / reference).mean()),
                frame_delta_over_before=(effective / reference).flatten().cpu().tolist(),
                token_max_over_teacher_rms=float(tokens.max() / scale),
                candidate_over_teacher_rms=float(proposed.mean() / scale))


def bounded_residual(candidate, base, current, weight, budget, first_known=False, velocity_scale=None, strength=1.):
    """All inputs [B,T,S,C]; two-level bound, excluding pure padding and known slice."""
    weight = weight.reshape(1, 1, -1, 1).to(candidate)
    valid = weight > 0
    u = candidate.float() * valid
    if first_known:
        u = torch.cat((torch.zeros_like(u[:, :1]), u[:, 1:]), dim=1)
    base_frame = rms(base, (-2, -1), weight).detach()
    if velocity_scale is None:
        frame_ref = torch.minimum(base_frame, rms(current, (-2, -1), weight).detach())
        token_ref = torch.minimum(rms(base).detach(), rms(current).detach())
    else:
        frame_ref = torch.maximum(base_frame, base_frame.new_tensor(.1) * velocity_scale)
        token_ref = rms(base).detach()
    token_ref = torch.maximum(token_ref, .1 * frame_ref)
    delta = soft_bound(u, token_ref, 2 * budget)
    delta = soft_bound(delta, frame_ref, budget, (-2, -1), weight)
    # Scale the completed, bounded intervention, not the bound's nonlinear budget.
    # This makes 0/.5/1 a direct intervention on the same site's candidate.
    if strength != 1.:
        delta = delta * strength
    # Addition remains FP32. Logging is done on the ACTUAL difference after addition.
    after = current.float() + delta if velocity_scale is None else base.float() + delta
    actual = after - (current.float() if velocity_scale is None else base.float())
    with torch.no_grad():
        fr = rms(actual, (-2, -1), weight) / frame_ref.clamp_min(1e-12)
        token = (rms(actual) / token_ref.clamp_min(1e-12)).squeeze(-1)[..., valid.flatten()]
        candidate_rms = rms(u, (-2, -1), weight)
        actual_rms = rms(actual, (-2, -1), weight)
        active_frames = fr[:, 1:] if first_known else fr
        stats = dict(frame_ratio=fr.flatten().cpu().tolist(), mean_ratio=float(fr.mean()),
                     token_p95=float(torch.quantile(token.float(), .95)), token_max=float(token.max()),
                     candidate_rms=float(candidate_rms.mean()), actual_rms=float(actual_rms.mean()),
                     compression=float((actual_rms / candidate_rms.clamp_min(1e-12)).mean()),
                     over_base=float((actual_rms / base_frame.clamp_min(1e-12)).mean()),
                     over_current=float((actual_rms / rms(current, (-2, -1), weight).clamp_min(1e-12)).mean()),
                     frame_over_base=(actual_rms / base_frame.clamp_min(1e-12)).flatten().cpu().tolist(),
                     frame_over_current=(actual_rms / rms(current, (-2, -1), weight).clamp_min(1e-12)).flatten().cpu().tolist(),
                     budget=float(budget), applied_strength=float(strength),
                     frame_near_budget_fraction=float((active_frames >= .95*budget).float().mean())
                         if float(budget)>0 and active_frames.numel() else 0.)
    return actual, stats


class FP32Norm(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(x.float(), self.normalized_shape, self.weight, self.bias, self.eps)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, query, context, valid=None):
        def split(x):
            return x.unflatten(-1, (self.heads, x.shape[-1] // self.heads)).transpose(-3, -2)
        q, k, v = split(self.q(query)), split(self.k(context)), split(self.v(context))
        mask = None if valid is None else valid[..., None, None, :].bool()
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        return self.out(result.transpose(-3, -2).flatten(-2))


class Cell(nn.Module):
    def __init__(self, width, heads, temporal=True):
        super().__init__()
        self.temporal = temporal
        self.norms = nn.ModuleList([FP32Norm(width) for _ in range(4)])
        self.spatial = Attention(width, heads)
        self.time = Attention(width, heads) if temporal else None
        self.text = Attention(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x, text, valid):
        b, t, s, c = x.shape
        mask = valid.reshape(1, s).expand(b * t, -1)
        norm = self.norms[0](x).reshape(b * t, s, c)
        x = x.float() + self.spatial(norm, norm, mask).reshape_as(x).float()
        if self.temporal:
            norm = self.norms[1](x).transpose(1, 2).reshape(b * s, t, c)
            x = x + self.time(norm, norm).reshape(b, s, t, c).transpose(1, 2).float()
        norm = self.norms[2](x).reshape(b * t, s, c)
        context = text[:, None].expand(b, t, -1, -1).reshape(b * t, text.shape[1], c)
        x = x + self.text(norm, context).reshape_as(x).float()
        x = x + self.mlp(self.norms[3](x)).float()
        return x * valid.reshape(1, 1, s, 1)


def zero_last(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class StateCore(nn.Module):
    """P queries H for observations, then learns a bounded delta_P in P space."""
    def __init__(self, width=1664, inner=512, hidden=3072, text_dim=4096, heads=8, depth=2, recompute=True):
        super().__init__()
        self.inner, self.recompute = inner, recompute
        self.p_norm, self.h_norm = FP32Norm(width), FP32Norm(hidden)
        self.p_proj, self.h_proj = nn.Linear(width, inner), nn.Linear(hidden, inner)
        self.text_proj = nn.Linear(text_dim, inner)
        self.sigma_proj, self.duration_proj = nn.Linear(inner, inner), nn.Linear(inner, inner)
        self.site_embedding = nn.Embedding(2, inner)
        self.read = Attention(inner, heads)
        self.cells = nn.ModuleList([Cell(inner, heads) for _ in range(depth)])
        self.output = nn.Sequential(FP32Norm(inner), nn.Linear(inner, width), nn.GELU(), nn.Linear(width, width))
        zero_last(self.output[-1])

    def forward(self, p, h, text, sigma, duration, site, geo, scale, return_metrics=False):
        b = p.shape[0]
        hp = h.reshape(b, len(geo.h_phase), -1, h.shape[-1])
        q = self.p_proj(self.p_norm(p)).float() + geo.position(geo.p_phase, geo.p_xy, self.inner)
        d = torch.as_tensor(duration, device=p.device, dtype=torch.float32).reshape(1)
        conditioning = (self.sigma_proj(sinusoid(sigma.reshape(1) * 1000, self.inner)) +
                        self.duration_proj(sinusoid(torch.log(d / 5), self.inner)) + self.site_embedding.weight[site])
        q = q + conditioning.float()[:, None, None]
        observed = self.h_proj(self.h_norm(hp)).float() + geo.position(geo.h_phase, geo.h_xy, self.inner)
        observed = interpolate_time(observed, geo.h_phase, geo.p_phase)
        t, s = len(geo.p_phase), len(geo.p_xy)
        observation = self.read(q.reshape(b * t, s, self.inner), observed.reshape(b * t, -1, self.inner),
                                (geo.h_weight > 0)[None].expand(b * t, -1)).reshape_as(q)
        x = q + .1 * observation.float()
        projected_text = self.text_proj(text)
        for cell in self.cells:
            if self.recompute and torch.is_grad_enabled():
                x = checkpoint(cell, x, projected_text, geo.p_weight > 0, use_reentrant=False)
            else:
                x = cell(x, projected_text, geo.p_weight > 0)
        candidate = self.output(x).float() * (.25 + .75 * (sigma / .15).clamp(0, 1))
        delta = soft_bound(candidate, scale, 1.) * (geo.p_weight > 0)[None, None, :, None]
        after = p.float() + delta
        if return_metrics:
            return after, state_update_metrics(candidate, p, after, geo.p_weight, scale)
        return after


class TextInit(nn.Module):
    def __init__(self, width=1664, inner=512, text_dim=4096, heads=8, depth=2, spatial=576):
        super().__init__()
        self.spatial_queries = nn.Parameter(torch.randn(1, 1, spatial, inner) * .02)
        self.mode_embedding = nn.Parameter(torch.zeros(1, 1, 1, inner))
        self.text_proj = nn.Linear(text_dim, inner)
        self.cells = nn.ModuleList([Cell(inner, heads, temporal=False) for _ in range(depth)])
        self.output = nn.Sequential(FP32Norm(inner), nn.Linear(inner, width), nn.GELU(), nn.Linear(width, width))
        zero_last(self.output[-1])

    def forward(self, text, mu, geo):
        x = self.spatial_queries + self.mode_embedding
        text = self.text_proj(text)
        for cell in self.cells:
            x = cell(x, text, geo.p_weight > 0)
        return mu[None, None].float() + self.output(x).float()


class ConditionTrajectoryInit(nn.Module):
    """Deployable trajectory P0: image/text, time positions and duration only.

    This replaces time replication; it adds no teacher inputs or extra loss.
    Zero output preserves the original anchor/mean at initialization.
    """
    def __init__(self, width=1664, inner=512, text_dim=4096, heads=8, depth=2,
                 spatial=576, recompute=True):
        super().__init__()
        self.inner, self.recompute = inner, recompute
        self.spatial_queries = nn.Parameter(torch.randn(1, 1, spatial, inner)*.02)
        self.mode_embedding = nn.Embedding(2, inner)
        self.anchor_norm, self.anchor_proj = FP32Norm(width), nn.Linear(width, inner)
        self.text_proj, self.duration_proj = nn.Linear(text_dim, inner), nn.Linear(inner, inner)
        self.cells = nn.ModuleList([Cell(inner, heads, temporal=True) for _ in range(depth)])
        self.output = nn.Sequential(FP32Norm(inner), nn.Linear(inner, width), nn.GELU(), nn.Linear(width, width))
        zero_last(self.output[-1])

    def forward(self, text, mu, geo, mode, anchor, duration, scale):
        if mode not in ('i2v', 't2v') or (mode == 'i2v') != (anchor is not None):
            raise ValueError('Trajectory initializer must use deployment mode/anchor conditions')
        base = (anchor.detach().float().reshape(1,1,-1,mu.shape[-1]) if mode == 'i2v'
                else mu[None,None].float())
        duration = torch.as_tensor(duration, device=text.device, dtype=torch.float32).reshape(1)
        if not bool(torch.isfinite(duration).all() & (duration>0).all()):
            raise ValueError('Initializer duration must be finite and positive')
        x = (self.spatial_queries + self.anchor_proj(self.anchor_norm(base)).float()
             + geo.position(geo.p_phase, geo.p_xy, self.inner)
             + self.mode_embedding.weight[0 if mode == 'i2v' else 1]
             + self.duration_proj(sinusoid(torch.log(duration/5),self.inner)).float()[:,None,None])
        context = self.text_proj(text)
        for cell in self.cells:
            if self.recompute and torch.is_grad_enabled():
                x = checkpoint(cell, x, context, geo.p_weight>0, use_reentrant=False)
            else:
                x = cell(x, context, geo.p_weight>0)
        delta = soft_bound(self.output(x).float(), scale, 1.)*(geo.p_weight>0)[None,None,:,None]
        return base + delta


class Writer(nn.Module):
    """H queries the updated P; a learned projection produces delta_H, not delta_P.

    The write gate scales the H-space candidate before its RMS bounds. Neither P
    nor the state core's increment is directly added to the Wan hidden state.
    """
    def __init__(self, width=1664, inner=512, hidden=3072, heads=8, gated=False):
        super().__init__()
        self.inner = inner
        self.h_norm, self.p_norm = FP32Norm(hidden), FP32Norm(width)
        self.h_proj, self.p_proj = nn.Linear(hidden, inner), nn.Linear(width, inner)
        self.attention = Attention(inner, heads)
        self.output = nn.Linear(inner, hidden)
        zero_last(self.output)
        # A nonzero sigmoid gate plus a zero writer preserves the base initially
        # while allowing the output matrix to receive FM gradients on step 1.
        # Missing config means the exact historical architecture/state_dict.
        self.write_gate = nn.Linear(inner, 1) if gated else None
        if self.write_gate is not None:
            zero_last(self.write_gate)

    def forward(self, h, p, base, sigma, ramp, first_known, site, geo, *, strength=1., monitor=False, maps=False, duration=5.):
        if not 0 <= strength <= 1:
            raise ValueError("Writer strength must be in [0, 1]")
        b, _, c = h.shape
        hgrid = h.reshape(b, len(geo.h_phase), -1, c)
        q = self.h_proj(self.h_norm(hgrid)).float() + geo.position(geo.h_phase, geo.h_xy, self.inner)
        kv = self.p_proj(self.p_norm(p)).float() + geo.position(geo.p_phase, geo.p_xy, self.inner)
        kv = interpolate_time(kv, geo.p_phase, geo.h_phase)
        t, s = q.shape[1:3]
        out = self.attention(q.reshape(b * t, s, self.inner), kv.reshape(b * t, -1, self.inner),
                             (geo.p_weight > 0)[None].expand(b * t, -1)).reshape_as(q)
        candidate = self.output(out).float()
        gate = self.write_gate(out).float().sigmoid() if self.write_gate is not None else None
        if gate is not None:
            candidate = candidate * gate
        floor, extra = ((.005, .025) if site == 5 else (.010, .040))
        budget = (floor + extra * noise_window(sigma)) * ramp
        delta, stats = bounded_residual(candidate, base.reshape_as(hgrid), hgrid, geo.h_weight,
                                        budget, first_known, strength=strength)
        if monitor:
            grid_hw = (geo.geo['h']//32, geo.geo['w']//32) if hasattr(geo, 'geo') else None
            stats['delta_h'] = residual_profile(delta, hgrid, geo.h_weight, geo.h_phase, duration,
                                               first_known=first_known, grid_hw=grid_hw, maps=maps)
        if gate is not None:
            stats.update(gate_mean=float(gate.detach().mean()), gate_min=float(gate.detach().min()),
                         gate_max=float(gate.detach().max()))
        return (hgrid.float() + delta).reshape_as(h), stats


class Corrector(nn.Module):
    """One site's independent reader/state core and writer; no cross-site weights."""
    def __init__(self, args, hidden, text_dim, depth, recompute, gated=False):
        super().__init__()
        self.core = StateCore(**args, hidden=hidden, text_dim=text_dim, depth=depth, recompute=recompute)
        self.writer = Writer(**args, hidden=hidden, gated=gated)

    @property
    def reader(self):
        # The reader and its projections are owned by this site's StateCore.
        return self.core.read


class Adapter(nn.Module):
    def __init__(self, cfg, stats, hidden=3072, text_dim=4096, spatial=576):
        super().__init__()
        state = cfg["state"]
        args = dict(width=state["width"], inner=state["inner"], heads=state["heads"])
        self.correctors = nn.ModuleDict({str(site): Corrector(
            args, hidden, text_dim, state["depth"], state["checkpoint"],
            gated=state.get("write_gate", False)) for site in state["sites"]})
        self.initializer = state.get('initializer', 'static_v1')
        if self.initializer == 'condition_trajectory_v1':
            self.text_init = ConditionTrajectoryInit(**args, text_dim=text_dim, depth=state['depth'],
                                                     spatial=spatial, recompute=state['checkpoint'])
        elif self.initializer == 'static_v1':
            self.text_init = TextInit(**args, text_dim=text_dim, depth=state["depth"], spatial=spatial)
        else:
            raise ValueError('Unknown initializer architecture')
        self.register_buffer("mu_first", stats["mu_first"].float())
        self.register_buffer("s_z", torch.as_tensor(stats["s_z"], dtype=torch.float32))
        self.register_buffer("s_v", torch.as_tensor(stats["s_v"], dtype=torch.float32))

    def initialize(self, text, geo, mode, anchor=None, duration=5.):
        if self.initializer == 'condition_trajectory_v1':
            return self.text_init(text, self.mu_first, geo, mode, anchor, duration, self.s_z)
        if mode == "i2v":
            if anchor is None:
                raise ValueError("I2V requires the deployment first-image anchor")
            a = anchor.detach().float().reshape(1, 1, -1, self.mu_first.shape[-1])
        elif mode == "t2v":
            if anchor is not None:
                raise ValueError("T2V cannot receive a sample image anchor")
            a = self.text_init(text, self.mu_first, geo)
        else:
            raise ValueError(mode)
        return a.expand(-1, len(geo.p_phase), -1, -1).contiguous()

    def auxiliary(self, p0, observations, text, sigma, duration, geo, detach_initial=True):
        initial = p0.detach() if detach_initial else p0
        p5 = self.correctors["5"].core(initial, observations[0].detach(), text.detach(), sigma, duration, 0, geo, self.s_z)
        p15 = self.correctors["15"].core(p5, observations[1].detach(), text.detach(), sigma, duration, 1, geo, self.s_z)
        return p15

    def core_parameters(self):
        for corrector in self.correctors.values():
            yield from corrector.core.parameters()

    def auxiliary_parameters(self):
        yield from self.core_parameters()
        yield from self.text_init.parameters()

    def groups(self):
        return dict(core5=self.correctors["5"].core, core15=self.correctors["15"].core,
                    text_init=self.text_init, writer5=self.correctors["5"].writer,
                    writer15=self.correctors["15"].writer)
