"""v4-style actual-path STRUCT/PRIOR and one backward, plus v4.2 writer repair."""
import torch

from .backbone import fix_first
from .geometry import Geometry
from .losses import fm_loss, struct_loss, v4_struct_coefficient, structure_components, correction_benefit
from .monitoring import frame_fm_profile
from .runtime import autocast
from .write_supervision import paired_repair_input, writer_repair_loss


def state_supervision(states, target, weight):
    """v4 raw dense MSE over actual states; pooled PRIOR preserves each time slot."""
    target = target.detach().float()
    scale = target.new_ones(())  # v4 does not divide STRUCT/PRIOR by teacher RMS.
    by_site = {name: struct_loss(states[name], target, scale, weight) for name in ('p5', 'p15')}
    struct = torch.stack(list(by_site.values())).mean()
    w = weight.reshape(1, 1, -1, 1).float()
    def pool(value):
        return (value.float()*w).sum(2) / w.sum(2).clamp_min(1e-8)
    prior = (pool(states['p0']) - pool(target)).square().mean()
    return struct, prior, by_site


def gt_conditioned_step(dropout, device):
    """Call once per optimizer step, never independently for each microbatch."""
    return bool(torch.rand((), device=device) >= dropout)


def gradient_diagnostics(fm, auxiliary, adapter):
    """Observe one microbatch before backward; never change .grad or project it."""
    params = list(adapter.parameters())
    gf = torch.autograd.grad(fm, params, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(auxiliary, params, retain_graph=True, allow_unused=True)
    lookup = {id(p): i for i, p in enumerate(params)}
    report = {}
    for name, module in adapter.groups().items():
        zero = fm.detach().new_zeros(())
        pairs = [(gf[lookup[id(p)]], ga[lookup[id(p)]]) for p in module.parameters()]
        f2 = sum((f.float().square().sum() for f, _ in pairs if f is not None), zero)
        a2 = sum((a.float().square().sum() for _, a in pairs if a is not None), zero)
        dot = sum(((f.float()*a.float()).sum() for f, a in pairs if f is not None and a is not None), zero)
        denominator = (f2*a2).sqrt()
        report[name] = dict(fm_norm=float(f2.sqrt()), auxiliary_norm=float(a2.sqrt()),
                            cosine=float(dot/denominator) if float(denominator)>0 else None,
                            conflict=bool(dot<0), fm_dot_joint=float(f2+dot), auxiliary_dot_joint=float(a2+dot))
    return dict(scope='first_micro_before_backward', auxiliary='weighted STRUCT + PRIOR + WRITE', groups=report)


def train_micro_joint(model, sample, exposure, step, accumulation, weights, *,
                      gt_conditioned, empty_text, text_dropout, repair, monitor=False, diagnose_gradients=False):
    a = model.adapter
    device = sample['latent'].device
    geo = Geometry.build(sample['record']['geometry'], device)
    first, xgt, text = sample['first'], sample['latent'], sample['text']
    dropped = bool(torch.rand((), device=device) < text_dropout)
    if dropped:
        text = empty_text
    duration = sample['record']['duration']
    sigma = torch.empty((), device=device).uniform_(.02, .999)
    noise = torch.randn_like(xgt)
    x = fix_first((1-sigma)*xgt + sigma*noise, first)
    gt = xgt if gt_conditioned else None
    with autocast(device):
        p0 = a.initialize(text, geo, exposure['mode'], sample['anchor'], duration=duration, gt_video=gt)
        result = model(x, first, text, sigma, duration, geo, p0, monitor=monitor, retain_state_graph=True)
        fm = fm_loss(result['cond'], noise-xgt, first is not None)
        base_fm = fm_loss(result['base'], noise-xgt, first is not None)
        struct, prior, by_site = state_supervision(result, sample['target'], geo.p_weight)
        warmup = min(1., step/weights['struct_warmup_steps'])
        coefficient = v4_struct_coefficient(sigma, step, weights)
        prior_coefficient = weights['prior']*warmup  # P0 is sigma-independent, as in v4.
        weighted_fm = weights['fm']*fm/accumulation
        weighted_struct = coefficient*struct/accumulation
        weighted_prior = prior_coefficient*prior/accumulation
    write = fm.new_zeros(())
    write_info, repair_info = {}, None
    write_coefficient = weights['write']*warmup*((1-sigma)/weights['struct_noise_width']).clamp(0,1)
    if weights['write'] > 0:
        _, bad, _, repair_info = paired_repair_input(xgt, noise, sigma, first, geo, repair)
        repair_states = (model.hidden_prefix(bad, first, text, sigma, duration, geo,
                         exposure['mode'], sample['anchor'], gt_video=gt) if repair_info['active'] else result)
        with autocast(device):
            for index, site in enumerate((5, 15)):
                loss, info = writer_repair_loss(a.correctors[str(site)].writer,
                    repair_states['observations'][index], repair_states['p5' if site==5 else 'p15'],
                    repair_states['base_observations'][index], result['write_outputs'][index],
                    sigma, geo, first is not None, site, duration)
                write = write + loss/2
                write_info[str(site)] = info
    weighted_write = write_coefficient*write/accumulation
    auxiliary = weighted_struct + weighted_prior + weighted_write
    diagnostics = gradient_diagnostics(weighted_fm, auxiliary, a) if diagnose_gradients else None
    (weighted_fm + auxiliary).backward()
    scalar = lambda value: float(value.detach())
    return dict(fm=scalar(fm), fm_base=scalar(base_fm), fm_delta=scalar(fm-base_fm),
                struct=scalar(struct), struct_by_site={k:scalar(v) for k,v in by_site.items()},
                prior=scalar(prior), write=scalar(write), temp=0.,
                weighted_fm=scalar(weighted_fm), weighted_struct=scalar(weighted_struct),
                weighted_prior=scalar(weighted_prior), weighted_write=scalar(weighted_write), weighted_temp=0.,
                prior_coefficient=prior_coefficient, write_coefficient=scalar(write_coefficient),
                struct_coefficient=scalar(coefficient), struct_warmup=warmup,
                struct_noise_weight=scalar(((1-sigma)/weights['struct_noise_width']).clamp(0,1)),
                struct_scheme='v4_limits', struct_definition='raw_mse_actual_p5_p15_mean',
                gt_conditioned=gt_conditioned, text_dropped=dropped, gradient_diagnostics=diagnostics,
                sigma=scalar(sigma), sigma_a=scalar(sigma), fm_active=True, temporal_active=False, repair=False,
                paired_repair=repair_info, write_supervision=write_info, out_of_range=None,
                structure_components=structure_components(result['p15'].detach(), sample['target'], a.s_z, geo.p_weight),
                correction_benefit=correction_benefit(result['cond'],result['base'],noise-xgt,first is not None),
                frame_fm=frame_fm_profile(result['cond'],result['base'],noise-xgt,first is not None) if monitor else None,
                interventions=result['metrics'])
