"""Verifier-to-C updates inside ONE ordinary Wan denoising trajectory.

There is no candidate selection, completed-video restart, or future reference.
At selected noise levels the existing denoiser evaluation also supplies a local
clean estimate. Its fixed-verifier derivative changes C for subsequent steps.
This local derivative is not the derivative of the eventual complete video.
"""
import torch


def feedback_policy(saved, *, enabled, step_size=None, sigmas=None):
    """Apply explicit inference options without discarding saved coordinates."""
    policy = dict(saved)
    policy['enabled'] = enabled
    if step_size is not None: policy['step_size'] = step_size
    if sigmas is not None: policy['sigmas'] = list(sigmas)
    policy.setdefault('step_size', .1)
    policy.setdefault('sigmas', [.85, .65])
    return policy


def latent_update(gradient, reference, step_size, metric):
    """Steepest descent for the specified coordinates of one video's C.

    With d = C.numel(), s² = mean(C_initial²), relative RMS displacement is
    ||delta||²/(d*s²). Its inverse metric is d*s², not s². Thus a mean-reduced
    objective has the same update when an equivalent latent is replicated.
    Gradient magnitude is retained; this does not force a fixed perturbation.
    Old saved policies without a metric retain their Euclidean convention.
    """
    scale_squared = reference.detach().float().square().mean()
    if metric == 'relative_rms':
        scale_squared = scale_squared*reference.numel()
    elif metric != 'euclidean':
        raise ValueError(f'Unknown C update metric: {metric}')
    return -step_size*scale_squared*gradient


def feedback_indices(sigmas, thresholds):
    thresholds = tuple(float(x) for x in thresholds)
    if not thresholds or any(not 0 < x < 1 for x in thresholds):
        raise ValueError('Feedback noise levels must be between zero and one')
    indices = set()
    for threshold in thresholds:
        available = (sigmas[:-2] <= threshold).nonzero().flatten()
        if not len(available):
            raise ValueError('Feedback needs a subsequent denoising step at this sampling resolution')
        indices.add(int(available[0]))
    return indices


def sample_with_feedback(model, conditions, positive, negative, text, null_text,
                         initial, sigmas, policy, strength):
    cfg = policy['latent_feedback']
    qwen_process = getattr(model, 'qwen_process', False)
    if not model.operator_training or (not qwen_process and model.process_encoder is None):
        raise ValueError('Load the conditional verifier and its observer for online C refinement')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Online C refinement requires frozen model and verifier parameters')
    if cfg['step_size'] <= 0:
        raise ValueError('Use a positive C refinement step size')
    indices = feedback_indices(sigmas, cfg['sigmas'])
    vcfg = model.cfg['verifier']
    fixed = None
    if not qwen_process:
        with torch.no_grad():
            first_rgb = model.decoder(conditions['first'], [0])
            fixed = model.knowledge.encode(first_rgb, conditions['caption'], conditions['clauses'],
                adapt=False, readout='content', duration=float(conditions['times'][-1]-conditions['times'][0]))['contextual']
    metric = cfg.get('metric', 'euclidean')
    # The reference scale stays fixed throughout this one sampling trajectory.
    scale_squared = positive.detach().float().square().mean()
    c, u, x = positive.detach().float(), negative.detach(), initial.detach()
    records = []
    for index in range(len(sigmas)-1):
        current, following = sigmas[index], sigmas[index+1]
        refine = index in indices
        with torch.set_grad_enabled(refine):
            c = c.detach().requires_grad_(refine)
            velocity = model.generator(x, conditions['first'], text, current, c, conditions['times'], strength)
            if policy['guidance'] != 1:
                with torch.no_grad():
                    uncond = model.generator(x, conditions['first'], null_text, current, u, conditions['times'], strength)
                velocity = uncond+policy['guidance']*(velocity-uncond)
            if refine:
                clean = torch.cat((conditions['first'], (x-current*velocity)[:, :, 1:]), 2)
                video = model.decoder(clean, resize=vcfg['size'])
                if qwen_process:
                    features = model.qwen_video_evidence(video, conditions)
                    loss = model.critic.generation_loss(features)
                else:
                    features = model.process_encoder(video)
                    loss = model.critic.generation_loss(features, fixed, conditions['times'], conditions['valid'])
                gradient = torch.autograd.grad(loss, c, allow_unused=True)[0] if loss.requires_grad else None
                if gradient is None: gradient = torch.zeros_like(c)
                update = latent_update(gradient, positive, cfg['step_size'], metric)
                if not torch.isfinite(update).all():
                    raise FloatingPointError('Nonfinite verifier gradient into C')
                records.append(dict(step=index, sigma=float(current), loss=float(loss.detach()),
                    gradient_rms=float(gradient.square().mean().sqrt()),
                    relative_C_update_rms=float(update.square().mean().sqrt()/scale_squared.sqrt().clamp_min(1e-12))))
                c = (c.detach()+update).detach()
                del clean, video, features, loss, gradient, update
        # Continue the SAME state with its already computed velocity. The new C
        # acts at the next step; no denoiser step or finished video is regenerated.
        x = torch.cat((conditions['first'], (x+(following-current)*velocity.detach())[:, :, 1:]), 2)
        del velocity
    model.last_sample_feedback = dict(kind='within_one_trajectory', sampling_steps=len(sigmas)-1,
        complete_trajectories=1, future_reference_used=False, metric=metric,
        latent_elements=positive.numel(), updates=records)
    return x.detach()
