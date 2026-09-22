"""Extra WRITE loss on V4's actual 3072-channel Wan residual, with no bottleneck.

The clean FM pass is the detached reference. A paired, lightly perturbed latent
pass supplies detached H/P inputs to the very same V4 write attention and writer.
This is reconstruction supervision, not a labelled physical counterexample.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def paired_input(clean, noisy, first, sigma, settings):
    b, c, t, h, w = clean.shape
    field = F.interpolate(torch.randn(b, 2, 3, 3, 3, device=clean.device),
                          size=(t, h, w), mode="trilinear", align_corners=True)
    field /= field.square().mean().sqrt().clamp_min(1e-8)
    ys = (torch.arange(h, device=clean.device) + .5) * 2 / h - 1
    xs = (torch.arange(w, device=clean.device) + .5) * 2 / w - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((xx, yy), -1)[None, None].expand(b, t, h, w, 2).clone()
    pixels = settings["warp_latent_pixels"]
    grid += field.permute(0, 2, 3, 4, 1) * grid.new_tensor([2 * pixels / w, 2 * pixels / h])
    source = clean.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    warped = F.grid_sample(source, grid.reshape(b * t, h, w, 2), mode="bilinear",
                           padding_mode="border", align_corners=False)
    warped = warped.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    drift = F.interpolate(torch.randn(b, c, 3, 2, 2, device=clean.device),
                          size=(t, h, w), mode="trilinear", align_corners=True)
    scale = clean.float().square().mean().sqrt().clamp_min(1e-6)
    drift = drift / drift.square().mean().sqrt().clamp_min(1e-8) * scale
    disturbance = .75 * (warped - clean) + .25 * drift
    if first is not None:
        disturbance[:, :, 0] = 0
    disturbance *= scale / disturbance.square().mean().sqrt().clamp_min(1e-8)
    strength = torch.rand((), device=clean.device) * settings["max_relative_strength"]
    student = noisy.detach() + sigma * (1 - sigma) * strength * disturbance
    if first is not None:
        student = torch.cat((first, student[:, :, 1:]), dim=2)
    return student, True


def native_write_loss(corrector, reference, student, sigma, coords, first_known):
    """Average raw native-H MSE over the three sites; target/H/P are detached.

    No budget normalization, target clipping, gate targets, or gradient projection.
    FM/STRUCT retain their original, fully connected V4 graph. This additional
    branch trains only parameters used by V4's write_residual, including retrieval.
    """
    if set(reference) != set(student) or len(reference) != len(corrector.a_blocks):
        raise ValueError("WRITE needs paired captures for every registered corrector")
    losses, metrics = [], {}
    first_tokens = len(coords.hidden[1]) * len(coords.hidden[2]) if first_known else 0
    for label, pair in student.items():
        hidden = pair["before"].detach()
        state = pair["process"].detach()
        reference_after = reference[label]["after"].detach().float()
        residual, _, _ = corrector.write_residual(pair["family"], hidden, state, sigma,
                                                  coords, pair["block"])
        desired = reference_after - hidden.float()
        # Match the actual V4 addition, including its autocast rounding. An
        # identical pair has exactly zero loss, even when H is BF16.
        error = (hidden + residual).float()[:, first_tokens:] - reference_after[:, first_tokens:]
        value = error.square().mean()
        losses.append(value)
        with torch.no_grad():
            baseline = desired[:, first_tokens:].square().mean()
            metrics.update({f"write/{label}/mse": value.detach(),
                f"write/{label}/no_write_mse": baseline,
                f"write/{label}/mse_gain": baseline - value.detach(),
                f"write/{label}/target_rms": baseline.sqrt(),
                f"write/{label}/residual_rms": residual.float()[:, first_tokens:].square().mean().sqrt()})
    return torch.stack(losses).mean(), metrics


def joint_gradient_diagnostics(terms, groups):
    """Observe weighted gradients on one microbatch, never modify optimizer .grad.

    All active parameters are included, grouped as initializer/A@5/A@15/A@25.
    This is not the cosine of an accumulated optimizer update or a quality metric.
    """
    names = list(terms)
    parameters = [p for values in groups.values() for p in values]
    gradients = {}
    for name, value in terms.items():
        gradients[name] = (torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                           if value.requires_grad else (None,) * len(parameters))
    result, offset = {}, 0
    device = next(iter(terms.values())).device
    for group, values in groups.items():
        vectors = {name: gradient[offset:offset + len(values)] for name, gradient in gradients.items()}
        offset += len(values)
        def dot(left, right):
            parts = [(a.detach().float() * b.detach().float()).sum()
                     for a, b in zip(vectors[left], vectors[right]) if a is not None and b is not None]
            return torch.stack(parts).sum() if parts else torch.zeros((), device=device)
        squares = {name: dot(name, name) for name in names}
        for name in names:
            result[f"loss_grad/{group}/{name}_norm"] = squares[name].sqrt()
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                product = dot(left, right)
                denominator = (squares[left] * squares[right]).sqrt()
                prefix = f"loss_grad/{group}/{left}_{right}"
                result[prefix + "_cosine"] = product / denominator.clamp_min(1e-30)
                result[prefix + "_both_nonzero"] = (denominator > 0).float()
        result[f"loss_grad/{group}/fm_dot_joint"] = sum(dot("fm", name) for name in names)
    result["loss_grad/single_micro_only"] = torch.ones((), device=device)
    return result
