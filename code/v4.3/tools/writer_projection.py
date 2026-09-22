"""Shared parameter proposal from Oracle displacements, not per-example writes.

For a frozen prefix and a linear writer, d = Z @ W.T. This computes a single
ridge-regression DeltaW across examples such that Z @ DeltaW.T approximates the
Oracle displacement. It does NOT accept a model update or certify FM/quality.
Real BF16 execution, downstream FM, held-out examples and video quality still
need separate evaluation. The first controlled use should be A@25 only.
"""
import math
import torch


class SharedWriterProjection:
    def __init__(self, input_width, output_width, *, device='cpu', dtype=torch.float64):
        if min(input_width, output_width) < 1 or dtype not in (torch.float32, torch.float64):
            raise ValueError('Positive dimensions and FP32/FP64 accumulation required')
        self.covariance = torch.zeros(input_width, input_width, device=device, dtype=dtype)
        self.cross = torch.zeros(input_width, output_width, device=device, dtype=dtype)
        self.target_energy = torch.zeros((), device=device, dtype=dtype)
        self.sample_weight = 0.
        self.examples = 0

    @torch.no_grad()
    def add(self, features, displacement, mask, *, sample_weight=1.):
        if not math.isfinite(sample_weight) or sample_weight <= 0:
            raise ValueError('sample_weight must be finite and positive')
        if (features.shape[:-1] != displacement.shape[:-1]
                or mask.shape != (*features.shape[:-1], 1)
                or features.shape[-1] != self.covariance.shape[0]
                or displacement.shape[-1] != self.cross.shape[1]):
            raise ValueError('Feature/displacement/mask geometry mismatch')
        z = features.detach().to(self.covariance).reshape(-1, features.shape[-1])
        e = displacement.detach().to(self.cross).reshape(-1, displacement.shape[-1])
        m = mask.detach().to(self.covariance).reshape(-1)
        if not all(bool(torch.isfinite(t).all()) for t in (z, e, m)) or bool((m < 0).any()):
            raise ValueError('Inputs must be finite and mask nonnegative')
        if not bool(m.sum() > 0):
            raise ValueError('No valid writable tokens')
        # Each video/state gets equal weight by default, independent of token count.
        root_weight = (sample_weight*m/m.sum()).sqrt().unsqueeze(1)
        zw, ew = z*root_weight, e*root_weight
        self.covariance.add_(zw.T @ zw)
        self.cross.add_(zw.T @ ew)
        self.target_energy.add_(ew.square().sum())
        self.sample_weight += sample_weight
        self.examples += 1

    @torch.no_grad()
    def propose(self, relative_ridge=1e-3):
        if self.examples < 1 or not math.isfinite(relative_ridge) or relative_ridge <= 0:
            raise ValueError('Need examples and a positive finite ridge value')
        c = self.covariance/self.sample_weight
        r = self.cross/self.sample_weight
        coefficient = relative_ridge*c.diagonal().mean().clamp_min(1e-12)
        system = (c+c.T)/2 + coefficient*torch.eye(c.shape[0], device=c.device, dtype=c.dtype)
        beta = torch.linalg.solve(system, r)
        # E ~= Z @ beta; nn.Linear stores the transpose of this coefficient.
        delta_weight = beta.T.contiguous()
        energy = self.target_energy/self.sample_weight
        error = energy - 2*(beta*r).sum() + (beta*(c@beta)).sum()
        # Sufficient-statistics subtraction can give a tiny negative roundoff
        # near an exact fit. Preserve the raw value and reject larger negatives.
        tolerance = 64*torch.finfo(c.dtype).eps*(energy.abs()+2*(beta*r).abs().sum()
                                                +(beta*(c@beta)).abs().sum()).clamp_min(1e-30)
        if not bool(torch.isfinite(error)) or bool(error < -tolerance):
            raise FloatingPointError('Invalid reconstructed fitting error')
        report = dict(examples=self.examples, coefficient=float(coefficient),
                      target_energy=float(energy), fitted_error=float(error.clamp_min(0)),
                      fitted_error_raw=float(error),
                      relative_fit=float(error.clamp_min(0)/energy) if float(energy)>0 else None,
                      delta_weight_rms=float(delta_weight.square().mean().sqrt()),
                      fm_verified=False, generalization_verified=False, video_quality_verified=False)
        return delta_weight, report
