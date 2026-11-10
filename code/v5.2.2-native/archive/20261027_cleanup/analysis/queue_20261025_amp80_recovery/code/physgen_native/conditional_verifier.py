"""Conditional process compatibility, with no video-only/condition-only head.

Training distinguishes process fidelity from condition matching. Neither cached
video differences nor a mismatched caption is declared a physical law violation.
The frozen observer sees actual videos, never supplies a future to the generator.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .temporal_critic import TemporalProcessCritic


class ConditionalProcessVerifier(nn.Module):
    def __init__(self, feature_width, condition_width, width, lags):
        super().__init__()
        self.lags = tuple(lags)
        self.feature = nn.Linear(feature_width, width, bias=False)
        self.condition = nn.Linear(condition_width, width, bias=False)
        self.pair = nn.Sequential(nn.Linear(3*width+3, width), nn.SiLU(), nn.Linear(width, width))
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(5.)))

    def forward(self, features, condition, times, valid, *, frozen=False):
        if frozen:
            weights = {name: p.detach() for name, p in self.named_parameters()}
            return torch.func.functional_call(self, weights, (features, condition.detach(), times, valid))
        normalize = TemporalProcessCritic.normalize
        x = self.feature(normalize(features)).flatten(1, 2)
        c = F.normalize(self.condition(normalize(condition.detach().reshape(-1, condition.shape[-1]))).float(), dim=-1)
        mask = F.interpolate(valid[None, None].float(), size=features.shape[1:3], mode='area').flatten()
        relative = (times-times[0])/(times[-1]-times[0]).clamp_min(1e-6)
        scores = {}
        for lag in self.lags:
            if lag >= len(x): continue
            a, b = x[:-lag], x[lag:]
            time = torch.stack((relative[:-lag], relative[lag:], torch.log1p(times[lag:]-times[:-lag])), -1)
            pair = self.pair(torch.cat((a, b, b-a, time[:, None].expand(-1, x.shape[1], -1)), -1))
            keys = F.normalize(self.key(pair).float(), dim=-1)
            attention = (5*torch.einsum('kd,pnd->pkn', c, keys)+mask.clamp_min(1e-8).log()[None, None]).softmax(-1)
            values = F.normalize(self.value(pair).float(), dim=-1)
            read = torch.einsum('pkn,pnd->pkd', attention, values)
            # Only an interaction score. A video-only realism head or a
            # condition-only prior cannot independently satisfy this readout.
            scores[lag] = self.log_scale.exp().clamp(max=20)*torch.einsum('pkd,kd->pk', read, c)
        return scores

    @staticmethod
    def preference_loss(first, second, first_error, second_error, temperature):
        return torch.stack([F.binary_cross_entropy_with_logits(
            first[h].mean(-1)-second[h].mean(-1),
            ((second_error[h]-first_error[h])/temperature).sigmoid().detach()) for h in first]).mean()

    @staticmethod
    def matching_loss(reference, candidate, other_reference, other_candidate,
                      reference_error, candidate_error, temperature):
        """Require a condition-dependent response to an observed process change.

        Difference in differences cancels ANY additive video-only score a(V)
        and ANY condition-only prior b(c). Ordinary matching can learn whether
        a caption agrees with the first image without reading later evolution.
        This contrast instead trains the association between the requested
        process and its actual development. It does not label an unobserved
        counterfactual video, or declare all alternate requests impossible.
        """
        losses = []
        for h in reference:
            relevant = reference[h].mean(-1)-candidate[h].mean(-1)
            other = other_reference[h].mean(-1)-other_candidate[h].mean(-1)
            preference = ((candidate_error[h]-reference_error[h])/temperature).sigmoid().detach()
            losses.append(F.binary_cross_entropy_with_logits(relevant-other, preference))
        return torch.stack(losses).mean()

    def generation_loss(self, features, condition, times, valid):
        scores = self(features, condition, times, valid, frozen=True)
        return -torch.stack([s.mean() for s in scores.values()]).mean()
