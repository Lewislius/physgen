"""A training-only process critic, calibrated from the existing videos.

Teacher preferences label edits; an edit is never assumed to violate physics.
Generator losses use detached critic weights and retain input derivatives.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ProcessCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg['model']; d, k = m['qwen_width'], m['critic_width']
        self.norm = nn.LayerNorm(d)
        self.first = nn.Linear(d, k)
        self.last = nn.Linear(k, 1)
        nn.init.normal_(self.last.weight, std=.01)
        nn.init.zeros_(self.last.bias)

    def forward(self, features, *, frozen=False):
        parameter = lambda x: x.detach() if frozen else x
        x = F.layer_norm(features.float(), self.norm.normalized_shape,
                         parameter(self.norm.weight), parameter(self.norm.bias), self.norm.eps)
        x = F.silu(F.linear(x, parameter(self.first.weight), parameter(self.first.bias)))
        return F.linear(x, parameter(self.last.weight), parameter(self.last.bias)).squeeze(-1).mean(-1)

    def calibration_loss(self, positive, alternative, preference):
        a, b = self(positive.detach()), self(alternative.detach())
        # The small common-offset penalty fixes the unidentifiable score offset.
        loss = F.binary_cross_entropy_with_logits(a-b, preference.detach())+.01*((a+b)/2).square().mean()
        return loss, dict(critic_margin=(a-b).mean().detach(), teacher_preference=preference.mean())

    def generation_loss(self, generated, reference):
        score = self(generated, frozen=True)
        target = self(reference.detach(), frozen=True).detach()
        return F.softplus(target-score).mean(), dict(physical_score=score.mean().detach(), reference_score=target.mean())


def evidence_indices(times, pairs):
    """Actual adjacent RGB pairs distributed over the complete native video."""
    last = 4*(len(times)-1)
    boundaries = torch.linspace(1, last, min(pairs, last)).round().long().tolist()
    # Keep duplicate positions at pair boundaries: temporal patch size remains two.
    return [index for end in boundaries for index in (end-1, end)]


def rgb_times(indices, latent_times):
    out = []
    for index in indices:
        left, fraction = divmod(index, 4)
        right = min(left+1, len(latent_times)-1)
        out.append(latent_times[left]+fraction/4*(latent_times[right]-latent_times[left]))
    return torch.stack(out)


def omitted_process(frames, sample, cfg, step):
    """A teacher-assessed alternative view; no invented physical failure label."""
    future = len(sample['times'])-1
    activity = sample.get('evidence_activity', frames.new_ones(future)).float().clamp_min(0)
    # Half uniform, half video-evidence sampling, with no inference dependency.
    probability = .5*torch.ones_like(activity)/len(activity)+.5*activity/activity.sum().clamp_min(1e-8)
    selected = int(torch.multinomial(probability, 1))
    length = cfg['perturb_lengths'][step % len(cfg['perturb_lengths'])]
    begin = max(1, min(selected+1-length//2, future-length+1))*4
    end = min(len(frames), begin+4*length)
    alternative = frames.detach().clone()
    if step % 3 == 0:
        alternative[begin:end] = alternative[begin-1:begin]
    elif step % 3 == 1:
        # Missing intermediate development, with the eventual outcome retained.
        middle = begin+(end-begin)//2
        alternative[begin:middle] = alternative[begin-1:begin]
        alternative[middle:end] = alternative[end-1:end]
    else:
        # Local temporal-order alternative; the teacher may rate it equally valid.
        alternative[begin:end] = alternative[begin:end].flip(0)
    return alternative
