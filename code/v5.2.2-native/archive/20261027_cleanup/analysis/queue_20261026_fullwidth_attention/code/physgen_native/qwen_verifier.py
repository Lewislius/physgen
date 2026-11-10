"""A calibrated readout of Qwen's joint video/requirement evidence.

The three contextual readouts concern prerequisites/responses, progression and
continuity. Targets come from frozen Qwen comparisons of actual videos, never
from JEPA distance or an unconditional label that stillness is wrong.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .attention import RMSNorm


class QwenProcessCritic(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = RMSNorm(width)
        self.score = nn.Linear(width, 1, bias=False)
        nn.init.normal_(self.score.weight, std=.01)

    def forward(self, evidence, *, frozen=False):
        if frozen:
            return torch.func.functional_call(self,
                {name: value.detach() for name, value in self.named_parameters()}, (evidence,))
        return self.score(self.norm(evidence.float())).squeeze(-1).mean(-1)

    def preference_loss(self, first, second, preference):
        return F.binary_cross_entropy_with_logits(self(first)-self(second), preference.detach().reshape(-1))

    def generation_loss(self, evidence):
        return -self(evidence, frozen=True).mean()
