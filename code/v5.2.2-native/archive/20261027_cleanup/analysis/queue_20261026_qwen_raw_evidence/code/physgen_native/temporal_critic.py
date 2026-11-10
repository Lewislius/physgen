"""Conditional distribution critic over visible, ordered process evidence.

This replaces an uninformative scene-level preference target in the experimental
recipe. Real/generated labels estimate a density ratio; they are NOT labels that
every generated video violates physics. No future state is predicted here.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalProcessCritic(nn.Module):
    def __init__(self, feature_width=4096, condition_width=4096, width=128,
                 lags=(1, 2, 4, 8, 16)):
        super().__init__()
        self.lags = tuple(lags)
        self.feature = nn.Linear(feature_width, width)
        self.condition = nn.Linear(condition_width, width)
        self.pair = nn.Sequential(nn.Linear(3*width+3, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU())
        self.key = nn.Linear(width, width, bias=False)
        self.head = nn.Sequential(nn.Linear(3*width, width), nn.SiLU(), nn.Linear(width, 1))

    @staticmethod
    def normalize(x):
        return x.float()*x.float().square().mean(-1,keepdim=True).clamp_min(1e-8).rsqrt()

    def forward(self, features, condition, times, valid, *, frozen=False):
        if frozen:
            # Detaching weights preserves image derivatives. The generator must
            # improve the video, not alter the critic's conditioning shortcut.
            weights = {name:p.detach() for name,p in self.named_parameters()}
            return torch.func.functional_call(self,weights,(features,condition.detach(),times,valid))
        x = self.feature(self.normalize(features))
        c = self.condition(self.normalize(condition.detach().reshape(-1,condition.shape[-1])))
        mask = F.interpolate(valid[None,None].float(),size=x.shape[1:3],mode='area').flatten()
        x = x.flatten(1,2)
        duration = (times[-1]-times[0]).clamp_min(1e-6)
        relative = (times-times[0])/duration
        scores = {}
        for lag in self.lags:
            if lag >= len(x): continue
            a,b = x[:-lag],x[lag:]
            # Actual seconds and normalized position remain distinct; neither is
            # a physical velocity nor an object trajectory.
            temporal = torch.stack((relative[:-lag],relative[lag:],
                                    torch.log1p(times[lag:]-times[:-lag])), -1)
            temporal = temporal[:,None].expand(-1,x.shape[1],-1)
            pair = self.pair(torch.cat((a,b,b-a,temporal),-1))
            logits = torch.einsum('kd,pnd->pkn',c,self.key(pair))/math.sqrt(c.shape[-1])
            attention = (logits+mask.clamp_min(1e-8).log()[None,None]).softmax(-1)
            read = torch.einsum('pkn,pnd->pkd',attention,pair)
            context = c[None].expand(len(read),-1,-1)
            # Keep every time pair and dependency slot until AFTER the loss.
            scores[lag] = self.head(torch.cat((read,context,read*context),-1)).squeeze(-1)
        return scores

    def calibration_loss(self, real, generated, condition, times, valid):
        a = self(real.detach(),condition,times,valid)
        b = self(generated.detach(),condition,times,valid)
        loss = torch.stack([F.softplus(-a[h]).mean()+F.softplus(b[h]).mean()+
                            .001*(a[h].square().mean()+b[h].square().mean()) for h in a]).mean()
        return loss, self.metrics(a,b)

    def generation_loss(self, generated, condition, times, valid):
        scores = self(generated,condition,times,valid,frozen=True)
        return torch.stack([F.softplus(-s).mean() for s in scores.values()]).mean()

    @staticmethod
    def metrics(real, generated):
        values = {}
        for lag in real:
            values[f'lag{lag}/margin'] = (real[lag]-generated[lag]).mean().detach()
            values[f'lag{lag}/pair_order_accuracy'] = (real[lag]>generated[lag]).float().mean().detach()
        values['real_score'] = torch.stack([s.mean() for s in real.values()]).mean().detach()
        values['generated_score'] = torch.stack([s.mean() for s in generated.values()]).mean().detach()
        return values
