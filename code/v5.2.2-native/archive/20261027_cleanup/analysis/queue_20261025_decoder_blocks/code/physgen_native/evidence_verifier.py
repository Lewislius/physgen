"""Learned conditional verifier calibrated on ordered process evidence.

Inputs at scoring time are the actual candidate video and legal conditions only.
Reference futures define training targets, never verifier inputs. The objective
calibrates visible process fidelity; it does not assert that every deviation from
a single recorded future is a physical violation. Physical validity and useful
generator feedback still require independent intervention and generation tests.
"""
import torch
from torch.nn import functional as F
from .temporal_critic import TemporalProcessCritic


class ProcessEvidenceVerifier(TemporalProcessCritic):
    def __init__(self,feature_width=1664,condition_width=4096,width=128,lags=(1,2,4,8,16,20)):
        super().__init__(feature_width,condition_width,width,lags)

    def energy(self,features,condition,times,valid,*,frozen=False):
        scores=self(features,condition,times,valid,frozen=frozen)
        # A single process target must not claim to label individual Qwen clauses.
        # Their joint readout predicts each temporal pair's nonnegative error.
        return {h:F.softplus(-s.mean(-1)) for h,s in scores.items()}

    def evidence_calibration_loss(self,features,targets,condition,times,valid):
        prediction=self.energy(features.detach(),condition.detach(),times,valid)
        loss=torch.stack([F.smooth_l1_loss(prediction[h],targets[h].detach().clamp_min(0),beta=.03)
                          for h in prediction]).mean()
        return loss,prediction

    def paired_calibration_loss(self,first,second,first_targets,second_targets,condition,times,valid):
        a=self(first.detach(),condition.detach(),times,valid)
        b=self(second.detach(),condition.detach(),times,valid)
        losses=[]
        for h in a:
            # Graded, symmetric evidence preference. Near-identical process
            # evidence is a tie, not an invented hard physical-error label.
            preference=((second_targets[h]-first_targets[h])/.03).sigmoid().detach()
            losses.append(F.binary_cross_entropy_with_logits((a[h]-b[h]).mean(-1),preference))
        return torch.stack(losses).mean()

    def generation_loss(self,features,condition,times,valid):
        energy=self.energy(features,condition,times,valid,frozen=True)
        return torch.stack([v.mean() for v in energy.values()]).mean()
