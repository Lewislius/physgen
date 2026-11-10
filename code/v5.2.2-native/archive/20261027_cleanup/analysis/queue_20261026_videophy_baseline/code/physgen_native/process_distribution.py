"""Match distributions of ordered visible transitions on actual sampled videos.

The reference is training supervision, never an inference condition. This energy
does not learn real/fake labels and cannot call a frozen photograph successful
merely because it looks authentic. Frozen video features are not physical state
variables: this is a process-supervision hypothesis, not a law-of-physics oracle.
"""
import torch
from torch.nn import functional as F


def _kernel(a,b):
    # Squared channel-mean distances; no matching/tracking of spatial points.
    dim=a.shape[-1]
    distance=(a.square().sum(-1)[:,:,None]+b.square().sum(-1)[:,None,:]-
              2*torch.bmm(a,b.transpose(1,2)))/dim
    distance=distance.clamp_min(0)
    return sum(torch.exp(-distance/(2*width*width)) for width in (.5,1.,2.))/3


def ordered_process_distribution(prediction,reference,valid,*,lags=(1,2,4,8,16,20),scale_floor=.03,return_pairs=False):
    """Weighted MMD at each ordered time pair; only spatial samples are pooled.

    Pairing endpoints before spatial distribution comparison retains their joint
    appearance/change information. Time pairs are not mixed into one bag. This
    permits different spatial realizations without rewarding absent late change.
    """
    reference=reference.detach().float();prediction=prediction.float()
    if prediction.shape!=reference.shape:raise ValueError('Use identical frozen feature geometry')
    weight=F.interpolate(valid[None,None].float(),size=reference.shape[1:3],mode='area').flatten()
    weight=weight/weight.sum().clamp_min(1e-8)
    x,y=prediction.flatten(1,2),reference.flatten(1,2)
    scale=(y.square().mean(-1)*weight).sum(-1).mean().sqrt().clamp_min(.1).detach()
    x,y=x/scale,y/scale
    pair_weight=weight[:,None]*weight[None,:]
    losses=[];metrics={};pairs={}
    # MMD cancellation is sensitive to BF16 arithmetic. The encoder and Wan may
    # use BF16; the statistical objective is evaluated and differentiated in FP32.
    with torch.autocast(prediction.device.type,enabled=False):
        for lag in lags:
            if lag>=len(y):continue
            dy=y[lag:]-y[:-lag];dx=x[lag:]-x[:-lag]
            magnitude=(dy.square().mean(-1)*weight).sum(-1).sqrt().clamp_min(scale_floor).detach()
            a=torch.cat((x[:-lag],x[lag:],dx/magnitude[:,None,None]),-1)
            b=torch.cat((y[:-lag],y[lag:],dy/magnitude[:,None,None]),-1)
            per_pair=((_kernel(a,a)+_kernel(b,b)-2*_kernel(a,b))*pair_weight).sum((-2,-1))
            loss=per_pair.mean();losses.append(loss)
            pairs[lag]=per_pair.detach()
            metrics[f'distribution/lag{lag}']=loss.detach()
            metrics[f'distribution/lag{lag}_late']=per_pair[(torch.arange(len(per_pair),device=x.device)+lag)>=int(.6*(len(y)-1))].mean().detach()
    if not losses:raise ValueError('At least one ordered temporal pair is required')
    result=(torch.stack(losses).mean(),metrics)
    return (*result,pairs) if return_pairs else result
