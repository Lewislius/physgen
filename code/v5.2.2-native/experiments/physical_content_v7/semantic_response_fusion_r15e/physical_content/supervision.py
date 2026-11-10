"""Real-video FM with spatial and temporal error reductions.

The temporal term compares predicted and target velocity differences. It is
not a penalty on generated motion, nor a motion reward or a different flow path.
"""
import math
import torch
from torch.nn import functional as F

def _reduce_error(error,valid,tail_fraction):
    mask=valid.float()[None,None].expand_as(error)
    mean=(error*mask).sum()/mask.sum().clamp_min(1)
    t,h,w=error.shape[-3:]
    windows=[(1,h,w),(min(4,t),h,w),(min(4,t),max(1,h//5),max(1,w//4)),(t,max(1,h//5),max(1,w//4))]
    terms=[]
    for window in windows:
        weighted=F.avg_pool3d((error*mask)[:,None],window,stride=window,ceil_mode=True)
        support=F.avg_pool3d(mask[:,None],window,stride=window,ceil_mode=True)
        values=(weighted/support.clamp_min(1e-8))[support>0]
        count=max(1,math.ceil(len(values)*tail_fraction))
        terms.append(values.topk(count).values.mean())
    tail=torch.stack(terms).mean()
    return mean,tail


def generation_loss(prediction,noise,clean,valid,tail_weight=.2,tail_fraction=.25,
                    temporal_weight=.2,temporal_lags=(1,2,4,8)):
    if min(tail_weight,temporal_weight)<0 or tail_weight+temporal_weight>1:
        raise ValueError('Generation weights must form a nonnegative mixture')
    if not 0<tail_fraction<=1:
        raise ValueError('Invalid high-error fraction')
    residual=prediction[:,:,1:].float()-(noise-clean)[:,:,1:].float()
    mean,tail=_reduce_error(residual.square().mean(1),valid,tail_fraction)
    temporal=mean.new_zeros(())
    if temporal_weight:
        terms=[]
        for lag in temporal_lags:
            if not isinstance(lag,int) or lag<1:raise ValueError('Temporal lags must be positive integers')
            if lag>=residual.shape[2]:continue
            # Dividing by 2 matches the variance of independent point errors.
            # A correct static or changing target has zero error alike. Known
            # frame0 is excluded, as in the existing ordinary FM objective.
            pair_error=(residual[:,:,lag:]-residual[:,:,:-lag]).square().mean(1)/2
            _,pair_tail=_reduce_error(pair_error,valid,tail_fraction)
            terms.append(pair_tail)
        if not terms:raise ValueError('No valid future-frame pairs for temporal supervision')
        temporal=torch.stack(terms).mean()
    total=(1-tail_weight-temporal_weight)*mean+tail_weight*tail+temporal_weight*temporal
    return total,dict(fm=mean,tail=tail,temporal=temporal)
