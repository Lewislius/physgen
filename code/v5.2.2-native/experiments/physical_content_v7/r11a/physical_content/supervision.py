"""Actual FM target, with an explicitly generic reduction of averaging dilution."""
import math
import torch
from torch.nn import functional as F

def generation_loss(prediction,noise,clean,valid,tail_weight=.2,tail_fraction=.25):
    error=(prediction[:,:,1:].float()-(noise-clean)[:,:,1:].float()).square().mean(1)
    mask=valid.float()[None,None].expand_as(error)
    mean=(error*mask).sum()/mask.sum().clamp_min(1)
    if tail_weight==0:return mean,dict(fm=mean,tail=mean)
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
    return (1-tail_weight)*mean+tail_weight*tail,dict(fm=mean,tail=tail)
