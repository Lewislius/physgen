"""Ordered changes in frozen semantic states, not RGB motion or physical speed.

Descriptive evaluation only. A static copy has zero predicted transition; its
relative squared error is one for a nonzero target transition. Spatial alignment
and one-reference trajectory remain assumptions, so this is not a physics judge.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def transition_diagnostics(predicted,target,valid=None):
    layers=[]
    for p,t in zip(predicted,target):
        p=F.normalize(p.float(),dim=-1);t=F.normalize(t.detach().float(),dim=-1)
        if p.shape!=t.shape or p.shape[0]!=3:raise ValueError('Use the same three observed semantic states')
        mask=torch.ones(t.shape[1:3],dtype=torch.bool,device=t.device)
        if valid is not None:
            mask=F.interpolate(valid.to(t.device).float()[None,None],size=t.shape[1:3],mode='area')[0,0]>.5
        rows=[]
        for a,b in [(0,1),(1,2),(0,2)]:
            dp=(p[b]-p[a])[mask];dt=(t[b]-t[a])[mask]
            target_energy=dt.square().sum();predicted_energy=dp.square().sum()
            error=(dp-dt).square().sum()
            norm=(target_energy*predicted_energy).sqrt()
            rows.append(dict(states=[a,b],
                relative_squared_error=float(error/target_energy) if target_energy>0 else None,
                direction_cosine=float((dp*dt).sum()/norm) if norm>0 else None,
                target_feature_energy_per_patch=float(target_energy/len(dt)),
                predicted_transition_is_zero=bool(predicted_energy==0)))
        layers.append(rows)
    errors=[r['relative_squared_error'] for layer in layers for r in layer if r['relative_squared_error'] is not None]
    directions=[r['direction_cosine'] for layer in layers for r in layer if r['direction_cosine'] is not None]
    return dict(mean_relative_squared_error=sum(errors)/len(errors) if errors else None,
        mean_direction_cosine=sum(directions)/len(directions) if directions else None,
        layers=layers,scope='Signed frozen-feature transitions between early/middle/final states; zero motion is not rewarded as matched evolution. Not a validated physical-process score; do not average or optimize without reviewing reference and alignment sensitivity.')
