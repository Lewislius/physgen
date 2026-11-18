"""Train-time observed states, differentiated through the real Wan decoder.

The target is the existing real training clip at matching observation times.
There is no scoring model, future-state predictor, event annotation, optical
flow, RGB difference target, or future-derived generation condition.
"""
from pathlib import Path
import importlib.util
import numpy as np
import torch


def observation_indices(cursor, seed):
    """Sample ordered pairs across the clip, always retaining its final state.

    Pairs are drawn from the established every-third-frame observation grid.
    Early and middle pairs vary across data positions; the final pair is 78/80.
    A private generator keeps resumed and matched-baseline sampling identical.
    """
    grid=list(range(0,81,3))+[80]
    pairs=[grid[i:i+2] for i in range(0,len(grid),2)]
    rng=torch.Generator().manual_seed(seed*3000017+cursor)
    early=int(torch.randint(0,7,(),generator=rng))
    middle=int(torch.randint(7,13,(),generator=rng))
    return pairs[early]+pairs[middle]+pairs[13]


def active_microstep(step, micro, sigma, accumulation, policy):
    return (micro==step%accumulation and
            policy['sigma_min']<=float(sigma)<=policy['sigma_max'])


def coefficient(step, policy):
    return policy['weight']*min(1.,(step+1)/policy['warmup'])


def clean_estimate(noisy, sigma, prediction, first):
    clean=noisy-sigma*prediction
    return torch.cat((first,clean[:,:,1:]),2)


class ProcessSupervisor:
    def __init__(self,cfg,device='cuda'):
        from physgen_native.decoder import VideoDecoder
        from .process_features import FrozenProcessFeatures
        self.cfg=cfg
        self.decoder=VideoDecoder(cfg,device=device)
        self.features=FrozenProcessFeatures(cfg,device=device)
        source=Path('/home/liuzhirui/Project/physGen/code/v5.2/physgen_v52/preparation.py')
        spec=importlib.util.spec_from_file_location('_process_r2_frames',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.read_frames=module.read_frames

    def __call__(self,sample,noisy,sigma,prediction,cursor):
        from .process_features import feature_distance
        if sample['row']['split']!='train':
            raise ValueError('Observed-state training targets must come from the existing training split')
        indices=observation_indices(cursor,self.cfg['train']['seed'])
        images=self.read_frames(sample['row'],indices)
        rgb=torch.from_numpy(np.stack([np.asarray(im) for im in images])).permute(0,3,1,2)
        rgb=rgb.to(noisy.device).float()/127.5-1
        with torch.no_grad():
            targets=self.features(rgb)
        del rgb,images
        predicted=clean_estimate(noisy,sigma,prediction,sample['first'])
        decoded=self.decoder(predicted,frame_indices=indices)
        value,parts=feature_distance(self.features(decoded),targets,sample['valid'])
        return value,dict(frame_indices=indices,layers=parts)
