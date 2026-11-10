"""Frozen, differentiable observed-video features; never a physical judge.

This candidate supervision utility is not enabled in the running r1 training.
It reads the existing Qwen vision weights only, without its language model,
text-based verdicts, generated scores, or trainable prediction heads.
"""
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from physgen_native.knowledge import visual_patches

class FrozenProcessFeatures(nn.Module):
    def __init__(self,cfg,device='cpu'):
        super().__init__();root=Path(cfg['paths']['qwen_checkpoint'])
        config=AutoConfig.from_pretrained(root,local_files_only=True).vision_config
        config._attn_implementation='sdpa'
        self.encoder=Qwen3VLVisionModel(config)
        index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        weights={};prefix='model.visual.'
        for shard in sorted(set(v for k,v in index.items() if k.startswith(prefix))):
            with safe_open(root/shard,framework='pt',device='cpu') as source:
                for key in source.keys():
                    if key.startswith(prefix):weights[key.removeprefix(prefix)]=source.get_tensor(key)
        self.encoder.load_state_dict(weights,strict=True,assign=True)
        self.encoder.to(device).eval().requires_grad_(False)
        # Checkpoint each frozen visual block while keeping evaluation behavior.
        # The outer checkpoint alone would retain all block activations when it
        # is recomputed for the derivative to the predicted video.
        for block in self.encoder.blocks:
            original=block.forward
            def wrap(forward):
                def run(*args,**kwargs):
                    if torch.is_grad_enabled() and any(torch.is_tensor(a) and a.requires_grad for a in args):
                        return checkpoint(forward,*args,use_reentrant=False,**kwargs)
                    return forward(*args,**kwargs)
                return run
            block.forward=wrap(original)
        self.patch=config.patch_size;self.merge=config.spatial_merge_size;self.temporal=config.temporal_patch_size
        self.size=cfg['knowledge']['video_size']

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    def forward(self,frames):
        if len(frames)%self.temporal:raise ValueError('Use complete ordered frame pairs')
        patches,grid=visual_patches(frames,self.size,self.patch,self.merge,self.temporal,video=True,native_geometry=True)
        def read(pixels):
            output=self.encoder(pixels.to(next(self.encoder.parameters()).dtype),grid_thw=grid)
            return output.pooler_output,*output.deepstack_features
        values=checkpoint(read,patches,use_reentrant=False) if torch.is_grad_enabled() and frames.requires_grad else read(patches)
        t,h,w=grid[0].tolist()
        return tuple(v.float().reshape(t,h//self.merge,w//self.merge,-1) for v in values)

def feature_distance(predicted,target,valid=None):
    """Match observed ordered features, retaining where and when errors occur.

    No frame-invariance term is used: real deformation, dissipation and newly
    appearing content remain valid target changes. This is a perceptual target,
    not an assertion that every nonzero distance is a physical error.
    """
    terms=[];rows=[]
    for p,t in zip(predicted,target):
        error=(F.normalize(p,dim=-1)-F.normalize(t.detach(),dim=-1)).square().sum(-1)/2
        mask=torch.ones_like(error,dtype=torch.bool)
        if valid is not None:
            support=F.interpolate(valid.float()[None,None],size=error.shape[-2:],mode='area')[0,0]>.5
            mask=support[None].expand_as(error)
        values=error[mask];tail=values.topk(max(1,(len(values)+3)//4)).values.mean()
        terms.append(.8*values.mean()+.2*tail)
        rows.append(dict(mean=float(values.detach().mean()),top_quarter=float(tail.detach()),
                         per_pair=[float(v[m].detach().mean()) for v,m in zip(error,mask)]))
    return torch.stack(terms).mean(),rows
