"""Frozen pretrained video evidence, with full-context versus pair-only control.

This is a training-time observer of an actual video, not a future-state predictor.
Its bidirectional features are never supplied as ordinary generation conditions.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F


class VideoProcessEncoder(nn.Module):
    def __init__(self, code, checkpoint, size=(192,320), device='cuda', checkpointing=True):
        super().__init__()
        sys.path.insert(0,str(Path(code)))
        from app.vjepa_2_1.models.vision_transformer import vit_gigantic_xformers
        from accelerate import init_empty_weights
        # Native construction also computes CPU scalar schedules with .item().
        # Empty parameters avoid a second multi-GB allocation without moving
        # those constructor computations onto the meta device.
        with init_empty_weights(include_buffers=False):
            model=vit_gigantic_xformers(img_size=(384,384),patch_size=16,num_frames=64,
                tubelet_size=2,use_sdpa=True,use_silu=False,wide_silu=True,uniform_power=False,
                use_rope=True,img_temporal_dim_size=1,interpolate_rope=True,
                use_activation_checkpointing=checkpointing,n_output_distillation=4)
        state=torch.load(checkpoint,mmap=True,map_location='cpu',weights_only=True)['target_encoder']
        model.load_state_dict({k.replace('module.','').replace('backbone.',''):v for k,v in state.items()},
                              strict=True,assign=True)
        self.model=model.eval().requires_grad_(False).to(device=device,dtype=torch.bfloat16)
        self.size=tuple(size)
        self.register_buffer('mean',torch.tensor([.485,.456,.406],device=device)[None,:,None,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225],device=device)[None,:,None,None,None])

    def train(self,mode=True):
        super().train(mode);self.model.eval();return self

    def forward(self,frames,*,context='full'):
        indices=[0,0]+[i for end in range(4,len(frames),4) for i in (end-1,end)]
        rgb=F.interpolate(frames[indices].float(),size=self.size,mode='bicubic',align_corners=False,antialias=True)
        rgb=(rgb.clamp(-1,1)+1)/2
        x=(rgb.permute(1,0,2,3)[None]-self.mean)/self.std
        if context=='full':
            features=self.model(x,training=False)
        elif context=='pairs':
            pairs=x.unflatten(2,(-1,2)).permute(0,2,1,3,4,5).flatten(0,1)
            features=torch.cat([self.model(batch,training=False) for batch in pairs.split(4)],0)
        else:raise ValueError('Choose full-video or isolated-pair evidence')
        return features.reshape(len(indices)//2,self.size[0]//16,self.size[1]//16,1664).float()
