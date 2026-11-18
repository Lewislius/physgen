"""Grounded full-width Q/C and state-dependent physical execution.

The frozen semantic interface is fitted only to training-side input reasoning.
There is no future-video condition, verifier, extra rollout, or inference T5 branch.
"""
import copy,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from physgen_native.message_conditioning import MessageAttention
from physgen_native.generator import load_wan
from physical_scaling.model import PhysicalModel as OriginalPhysicalModel,install_execution_lora,time_features
from physical_scaling.supervision import group_log_prior
from .evidence import resample_words


class ContentBlock(nn.Module):
    def __init__(self,width,heads,grounded_heads):
        super().__init__()
        if not 0 < grounded_heads < heads:
            raise ValueError('Retain both relation-grounded and full-context reads')
        self.grounded_heads=grounded_heads
        self.query_norm=nn.LayerNorm(width);self.memory_norm=nn.LayerNorm(width)
        self.cross=MessageAttention(width,heads)
        self.exchange_norm=nn.LayerNorm(width)
        self.exchange=MessageAttention(width,heads)
        self.ffn_norm=nn.LayerNorm(width)
        self.ffn=nn.Sequential(nn.Linear(width,2*width,bias=False),nn.GELU(),nn.Linear(2*width,width,bias=False))
        self.update_norm=nn.RMSNorm(width,eps=1e-6)

    def read_heads(self,query,memory,memory_mask,words,word_groups,groups):
        attention=self.cross
        split=lambda x:x.unflatten(-1,(attention.heads,attention.head_dim)).transpose(1,2)
        q=split(attention.q(query));local=self.grounded_heads
        k=split(attention.k(memory));v=split(attention.v(memory))
        wk=split(attention.k(words));wv=split(attention.v(words))
        allowed=groups[:,None]==word_groups[None,:]
        grounded=F.scaled_dot_product_attention(q[:,:local],wk[:,:local],wv[:,:local],
            attn_mask=allowed[None,None])
        contextual=F.scaled_dot_product_attention(q[:,local:],k[:,local:],v[:,local:],
            attn_mask=memory_mask[:,None,None,:])
        return torch.cat((grounded,contextual),1)

    def forward(self,state,anchor,memory,memory_mask,groups,words,word_groups):
        read=self.read_heads(self.query_norm(state),self.memory_norm(memory),memory_mask,
            self.memory_norm(words),word_groups,groups)
        state=state+self.cross.o(read.transpose(1,2).flatten(2))
        normalized=self.exchange_norm(state)
        summaries=torch.stack([normalized[:,groups==r].mean(1) for r in range(int(groups.max())+1)],1)
        exchange_memory=torch.cat((normalized,summaries),1)
        local=groups[:,None]==groups[None,:]
        allowed=torch.cat((local,torch.ones(len(groups),len(summaries[0]),device=state.device,dtype=torch.bool)),1)
        attention=self.exchange
        split=lambda x:x.unflatten(-1,(attention.heads,attention.head_dim)).transpose(1,2)
        values=F.scaled_dot_product_attention(split(attention.q(normalized)),split(attention.k(exchange_memory)),
                   split(attention.v(exchange_memory)),attn_mask=allowed[None,None])
        state=state+attention.o(values.transpose(1,2).flatten(2))
        state=state+self.ffn(self.ffn_norm(state))
        return anchor+self.update_norm(state-anchor)


class FixedProjection(nn.Module):
    """Keep the fitted native coordinates; train the physical update before them."""
    def __init__(self,weight):
        super().__init__()
        self.register_buffer('weight',weight.detach().clone())

    def forward(self,value):
        return F.linear(value,self.weight)


class GroundedContentWrite(nn.Module):
    """Make the Q update act on the actual input content in its fitted coordinates.

    A shared Q update becomes a shared feature transform of different physical
    words, rather than an independent common vector added to every C token.
    No output diversity, orthogonality, clipping or scalar gate is imposed.
    """
    def forward(self,anchor,update):
        return anchor*update


class Enhancement(nn.Module):
    def __init__(self,cfg,interface=None):
        super().__init__();m=cfg['model']
        if interface is None:interface=torch.load(cfg['paths']['interface'],map_location='cpu',weights_only=True)
        for name in ['mean','scale','weight','bias','target_scale']:
            self.register_buffer('anchor_'+name,interface[name].float())
        self.seed=nn.Parameter(torch.randn(1,m['q_tokens'],m['qwen_width'])*.02)
        self.block=ContentBlock(m['qwen_width'],m['q_heads'],m['grounded_read_heads'])
        project=nn.Linear(m['qwen_width'],m['width'],bias=False)
        # Match R4's initial values and RNG consumption exactly. The semantic
        # interface is now a saved constant, so generation credit must update
        # the full-width Q reader instead of changing this coordinate map.
        with torch.no_grad():
            project.weight.copy_(self.anchor_weight*m['content_projection_init_scale'])
        self.project=FixedProjection(project.weight)
        self.content_write=GroundedContentWrite()
        self.rounds=m['rounds'];self.checkpointing=cfg['train']['gradient_checkpointing']

    def forward(self,evidence,update_override=None):
        words=evidence['words'].detach().float()
        # Use the same training-input coordinates for local evidence and its
        # query anchor. Per-token LayerNorm alone retains the strong common
        # direction in raw Qwen word features; this is not per-sample centering.
        grounded_words=(words-self.anchor_mean)/self.anchor_scale
        anchor,groups,_=resample_words(grounded_words,evidence['word_groups'],self.seed.shape[1])
        if update_override is None:
            state=anchor+self.seed
            memory=evidence['memory'].detach().float();mask=evidence['memory_mask'].bool()
            physical_words=grounded_words[None] if grounded_words.ndim==2 else grounded_words
            for _ in range(self.rounds):
                if self.checkpointing and torch.is_grad_enabled():
                    state=checkpoint(self.block,state,anchor,memory,mask,groups,physical_words,
                        evidence['word_groups'],use_reentrant=False)
                else:state=self.block(state,anchor,memory,mask,groups,physical_words,evidence['word_groups'])
            update=state-anchor
        else:
            if update_override.shape!=anchor.shape:
                raise ValueError('A fixed-update control must retain the full Q-token/channel shape')
            update=update_override.detach().to(anchor)
        native=F.linear(anchor,self.anchor_weight,self.anchor_bias)
        return native+self.project(self.content_write(anchor,update)),groups

    def content_loss(self,c,evidence):
        target,_,_=resample_words(evidence['native_words'].detach().float(),evidence['word_groups'],c.shape[1])
        return (c.float()-target).square().mean()/self.anchor_target_scale.square().clamp_min(1e-8)


class Fusion(nn.Module):
    def __init__(self,attention,output_init_scale):
        super().__init__();width=attention.dim
        self.key=nn.Linear(width,width,bias=False,device=attention.k.weight.device)
        self.value=nn.Linear(width,width,bias=False,device=attention.v.weight.device)
        with torch.no_grad():
            self.key.weight.copy_(attention.k.weight);self.value.weight.copy_(attention.v.weight)
        self.key_norm=copy.deepcopy(attention.norm_k).float().requires_grad_(True)
        self.output=nn.Linear(width,width,bias=False,device=attention.q.weight.device)
        self.state_value=nn.Linear(width,width,bias=False,device=attention.q.weight.device)
        self.time=nn.Linear(256,width,bias=False,device=attention.q.weight.device)
        # Start in Wan's pretrained value-to-latent coordinates. In R1 zero O
        # made the initial generation derivative to C and its reader zero.
        # The independent physical O remains trainable after this initializer.
        with torch.no_grad():
            self.output.weight.copy_(attention.o.weight.float()*output_init_scale)
        for layer in [self.state_value,self.time]:nn.init.zeros_(layer.weight)
        self.reference_key=nn.Parameter(torch.zeros(1,1,width,device=attention.q.weight.device))

    def forward(self,x,context,c,times,geometry,attention,groups=None,target=None,valid=None):
        batch,length,width=x.shape;known=geometry[1]*geometry[2]
        heads,dim=attention.num_heads,attention.head_dim
        split=lambda value:value.reshape(batch,-1,heads,dim)
        raw=attention.q(x)
        qt=split(attention.norm_q(raw));kt=split(attention.norm_k(attention.k(context)));vt=split(attention.v(context))
        kernel=attention.forward.__func__.__globals__.get('flash_attention')
        if kernel is None:
            text=F.scaled_dot_product_attention(qt.transpose(1,2),kt.transpose(1,2).to(qt),vt.transpose(1,2).to(qt)).transpose(1,2)
        else:text=kernel(qt,kt,vt,k_lens=None)
        text=attention.o(text.flatten(2))
        time=self.time(time_features(times))[:,1:].repeat_interleave(known,1)
        q=split(attention.norm_q(raw[:,known:]+time)).transpose(1,2)
        k=split(self.key_norm(self.key(c))).transpose(1,2).to(q)
        v=split(self.value(c)).transpose(1,2).to(q)
        reference=split(self.reference_key.expand(batch,-1,-1)).transpose(1,2).to(q)
        k=torch.cat((k,reference),2);v=torch.cat((v,torch.zeros_like(reference)),2)
        scores=(q.float()@k.float().transpose(-1,-2))/math.sqrt(dim)+group_log_prior(groups).to(q)
        z=(scores.softmax(-1).to(v)@v).transpose(1,2).flatten(2)
        state=F.layer_norm(x[:,known:].float(),(width,))
        delta=self.output(z+self.state_value(state)*z)
        return text+torch.cat((torch.zeros_like(text[:,:known]),delta),1),x.new_zeros((),dtype=torch.float32)


class EmptyEnhancement(nn.Module):
    def forward(self,evidence):
        memory=evidence['memory']
        return memory.new_zeros((memory.shape[0],0,3072)),None


class PhysicalModel(OriginalPhysicalModel):
    def __init__(self,cfg,device='cuda',wan=None):
        nn.Module.__init__(self);self.cfg=cfg
        self.wan=load_wan(cfg,device) if wan is None else wan.requires_grad_(False)
        if cfg['variant']=='native_lora':
            self.enhancement=EmptyEnhancement();self.fusions=nn.ModuleDict()
        else:
            self.enhancement=Enhancement(cfg).to(device)
            self.fusions=nn.ModuleDict({str(i):Fusion(block.cross_attn,cfg['model']['native_output_init_scale']) for i,block in enumerate(self.wan.blocks)
                if (i+1)%cfg['model']['condition_every']==0}).to(device)
        devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg['train']['seed']+731)
            if devices:torch.cuda.manual_seed_all(cfg['train']['seed']+731)
            install_execution_lora(self.wan,cfg['model']['lora_rank'])
        self.learned_names=tuple(name for name,p in self.named_parameters() if p.requires_grad)
        self.checkpointing=cfg['train']['gradient_checkpointing']

    def constant_state(self):
        return {name:value.detach().cpu() for name,value in self.named_buffers() if name.startswith('enhancement.')}
