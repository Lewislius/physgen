"""Full-width semantic routing and Q-composed physical execution.

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
from physgen_native.context_alignment import native_text_context
from physgen_native.generator import sinusoidal_embedding_1d


def map_content(content, function):
    """Transform both explicit memories without hidden per-forward state."""
    return {name: function(content[name]) for name in ('routing', 'execution')}


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


class Enhancement(nn.Module):
    def __init__(self,cfg,interface=None):
        super().__init__();m=cfg['model']
        if interface is None:interface=torch.load(cfg['paths']['interface'],map_location='cpu',weights_only=True)
        for name in ['mean','scale','weight','bias','target_scale']:
            self.register_buffer('anchor_'+name,interface[name].float())
        self.seed=nn.Parameter(torch.randn(1,m['q_tokens'],m['qwen_width'])*.02)
        self.block=ContentBlock(m['qwen_width'],m['q_heads'],m['grounded_read_heads'])
        self.project=nn.Linear(m['qwen_width'],m['width'],bias=False)
        nn.init.zeros_(self.project.weight)
        self.rounds=m['rounds'];self.checkpointing=cfg['train']['gradient_checkpointing']
        self.shared_offset=nn.Parameter(torch.zeros(m['width']))
        # Preserve all existing initializer draws, including the later fusion.
        with torch.random.fork_rng(devices=[]):
            self.execution_project=nn.Linear(m['qwen_width'],m['width'],bias=False)
            nn.init.zeros_(self.execution_project.weight)
            self.request_project=nn.Linear(m['qwen_width'],m['width'],bias=True)
            with torch.no_grad():
                self.request_project.weight.copy_(self.anchor_weight)
                self.request_project.bias.copy_(self.anchor_bias)

    def request_memory(self,evidence):
        """Keep literal, image-contextualized request words outside Q/C pooling.

        Empty requests remain zero-length memories. The one full-width interface
        starts from the existing fitted map and learns from real-video generation.
        """
        words=evidence['request_words'].detach().float()
        if words.ndim==2:words=words[None]
        return self.request_project((words-self.anchor_mean)/self.anchor_scale)

    def read_values(self, state, memory, words, groups, word_groups):
        """Use refined Q for weights; values project the fitted input words.

        There is no seed/FFN residual added after this read and no output RMSNorm
        that fixes its magnitude. The existing full-width project stays learned.
        The bias-free shared value map sees all input channels for every head;
        values bypass memory LayerNorm and its learned affine offset. This does
        not prevent learned attention or projection from ignoring data.
        """
        attention = self.block.cross
        split = lambda x: x.unflatten(-1, (attention.heads, attention.head_dim)).transpose(1, 2)
        q = split(attention.q(self.block.query_norm(state)))
        keys = split(attention.k(self.block.memory_norm(memory)))
        word_keys = split(attention.k(self.block.memory_norm(words)))
        values = split(attention.v(memory))
        word_values = split(attention.v(words))
        local = self.block.grounded_heads
        allowed = groups[:, None] == word_groups[None, :]
        grounded = F.scaled_dot_product_attention(
            q[:, :local], word_keys[:, :local], word_values[:, :local],
            attn_mask=allowed[None, None])
        contextual = F.scaled_dot_product_attention(
            q[:, local:], keys[:, local:], values[:, local:])
        return torch.cat((grounded, contextual), 1).transpose(1, 2).flatten(2)

    def forward(self, evidence, *, anchor_only=False, routing_only=False, anchor_routing=False):
        words = evidence['words'].detach().float()
        words = (words - self.anchor_mean) / self.anchor_scale
        anchor, groups, _ = resample_words(words, evidence['word_groups'], self.seed.shape[1])
        native = F.linear(anchor, self.anchor_weight, self.anchor_bias)
        if anchor_only:
            return dict(routing=native, execution=native), groups
        words = words[None] if words.ndim == 2 else words
        request = evidence['request_words'].detach().float()
        request = (request - self.anchor_mean) / self.anchor_scale
        request = request[None] if request.ndim == 2 else request
        memory = torch.cat((request, words), 1)
        mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        state = anchor + self.seed
        for _ in range(self.rounds):
            arguments = (state, anchor, memory, mask, groups, words, evidence['word_groups'])
            state = (checkpoint(self.block, *arguments, use_reentrant=False)
                     if self.checkpointing and torch.is_grad_enabled() else self.block(*arguments))
        values = self.read_values(state, memory, words, groups, evidence['word_groups'])
        routing = native if anchor_routing else native + self.shared_offset + self.project(values)
        execution = native if routing_only else native + self.execution_project(state-anchor)
        return dict(routing=routing, execution=execution), groups

    def content_loss(self,c,evidence):
        routing=c['routing']
        target,_,_=resample_words(evidence['native_words'].detach().float(),evidence['word_groups'],routing.shape[1])
        return (routing.float()-target).square().mean()/self.anchor_target_scale.square().clamp_min(1e-8)


class Fusion(nn.Module):
    def __init__(self,attention):
        super().__init__();width=attention.dim
        self.key=nn.Linear(width,width,bias=False,device=attention.k.weight.device)
        self.value=nn.Linear(width,width,bias=False,device=attention.v.weight.device)
        with torch.no_grad():
            self.key.weight.copy_(attention.k.weight);self.value.weight.copy_(attention.v.weight)
        self.key_norm=copy.deepcopy(attention.norm_k).float().requires_grad_(True)
        self.output=nn.Linear(width,width,bias=False,device=attention.q.weight.device)
        self.state_value=nn.Linear(width,width,bias=False,device=attention.q.weight.device)
        self.time=nn.Linear(256,width,bias=False,device=attention.q.weight.device)
        for layer in [self.output,self.state_value,self.time]:nn.init.zeros_(layer.weight)
        self.reference_key=nn.Parameter(torch.zeros(1,1,width,device=attention.q.weight.device))
        # Preserve every original R10A draw and physical read/write. This branch
        # reads an independent Qwen request memory, with a zero-initial write.
        device=attention.q.weight.device
        devices=[device.index or 0] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.request_output=nn.Linear(width,width,bias=False,device=device)
            nn.init.zeros_(self.request_output.weight)

    def read_request(self,query,memory,attention):
        """Native video queries address Qwen request words through frozen K/V.

        Unlike the previous branch, this does not reread native T5 values or the
        same physical C. There are no padding tokens or fabricated null words.
        """
        if memory.shape[1]==0:
            return query.new_zeros(query.shape[0],query.shape[2],attention.dim)
        split=lambda value:value.unflatten(-1,(attention.num_heads,attention.head_dim)).transpose(1,2)
        keys=split(attention.norm_k(attention.k(memory))).to(query)
        values=split(attention.v(memory)).to(query)
        message=F.scaled_dot_product_attention(query,keys,values)
        return self.request_output(message.transpose(1,2).flatten(2))

    def forward(self,x,context,c,times,geometry,attention,groups=None,target=None,valid=None,
                *,request_memory=None,request_enabled=True):
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
        k=split(self.key_norm(self.key(c['routing']))).transpose(1,2).to(q)
        v=split(self.value(c['execution'])).transpose(1,2).to(q)
        reference=split(self.reference_key.expand(batch,-1,-1)).transpose(1,2).to(q)
        k=torch.cat((k,reference),2);v=torch.cat((v,torch.zeros_like(reference)),2)
        prior=group_log_prior(groups).to(q)
        scores=(q.float()@k.float().transpose(-1,-2))/math.sqrt(dim)+prior
        z=(scores.softmax(-1).to(v)@v).transpose(1,2).flatten(2)
        state=F.layer_norm(x[:,known:].float(),(width,))
        delta=self.output(z+self.state_value(state)*z)
        if request_enabled:
            if request_memory is None:raise ValueError('Provide the explicit request memory, including an empty tensor for a null request')
            delta=delta+self.read_request(qt[:,known:].transpose(1,2),request_memory,attention)
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
            self.fusions=nn.ModuleDict({str(i):Fusion(block.cross_attn) for i,block in enumerate(self.wan.blocks)
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

    def optim_groups(self):
        groups=super().optim_groups()
        read=next(g for g in groups if g['name']=='fusion_read')
        write=next(g for g in groups if g['name']=='fusion_write')
        request={id(p) for name,p in self.named_parameters() if '.request_output.' in name}
        write['params'].extend(p for p in read['params'] if id(p) in request)
        read['params']=[p for p in read['params'] if id(p) not in request]
        return groups

    def set_phase(self,step):
        joint=step>=self.cfg['stages']['interface']
        self.enhancement.requires_grad_(True)
        self.fusions.requires_grad_(True)
        for name,p in self.wan.named_parameters():
            p.requires_grad_(joint and name.endswith(('lora_a','lora_b')))
        return 'joint' if joint else 'interface'

    def velocity(self,noisy,first,text,sigma,c,times,groups=None,target=None,valid=None,
                 *,physical_enabled=True,request_enabled=True,request_memory=None):
        """R10A parallel fusion; native-only pass shares its actual Wan parameters.

        The explicit per-call flag is captured during checkpoint recomputation;
        no temporary parameter zeroing or module-dictionary swapping is used.
        """
        wan=self.wan
        if physical_enabled and self.fusions and request_enabled and request_memory is None:
            raise ValueError('Real generation requires an explicit Qwen request memory')
        if noisy.shape[0]!=1:raise ValueError('One video per GPU; use gradient accumulation')
        if wan.freqs.device!=noisy.device:wan.freqs=wan.freqs.to(noisy.device)
        x=wan.patch_embedding(torch.cat((first,noisy[:,:,1:]),2))
        geometry=tuple(x.shape[2:]);x=x.flatten(2).transpose(1,2)
        length,known=x.shape[1],geometry[1]*geometry[2]
        position=torch.cat((sigma.new_zeros(known),sigma.expand(length-known)*1000))
        with torch.autocast(noisy.device.type,enabled=False):
            e=wan.time_embedding(sinusoidal_embedding_1d(wan.freq_dim,position).float()[None])
            modulation=wan.time_projection(e).unflatten(-1,(6,wan.dim))
        context=native_text_context(wan,text)
        grid=torch.tensor([geometry],dtype=torch.long);lengths=torch.tensor([length],dtype=torch.long)
        auxiliary=x.new_zeros((),dtype=torch.float32)
        for index,block in enumerate(wan.blocks):
            def run(hidden,condition,semantic,layer=block,site=str(index),enabled=physical_enabled,
                    use_request=request_enabled):
                if not enabled or site not in self.fusions:
                    return layer(hidden,modulation,lengths,grid,wan.freqs,context,None),hidden.new_zeros((),dtype=torch.float32)
                parts=(layer.modulation.unsqueeze(0).float()+modulation).chunk(6,dim=2)
                y=layer.self_attn(layer.norm1(hidden).float()*(1+parts[1].squeeze(2))+parts[0].squeeze(2),lengths,grid,wan.freqs)
                hidden=hidden.float()+y.float()*parts[2].squeeze(2)
                update,loss=self.fusions[site](layer.norm3(hidden),context,condition,times,geometry,
                                               layer.cross_attn,groups,request_memory=semantic,
                                               request_enabled=use_request)
                hidden=hidden+update
                y=layer.ffn(layer.norm2(hidden).float()*(1+parts[4].squeeze(2))+parts[3].squeeze(2))
                return hidden+y.float()*parts[5].squeeze(2),loss
            if self.checkpointing and torch.is_grad_enabled():
                x,loss=checkpoint(run,x,c,request_memory,use_reentrant=False)
            else:x,loss=run(x,c,request_memory)
            auxiliary=auxiliary+loss
        velocity=wan.unpatchify(wan.head(x,e),grid)[0][None].float()
        return velocity,auxiliary/max(1,len(self.fusions))
