"""Q enhancement plus independent, full-width physical fusion inside native Wan."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from physgen_native.message_conditioning import MessageBlock
from physgen_native.generator import load_wan, sinusoidal_embedding_1d
from physgen_native.context_alignment import native_text_context
from physgen_native.lora import LoRALinear
from .supervision import (slot_groups, group_log_prior, grouped_probabilities,
                          association_kl, write_regression)


class Enhancement(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg['model']
        self.grouped = cfg['variant'] == 'v6_assoc'
        self.rounds = m['rounds']
        self.seed = nn.Parameter(torch.empty(1,m['q_tokens'],m['qwen_width']))
        nn.init.normal_(self.seed, std=.02)
        self.memory_norm = nn.RMSNorm(m['qwen_width'], eps=1e-6)
        self.relation_norm = nn.RMSNorm(m['qwen_width'], eps=1e-6) if self.grouped else None
        self.block = MessageBlock(m['qwen_width'], m['q_heads'], 2, exchange=True)
        self.final_norm = nn.RMSNorm(m['qwen_width'], eps=1e-6)
        self.project = nn.Linear(m['qwen_width'],m['width'],bias=False)
        self.output_norm = nn.RMSNorm(m['width'],eps=1e-6)
        self.checkpointing = cfg['train']['gradient_checkpointing']

    def forward(self, evidence):
        memory = self.memory_norm(evidence['memory'].detach())
        mask = evidence['memory_mask'].bool()
        state = self.seed.expand(memory.shape[0],-1,-1)
        groups = None
        if self.grouped:
            relation = self.relation_norm(evidence['relation_vectors'].detach())
            groups = slot_groups(state.shape[1], relation.shape[1], state.device)
            state = state + relation[:,groups]
        state_mask = torch.ones(state.shape[:2],dtype=torch.bool,device=state.device)
        for _ in range(self.rounds):
            if self.checkpointing and torch.is_grad_enabled():
                state = checkpoint(self.block,state,memory,mask,state_mask,use_reentrant=False)
            else:
                state = self.block(state,memory,mask,state_mask)
        return self.output_norm(self.project(self.final_norm(state))), groups


def time_features(times):
    delta = torch.cat((times[:,:1]*0, times[:,1:]-times[:,:-1]),1)
    a = sinusoidal_embedding_1d(128,times.flatten())
    b = sinusoidal_embedding_1d(128,delta.flatten())
    return torch.cat((a,b),-1).reshape(*times.shape,256).float()


class Fusion(nn.Module):
    def __init__(self, attention, variant, rank=32):
        super().__init__()
        width = attention.dim
        self.variant = variant
        # Independent parameters, copied ONLY from the official text weights.
        key = nn.Linear(width,width,bias=False,device=attention.k.weight.device)
        value = nn.Linear(width,width,bias=False,device=attention.v.weight.device)
        with torch.no_grad():
            key.weight.copy_(attention.k.weight)
            value.weight.copy_(attention.v.weight)
        if variant != 'v5_ce':
            # Frozen bf16 full-width bases; trainable FP32 low-rank increments.
            key, value = key.to(torch.bfloat16), value.to(torch.bfloat16)
            key, value = LoRALinear(key,rank,rank), LoRALinear(value,rank,rank)
        self.key, self.value = key, value
        self.key_norm = copy.deepcopy(attention.norm_k).float().requires_grad_(True)
        self.output = nn.Linear(width,width,bias=False,device=attention.q.weight.device)
        nn.init.zeros_(self.output.weight)
        self.time = nn.Linear(256,width,bias=False,device=attention.q.weight.device)
        nn.init.zeros_(self.time.weight)
        self.reference_key = nn.Parameter(torch.zeros(1,1,width,device=attention.q.weight.device)) if variant=='v6_assoc' else None

    def forward(self, x, context, c, times, geometry, attention, groups=None, target=None, valid=None):
        batch,length,width = x.shape
        t,h,w = geometry
        known = h*w
        heads, dim = attention.num_heads, attention.head_dim
        split = lambda v: v.reshape(batch,-1,heads,dim)
        q_raw = attention.q(x)
        qt = split(attention.norm_q(q_raw))
        kt = split(attention.norm_k(attention.k(context)))
        vt = split(attention.v(context))
        kernel = attention.forward.__func__.__globals__.get('flash_attention')
        if kernel is None:
            # CPU synthetic tests only; deployed native class always has kernel.
            text = F.scaled_dot_product_attention(qt.transpose(1,2),kt.transpose(1,2).to(qt),vt.transpose(1,2).to(qt)).transpose(1,2)
        else:
            text = kernel(qt,kt,vt,k_lens=None)
        text = attention.o(text.flatten(2))
        te = self.time(time_features(times))[:,1:].repeat_interleave(known,dim=1)
        qc = split(attention.norm_q(q_raw[:,known:]+te)).transpose(1,2)
        kc = split(self.key_norm(self.key(c))).transpose(1,2).to(qc)
        vc = split(self.value(c)).transpose(1,2).to(qc)
        auxiliary = x.new_zeros((),dtype=torch.float32)
        if self.variant == 'v6_assoc':
            if groups is None:
                raise ValueError('Actual C attention needs named relation groups')
            reference = split(self.reference_key.expand(batch,-1,-1)).transpose(1,2).to(qc)
            kc = torch.cat((kc,reference),2)
            vc = torch.cat((vc,torch.zeros_like(reference)),2)
            logits = torch.matmul(qc.float(),kc.float().transpose(-1,-2)).float()/math.sqrt(dim)
            logits = logits + group_log_prior(groups).to(logits)
            probabilities = logits.softmax(-1)
            message = torch.matmul(probabilities.to(vc),vc).transpose(1,2).flatten(2)
            if target is not None:
                auxiliary = association_kl(grouped_probabilities(probabilities,groups),target['association'],valid)
        else:
            message = F.scaled_dot_product_attention(qc,kc,vc).transpose(1,2).flatten(2)
        delta = self.output(message)
        if self.variant == 'v6_write' and target is not None:
            # Reproduce the requested initial V6 equation, not the corrected KL.
            with torch.no_grad():
                anchor = F.linear(target['text_anchors'].to(attention.v.weight),attention.v.weight)
                anchor = F.linear(anchor,attention.o.weight).float()
                desired = torch.matmul(target['support'].float(),anchor)
            auxiliary = write_regression(delta,desired,valid)
        return text + torch.cat((torch.zeros_like(delta[:,:known]),delta),1), auxiliary


def install_execution_lora(wan, rank):
    for block in wan.blocks:
        for name in ('q','k','v','o'):
            setattr(block.self_attn,name,LoRALinear(getattr(block.self_attn,name),rank,rank))
        for index,layer in enumerate(block.ffn):
            if isinstance(layer,nn.Linear):
                block.ffn[index] = LoRALinear(layer,rank,rank)


class PhysicalModel(nn.Module):
    def __init__(self,cfg,device='cuda',wan=None):
        super().__init__()
        self.cfg = cfg
        self.wan = load_wan(cfg,device) if wan is None else wan.requires_grad_(False)
        self.enhancement = Enhancement(cfg).to(device)
        self.fusions = nn.ModuleDict({str(i): Fusion(block.cross_attn,cfg['variant'],cfg['model']['lora_rank'])
                                     for i,block in enumerate(self.wan.blocks)
                                     if (i+1)%cfg['model']['condition_every']==0}).to(device)
        install_execution_lora(self.wan,cfg['model']['lora_rank'])
        self.bridge = nn.Linear(cfg['model']['width'],cfg['model']['qwen_width'],bias=False).to(device) if cfg['variant']=='v5_ce' else None
        self.learned_names = tuple(name for name,p in self.named_parameters() if p.requires_grad)
        self.checkpointing = cfg['train']['gradient_checkpointing']

    def set_phase(self,step):
        joint = step >= self.cfg['stages']['interface']
        for name,p in self.wan.named_parameters():
            if name.endswith(('lora_a','lora_b')):
                p.requires_grad_(joint)
        return 'joint' if joint else 'interface'

    def optim_groups(self):
        named = dict(self.named_parameters())
        new,wan = [],[]
        for name in self.learned_names:
            (wan if name.startswith('wan.') else new).append(named[name])
        return [dict(name='physical',params=new,lr=self.cfg['train']['new_lr']),
                dict(name='wan_lora',params=wan,lr=self.cfg['train']['wan_lora_lr'])]

    def velocity(self,noisy,first,text,sigma,c,times,groups=None,target=None,valid=None):
        wan = self.wan
        if noisy.shape[0] != 1:
            raise ValueError('One video per allocated GPU; use gradient accumulation')
        if wan.freqs.device != noisy.device:
            wan.freqs = wan.freqs.to(noisy.device)
        x = wan.patch_embedding(torch.cat((first,noisy[:,:,1:]),2))
        geometry = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1,2)
        length,known = x.shape[1],geometry[1]*geometry[2]
        position = torch.cat((sigma.new_zeros(known),sigma.expand(length-known)*1000))
        with torch.autocast(noisy.device.type,enabled=False):
            e = wan.time_embedding(sinusoidal_embedding_1d(wan.freq_dim,position).float()[None])
            modulation = wan.time_projection(e).unflatten(-1,(6,wan.dim))
        context = native_text_context(wan,text)
        grid = torch.tensor([geometry],dtype=torch.long)
        lengths = torch.tensor([length],dtype=torch.long)
        aux = x.new_zeros((),dtype=torch.float32)
        patch_valid = None
        if valid is not None:
            patch_valid = F.avg_pool2d(valid[None,None].float(),2).flatten()[None].repeat(1,geometry[0]-1)
        for index,block in enumerate(wan.blocks):
            def run(hidden,condition,layer=block,site=str(index)):
                if site not in self.fusions:
                    return layer(hidden,modulation,lengths,grid,wan.freqs,context,None),hidden.new_zeros((),dtype=torch.float32)
                parts = (layer.modulation.unsqueeze(0).float()+modulation).chunk(6,dim=2)
                y = layer.self_attn(layer.norm1(hidden).float()*(1+parts[1].squeeze(2))+parts[0].squeeze(2),lengths,grid,wan.freqs)
                hidden = hidden.float()+y.float()*parts[2].squeeze(2)
                update,loss = self.fusions[site](layer.norm3(hidden),context,condition,times,geometry,
                                               layer.cross_attn,groups,target,patch_valid)
                hidden = hidden+update
                y = layer.ffn(layer.norm2(hidden).float()*(1+parts[4].squeeze(2))+parts[3].squeeze(2))
                return hidden+y.float()*parts[5].squeeze(2),loss
            if self.checkpointing and torch.is_grad_enabled():
                x,loss = checkpoint(run,x,c,use_reentrant=False)
            else:
                x,loss = run(x,c)
            aux = aux+loss
        velocity = wan.unpatchify(wan.head(x,e),grid)[0][None].float()
        return velocity,aux/max(1,len(self.fusions))


def semantic_ce(model,knowledge,evidence,answer):
    """C-only input-gradient training; frozen decoder, no image/H/prompt bypass."""
    c,_ = model.enhancement(evidence)
    state = model.bridge(c)
    tokenizer,backbone = knowledge.tokenizer,knowledge.backbone
    prefix = tokenizer.apply_chat_template([{'role':'user','content':
        'Describe the relevant physical conditions, response, and allowed or preserved properties.'}],
        tokenize=False,add_generation_prompt=True)
    qids = tokenizer.encode(prefix,add_special_tokens=False)
    aids = tokenizer.encode(answer+tokenizer.eos_token,add_special_tokens=False)
    ids = torch.tensor([qids+aids],device=state.device)
    language = backbone.model.language_model
    embedded = torch.cat((state.to(language.embed_tokens.weight.dtype),language.embed_tokens(ids)),1)
    hidden = language(inputs_embeds=embedded,use_cache=False,return_dict=True).last_hidden_state
    start = state.shape[1]+len(qids)
    logits = backbone.lm_head(hidden[:,start-1:-1])
    return F.cross_entropy(logits.float().flatten(0,1),ids[:,len(qids):].flatten())


def enable_frozen_decoder_checkpointing(backbone):
    # Frozen parameters stay in eval; gradients to the C input are retained.
    from functools import wraps
    for layer in backbone.model.language_model.layers:
        original = layer.forward
        def wrap(fn):
            @wraps(fn)
            def run(*args,**kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(fn,*args,use_reentrant=False,**kwargs)
                return fn(*args,**kwargs)
            return run
        layer.forward = wrap(original)
