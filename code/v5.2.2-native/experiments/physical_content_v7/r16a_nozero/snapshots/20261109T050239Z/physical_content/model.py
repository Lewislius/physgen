"""Direct physical C read with a native-request prior; native text is unchanged."""
import copy,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from physgen_native.message_conditioning import MessageAttention
from physgen_native.generator import load_wan,sinusoidal_embedding_1d
from physgen_native.context_alignment import native_text_context
from physical_scaling.model import PhysicalModel as OriginalPhysicalModel,install_execution_lora,time_features
from physical_scaling.supervision import group_log_prior
from .evidence import resample_words


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

    def forward(self, evidence, *, anchor_only=False, routing_only=False):
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
        routing = native + self.shared_offset + self.project(values)
        execution = native if routing_only else native + self.execution_project(state-anchor)
        return dict(routing=routing, execution=execution), groups

    def content_loss(self,c,evidence):
        routing=c['routing']
        target,_,_=resample_words(evidence['native_words'].detach().float(),evidence['word_groups'],routing.shape[1])
        return (routing.float()-target).square().mean()/self.anchor_target_scale.square().clamp_min(1e-8)


class Fusion(nn.Module):
    def __init__(self, attention):
        super().__init__()
        width, device = attention.dim, attention.q.weight.device
        self.key = nn.Linear(width, width, bias=False, device=device)
        self.value = nn.Linear(width, width, bias=False, device=device)
        with torch.no_grad():
            self.key.weight.copy_(attention.k.weight)
            self.value.weight.copy_(attention.v.weight)
        self.key_norm = copy.deepcopy(attention.norm_k).float().requires_grad_(True)
        self.output = nn.Linear(width, width, bias=False, device=device)
        self.state_value = nn.Linear(width, width, bias=False, device=device)
        self.time = nn.Linear(256, width, bias=False, device=device)
        for layer in (self.output, self.state_value, self.time):
            nn.init.zeros_(layer.weight)
        self.query = copy.deepcopy(attention.q).float().requires_grad_(True)
        self.query_norm = copy.deepcopy(attention.norm_q).float().requires_grad_(True)

    def forward(self, x, context, c, times, geometry, attention, groups, *,
                context_length, memory_mode='joint', native_physical_query=False,
                state_value_enabled=True, shared_physical_query=False):
        from .joint_read import read_physical
        batch, length, width = x.shape
        known = geometry[1] * geometry[2]
        split = lambda value: value.reshape(batch, -1, attention.num_heads, attention.head_dim)
        raw_text_query = attention.q(x)
        text_query = split(attention.norm_q(raw_text_query))
        text_keys = split(attention.norm_k(attention.k(context)))
        text_values = split(attention.v(context))
        kernel = attention.forward.__func__.__globals__.get('flash_attention')
        if kernel is None:
            native_read = F.scaled_dot_product_attention(text_query.transpose(1, 2),
                text_keys.transpose(1, 2).to(text_query), text_values.transpose(1, 2).to(text_query)).transpose(1, 2)
        else:
            native_read = kernel(text_query, text_keys, text_values, k_lens=None)
        text = attention.o(native_read.flatten(2))
        raw = raw_text_query if native_physical_query else self.query(x)
        norm = attention.norm_q if native_physical_query else self.query_norm
        temporal = self.time(time_features(times))[:, 1:].repeat_interleave(known, 1)
        q = split(norm(raw[:, known:] + temporal)).transpose(1, 2)
        k = split(self.key_norm(self.key(c['routing']))).transpose(1, 2).to(q)
        v = split(self.value(c['execution'])).transpose(1, 2).to(q)
        if not 1 <= context_length <= context.shape[1]:
            raise ValueError('Use real cached T5 tokens including EOS, excluding artificial padding')
        request_keys = text_keys[:, :context_length].transpose(1, 2).to(q)
        prior = group_log_prior(groups)[:-1].to(q)
        request_query = q if shared_physical_query else text_query[:, known:].transpose(1, 2).to(q)
        message = read_physical(q, k, v, request_keys, prior, memory_mode,
            request_query=request_query, relation_count=int(groups.max()) + 1).transpose(1, 2).flatten(2)
        state = F.layer_norm(x[:, known:].float(), (width,))
        physical = message + self.state_value(state) * message if state_value_enabled else message
        delta = self.output(physical)
        return text + torch.cat((torch.zeros_like(text[:, :known]), delta), 1)


class EmptyEnhancement(nn.Module):
    def forward(self, evidence):
        memory = evidence['memory']
        return memory.new_zeros((memory.shape[0], 0, 3072)), None


class PhysicalModel(OriginalPhysicalModel):
    def __init__(self, cfg, device='cuda', wan=None):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.wan = load_wan(cfg, device) if wan is None else wan.requires_grad_(False)
        if cfg['variant'] == 'native_lora':
            self.enhancement = EmptyEnhancement()
            self.fusions = nn.ModuleDict()
        else:
            self.enhancement = Enhancement(cfg).to(device)
            self.fusions = nn.ModuleDict({str(i): Fusion(block.cross_attn)
                for i, block in enumerate(self.wan.blocks)
                if (i + 1) % cfg['model']['condition_every'] == 0}).to(device)
        devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg['train']['seed'] + 731)
            if devices:
                torch.cuda.manual_seed_all(cfg['train']['seed'] + 731)
            install_execution_lora(self.wan, cfg['model']['lora_rank'])
        self.learned_names = tuple(name for name, p in self.named_parameters() if p.requires_grad)
        self.checkpointing = cfg['train']['gradient_checkpointing']

    def constant_state(self):
        return {name: value.detach().cpu() for name, value in self.named_buffers()
                if name.startswith('enhancement.')}

    def velocity(self, noisy, first, text, sigma, c, times, groups=None, target=None, valid=None, *,
                 physical_enabled=True, native_physical_query=False,
                 memory_mode="joint", state_value_enabled=True, shared_physical_query=False):
        wan = self.wan
        if noisy.shape[0] != 1:
            raise ValueError('One video per GPU; retain gradient accumulation')
        if wan.freqs.device != noisy.device:
            wan.freqs = wan.freqs.to(noisy.device)
        x = wan.patch_embedding(torch.cat((first, noisy[:, :, 1:]), 2))
        geometry = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)
        known = geometry[1] * geometry[2]
        position = torch.cat((sigma.new_zeros(known), sigma.expand(x.shape[1] - known) * 1000))
        with torch.autocast(noisy.device.type, enabled=False):
            e = wan.time_embedding(sinusoidal_embedding_1d(wan.freq_dim, position).float()[None])
            modulation = wan.time_projection(e).unflatten(-1, (6, wan.dim))
        context = native_text_context(wan, text)
        context_length = min(text.shape[1], wan.text_len)
        grid = torch.tensor([geometry], dtype=torch.long)
        lengths = torch.tensor([x.shape[1]], dtype=torch.long)
        for index, block in enumerate(wan.blocks):
            def run(hidden, condition, layer=block, site=str(index)):
                if not physical_enabled or site not in self.fusions:
                    return layer(hidden, modulation, lengths, grid, wan.freqs, context, None)
                parts = (layer.modulation.unsqueeze(0).float() + modulation).chunk(6, dim=2)
                y = layer.self_attn(layer.norm1(hidden).float() * (1 + parts[1].squeeze(2))
                    + parts[0].squeeze(2), lengths, grid, wan.freqs)
                hidden = hidden.float() + y.float() * parts[2].squeeze(2)
                combined = self.fusions[site](layer.norm3(hidden), context, condition, times,
                    geometry, layer.cross_attn, groups,
                    context_length=context_length, native_physical_query=native_physical_query,
                    memory_mode=memory_mode, state_value_enabled=state_value_enabled,
                    shared_physical_query=shared_physical_query)
                hidden = hidden + combined
                y = layer.ffn(layer.norm2(hidden).float() * (1 + parts[4].squeeze(2)) + parts[3].squeeze(2))
                return hidden + y.float() * parts[5].squeeze(2)
            x = checkpoint(run, x, c, use_reentrant=False) if self.checkpointing and torch.is_grad_enabled() else run(x, c)
        prediction = wan.unpatchify(wan.head(x, e), grid)[0][None].float()
        return prediction, prediction.new_zeros(())


