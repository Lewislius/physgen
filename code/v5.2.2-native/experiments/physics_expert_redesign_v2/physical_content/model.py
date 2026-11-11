"""Bind physical execution values to the native request before writing to video.

R10A enhancement and state-dependent execution are retained. Native text Q/K/V/O
remain frozen and unchanged. The physical branch uses native request keys to bind
C values to the prompt, then reads those values with each video's native query.
There is no additional request writer, native-attention modulation or new encoder.
"""
import copy,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from physgen_native.message_conditioning import MessageAttention
from physgen_native.lora import LoRALinear
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



class RequestKeyEnhancer(nn.Module):
    """Shared request-conditioned physical key enhancer (P1 only)."""
    def __init__(self, width=3072, heads=24, eps=1e-6):
        super().__init__()
        if width % heads:
            raise ValueError('width must divide heads')
        self.heads, self.head_dim = heads, width // heads
        self.route_norm = nn.LayerNorm(width, eps=eps, elementwise_affine=False)
        self.text_norm = nn.LayerNorm(width, eps=eps, elementwise_affine=False)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        for layer in (self.q, self.k, self.v):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(self.o.weight)

    def split(self, x):
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)

    def forward(self, route, text, text_mask):
        q = self.split(self.q(self.route_norm(route)))
        t = self.text_norm(text)
        k, v = self.split(self.k(t)), self.split(self.v(t))
        message = F.scaled_dot_product_attention(
            q, k, v, attn_mask=text_mask[:, None, None, :], dropout_p=0.0)
        return route + self.o(message.transpose(1, 2).flatten(2))


def frozen_base(linear, x):
    return linear.base(x) if isinstance(linear, LoRALinear) else linear(x)


class PhysicsReader(nn.Module):
    """Full-width direct C reader; P1 keeps the state-value interaction."""
    def __init__(self, attention, *, state_value=False):
        super().__init__()
        width = attention.dim
        self.key = nn.Linear(width, width, bias=False, device=attention.k.weight.device)
        self.value = nn.Linear(width, width, bias=False, device=attention.v.weight.device)
        with torch.no_grad():
            self.key.weight.copy_(attention.k.weight)
            self.value.weight.copy_(attention.v.weight)
        self.key_norm = copy.deepcopy(attention.norm_k).float().requires_grad_(True)
        self.output = nn.Linear(width, width, bias=False, device=attention.q.weight.device)
        self.time = nn.Linear(256, width, bias=False, device=attention.q.weight.device)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.time.weight)
        self.reference_key = nn.Parameter(torch.zeros(1, 1, width, device=attention.q.weight.device))
        self.state_value = None
        if state_value:
            self.state_value = nn.Linear(width, width, bias=False, device=attention.q.weight.device)
            nn.init.zeros_(self.state_value.weight)

    def forward(self, hidden, norm3, route, execute, groups, times, geometry,
                attention, known):
        batch, length, width = hidden.shape
        heads, dim = attention.num_heads, attention.head_dim
        split = lambda value: value.reshape(batch, -1, heads, dim)
        time = self.time(time_features(times))[:, 1:].repeat_interleave(known, 1)
        raw = norm3(hidden)
        q = split(attention.norm_q(frozen_base(attention.q, raw[:, known:]) + time)).transpose(1, 2)
        k = split(self.key_norm(self.key(route))).transpose(1, 2).to(q)
        v = split(self.value(execute)).transpose(1, 2).to(q)
        reference = split(self.reference_key.expand(batch, -1, -1)).transpose(1, 2).to(q)
        prior = group_log_prior(groups).to(q)
        keys = torch.cat((k, reference), 2)
        values = torch.cat((v, torch.zeros_like(reference)), 2)
        weights = ((q.float() @ keys.float().transpose(-1, -2)) / math.sqrt(dim) + prior).softmax(-1)
        message = (weights.to(values) @ values).transpose(1, 2).flatten(2)
        state = F.layer_norm(hidden[:, known:].float(), (width,))
        physical = message
        if self.state_value is not None:
            physical = message + self.state_value(state) * message
        delta = self.output(physical)
        message = torch.cat((hidden.new_zeros(hidden.shape[0], known, width), message), 1)
        delta = torch.cat((hidden.new_zeros(hidden.shape[0], known, width), delta), 1)
        return message, delta


class SourceConditionedExpert(nn.Module):
    """Full-width source-conditioned expert; no 3072->512->3072 path."""
    def __init__(self, width=3072, eps=1e-6):
        super().__init__()
        self.width, self.eps = width, eps
        self.state = nn.Linear(width, width, bias=False)
        self.source = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        nn.init.kaiming_uniform_(self.state.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.source.weight, a=math.sqrt(5))
        nn.init.zeros_(self.output.weight)

    def forward(self, u, source):
        source = F.layer_norm(source.float(), (self.width,), eps=self.eps)
        joint = F.gelu(self.state(u).float(), approximate='tanh') * self.source(source)
        return self.output(joint)


def install_text_qo_lora(wan, rank=16, alpha=16):
    for block in wan.blocks:
        for name in ('q', 'o'):
            base = getattr(block.cross_attn, name)
            if not isinstance(base, LoRALinear):
                setattr(block.cross_attn, name, LoRALinear(base, rank, alpha))


def install_self_ffn_lora(wan, rank):
    for block in wan.blocks:
        for name in ('q', 'k', 'v', 'o'):
            base = getattr(block.self_attn, name)
            if not isinstance(base, LoRALinear):
                setattr(block.self_attn, name, LoRALinear(base, rank, rank))
        for index, layer in enumerate(block.ffn):
            if isinstance(layer, nn.Linear):
                block.ffn[index] = LoRALinear(layer, rank, rank)


class PhysicalModel(nn.Module):
    def __init__(self, cfg, device='cuda', wan=None):
        super().__init__()
        self.cfg = cfg
        self.variant = cfg['variant']
        self.wan = load_wan(cfg, device) if wan is None else wan.requires_grad_(False)
        self.enhancement = Enhancement(cfg).to(device)
        every = int(cfg['model'].get('condition_every', 3))
        sites = [i for i in range(len(self.wan.blocks)) if (i + 1) % every == 0]
        state_value = self.variant == 'p1_request_key'
        self.readers = nn.ModuleDict({str(i): PhysicsReader(self.wan.blocks[i].cross_attn,
                                                            state_value=state_value)
                                      for i in sites}).to(device)
        self.request_key = None
        if self.variant == 'p1_request_key':
            self.request_key = RequestKeyEnhancer(cfg['model']['width'], self.wan.blocks[0].cross_attn.num_heads).to(device)
        self.experts = None
        if self.variant == 'p4_source_conditioned_experts':
            width = cfg['model']['width']
            self.experts = nn.ModuleDict({
                str(i): nn.ModuleDict({
                    'physics': SourceConditionedExpert(width).to(device),
                    'context': SourceConditionedExpert(width).to(device),
                }) for i in sites}).to(device)
        native_rank = 64 if self.variant == 'p2_pre_self' else 32
        install_self_ffn_lora(self.wan, native_rank)
        install_text_qo_lora(self.wan, 16, 16)
        self.learned_names = tuple(name for name, p in self.named_parameters() if p.requires_grad)
        self.checkpointing = cfg['train']['gradient_checkpointing']

    def constant_state(self):
        return {name: value.detach().cpu() for name, value in self.named_buffers()
                if name.startswith('enhancement.')}

    def set_phase(self, step):
        # joint400: selected adapters are already trainable; never unfreeze
        # native bases or Qwen/text base weights.
        self.wan.eval()
        return 'joint'

    def optim_groups(self):
        named = dict(self.named_parameters())
        groups = {g: [] for g in ('enhancement', 'physics_read', 'physics_write',
                                  'native_self_ffn_lora', 'native_text_qo_lora',
                                  'request_key', 'expert_in', 'expert_out')}
        for name in self.learned_names:
            if name.startswith('enhancement.'):
                role = 'enhancement'
            elif name.startswith('request_key.'):
                role = 'request_key'
            elif name.startswith('experts.') and '.output.' in name:
                role = 'expert_out'
            elif name.startswith('experts.'):
                role = 'expert_in'
            elif name.startswith('readers.') and '.output.' in name:
                role = 'physics_write'
            elif name.startswith('readers.'):
                role = 'physics_read'
            elif name.startswith('wan.') and 'cross_attn' in name and ('.q.' in name or '.o.' in name):
                role = 'native_text_qo_lora'
            elif name.startswith('wan.'):
                role = 'native_self_ffn_lora'
            else:
                raise ValueError(f'Unassigned parameter: {name}')
            groups[role].append(named[name])
        return [dict(name=role, params=groups[role],
                     lr=self.cfg['train']['peak_lrs'][role]) for role in groups]

    def _masked(self, value, known):
        return torch.cat((torch.zeros_like(value[:, :known]), value[:, known:]), 1)

    def velocity(self, noisy, first, text, sigma, c, times, groups=None,
                 target=None, valid=None, *, physical_enabled=True,
                 direct_physical_read=False, state_value_enabled=True, **legacy):
        wan = self.wan
        if noisy.shape[0] != 1:
            raise ValueError('One video per GPU')
        if wan.freqs.device != noisy.device:
            wan.freqs = wan.freqs.to(noisy.device)
        x = wan.patch_embedding(torch.cat((first, noisy[:, :, 1:]), 2))
        geometry = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)
        length, known = x.shape[1], geometry[1] * geometry[2]
        position = torch.cat((sigma.new_zeros(known), sigma.expand(length - known) * 1000))
        with torch.autocast(noisy.device.type, enabled=False):
            e = wan.time_embedding(sinusoidal_embedding_1d(wan.freq_dim, position).float()[None])
            modulation = wan.time_projection(e).unflatten(-1, (6, wan.dim))
        context = native_text_context(wan, text)
        context_length = min(text.shape[1], wan.text_len)
        text_mask = torch.zeros((1, context.shape[1]), dtype=torch.bool, device=context.device)
        text_mask[:, :context_length] = True
        route_raw = c['routing']
        route_effective = route_raw
        if self.request_key is not None:
            route_effective = self.request_key(route_raw, context, text_mask)
        execute = c['execution']
        grid = torch.tensor([geometry], dtype=torch.long)
        lengths = torch.tensor([length], dtype=torch.long)
        aux = x.new_zeros((), dtype=torch.float32)
        for index, block in enumerate(wan.blocks):
            site = str(index)

            def run(hidden, route, execute_, layer=block, site=site):
                if physical_enabled and site in self.readers:
                    if self.variant == 'p2_pre_self':
                        _, delta = self.readers[site](hidden, layer.norm3, route, execute_,
                                                       groups, times, geometry,
                                                       layer.cross_attn, known)
                        hidden = hidden.float() + self._masked(delta, known).float()
                        return layer(hidden, modulation, lengths, grid, wan.freqs, context, None)
                    parts = (layer.modulation.unsqueeze(0).float() + modulation).chunk(6, dim=2)
                    y = layer.self_attn(
                        layer.norm1(hidden).float() * (1 + parts[1].squeeze(2)) + parts[0].squeeze(2),
                        lengths, grid, wan.freqs)
                    h_sa = hidden.float() + y.float() * parts[2].squeeze(2)
                    message, delta = self.readers[site](
                        h_sa, layer.norm3, route, execute_, groups, times,
                        geometry, layer.cross_attn, known)
                    h_p = h_sa + self._masked(delta, known).float()
                    # P1 is deliberately parallel: text attention reads h_sa,
                    # before the physical delta; P4 reads h_p after the write.
                    text_source = h_sa if self.variant == 'p1_request_key' else h_p
                    text_update = layer.cross_attn(layer.norm3(text_source), context, None)
                    h_pt = h_p + text_update
                    if self.variant == 'p1_request_key':
                        h_mix = h_sa + text_update + self._masked(delta, known).float()
                        y = layer.ffn(
                            layer.norm2(h_mix).float() * (1 + parts[4].squeeze(2)) + parts[3].squeeze(2))
                        return h_mix + y.float() * parts[5].squeeze(2)
                    u = layer.norm2(h_pt).float() * (1 + parts[4].squeeze(2)) + parts[3].squeeze(2)
                    ffn = layer.ffn(u).float()
                    ep = self.experts[site]['physics'](u, message)
                    et = self.experts[site]['context'](u, text_update)
                    ep = self._masked(ep, known)
                    return h_pt + parts[5].squeeze(2) * (ffn + ep + et)
                return layer(hidden, modulation, lengths, grid, wan.freqs, context, None)

            if self.checkpointing and torch.is_grad_enabled():
                x = checkpoint(run, x, route_effective, execute, use_reentrant=False)
            else:
                x = run(x, route_effective, execute)
        prediction = wan.unpatchify(wan.head(x, e), grid)[0][None].float()
        return prediction, aux / max(1, len(self.readers))
