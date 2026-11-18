"""R15Bt: R15B request-bound physical fusion for text-to-video.

R15B Enhancement, R10A state-value execution, and the B400 joint direct/
request-bound physical read are retained byte-for-byte in structure.  The only
task change is that T2V has no first latent frame: the full 21-frame latent is
noisy, every token receives a native time position, and the physical write is
applied to all frames.  ``first`` must be None.
"""
import copy
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from physgen_native.message_conditioning import MessageAttention
from physgen_native.generator import load_wan, sinusoidal_embedding_1d
from physgen_native.context_alignment import native_text_context
from physical_scaling.model import (PhysicalModel as OriginalPhysicalModel,
                                    install_execution_lora, time_features)
from physical_scaling.supervision import group_log_prior

from .evidence import resample_words


def read_physical(q, k, v, request_keys, reference, prior, mode='joint', *,
                  relation_count=None):
    """B400 joint read over direct C and request-associated C.

    ``q`` is [B,H,N,d], ``k``/``v`` are [B,H,S,d], ``request_keys`` are
    [B,H,L,d], ``reference`` is [B,H,1,d], and ``prior`` has R+1 entries.
    The two physical memories share one zero-value reference and the original
    total relation prior; the bound memory preserves the equal-relation mass.
    """
    scale = math.sqrt(q.shape[-1])
    if mode not in {'direct', 'bound', 'joint'}:
        raise ValueError(mode)
    if mode != 'direct':
        binding = ((request_keys.float() @ k.float().transpose(-1, -2)) / scale
                   + prior[:-1]).softmax(-1)
        bound_values = binding.to(v) @ v
        log_relations = (math.log(relation_count) if relation_count is not None
                         else float(prior[:-1].float().logsumexp(-1)))
        bound_prior = prior.new_full((request_keys.shape[-2],),
                                     log_relations - math.log(request_keys.shape[-2]))
    if mode == 'direct':
        keys, values, physical_prior = k, v, prior[:-1]
    elif mode == 'bound':
        keys, values, physical_prior = request_keys, bound_values, bound_prior
    else:
        keys = torch.cat((k, request_keys), -2)
        values = torch.cat((v, bound_values), -2)
        physical_prior = torch.cat((prior[:-1], bound_prior), -1).float() - math.log(2.)
    keys = torch.cat((keys, reference), -2)
    values = torch.cat((values, torch.zeros_like(reference)), -2)
    logits = q.float() @ keys.float().transpose(-1, -2) / scale
    logits = logits + torch.cat((physical_prior, prior[-1:]), -1)
    return logits.softmax(-1).to(values) @ values


def map_content(content, function):
    """Transform both explicit memories without hidden per-forward state."""
    return {name: function(content[name]) for name in ('routing', 'execution')}


class ContentBlock(nn.Module):
    def __init__(self, width, heads, grounded_heads):
        super().__init__()
        if not 0 < grounded_heads < heads:
            raise ValueError('Retain both relation-grounded and full-context reads')
        self.grounded_heads = grounded_heads
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross = MessageAttention(width, heads)
        self.exchange_norm = nn.LayerNorm(width)
        self.exchange = MessageAttention(width, heads)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width, bias=False),
                                 nn.GELU(),
                                 nn.Linear(2 * width, width, bias=False))
        self.update_norm = nn.RMSNorm(width, eps=1e-6)

    def read_heads(self, query, memory, memory_mask, words, word_groups, groups):
        attention = self.cross
        split = lambda x: x.unflatten(-1, (attention.heads, attention.head_dim)).transpose(1, 2)
        q = split(attention.q(query))
        local = self.grounded_heads
        k = split(attention.k(memory))
        v = split(attention.v(memory))
        wk = split(attention.k(words))
        wv = split(attention.v(words))
        allowed = groups[:, None] == word_groups[None, :]
        grounded = F.scaled_dot_product_attention(
            q[:, :local], wk[:, :local], wv[:, :local], attn_mask=allowed[None, None])
        contextual = F.scaled_dot_product_attention(
            q[:, local:], k[:, local:], v[:, local:], attn_mask=memory_mask[:, None, None, :])
        return torch.cat((grounded, contextual), 1)

    def forward(self, state, anchor, memory, memory_mask, groups, words, word_groups):
        read = self.read_heads(self.query_norm(state), self.memory_norm(memory),
                               memory_mask, self.memory_norm(words), word_groups, groups)
        state = state + self.cross.o(read.transpose(1, 2).flatten(2))
        normalized = self.exchange_norm(state)
        summaries = torch.stack(
            [normalized[:, groups == r].mean(1) for r in range(int(groups.max()) + 1)], 1)
        exchange_memory = torch.cat((normalized, summaries), 1)
        local = groups[:, None] == groups[None, :]
        allowed = torch.cat(
            (local, torch.ones(len(groups), len(summaries[0]),
                               device=state.device, dtype=torch.bool)), 1)
        attention = self.exchange
        split = lambda x: x.unflatten(-1, (attention.heads, attention.head_dim)).transpose(1, 2)
        values = F.scaled_dot_product_attention(
            split(attention.q(normalized)), split(attention.k(exchange_memory)),
            split(attention.v(exchange_memory)), attn_mask=allowed[None, None])
        state = state + attention.o(values.transpose(1, 2).flatten(2))
        state = state + self.ffn(self.ffn_norm(state))
        return anchor + self.update_norm(state - anchor)


class Enhancement(nn.Module):
    def __init__(self, cfg, interface=None):
        super().__init__()
        m = cfg['model']
        if interface is None:
            interface = torch.load(Path(cfg['paths']['interface']),
                                   map_location='cpu', weights_only=True)
        for name in ('mean', 'scale', 'weight', 'bias', 'target_scale'):
            if name not in interface:
                raise KeyError(f'Interface is missing {name}')
            self.register_buffer(f'anchor_{name}', interface[name].float())
        self.seed = nn.Parameter(torch.randn(1, m['q_tokens'], m['qwen_width']) * .02)
        self.block = ContentBlock(m['qwen_width'], m['q_heads'], m['grounded_read_heads'])
        self.project = nn.Linear(m['qwen_width'], m['width'], bias=False)
        nn.init.zeros_(self.project.weight)
        self.rounds = m['rounds']
        self.checkpointing = cfg['train']['gradient_checkpointing']
        self.shared_offset = nn.Parameter(torch.zeros(m['width']))
        # Preserve all existing initializer draws, including the later fusion.
        with torch.random.fork_rng(devices=[]):
            self.execution_project = nn.Linear(m['qwen_width'], m['width'], bias=False)
            nn.init.zeros_(self.execution_project.weight)

    def read_values(self, state, memory, words, groups, word_groups):
        """Use refined Q for weights; values project the fitted input words.

        No seed/FFN residual and no output RMSNorm after this read.  The
        bias-free shared value map sees all input channels for every head;
        values bypass memory LayerNorm and its learned affine offset.
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
        execution = native if routing_only else native + self.execution_project(state - anchor)
        return dict(routing=routing, execution=execution), groups

    def content_loss(self, c, evidence):
        routing = c['routing']
        target, _, _ = resample_words(evidence['native_words'].detach().float(),
                                      evidence['word_groups'], routing.shape[1])
        return ((routing.float() - target).square().mean()
                / self.anchor_target_scale.square().clamp_min(1e-8))


class Fusion(nn.Module):
    def __init__(self, attention):
        super().__init__()
        width = attention.dim
        self.key = nn.Linear(width, width, bias=False, device=attention.k.weight.device)
        self.value = nn.Linear(width, width, bias=False, device=attention.v.weight.device)
        with torch.no_grad():
            self.key.weight.copy_(attention.k.weight)
            self.value.weight.copy_(attention.v.weight)
        self.key_norm = copy.deepcopy(attention.norm_k).float().requires_grad_(True)
        self.output = nn.Linear(width, width, bias=False, device=attention.q.weight.device)
        self.state_value = nn.Linear(width, width, bias=False, device=attention.q.weight.device)
        self.time = nn.Linear(256, width, bias=False, device=attention.q.weight.device)
        for layer in (self.output, self.state_value, self.time):
            nn.init.zeros_(layer.weight)
        self.reference_key = nn.Parameter(
            torch.zeros(1, 1, width, device=attention.q.weight.device))
        self.memory_mode = 'joint'

    def forward(self, x, context, c, times, geometry, attention, groups=None,
                target=None, valid=None, *, context_length=None,
                direct_physical_read=False, state_value_enabled=True,
                known_frames=None):
        batch, length, width = x.shape
        spatial = geometry[1] * geometry[2]
        known = spatial if known_frames is None else int(known_frames)
        if known not in (0, spatial):
            raise ValueError('known_frames must be 0 (T2V) or one spatial frame (I2V)')
        heads, dim = attention.num_heads, attention.head_dim
        split = lambda value: value.reshape(batch, -1, heads, dim)
        raw = attention.q(x)
        qt = split(attention.norm_q(raw))
        kt = split(attention.norm_k(attention.k(context)))
        vt = split(attention.v(context))
        kernel = attention.forward.__func__.__globals__.get('flash_attention')
        if kernel is None:
            text = F.scaled_dot_product_attention(
                qt.transpose(1, 2), kt.transpose(1, 2).to(qt),
                vt.transpose(1, 2).to(qt)).transpose(1, 2)
        else:
            text = kernel(qt, kt, vt, k_lens=None)
        text = attention.o(text.flatten(2))
        # I2V drops times[0] because the first frame is known.  T2V has no
        # known frame, so every one of the 21 time points is consumed.
        time = time_features(times)
        if known:
            time = time[:, 1:]
        time = self.time(time).repeat_interleave(spatial, 1)
        q = split(attention.norm_q(raw[:, known:] + time)).transpose(1, 2)
        k = split(self.key_norm(self.key(c['routing']))).transpose(1, 2).to(q)
        v = split(self.value(c['execution'])).transpose(1, 2).to(q)
        reference = split(self.reference_key.expand(batch, -1, -1)).transpose(1, 2).to(q)
        if context_length is None or not 1 <= context_length <= context.shape[1]:
            raise ValueError('Use the actual cached T5 length excluding added padding')
        request_keys = kt[:, :context_length].transpose(1, 2).to(q)
        if groups is None:
            raise ValueError('The joint physical read requires relation groups')
        prior = group_log_prior(groups).to(q)
        mode = 'direct' if direct_physical_read else self.memory_mode
        z = read_physical(q, k, v, request_keys, reference, prior, mode,
                          relation_count=int(groups.max()) + 1).transpose(1, 2).flatten(2)
        state = F.layer_norm(x[:, known:].float(), (width,))
        physical = z + self.state_value(state) * z if state_value_enabled else z
        delta = self.output(physical)
        return text + torch.cat((torch.zeros_like(text[:, :known]), delta), 1), \
            x.new_zeros((), dtype=torch.float32)


class PhysicalModel(OriginalPhysicalModel):
    def __init__(self, cfg, device='cuda', wan=None, interface=None):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.task = cfg['task']
        if self.task != 't2v':
            raise ValueError('R15Bt is the T2V entry only')
        self.wan = load_wan(cfg, device) if wan is None else wan.requires_grad_(False)
        self.enhancement = Enhancement(cfg, interface).to(device)
        self.fusions = nn.ModuleDict({
            str(i): Fusion(block.cross_attn) for i, block in enumerate(self.wan.blocks)
            if (i + 1) % cfg['model']['condition_every'] == 0}).to(device)
        devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg['train']['seed'] + 731)
            if devices:
                torch.cuda.manual_seed_all(cfg['train']['seed'] + 731)
            install_execution_lora(self.wan, cfg['model']['lora_rank'])
        self.learned_names = tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad)
        self.checkpointing = cfg['train']['gradient_checkpointing']

    def constant_state(self):
        return {name: value.detach().cpu() for name, value in self.named_buffers()
                if name.startswith('enhancement.')}

    def velocity(self, noisy, first, text, sigma, c, times, groups=None,
                 target=None, valid=None, *, physical_enabled=True,
                 direct_physical_read=False, state_value_enabled=True):
        if first is not None:
            raise ValueError('R15Bt is T2V: no first frame or first latent is allowed')
        wan = self.wan
        if noisy.shape[0] != 1:
            raise ValueError('One video per allocated GPU; use gradient accumulation')
        if wan.freqs is not None and wan.freqs.device != noisy.device:
            wan.freqs = wan.freqs.to(noisy.device)
        x = wan.patch_embedding(noisy)
        geometry = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)
        length = x.shape[1]
        known = 0
        # T2V: every latent frame is unknown and receives sigma*1000.
        position = sigma.expand(length) * 1000
        with torch.autocast(noisy.device.type, enabled=False):
            e = wan.time_embedding(
                sinusoidal_embedding_1d(wan.freq_dim, position).float()[None])
            modulation = wan.time_projection(e).unflatten(-1, (6, wan.dim))
        context = native_text_context(wan, text)
        context_length = min(text.shape[1], wan.text_len)
        grid = torch.tensor([geometry], dtype=torch.long)
        lengths = torch.tensor([length], dtype=torch.long)
        aux = x.new_zeros((), dtype=torch.float32)
        patch_valid = None
        if valid is not None:
            patch_valid = F.avg_pool2d(valid[None, None].float(), 2).flatten()[None]
            patch_valid = patch_valid.repeat(1, geometry[0])
        for index, block in enumerate(wan.blocks):
            def run(hidden, condition, layer=block, site=str(index)):
                if site not in self.fusions or not physical_enabled:
                    return layer(hidden, modulation, lengths, grid, wan.freqs,
                                 context, None), hidden.new_zeros((), dtype=torch.float32)
                parts = (layer.modulation.unsqueeze(0).float() + modulation).chunk(6, dim=2)
                y = layer.self_attn(
                    layer.norm1(hidden).float() * (1 + parts[1].squeeze(2))
                    + parts[0].squeeze(2), lengths, grid, wan.freqs)
                hidden = hidden.float() + y.float() * parts[2].squeeze(2)
                update, loss = self.fusions[site](
                    layer.norm3(hidden), context, condition, times, geometry,
                    layer.cross_attn, groups, target, patch_valid,
                    context_length=context_length, known_frames=known,
                    direct_physical_read=direct_physical_read,
                    state_value_enabled=state_value_enabled)
                hidden = hidden + update
                y = layer.ffn(
                    layer.norm2(hidden).float() * (1 + parts[4].squeeze(2))
                    + parts[3].squeeze(2))
                return hidden + y.float() * parts[5].squeeze(2), loss
            if self.checkpointing and torch.is_grad_enabled():
                x, loss = checkpoint(run, x, c, use_reentrant=False)
            else:
                x, loss = run(x, c)
            aux = aux + loss
        velocity = wan.unpatchify(wan.head(x, e), grid)[0][None].float()
        return velocity, aux / max(1, len(self.fusions))
