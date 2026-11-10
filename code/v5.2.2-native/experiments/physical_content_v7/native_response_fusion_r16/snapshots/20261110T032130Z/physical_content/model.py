"""Multi-read state-gated global execution plus a local video-state operator.

The native text Q/K/V/O stay frozen and unchanged. R10A/R15B routing and
execution C are read through three independently normalized operators:

* direct C read (R10A): the native video query reads C execution values;
* request-bound C read (R15B): native request keys bind C values to the
  literal request, then the same video query reads the bound values;
* joint direct+request read (B400): both physical memories share one null.

The local operator is the proven R15C construction: its query is shifted by
a zero-init full-width direct C condition, while K and V remain the native
video-state projections and a 5-frame x 3x3 local space-time attention carries
the dynamics. V is never overwritten by C, so appearance is not replaced.

Each selected global read is written through its own zero-init full-width
output and its own zero-init full-width state gate, exactly the multiplicative
state interaction used by R10A/R15B. The global execution write and the local
operator are additive, both zero-init, and the first frame remains exact.

Every version keeps all three clean global reads. R16G conditions the local
query on the direct read alone. R16H also adds a zero-init full-width joint
local-query condition. R16I additionally adds a zero-init full-width request
local-query condition and the R16E zero-init full-width request-bound V
correction. This is an ablation of local physical conditioning on top of the
maximal clean global execution set. No map is high-dimensional-to-low-
dimensional-to-high; all new maps are 3072->3072. No raw Qwen word residual
is written to appearance. Training and normal inference use the same forward.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from physgen_native.context_alignment import native_text_context
from physgen_native.generator import load_wan, sinusoidal_embedding_1d
from physgen_native.message_conditioning import MessageAttention
from physical_scaling.model import (
    PhysicalModel as OriginalPhysicalModel,
    install_execution_lora,
    time_features,
)
from physical_scaling.supervision import group_log_prior

from .evidence import resample_words
from .legacy_joint_read import read_physical as legacy_joint_read
from .local_attention import local_attention


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
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width, bias=False), nn.GELU(),
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
        read = self.read_heads(self.query_norm(state), self.memory_norm(memory), memory_mask,
                               self.memory_norm(words), word_groups, groups)
        state = state + self.cross.o(read.transpose(1, 2).flatten(2))
        normalized = self.exchange_norm(state)
        summaries = torch.stack(
            [normalized[:, groups == r].mean(1) for r in range(int(groups.max()) + 1)], 1)
        exchange_memory = torch.cat((normalized, summaries), 1)
        local = groups[:, None] == groups[None, :]
        allowed = torch.cat((local, torch.ones(len(groups), len(summaries[0]),
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
            interface = torch.load(cfg['paths']['interface'], map_location='cpu', weights_only=True)
        for name in ['mean', 'scale', 'weight', 'bias', 'target_scale']:
            self.register_buffer('anchor_' + name, interface[name].float())
        self.seed = nn.Parameter(torch.randn(1, m['q_tokens'], m['qwen_width']) * .02)
        self.block = ContentBlock(m['qwen_width'], m['q_heads'], m['grounded_read_heads'])
        self.project = nn.Linear(m['qwen_width'], m['width'], bias=False)
        nn.init.zeros_(self.project.weight)
        self.rounds = m['rounds']
        self.checkpointing = cfg['train']['gradient_checkpointing']
        self.shared_offset = nn.Parameter(torch.zeros(m['width']))
        with torch.random.fork_rng(devices=[]):
            self.execution_project = nn.Linear(m['qwen_width'], m['width'], bias=False)
            nn.init.zeros_(self.execution_project.weight)
        self.response_mode = m.get('response_mode', '')
        self.request_project = None
        if self.response_mode.startswith('ecr_'):
            # Literal request evidence is an input field, not a persistent
            # memory. The full-width map starts from the fitted anchor and
            # receives real-video generation gradients. Fork the RNG so the
            # new module does not perturb the historical fusion initializer
            # draws used by the clean comparison.
            with torch.random.fork_rng(devices=[]):
                self.request_project = nn.Linear(m['qwen_width'], m['width'], bias=True)
                with torch.no_grad():
                    self.request_project.weight.copy_(self.anchor_weight)
                    self.request_project.bias.copy_(self.anchor_bias)

    def read_values(self, state, memory, words, groups, word_groups):
        """Use refined Q for weights; values project the fitted input words."""
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
            q[:, :local], word_keys[:, :local], word_values[:, :local], attn_mask=allowed[None, None])
        contextual = F.scaled_dot_product_attention(
            q[:, local:], keys[:, local:], values[:, local:])
        return torch.cat((grounded, contextual), 1).transpose(1, 2).flatten(2)

    def forward(self, evidence, *, anchor_only=False, routing_only=False):
        words = evidence['words'].detach().float()
        words = (words - self.anchor_mean) / self.anchor_scale
        anchor, groups, _ = resample_words(words, evidence['word_groups'], self.seed.shape[1])
        native = F.linear(anchor, self.anchor_weight, self.anchor_bias)
        if anchor_only:
            literal = None
            if self.request_project is not None:
                raw = evidence['request_words'].detach().float()
                raw = (raw - self.anchor_mean) / self.anchor_scale
                raw = raw[None] if raw.ndim == 2 else raw
                literal = (self.request_project(raw) if raw.shape[1] else
                           raw.new_zeros((raw.shape[0], 0, self.request_project.out_features)))
            return dict(routing=native, execution=native, literal_request=literal), groups
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
        literal = None
        if self.request_project is not None:
            literal = (self.request_project(request) if request.shape[1] else
                       request.new_zeros((request.shape[0], 0, self.request_project.out_features)))
        return dict(routing=routing, execution=execution, literal_request=literal), groups

    def content_loss(self, c, evidence):
        routing = c['routing']
        target, _, _ = resample_words(evidence['native_words'].detach().float(),
                                      evidence['word_groups'], routing.shape[1])
        return (routing.float() - target).square().mean() / self.anchor_target_scale.square().clamp_min(1e-8)


class Fusion(nn.Module):
    """R15C local state operator plus state-gated global execution writes."""

    def __init__(self, attention, video_attention, *, shift=0, window=(5, 3, 3),
                 response_mode='direct_plus_request'):
        super().__init__()
        width = attention.dim
        self.response_mode = response_mode
        self.evidence_mode = response_mode.startswith('ecr_')
        self.specialized_heads = response_mode in ('ecr_specialized_heads', 'ecr_iterative')
        self.iterative_query = response_mode == 'ecr_iterative'
        self.key = nn.Linear(width, width, bias=False, device=attention.k.weight.device)
        self.value = nn.Linear(width, width, bias=False, device=attention.v.weight.device)
        with torch.no_grad():
            self.key.weight.copy_(attention.k.weight)
            self.value.weight.copy_(attention.v.weight)
        self.key_norm = copy.deepcopy(attention.norm_k).float().requires_grad_(True)
        # R15C local operator: the only local Q condition is the direct C read.
        self.output = nn.Linear(width, width, bias=False, device=attention.q.weight.device)
        self.condition_query = nn.Linear(width, width, bias=False, device=attention.q.weight.device)
        self.request_condition_query = nn.Linear(width, width, bias=False,
                                                 device=attention.q.weight.device)
        self.joint_condition_query = nn.Linear(width, width, bias=False,
                                               device=attention.q.weight.device)
        self.time = nn.Linear(256, width, bias=False, device=attention.q.weight.device)
        for layer in (self.output, self.condition_query, self.request_condition_query,
                      self.joint_condition_query, self.time):
            nn.init.zeros_(layer.weight)
        self.reference_key = nn.Parameter(torch.zeros(1, 1, width, device=attention.q.weight.device))
        # Each global read has its own zero-init full-width write and its own
        # zero-init full-width state gate. No low-dimensional bottleneck.
        for name in ('direct', 'request', 'joint'):
            setattr(self, name + '_output', nn.Linear(
                width, width, bias=False, device=attention.q.weight.device))
            setattr(self, name + '_state_value', nn.Linear(
                width, width, bias=False, device=attention.q.weight.device))
            nn.init.zeros_(getattr(self, name + '_output').weight)
            nn.init.zeros_(getattr(self, name + '_state_value').weight)
        # Gated single-write fusion: full-width state-conditioned gates select
        # and combine physical reads before one output. No per-read additive
        # outputs. The final local/global gates balance the isolated local
        # state operator against the fused global execution write.
        for name in ('gate_direct', 'gate_request', 'gate_joint'):
            layer = nn.Linear(width, width, bias=True, device=attention.q.weight.device)
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, 1.)
            setattr(self, name, layer)
        self.gate_single = nn.Linear(width, width, bias=True, device=attention.q.weight.device)
        nn.init.zeros_(self.gate_single.weight)
        nn.init.constant_(self.gate_single.bias, 1.)
        self.local_value_gate = nn.Linear(width, width, bias=True, device=attention.q.weight.device)
        nn.init.zeros_(self.local_value_gate.weight)
        nn.init.constant_(self.local_value_gate.bias, 1.)
        self.fused_output = nn.Linear(width, width, bias=False, device=attention.q.weight.device)
        self.fused_state_value = nn.Linear(width, width, bias=False, device=attention.q.weight.device)
        self.final_local_gate = nn.Linear(width, width, bias=True, device=attention.q.weight.device)
        self.final_global_gate = nn.Linear(width, width, bias=True, device=attention.q.weight.device)
        for layer in (self.fused_output, self.fused_state_value):
            nn.init.zeros_(layer.weight)
        for layer in (self.final_local_gate, self.final_global_gate):
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, 1.)
        if self.evidence_mode:
            # A single evidence-complete physical read: one output and one
            # state gate. Literal request words are evidence, never a global
            # residual. The local request condition only shifts local Q.
            self.ecr_output = nn.Linear(width, width, bias=False,
                                        device=attention.q.weight.device)
            self.ecr_state_value = nn.Linear(width, width, bias=False,
                                             device=attention.q.weight.device)
            self.literal_condition_query = nn.Linear(width, width, bias=False,
                                                     device=attention.q.weight.device)
            for layer in (self.ecr_output, self.ecr_state_value,
                          self.literal_condition_query):
                nn.init.zeros_(layer.weight)
            if self.iterative_query:
                self.query_refine = nn.Linear(width, width, bias=False,
                                              device=attention.q.weight.device)
                nn.init.zeros_(self.query_refine.weight)
        for name in ('q', 'k', 'v', 'norm_q', 'norm_k'):
            setattr(self, 'video_' + name,
                    copy.deepcopy(getattr(video_attention, name)).float().requires_grad_(True))
        self.shift = shift
        self.window = tuple(window)

    @property
    def global_reads(self):
        return {
            'gated_direct_request': {'direct', 'request'},
            'gated_triple': {'direct', 'request', 'joint'},
            'gated_union': {'direct', 'request', 'joint', 'single'},
            'gated_joint_only': {'joint'},
        }.get(self.response_mode, set())

    @property
    def use_request_read(self):
        return self.response_mode != 'gated_joint_only'

    @property
    def use_single_read(self):
        return self.response_mode == 'gated_union'

    @property
    def use_joint_read(self):
        return self.response_mode in ('triple_global_pure_local',
                                      'triple_global_joint_local',
                                      'triple_global_full_local', 'gated_triple',
                                      'gated_union', 'gated_joint_only')

    def _direct_read(self, q, k, v, reference, prior):
        keys = torch.cat((k, reference), 2)
        values = torch.cat((v, torch.zeros_like(reference)), 2)
        weights = ((q.float() @ keys.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
                   + prior).softmax(-1)
        return (weights.to(values) @ values).transpose(1, 2).flatten(2)

    def _request_read(self, q, k, v, request_keys, reference, prior, relation_count):
        binding = ((request_keys.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
                   + prior[:-1]).softmax(-1)
        request_values = binding.to(v) @ v
        request_prior = prior.new_full(
            (request_keys.shape[-2],),
            math.log(relation_count) - math.log(request_keys.shape[-2]))
        keys = torch.cat((request_keys, reference), 2)
        values = torch.cat((request_values, torch.zeros_like(reference)), 2)
        logits = ((q.float() @ keys.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
                  + torch.cat((request_prior, prior[-1:]), -1))
        return (logits.softmax(-1).to(values) @ values).transpose(1, 2).flatten(2)

    def _evidence_read(self, q, k, v, request_keys, literal_keys, literal_values,
                       reference, prior, attention, relation_count, *,
                       direct_enabled=True, bound_enabled=True, literal_enabled=True,
                       iterative_enabled=True):
        """One physical softmax over evidence banks, with static head masks.

        Banks are: direct C, request-bound C, literal request words, and one
        shared null. Ordinary ECR lets every head see every bank. The
        specialized variant gives half the heads to physical C evidence and
        half to literal request evidence. No bank is summed after the read.
        """
        pieces = []
        keys = []
        values = []
        bank_ids = []
        if direct_enabled:
            keys.append(k)
            values.append(v)
            pieces.append(prior[:-1])
            bank_ids.extend([0] * k.shape[2])
        if bound_enabled and request_keys is not None:
            binding = ((request_keys.float() @ k.float().transpose(-1, -2))
                       / math.sqrt(q.shape[-1]) + prior[:-1]).softmax(-1)
            bound_values = binding.to(v) @ v
            bound_prior = prior.new_full(
                (request_keys.shape[-2],),
                math.log(relation_count) - math.log(request_keys.shape[-2]))
            keys.append(request_keys)
            values.append(bound_values)
            pieces.append(bound_prior)
            bank_ids.extend([1] * request_keys.shape[-2])
        if literal_enabled and literal_keys is not None and literal_keys.shape[2] > 0:
            count = literal_keys.shape[2]
            literal_prior = prior.new_full(
                (count,), math.log(relation_count) - math.log(count))
            keys.append(literal_keys)
            values.append(literal_values)
            pieces.append(literal_prior)
            bank_ids.extend([2] * count)
        keys.append(reference)
        values.append(torch.zeros_like(reference))
        pieces.append(prior[-1:])
        bank_ids.extend([3])
        all_keys = torch.cat(keys, 2)
        all_values = torch.cat(values, 2)
        all_prior = torch.cat(pieces, -1).to(q)
        logits = (q.float() @ all_keys.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        logits = logits + all_prior
        ids = torch.tensor(bank_ids, device=q.device, dtype=torch.long)
        heads = q.shape[1]
        if self.specialized_heads:
            physical = max(1, heads // 2)
            physical_mask = ((ids == 0) | (ids == 1) | (ids == 3))[None, :]
            request_mask = ((ids == 2) | (ids == 3))[None, :]
            mask = torch.cat((
                physical_mask.expand(physical, -1),
                request_mask.expand(heads - physical, -1)), 0)
        else:
            mask = torch.ones((heads, ids.numel()), dtype=torch.bool, device=q.device)
        logits = logits.masked_fill(~mask[None, :, None, :], -torch.inf)
        read = (logits.softmax(-1).to(all_values) @ all_values).transpose(1, 2).flatten(2)
        if self.iterative_query and iterative_enabled:
            q2 = q + self.query_refine(read).unflatten(-1, (heads, -1)).transpose(1, 2)
            return self._evidence_read(q2, k, v, request_keys, literal_keys, literal_values,
                                       reference, prior, attention, relation_count,
                                       direct_enabled=direct_enabled, bound_enabled=bound_enabled,
                                       literal_enabled=literal_enabled, iterative_enabled=False)
        return read

    def forward(self, x, video_state, c, times, geometry, attention, groups, grid, freqs,
                rope_apply, context, context_length, value_correction, *,
                condition_enabled=True, temporal_enabled=True, local_enabled=True,
                global_enabled=True, global_direct_enabled=True,
                global_request_enabled=True, global_literal_enabled=True,
                global_joint_enabled=True, global_single_enabled=True,
                iterative_enabled=True,
                local_request_condition_enabled=True,
                local_joint_condition_enabled=True,
                value_correction_enabled=True,
                local_value_correction_enabled=True):
        batch, length, width = x.shape
        known = geometry[1] * geometry[2]
        heads, dim = attention.num_heads, attention.head_dim
        split = lambda value: value.reshape(batch, -1, heads, dim)

        conditioned = torch.zeros_like(video_state)
        direct = request = joint = single = None
        ecr_read = None
        literal_keys = literal_values = None
        request_keys = None
        if condition_enabled:
            time = self.time(time_features(times))[:, 1:].repeat_interleave(known, 1)
            raw = attention.q(x[:, known:])
            q_global = split(attention.norm_q(raw + time)).transpose(1, 2)
            k = split(self.key_norm(self.key(c['routing']))).transpose(1, 2).to(q_global)
            v = split(self.value(c['execution'])).transpose(1, 2).to(q_global)
            reference = split(self.reference_key.expand(batch, -1, -1)).transpose(1, 2).to(q_global)
            prior = group_log_prior(groups).to(q_global)
            relations = int(groups.max()) + 1
            direct = self._direct_read(q_global, k, v, reference, prior)
            condition = self.condition_query(direct)
            if not 1 <= context_length <= context.shape[1]:
                raise ValueError('Use the actual cached T5 length including EOS')
            text_keys = split(attention.norm_k(attention.k(context)))
            request_keys = text_keys[:, :context_length].transpose(1, 2).to(q_global)
            if self.evidence_mode:
                literal = c.get('literal_request')
                if literal is not None and literal.shape[1] > 0:
                    literal_keys = split(attention.norm_k(attention.k(literal))).transpose(1, 2).to(q_global)
                    literal_values = split(attention.v(literal)).transpose(1, 2).to(q_global)
                if global_enabled:
                    ecr_read = self._evidence_read(
                        q_global, k, v, request_keys, literal_keys, literal_values,
                        reference, prior, attention, relations,
                        direct_enabled=global_direct_enabled,
                        bound_enabled=global_request_enabled,
                        literal_enabled=global_literal_enabled,
                        iterative_enabled=iterative_enabled)
                else:
                    ecr_read = None
            else:
                if self.use_request_read:
                    request = self._request_read(q_global, k, v, request_keys, reference, prior, relations)
                    if (local_request_condition_enabled
                            and self.response_mode == 'triple_global_full_local'
                            and not self.response_mode.startswith('gated_')):
                        condition = condition + self.request_condition_query(request)
                if self.use_joint_read:
                    joint = legacy_joint_read(
                        q_global, k, v, request_keys, reference, prior, 'joint',
                        relation_count=relations).transpose(1, 2).flatten(2)
                    if (local_joint_condition_enabled
                            and self.response_mode in ('triple_global_joint_local',
                                                       'triple_global_full_local')
                            and not self.response_mode.startswith('gated_')):
                        condition = condition + self.joint_condition_query(joint)
                if self.use_single_read:
                    k_single = split(self.key_norm(self.key(c['execution']))).transpose(1, 2).to(q_global)
                    v_single = split(self.value(c['execution'])).transpose(1, 2).to(q_global)
                    single = self._direct_read(q_global, k_single, v_single, reference, prior)
            conditioned = torch.cat((torch.zeros_like(video_state[:, :known]), condition), 1)

        global_state = F.layer_norm(x[:, known:].float(), (width,))
        q = split(self.video_norm_q(self.video_q(video_state) + conditioned))
        if (self.evidence_mode and local_request_condition_enabled and condition_enabled
                and literal_keys is not None and literal_keys.shape[2] > 0):
            local_context = F.scaled_dot_product_attention(
                q.transpose(1, 2), literal_keys, literal_values)
            q = q + split(self.literal_condition_query(
                local_context.transpose(1, 2).flatten(2)))
        k = split(self.video_norm_k(self.video_k(video_state)))
        v_base = self.video_v(video_state)
        if (value_correction_enabled and condition_enabled and request is not None
                and self.response_mode == 'triple_global_full_local'):
            v_base = v_base + torch.cat(
                (torch.zeros_like(v_base[:, :known]), value_correction(request)), 1)
        if (value_correction_enabled and local_value_correction_enabled
                and condition_enabled and request is not None
                and self.response_mode == 'gated_union'):
            correction_gate = torch.sigmoid(self.local_value_gate(global_state))
            v_base = v_base + torch.cat(
                (torch.zeros_like(v_base[:, :known]),
                 correction_gate * value_correction(request)), 1)
        v = split(v_base)
        # RoPE indexes the real video grid, never learned C slots.
        q = rope_apply(q, grid, freqs)
        k = rope_apply(k, grid, freqs)
        state = local_attention(q, k, v, geometry, window=self.window, shift=self.shift,
                                temporal_enabled=temporal_enabled).flatten(2)
        delta = self.output(state) if local_enabled else torch.zeros_like(state)

        if self.evidence_mode:
            if condition_enabled and global_enabled and ecr_read is not None:
                fused = ecr_read + self.ecr_state_value(global_state) * ecr_read
                global_delta = self.ecr_output(fused)
            else:
                global_delta = torch.zeros_like(delta[:, known:])
            local_delta = delta[:, known:] if local_enabled else torch.zeros_like(global_delta)
            return torch.cat((torch.zeros_like(delta[:, :known]),
                              local_delta + global_delta), 1)

        if self.response_mode.startswith('gated_'):
            fused = torch.zeros_like(global_state)
            if condition_enabled and global_enabled:
                def read_gate(name, value):
                    return torch.sigmoid(getattr(self, 'gate_' + name)(global_state))
                if 'direct' in self.global_reads and global_direct_enabled and direct is not None:
                    fused = fused + read_gate('direct', direct) * direct
                if ('request' in self.global_reads and global_request_enabled
                        and self.use_request_read and request is not None):
                    fused = fused + read_gate('request', request) * request
                if ('joint' in self.global_reads and global_joint_enabled
                        and self.use_joint_read and joint is not None):
                    fused = fused + read_gate('joint', joint) * joint
                if ('single' in self.global_reads and global_single_enabled
                        and self.use_single_read and single is not None):
                    fused = fused + read_gate('single', single) * single
                fused_physical = fused + self.fused_state_value(global_state) * fused
                global_delta = self.fused_output(fused_physical)
            else:
                global_delta = torch.zeros_like(fused)
            local_gate = torch.sigmoid(self.final_local_gate(global_state))
            global_gate = torch.sigmoid(self.final_global_gate(global_state))
            local_part = torch.zeros_like(global_state)
            global_part = torch.zeros_like(global_state)
            if local_enabled:
                local_part = local_gate * delta[:, known:]
            if global_enabled:
                global_part = global_gate * global_delta
            return (torch.cat((torch.zeros_like(delta[:, :known]), global_part), 1),
                    torch.cat((torch.zeros_like(delta[:, :known]), local_part), 1))
        extra = torch.zeros_like(delta[:, known:])
        if condition_enabled and global_enabled:
            if global_direct_enabled:
                extra = extra + self.direct_output(
                    direct + self.direct_state_value(global_state) * direct)
            if (global_request_enabled and self.use_request_read
                    and request is not None):
                extra = extra + self.request_output(
                    request + self.request_state_value(global_state) * request)
            if (global_joint_enabled and self.use_joint_read
                    and joint is not None):
                extra = extra + self.joint_output(
                    joint + self.joint_state_value(global_state) * joint)
        return torch.cat((torch.zeros_like(delta[:, :known]),
                          delta[:, known:] + extra), 1)


class EmptyEnhancement(nn.Module):
    def forward(self, evidence):
        memory = evidence['memory']
        return memory.new_zeros((memory.shape[0], 0, 3072)), None


class PhysicalModel(OriginalPhysicalModel):
    def __init__(self, cfg, device='cuda', wan=None):
        nn.Module.__init__(self)
        self.cfg = cfg
        import os, time as _time
        _debug = os.getenv('R16_DEBUG_INIT') == '1'
        _t = _time.time()
        if _debug: print('debug_init: loading wan', flush=True)
        self.wan = load_wan(cfg, device) if wan is None else wan.requires_grad_(False)
        if _debug: print(f'debug_init: wan loaded { _time.time()-_t:.1f}s', flush=True)
        if cfg['variant'] == 'native_lora':
            self.enhancement = EmptyEnhancement()
            self.fusions = nn.ModuleDict()
        else:
            self.enhancement = Enhancement(cfg).to(device)
            self.fusions = nn.ModuleDict({
                str(i): Fusion(block.cross_attn, block.self_attn,
                               shift=((i + 1) // cfg['model']['condition_every'] - 1) % 2,
                               window=cfg['model']['physical_window'],
                               response_mode=cfg['model']['response_mode'])
                for i, block in enumerate(self.wan.blocks)
                if (i + 1) % cfg['model']['condition_every'] == 0}).to(device)
            width = cfg['model']['width']
            # Shared full-width zero-init request-bound V correction. It is
            # used only by R16I; no low-dimensional bottleneck.
            self.value_correction = nn.Linear(width, width, bias=False, device=device)
            nn.init.zeros_(self.value_correction.weight)
            if _debug: print(f'debug_init: enhancement+fusions { _time.time()-_t:.1f}s', flush=True)
        devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg['train']['seed'] + 731)
            if devices:
                torch.cuda.manual_seed_all(cfg['train']['seed'] + 731)
            install_execution_lora(self.wan, cfg['model']['lora_rank'])
        self.learned_names = tuple(name for name, p in self.named_parameters() if p.requires_grad)
        self.checkpointing = cfg['train']['gradient_checkpointing']
        if _debug: print(f'debug_init: lora+names { _time.time()-_t:.1f}s params={sum(dict(self.named_parameters())[n].numel() for n in self.learned_names)}', flush=True)

    def constant_state(self):
        return {name: value.detach().cpu() for name, value in self.named_buffers()
                if name.startswith('enhancement.')}

    def optim_groups(self):
        named = dict(self.named_parameters())
        from physical_scaling.schedule import GROUPS
        groups = {name: [] for name in GROUPS}
        for name in self.learned_names:
            if name.startswith('wan.'):
                role = 'wan_lora'
            elif name.startswith('enhancement.'):
                role = 'enhancement'
            elif (name.startswith('fusions.') and len(name.split('.')) == 4
                  and name.split('.')[2] == 'output'
                  and self.cfg['model'].get('local_output_group') == 'fusion_read'):
                role = 'fusion_read'
            elif name.startswith('fusions.') and any(
                    marker in name for marker in
                    ('.output.', '.direct_output.', '.request_output.', '.joint_output.',
                     '.ecr_output.', '.ecr_state_value.', '.fused_output.')):
                role = 'fusion_write'
            elif name.startswith(('fusions.', 'value_correction.')):
                role = 'fusion_read'
            else:
                raise ValueError(f'Unassigned trainable parameter: {name}')
            groups[role].append(named[name])
        return [dict(name=role, params=groups[role], lr=self.cfg['train']['peak_lrs'][role])
                for role in GROUPS]

    def velocity(self, noisy, first, text, sigma, c, times, groups=None, target=None, valid=None, *,
                 physical_enabled=True, condition_enabled=True, local_enabled=True,
                 global_enabled=True, global_direct_enabled=True,
                 global_request_enabled=True, global_literal_enabled=True,
                 global_joint_enabled=True, global_single_enabled=True,
                 iterative_enabled=True,
                 local_request_condition_enabled=True,
                 local_joint_condition_enabled=True,
                 value_correction_enabled=True, local_value_correction_enabled=True,
                 temporal_enabled=True, **legacy):
        """Matched forward; old evaluation flags are explicit no-ops."""
        if legacy:
            unknown = set(legacy) - {'native_physical_query', 'shared_physical_query',
                                     'memory_mode', 'state_value_enabled'}
            if unknown:
                raise ValueError(f'Unsupported execution flag(s): {sorted(unknown)}')
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
        gated = self.cfg['model']['response_mode'].startswith('gated_')
        local_sum = x.new_zeros(x.shape)
        grid = torch.tensor([geometry], dtype=torch.long)
        lengths = torch.tensor([x.shape[1]], dtype=torch.long)
        aux = x.new_zeros((), dtype=torch.float32)
        for index, block in enumerate(wan.blocks):
            def run(hidden, condition, layer=block, site=str(index)):
                if site not in self.fusions or not physical_enabled:
                    result = layer(hidden, modulation, lengths, grid, wan.freqs, context, None)
                    zero = hidden.new_zeros((), dtype=torch.float32)
                    return (result, zero, hidden.new_zeros(hidden.shape)) if gated else (result, zero)
                parts = (layer.modulation.unsqueeze(0).float() + modulation).chunk(6, dim=2)
                y = layer.self_attn(
                    layer.norm1(hidden).float() * (1 + parts[1].squeeze(2)) + parts[0].squeeze(2),
                    lengths, grid, wan.freqs)
                hidden = hidden.float() + y.float() * parts[2].squeeze(2)
                normalized = layer.norm3(hidden)
                text_update = layer.cross_attn(normalized, context, None)
                state = layer.norm1(hidden + text_update).float() * (1 + parts[1].squeeze(2)) + parts[0].squeeze(2)
                rope_apply = layer.self_attn.forward.__func__.__globals__['rope_apply']
                out = self.fusions[site](
                    normalized, state, condition, times, geometry, layer.cross_attn, groups,
                    grid, wan.freqs, rope_apply, context, context_length,
                    self.value_correction,
                    condition_enabled=condition_enabled,
                    local_enabled=local_enabled,
                    global_enabled=global_enabled,
                    global_direct_enabled=global_direct_enabled,
                    global_request_enabled=global_request_enabled,
                    global_literal_enabled=global_literal_enabled,
                    global_joint_enabled=global_joint_enabled,
                    global_single_enabled=global_single_enabled,
                    iterative_enabled=iterative_enabled,
                    local_request_condition_enabled=local_request_condition_enabled,
                    local_joint_condition_enabled=local_joint_condition_enabled,
                    value_correction_enabled=value_correction_enabled,
                    local_value_correction_enabled=local_value_correction_enabled,
                    temporal_enabled=temporal_enabled)
                if gated:
                    delta, local_part = out
                else:
                    delta, local_part = out, hidden.new_zeros(hidden.shape)
                hidden = hidden + text_update + delta
                y = layer.ffn(layer.norm2(hidden).float() * (1 + parts[4].squeeze(2)) + parts[3].squeeze(2))
                result = hidden + y.float() * parts[5].squeeze(2)
                zero = hidden.new_zeros((), dtype=torch.float32)
                return (result, zero, local_part) if gated else (result, zero)
            if self.checkpointing and torch.is_grad_enabled():
                output = checkpoint(run, x, c, use_reentrant=False)
            else:
                output = run(x, c)
            if gated:
                x, loss, local_part = output
                local_sum = local_sum + local_part
            else:
                x, loss = output
            aux = aux + loss
        if gated:
            x = x + local_sum
        prediction = wan.unpatchify(wan.head(x, e), grid)[0][None].float()
        return prediction, aux / max(1, len(self.fusions))
