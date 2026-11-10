"""One physical read over direct C and request-associated C; native text is unchanged.

This is an inference-only R15B400 mechanism probe, not a trained R15I model.
The two physical memories share one null and the original total relation prior.
"""
import math

import torch
from torch.nn import functional as F


def read_physical(q, k, v, request_keys, reference, prior, mode="joint", *, relation_count=None):
    scale = math.sqrt(q.shape[-1])
    if mode not in {"direct", "bound", "joint"}:
        raise ValueError(mode)
    if mode != "direct":
        binding = ((request_keys.float() @ k.float().transpose(-1, -2)) / scale
                   + prior[:-1]).softmax(-1)
        bound_values = binding.to(v) @ v
        # exp(prior[:-1]).sum() equals the number of relations, independently
        # of the number of slots assigned to each relation.
        log_relations = (math.log(relation_count) if relation_count is not None
                         else float(prior[:-1].float().logsumexp(-1)))
        bound_prior = prior.new_full((request_keys.shape[-2],),
                                     log_relations - math.log(request_keys.shape[-2]))
    if mode == "direct":
        keys, values, physical_prior = k, v, prior[:-1]
    elif mode == "bound":
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


def fusion_forward(self, x, context, c, times, geometry, attention, groups=None,
                   target=None, valid=None, *, context_length=None,
                   direct_physical_read=False, state_value_enabled=True):
    # Imported from the immutable R15B module selected by the probe launcher.
    from physical_content.model import time_features, group_log_prior

    batch, _, width = x.shape
    known = geometry[1] * geometry[2]
    heads, dim = attention.num_heads, attention.head_dim
    split = lambda value: value.reshape(batch, -1, heads, dim)
    raw = attention.q(x)
    qt = split(attention.norm_q(raw))
    kt = split(attention.norm_k(attention.k(context)))
    vt = split(attention.v(context))
    kernel = attention.forward.__func__.__globals__.get("flash_attention")
    if kernel is None:
        text = F.scaled_dot_product_attention(
            qt.transpose(1, 2), kt.transpose(1, 2).to(qt), vt.transpose(1, 2).to(qt)
        ).transpose(1, 2)
    else:
        text = kernel(qt, kt, vt, k_lens=None)
    text = attention.o(text.flatten(2))
    temporal = self.time(time_features(times))[:, 1:].repeat_interleave(known, 1)
    q = split(attention.norm_q(raw[:, known:] + temporal)).transpose(1, 2)
    k = split(self.key_norm(self.key(c["routing"]))).transpose(1, 2).to(q)
    v = split(self.value(c["execution"])).transpose(1, 2).to(q)
    reference = split(self.reference_key.expand(batch, -1, -1)).transpose(1, 2).to(q)
    if context_length is None or not 1 <= context_length <= context.shape[1]:
        raise ValueError("Require the cached T5 length, excluding added padding")
    request_keys = kt[:, :context_length].transpose(1, 2).to(q)
    prior = group_log_prior(groups).to(q)
    mode = "direct" if direct_physical_read else getattr(self, "memory_mode", "joint")
    z = read_physical(q, k, v, request_keys, reference, prior, mode,
                      relation_count=int(groups.max()) + 1).transpose(1, 2).flatten(2)
    state = F.layer_norm(x[:, known:].float(), (width,))
    physical = z + self.state_value(state) * z if state_value_enabled else z
    delta = self.output(physical)
    return text + torch.cat((torch.zeros_like(text[:, :known]), delta), 1), x.new_zeros((), dtype=torch.float32)
