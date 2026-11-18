"""B400 joint direct/request C values, with separate physical/native queries."""
import math
import torch


def read_physical(q,k,v,request_keys,reference,prior,mode='joint',*,request_query,relation_count):
    if mode not in {'joint','direct','request','absolute'}:raise ValueError(mode)
    scale=math.sqrt(q.shape[-1])
    direct=q.float()@k.float().transpose(-1,-2)/scale+prior[:-1]
    null=q.float()@reference.float().transpose(-1,-2)/scale+prior[-1:]
    if mode!='direct':
        binding=(request_keys.float()@k.float().transpose(-1,-2)/scale+prior[:-1]).softmax(-1)
        request_values=binding.to(v)@v
        request=request_query.float()@request_keys.float().transpose(-1,-2)/scale
        request=request+math.log(relation_count)-math.log(request_keys.shape[-2])
    if mode=='direct':logits,values=direct,v
    elif mode=='request':logits,values=request,request_values
    else:
        logits=torch.cat((direct,request),-1)-math.log(2.)
        values=torch.cat((v,request_values),-2)
    logits=torch.cat((logits,null),-1)
    values=torch.cat((values,torch.zeros_like(reference)),-2)
    return logits.softmax(-1).to(values)@values

from torch.nn import functional as F

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
                      request_query=qt[:, known:].transpose(1,2).to(q), relation_count=int(groups.max()) + 1).transpose(1, 2).flatten(2)
    state = F.layer_norm(x[:, known:].float(), (width,))
    interaction=self.state_value(state) if state_value_enabled else torch.zeros_like(z)
    physical=(z-z.mean(1,True))+(interaction*z-interaction.mean(1,True)*z.mean(1,True))
    delta = self.output(physical)
    return text + torch.cat((torch.zeros_like(text[:, :known]), delta), 1), x.new_zeros((), dtype=torch.float32)
