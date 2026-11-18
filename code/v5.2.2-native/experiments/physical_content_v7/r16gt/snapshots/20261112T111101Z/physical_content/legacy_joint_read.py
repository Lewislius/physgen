"""One physical read over direct C and request-associated C; native text is unchanged.

The same operator passed the declared R15B400 fixed16 inference comparison.
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

