"""One physical read over direct C and request-associated C; native text is unchanged.

Evaluation intervention only: competitive direct/request read with the reference
removed consistently with this candidate's architecture. No zero-valued slot.
"""
import math

import torch
from torch.nn import functional as F


def read_physical(q, k, v, request_keys, prior, mode="joint", *, relation_count=None):
    scale = math.sqrt(q.shape[-1])
    if mode not in {"direct", "bound", "joint"}:
        raise ValueError(mode)
    if mode != "direct":
        binding = ((request_keys.float() @ k.float().transpose(-1, -2)) / scale
                   + prior).softmax(-1)
        bound_values = binding.to(v) @ v
        # exp(prior).sum() equals the number of relations, independently
        # of the number of slots assigned to each relation.
        log_relations = (math.log(relation_count) if relation_count is not None
                         else float(prior.float().logsumexp(-1)))
        bound_prior = prior.new_full((request_keys.shape[-2],),
                                     log_relations - math.log(request_keys.shape[-2]))
    if mode == "direct":
        keys, values, physical_prior = k, v, prior
    elif mode == "bound":
        keys, values, physical_prior = request_keys, bound_values, bound_prior
    else:
        keys = torch.cat((k, request_keys), -2)
        values = torch.cat((v, bound_values), -2)
        physical_prior = torch.cat((prior, bound_prior), -1).float() - math.log(2.)
    logits = q.float() @ keys.float().transpose(-1, -2) / scale
    logits = logits + physical_prior
    return logits.softmax(-1).to(values) @ values

