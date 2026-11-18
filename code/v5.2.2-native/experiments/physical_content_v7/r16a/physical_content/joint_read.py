"""Native request attention supplies a prior for a direct physical C read.

Physical queries select original C values. No extra learned scalar or loss.
"""
import math
import torch


def read_physical(q, k, v, request_keys, reference, prior, mode='joint', *,
                  request_query, relation_count):
    if mode not in {'joint', 'direct', 'request_prior', 'legacy_joint'}:
        raise ValueError(mode)
    if mode == 'legacy_joint':
        from .legacy_joint_read import read_physical as legacy
        return legacy(q, k, v, request_keys, reference, prior, 'joint', relation_count=relation_count)
    scale = math.sqrt(q.shape[-1])
    if mode == 'direct':
        content_prior = prior[:-1]
    else:
        # Native video-to-request coordinates; actual tokens including EOS.
        request_attention = (request_query.float() @ request_keys.float().transpose(-1, -2) / scale).softmax(-1)
        request_to_content = (request_keys.float() @ k.float().transpose(-1, -2) / scale + prior[:-1]).softmax(-1)
        content_probability = request_attention @ request_to_content
        # Preserve original relation:null prior mass R:1. Uninformative request
        # association reduces exactly to the original direct-C group prior.
        content_prior = content_probability.clamp_min(torch.finfo(torch.float32).tiny).log() + math.log(relation_count)
    direct_logits = q.float() @ k.float().transpose(-1, -2) / scale
    content_logits = content_prior if mode == 'request_prior' else direct_logits + content_prior
    reference_logit = q.float() @ reference.float().transpose(-1, -2) / scale + prior[-1:]
    weights = torch.cat((content_logits, reference_logit), -1).softmax(-1)
    values = torch.cat((v, torch.zeros_like(reference)), -2)
    return weights.to(values) @ values
