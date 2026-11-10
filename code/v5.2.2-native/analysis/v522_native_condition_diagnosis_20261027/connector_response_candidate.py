"""Isolated candidate loss for the existing full-width C connector.

NOT wired into V5.2.2-native training. NOT a trained checkpoint or quality gain.
Use only inside the existing frozen-Wan interface phase of a new revision.
The target is the existing Wan reader's conditional response, not a physics
label, evaluator reward, ranking objective, or external teacher model.
"""
import torch
from torch.nn import functional as F


def reader_response(attention, query, context, mask=None):
    """Same native K/V and value-bias convention as NativeAttentionInterface."""
    batch, _, heads, dim = query.shape
    if heads != attention.num_heads or heads*dim != attention.dim:
        raise ValueError('Query must retain the actual native attention width')
    if context.ndim != 3 or context.shape[0] != batch or context.shape[-1] != attention.dim:
        raise ValueError('C and native text context must have native Wan width')
    k = attention.norm_k(attention.k(context)).reshape(batch, -1, heads, dim)
    value = attention.v(context)
    base = getattr(attention.v, 'base', attention.v)
    if base.bias is not None:
        value = value-base.bias.to(value)
    value = value.reshape(batch, -1, heads, dim)
    if mask is not None:
        if mask.shape != context.shape[:2] or mask.dtype != torch.bool or not bool(mask.any(-1).all()):
            raise ValueError('Each context requires at least one valid token')
        mask = mask[:, None, None, :]
    return F.scaled_dot_product_attention(query.transpose(1, 2), k.transpose(1, 2).to(query),
        value.transpose(1, 2).to(query), attn_mask=mask).transpose(1, 2).flatten(2)


def connector_response_loss(attention, native_query, positive_C, alternative_C,
                            positive_text_context, alternative_text_context,
                            *, positive_mask=None, alternative_mask=None,
                            query_samples=64, energy_floor=1e-4):
    """Regress conditional response differences at fixed native Wan queries.

    The caller supplies FUTURE-frame query positions and two captions with the
    SAME first image. Both text contexts are after native_text_context, including
    native padding. No alternate future video is used. C retains its full token
    count and width; only diagnostic query positions are sampled for this loss.
    """
    if query_samples < 1 or energy_floor <= 0:
        raise ValueError('Positive query budget and energy floor required')
    if any(p.requires_grad for p in attention.parameters()):
        raise ValueError('Candidate is restricted to the frozen-Wan interface phase')
    query = native_query.detach()
    if query.shape[1] > query_samples:
        positions = torch.linspace(0, query.shape[1]-1, query_samples,
                                   device=query.device).long()
        query = query[:, positions]
    with torch.no_grad():
        target = (reader_response(attention, query, positive_text_context.detach())-
                  reader_response(attention, query, alternative_text_context.detach())).float()
    predicted = (reader_response(attention, query, positive_C, positive_mask)-
                 reader_response(attention, query, alternative_C, alternative_mask)).float()
    energy = target.square().mean().clamp_min(energy_floor).detach()
    loss = (predicted-target).square().mean()/energy
    metrics = dict(connector_response_mse=(predicted-target).square().mean().detach(),
                   connector_reference_energy=energy,
                   connector_response_rms=predicted.square().mean().sqrt().detach(),
                   connector_reference_rms=target.square().mean().sqrt().detach())
    return loss, metrics
