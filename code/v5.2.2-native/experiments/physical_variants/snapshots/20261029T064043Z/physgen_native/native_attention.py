"""Full-width C attention inside the unchanged native Wan block traversal.

Reuse the native query once, separate text/C softmax normalizers, then use the
native full-width output projection once. The block's own residual and FFN run
after this module. No new q/k/v/o matrices or rank-compressed message pathway.
"""
from dataclasses import dataclass
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class NativeAttentionContext:
    text: torch.Tensor
    condition: torch.Tensor
    interface: object
    geometry: tuple
    strength: float
    index: int
    condition_mask: torch.Tensor | None = None


def _conditioned_attention(module, x, context, context_lens):
    if not isinstance(context, NativeAttentionContext):
        return module._original_context_forward(x, context, context_lens)
    return context.interface.attend(module, x, context, context_lens)


class NativeAttentionInterface(nn.Module):
    def __init__(self, cfg, depth):
        super().__init__()
        width=cfg['model']['width']
        self.width=width;self.heads=cfg['model']['attention_heads']
        self.layers=tuple(i for i in range(depth) if (i+1)%cfg['model']['condition_every']==0)
        self.capture=False
        self.observations=[]

    def condition_query(self, query, packet):
        # The native query from norm3(H_s), without a second projection or a.
        return query

    def attach(self, blocks):
        for index in self.layers:
            attention=blocks[index].cross_attn
            if attention.dim!=self.width or attention.num_heads!=self.heads:
                raise ValueError('C attention must use the actual native Wan width and head count')
            kernel=attention.forward.__func__.__globals__.get('flash_attention')
            if kernel is None:raise TypeError('Expected the native Wan attention kernel')
            # Preserve parameter ownership/names: no nested copy of the backbone.
            attention._original_context_forward=attention.forward
            attention._native_context_kernel=kernel
            attention.forward=MethodType(_conditioned_attention,attention)

    def attend(self, attention, x, packet, context_lens):
        batch,length,width=x.shape;t,h,w=packet.geometry;known=h*w
        if length!=t*h*w:raise ValueError('Native C query geometry differs')
        if packet.strength==0:
            return attention._original_context_forward(x,packet.text,context_lens)
        heads,dim=attention.num_heads,attention.head_dim
        reshape=lambda value:value.reshape(batch,-1,heads,dim)
        # Exactly the pretrained native projections. Q is computed only once.
        q=reshape(attention.norm_q(attention.q(x)))
        kt=reshape(attention.norm_k(attention.k(packet.text)))
        vt=reshape(attention.v(packet.text))
        text_message=attention._native_context_kernel(q,kt,vt,k_lens=context_lens)
        kc=reshape(attention.norm_k(attention.k(packet.condition)))
        vc=attention.v(packet.condition)
        base=getattr(attention.v,'base',attention.v)
        if base.bias is not None:vc=vc-base.bias.to(vc)
        vc=reshape(vc)
        # Native V/O biases are already included once through the text branch.
        # Do not turn a zero C into another condition-independent value bias.
        qc=self.condition_query(q[:,known:],packet)
        mask = packet.condition_mask
        if mask is not None:
            if mask.shape != packet.condition.shape[:2] or mask.dtype != torch.bool:
                raise ValueError('C mask must be boolean [B,N], True for valid tokens')
            if not bool(mask.any(-1).all()): raise ValueError('C needs at least one valid token per sample')
            mask = mask[:, None, None].to(qc.device)
        cm=F.scaled_dot_product_attention(qc.transpose(1,2),kc.transpose(1,2).to(qc),
                                         vc.transpose(1,2).to(qc), attn_mask=mask).transpose(1,2)
        message=torch.cat((text_message[:,:known],text_message[:,known:]+packet.strength*cm.to(text_message)),1)
        result=attention.o(message.flatten(2))
        if self.capture:
            self.observations.append(dict(layer=packet.index+1,q_shape=list(q.shape),
                c_key_shape=list(kc.shape),c_value_shape=list(vc.shape),c_message_shape=list(cm.shape),
                c_message_rms=float(cm.detach().float().square().mean().sqrt()),
                text_message_rms=float(text_message[:,known:].detach().float().square().mean().sqrt()),
                first_frame_has_direct_C_update=False))
        return result
