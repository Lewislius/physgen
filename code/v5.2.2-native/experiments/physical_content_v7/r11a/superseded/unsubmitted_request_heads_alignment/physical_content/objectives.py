"""Routing representation alignment, followed by joint full video generation."""
import torch
from .supervision import generation_loss


def backward_alignment(model,evidence,accumulation):
    content,_=model.enhancement(evidence)
    alignment=model.enhancement.content_loss(content,evidence)
    if not torch.isfinite(alignment):raise FloatingPointError('Nonfinite representation alignment')
    (alignment/accumulation).backward()
    return dict(alignment=float(alignment.detach()))


def backward_sample(model,evidence,sample,text,sigma,noise,coefficient,accumulation):
    noisy=(1-sigma)*sample['latent']+sigma*noise
    arguments=(noisy,sample['first'],text,sigma)
    # Called only after the representation stage. Execution C, fusion and the
    # original Wan self-attention/FFN LoRA learn the real video target together.
    content,groups=model.enhancement(evidence)
    pred,_=model.velocity(*arguments,content,sample['times'],groups)
    generation,parts=generation_loss(pred,noise,sample['latent'],sample['valid'],
                                      **model.cfg['loss']['generation'])
    auxiliary=model.enhancement.content_loss(content,evidence)
    objective=generation+coefficient*auxiliary
    if not torch.isfinite(objective):raise FloatingPointError('Nonfinite full generation objective')
    (objective/accumulation).backward()
    value=content['execution'].detach().float()[0]
    normed=torch.nn.functional.normalize(value,dim=-1);n=len(value)
    return dict(fm=float(parts['fm'].detach()),tail=float(parts['tail'].detach()),
        generation=float(generation.detach()),aux=float(auxiliary.detach()),
        weighted_aux=float(coefficient)*float(auxiliary.detach()),
        c_pair_cosine=float((normed.sum(0).square().sum()-n)/(n*(n-1))),
        c_centered_energy=float((value-value.mean(0)).square().sum()/value.square().sum().clamp_min(1e-20)))
