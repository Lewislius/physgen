"""Separate parameter credit using two real-video generation objectives."""
import torch
from .supervision import generation_loss


def backward_sample(model,evidence,sample,text,sigma,noise,coefficient,accumulation):
    noisy=(1-sigma)*sample['latent']+sigma*noise
    arguments=(noisy,sample['first'],text,sigma)
    native,_=model.velocity(*arguments,None,sample['times'],physical_enabled=False)
    native_loss,native_parts=generation_loss(native,noise,sample['latent'],sample['valid'],
                                             **model.cfg['loss']['generation'])
    if not torch.isfinite(native_loss):raise FloatingPointError('Nonfinite native generation loss')
    (native_loss/accumulation).backward()
    native_value=float(native_loss.detach());native_fm=float(native_parts['fm'].detach())
    del native,native_loss,native_parts
    # No optimizer step between the two views: same sample/noise/sigma/weights.
    with model.physical_training():
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
        native_generation=native_value,native_fm=native_fm,
        full_minus_native_generation=float(generation.detach())-native_value,
        c_pair_cosine=float((normed.sum(0).square().sum()-n)/(n*(n-1))),
        c_centered_energy=float((value-value.mean(0)).square().sum()/value.square().sum().clamp_min(1e-20)))
