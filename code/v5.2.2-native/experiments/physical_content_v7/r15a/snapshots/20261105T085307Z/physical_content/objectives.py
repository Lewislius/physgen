"""One full-generation objective in a short interface phase then joint training."""
import math
import torch
from .supervision import generation_loss


def execution_alignment_weight(cfg,step):
    """Same semantic target early; free execution values by the end of joint training."""
    total=sum(cfg['stages'].values());interface=cfg['stages']['interface']
    if not 0<=step<total:raise ValueError('Alignment schedule step outside training')
    if step<interface:return 1.
    fraction=(step-interface)/max(1,total-interface-1)
    return .5*(1.+math.cos(math.pi*fraction))


def backward_sample(model,evidence,sample,text,sigma,noise,coefficient,accumulation,*,execution_alignment=1.):
    noisy=(1-sigma)*sample['latent']+sigma*noise
    arguments=(noisy,sample['first'],text,sigma)
    # The interface phase learns enhancement and fusion for128 updates with
    # real-video generation gradients. The remaining272 also update Wan LoRA.
    content,groups=model.enhancement(evidence)
    pred,_=model.velocity(*arguments,content,sample['times'],groups)
    generation,parts=generation_loss(pred,noise,sample['latent'],sample['valid'],
                                      **model.cfg['loss']['generation'])
    losses=model.enhancement.content_losses(content,evidence)
    # Normalize the two targets so equal C tensors have the original total
    # alignment strength. This weight is a training schedule, not an inference gate.
    auxiliary=(losses['routing']+execution_alignment*losses['execution'])/(1.+execution_alignment)
    objective=generation+coefficient*auxiliary
    if not torch.isfinite(objective):raise FloatingPointError('Nonfinite full generation objective')
    (objective/accumulation).backward()
    value=content['execution'].detach().float()[0]
    normed=torch.nn.functional.normalize(value,dim=-1);n=len(value)
    return dict(fm=float(parts['fm'].detach()),tail=float(parts['tail'].detach()),
        generation=float(generation.detach()),aux=float(auxiliary.detach()),
        weighted_aux=float(coefficient)*float(auxiliary.detach()),
        route_aux=float(losses['routing'].detach()),execution_aux=float(losses['execution'].detach()),
        weighted_route_aux=float(coefficient)*float(losses['routing'].detach())/(1.+execution_alignment),
        weighted_execution_aux=float(coefficient)*execution_alignment*float(losses['execution'].detach())/(1.+execution_alignment),
        c_pair_cosine=float((normed.sum(0).square().sum()-n)/(n*(n-1))),
        c_centered_energy=float((value-value.mean(0)).square().sum()/value.square().sum().clamp_min(1e-20)))
