"""Real-video FM/tail loss plus routing-word alignment, one forward per sample."""
import torch
from .supervision import generation_loss


def backward_sample(model,evidence,sample,text,sigma,noise,coefficient,accumulation):
    noisy=(1-sigma)*sample['latent']+sigma*noise
    arguments=(noisy,sample['first'],text,sigma)
    # The interface phase learns enhancement and fusion for128 updates with
    # real-video generation gradients. The remaining272 also update Wan LoRA.
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
        route_aux=float(auxiliary.detach()),
        weighted_route_aux=float(coefficient)*float(auxiliary.detach()),
        c_pair_cosine=float((normed.sum(0).square().sum()-normed.square().sum())/(n*(n-1))),
        c_nonzero_token_fraction=float((value.norm(dim=-1)>1e-12).float().mean()),
        c_rms=float(value.square().mean().sqrt()),
        c_centered_energy=float((value-value.mean(0)).square().sum()/value.square().sum().clamp_min(1e-20)))
