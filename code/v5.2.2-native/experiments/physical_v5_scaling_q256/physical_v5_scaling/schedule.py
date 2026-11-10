"""Warm up, hold useful learning rates, then decay; steps are optimizer updates."""
import math

GROUPS=('enhancement','fusion_read','fusion_write','semantic_bridge','wan_lora')


def cosine_tail(step,start,total,floor):
    if step < start:
        return 1.
    fraction=(step-start)/max(1,total-start-1)
    return floor+(1-floor)*.5*(1+math.cos(math.pi*min(1.,fraction)))


def learning_rates(cfg,step):
    total=sum(cfg['stages'].values())
    if not 0<=step<total:
        raise ValueError('Step must be a zero-based optimizer update before completion')
    settings=cfg['train']
    tail=cosine_tail(step,settings['decay_start'],total,settings['minimum_lr_ratio'])
    warm=min(1.,(step+1)/settings['warmup'])
    result={g:settings['peak_lrs'][g]*warm*tail for g in GROUPS if g!='wan_lora'}
    j=step-cfg['stages']['interface']
    result['wan_lora']=(0. if j<0 else settings['peak_lrs']['wan_lora']*
                        min(1.,(j+1)/settings['wan_warmup'])*tail)
    return result


def auxiliary_coefficient(cfg,step):
    total=sum(cfg['stages'].values())
    if not 0<=step<total:
        raise ValueError('Auxiliary schedule step outside training')
    loss=cfg['loss']
    warm=min(1.,(step+1)/loss['warmup'])
    tail=cosine_tail(step,loss['decay_start'],total,loss['final_weight']/loss['peak_weight'])
    return loss['peak_weight']*warm*tail


def rates(optimizer,cfg,step):
    values=learning_rates(cfg,step)
    for group in optimizer.param_groups:
        group['lr']=values[group['name']]
