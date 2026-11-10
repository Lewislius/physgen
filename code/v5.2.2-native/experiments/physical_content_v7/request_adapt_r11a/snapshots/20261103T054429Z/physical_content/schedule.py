"""Keep R10A physical/backbone rates; train native text from the first update."""
from physical_scaling.schedule import learning_rates as original_rates, auxiliary_coefficient
from physical_scaling.schedule import cosine_tail


def learning_rates(cfg,step):
    values=original_rates(cfg,step)
    train=cfg['train']
    warm=min(1.,(step+1)/train['native_text_warmup'])
    tail=cosine_tail(step,train['decay_start'],sum(cfg['stages'].values()),train['minimum_lr_ratio'])
    values['native_text']=train['peak_lrs']['native_text']*warm*tail
    return values


def rates(optimizer,cfg,step):
    values=learning_rates(cfg,step)
    for group in optimizer.param_groups:group['lr']=values[group['name']]
