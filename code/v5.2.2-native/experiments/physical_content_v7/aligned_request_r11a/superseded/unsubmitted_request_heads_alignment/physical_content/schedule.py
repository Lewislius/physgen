"""Separate alignment and generation clocks; no generator updates in stage one."""
def learning_rates(cfg,step):
    if not 0<=step<sum(cfg['stages'].values()):raise ValueError('Step outside this400-update run')
    train=cfg['train'];joint_step=step-cfg['stages']['alignment']
    warm=lambda n,d:min(1.,(n+1)/d)
    rates={'enhancement':train['peak_lrs']['enhancement']*warm(step,train['warmup'])}
    for group in ('fusion_read','fusion_write','wan_lora'):
        duration=train['wan_warmup'] if group=='wan_lora' else train['warmup']
        rates[group]=0. if joint_step<0 else train['peak_lrs'][group]*warm(joint_step,duration)
    return rates


def auxiliary_coefficient(cfg,step):
    joint_step=step-cfg['stages']['alignment']
    return 1. if joint_step<0 else cfg['loss']['peak_weight']*min(1.,(joint_step+1)/cfg['loss']['warmup'])


def rates(optimizer,cfg,step):
    values=learning_rates(cfg,step)
    for group in optimizer.param_groups:group['lr']=values[group['name']]
