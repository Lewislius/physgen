"""Single-device optimizer ownership, continuous stages, and atomic restart state."""
from pathlib import Path
import json
import math
import os
import random
import torch
from .config import REVISIONS, OPERATOR_REVISIONS


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def owner(name):
    if name.startswith('conditioner.'): return 'condition'
    if name.startswith('critic.'): return 'critic'
    if name.startswith('generator.interface.'): return 'interface'
    if name.endswith(('lora_a', 'lora_b')):
        return 'knowledge' if name.startswith('knowledge.') else 'wan_lora'
    return None


def parameter_groups(model, cfg):
    t = cfg['train']
    rates = {'condition': t['new_lr'], 'interface': t['new_lr'], 'critic': t['critic_lr'],
             'knowledge': t['qwen_lr'], 'wan_lora': t['wan_lora_lr']}
    groups = {name: dict(name=name, params=[], param_names=[], lr=lr) for name, lr in rates.items()}
    for name, parameter in model.named_parameters():
        group = owner(name)
        if group is not None:
            groups[group]['params'].append(parameter); groups[group]['param_names'].append(name)
    return [g for g in groups.values() if g['params']]


def schedule(optimizer, cfg):
    if cfg['revision'] in OPERATOR_REVISIONS:
        active = {'condition': ('interface', 'joint'), 'interface': ('interface', 'joint'),
                  'knowledge': ('interface', 'joint'), 'critic': ('calibration',), 'wan_lora': ('joint',)}
        curves = []
        for group in optimizer.param_groups:
            phases = active[group['name']]
            total = sum(n for p, n in cfg['stages'].items() if p in phases)
            warm = min(cfg['train']['warmup'], max(1, total))
            floor = cfg['train']['minimum_lr_ratio']
            def curve(step, phases=phases, total=total, warm=warm, floor=floor):
                elapsed, consumed = 0, 0
                for phase, count in cfg['stages'].items():
                    if phase in phases: elapsed += min(count, max(0, step-consumed))
                    if consumed <= step < consumed+count and phase not in phases: return 0.
                    consumed += count
                if elapsed < warm: return (elapsed+1)/warm
                fraction = min(1., (elapsed-warm)/max(1, total-warm))
                return floor+(1-floor)*.5*(1+math.cos(math.pi*fraction))
            curves.append(curve)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, curves)
    end = sum(cfg['stages'].values()); joint = cfg['stages']['interface']
    curves = []
    for group in optimizer.param_groups:
        start = joint if group['name'] == 'wan_lora' else 0
        warm = min(cfg['train']['warmup'], max(1, end-start))
        floor = cfg['train']['minimum_lr_ratio']
        def curve(step, begin=start, warmup=warm, minimum=floor):
            if step < begin: return 0.
            if step < begin+warmup: return (step-begin+1)/warmup
            fraction = min(1., (step-begin-warmup)/max(1, end-begin-warmup))
            return minimum+(1-minimum)*.5*(1+math.cos(math.pi*fraction))
        curves.append(curve)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, curves)


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def atomic_save(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with temp.open('wb') as stream:
            torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_checkpoint(model, optimizer, scheduler, cfg, step, cursor, path):
    learned = {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if owner(name) is not None}
    state = dict(revision=cfg['revision'], config=cfg, step=step, cursor=cursor, model=learned,
                 optimizer=cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                 rng=dict(torch=torch.get_rng_state(), python=random.getstate(),
                          cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
    # CPU copies above are serialization only, never training offload.
    atomic_save(state, path)


def load_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('revision') not in REVISIONS: raise ValueError('Cannot load original V5.2.1 weights as this revision')
    if state['revision'] != state['config']['revision']:raise ValueError('Checkpoint/config revisions differ')
    return state


def restore(model, optimizer, scheduler, state, cfg):
    if state['config'] != cfg: raise ValueError('Resume must use the saved, identical configuration')
    current = {name: parameter for name, parameter in model.named_parameters() if owner(name) is not None}
    learned = state['model']
    if not getattr(model, 'training_observers', True):
        if optimizer is not None: raise ValueError('A generation-only instance cannot resume training')
        learned = {k:v for k,v in learned.items() if not k.startswith('critic.')}
    if set(current) != set(learned): raise ValueError('Trainable checkpoint parameter names differ')
    with torch.no_grad():
        for name, parameter in current.items(): parameter.copy_(learned[name].to(parameter))
    if optimizer is not None:
        previous = state['optimizer']['param_groups']
        if [g['param_names'] for g in previous] != [g['param_names'] for g in optimizer.param_groups]:
            raise ValueError('Optimizer parameter ownership changed')
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng']['torch']); random.setstate(state['rng']['python'])
        if state['rng']['cuda']: torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['step'], state['cursor']


class Logger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.stream = Path(path).open('a', buffering=1)
    def write(self, kind, step, **values):
        def convert(value):
            if torch.is_tensor(value): return float(value.detach().cpu())
            if isinstance(value, dict): return {k: convert(v) for k, v in value.items()}
            return value
        text = json.dumps(dict(kind=kind, step=step, **convert(values)), ensure_ascii=False)
        self.stream.write(text+'\n'); print(text, flush=True)
    def close(self): self.stream.close()
