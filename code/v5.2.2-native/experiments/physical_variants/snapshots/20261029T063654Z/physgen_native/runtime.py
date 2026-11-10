"""Single-device optimizer ownership, continuous stages, and atomic restart state."""
from pathlib import Path
from contextlib import contextmanager
import json
import math
import os
import random
import torch
from .config import REVISIONS, OPERATOR_REVISIONS, FULLWIDTH_REVISION, ALIGNED_REVISION, ALIGNED_REVISIONS, MULTIROUND_REVISION, MESSAGE_REVISIONS, SIMPLE_MESSAGE_VARIANTS, SIMPLE_ARCHITECTURE_REVISIONS


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def owner(name):
    if name.startswith('knowledge.') and name.endswith(('verifier_lora_a', 'verifier_lora_b')): return 'verifier'
    if name.startswith('knowledge.reasoner.'): return 'reasoning'
    if name.startswith('conditioner.'): return 'condition'
    if name.startswith('critic.'): return 'critic'
    if name.startswith('generator.interface.'): return 'interface'
    if name.endswith(('lora_a', 'lora_b')):
        return 'knowledge' if name.startswith('knowledge.') else 'wan_lora'
    return None


def parameter_groups(model, cfg):
    t = cfg['train']
    if cfg['revision'] in SIMPLE_ARCHITECTURE_REVISIONS:
        groups = {}
        rates = {'condition': t['new_lr'], 'alignment': t['alignment_lr'],
                 'wan_lora': t['wan_lora_lr'], 'verifier': t.get('verifier_lr', 0.)}
        for name, parameter in model.named_parameters():
            role = owner(name)
            if role is None: continue
            if role not in ('condition', 'wan_lora', 'verifier'):
                raise ValueError(f'Unexpected simple-recipe parameter owner: {role}')
            group = ('alignment' if name.startswith(('conditioner.norm.', 'conditioner.alignment.'))
                     else role)
            decay = (t['weight_decay'] if parameter.ndim >= 2 and
                     not name.endswith('query_seed') and group != 'alignment' else 0.)
            key = group + ('_no_decay' if decay == 0. and group != 'alignment' else '')
            if key not in groups:
                groups[key] = dict(name=key, role=role, schedule_group=group, params=[], param_names=[],
                                   lr=rates[group], weight_decay=decay)
            groups[key]['params'].append(parameter)
            groups[key]['param_names'].append(name)
        return list(groups.values())
    rates = {'condition': t['new_lr'], 'interface': t['new_lr'], 'critic': t['critic_lr'],
             'knowledge': t['qwen_lr'], 'reasoning': t['new_lr'], 'wan_lora': t['wan_lora_lr'],
             'verifier': t.get('verifier_lr', 0.)}
    groups = {name: dict(name=name, params=[], param_names=[], lr=lr) for name, lr in rates.items()}
    for name, parameter in model.named_parameters():
        group = owner(name)
        if group is not None:
            groups[group]['params'].append(parameter); groups[group]['param_names'].append(name)
    return [g for g in groups.values() if g['params']]


def schedule(optimizer, cfg):
    if cfg['revision'] in SIMPLE_ARCHITECTURE_REVISIONS:
        return simple_schedule(optimizer, cfg)
    if cfg['revision'] in OPERATOR_REVISIONS:
        active = {'condition': ('interface', 'joint'), 'interface': ('interface', 'joint'),
                  'knowledge': ('interface', 'joint'), 'reasoning': ('interface', 'joint'),
                  'critic': ('calibration',), 'verifier': ('calibration',), 'wan_lora': ('joint',)}
        if cfg['train'].get('wan_lora_in_interface'): active['wan_lora'] = ('interface', 'joint')
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


def simple_schedule(optimizer, cfg):
    """C reaches its joint LR smoothly; calibration pauses its schedule.

    Wan warms up only when unfrozen. Calibration has its own short warmup;
    inactive phases consume neither the Adam step nor the active LR budget.
    """
    t, stages = cfg['train'], cfg['stages']
    interface, calibration, joint = stages['interface'], stages.get('calibration', 0), stages['joint']
    joint_start = interface + calibration
    def cosine(index, count, floor):
        fraction = min(1., max(0., index / max(1, count - 1)))
        return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * fraction))
    def warmed(index, count, warmup):
        warm = min(warmup, count)
        if index < warm: return (index + 1) / warm
        return cosine(index - warm, count - warm, t['minimum_lr_ratio'])
    curves = []
    for group in optimizer.param_groups:
        kind = group['schedule_group']
        joint_ratio = (t['joint_alignment_lr'] / t['alignment_lr'] if kind == 'alignment'
                       else t['joint_lr'] / t['new_lr'])
        def curve(step, kind=kind, joint_ratio=joint_ratio):
            if kind == 'verifier':
                return (warmed(step - interface, calibration, t['verifier_warmup'])
                        if interface <= step < joint_start else 0.)
            if kind == 'wan_lora':
                return warmed(step - joint_start, joint, t['warmup']) if step >= joint_start else 0.
            if step < interface:
                warm = min(t['warmup'], interface)
                if step < warm: return (step + 1) / warm
                return cosine(step - warm, interface - warm, joint_ratio)
            if step < joint_start: return 0.
            return joint_ratio * cosine(step - joint_start, joint, t['minimum_lr_ratio'])
        curves.append(curve)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, curves)


def clip_owned_gradients(optimizer, cfg):
    """Preserve C-only reward ownership even when clipping large gradients."""
    buckets = {}
    for group in optimizer.param_groups:
        role = group.get('role', group['name'])
        if not cfg['train'].get('clip_by_owner', False):
            role = 'critic' if role in ('critic', 'verifier') else 'main'
        buckets.setdefault(role, []).extend(group['params'])
    return {role: torch.nn.utils.clip_grad_norm_(params, cfg['train']['grad_clip'], error_if_nonfinite=True)
            for role, params in buckets.items()}


@contextmanager
def feedback_gradient_metrics(optimizer, cfg):
    """Measure the actual extra C gradient without another forward/backward."""
    metrics, handles, contributions = {}, [], []
    try:
        if cfg['revision'] in SIMPLE_MESSAGE_VARIANTS:
            parameters = [p for g in optimizer.param_groups if g.get('role') == 'condition' for p in g['params']]
            before = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
            metrics['paired_C_grad_norm'] = torch.stack(before).sum().sqrt() if before else 0.
            def capture(gradient):
                contributions.append(gradient.detach().float().square().sum())
            handles = [p.register_hook(capture) for p in parameters if p.requires_grad]
        yield metrics
    finally:
        for handle in handles: handle.remove()
        if contributions:
            metrics['feedback_C_grad_norm'] = torch.stack(contributions).sum().sqrt()
            metrics['feedback_to_paired_C_grad_ratio'] = (
                metrics['feedback_C_grad_norm'] / (metrics['paired_C_grad_norm'] + 1e-12))


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
    if getattr(model, 'criterion_verifier', False): state['verifier_state'] = dict(model.verifier_state)
    # CPU copies above are serialization only, never training offload.
    atomic_save(state, path)


def load_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('revision') not in REVISIONS: raise ValueError('Cannot load original V5.2.1 weights as this revision')
    if state['revision'] != state['config']['revision']:raise ValueError('Checkpoint/config revisions differ')
    return state


def initialize_generator(model, cfg, path):
    """Initialize a new run; explicitly retire the old compact C reader in v12.

    This is not resume and does not reuse its optimizer or its verifier. The
    Qwen-to-Wan semantic alignment must be trained again for the native reader.
    """
    source=load_checkpoint(path)
    for key in ('wan_checkpoint','qwen_checkpoint'):
        if Path(source['config']['paths'][key]).resolve()!=Path(cfg['paths'][key]).resolve():
            raise ValueError('Initialization base model differs')
    for section,keys in (('wan_lora',('rank','alpha')),('knowledge',('lora_rank','lora_alpha'))):
        if section == 'knowledge' and cfg['revision'] in MESSAGE_REVISIONS:
            continue
        if any(source['config'][section][key]!=cfg[section][key] for key in keys):
            raise ValueError('Initialization LoRA rank or scale differs')
    current={n:p for n,p in model.named_parameters() if owner(n) not in (None,'critic','verifier')}
    saved={n:p for n,p in source['model'].items() if owner(n) not in ('critic','verifier')}
    migration=(cfg['revision']==FULLWIDTH_REVISION and
               source['config']['model'].get('injection')=='conditional_operator')
    reasoning_migration = cfg['revision'] == MULTIROUND_REVISION and source['revision'] != MULTIROUND_REVISION
    alignment_migration=(cfg['revision'] in ALIGNED_REVISIONS and source['revision'] not in ALIGNED_REVISIONS)
    message_migration = cfg['revision'] in MESSAGE_REVISIONS and source['revision'] != cfg['revision']
    if message_migration:
        # Keep the trained native reader and compatible projection/affine.
        # Old Qwen adapters and independent-query workspaces are explicitly retired.
        transferable = ('generator.wan.', 'conditioner.norm.', 'conditioner.project.', 'conditioner.alignment.')
        same_message = (source['config']['knowledge'].get('reasoning_mode') == cfg['knowledge']['reasoning_mode'])
        output_projection_migration = (cfg['revision'] in SIMPLE_ARCHITECTURE_REVISIONS and
            source['config'].get('message', {}).get('projection', 'input') != 'output')
        if output_projection_migration:
            # P keeps its compatible 4096->3072 shape, but is now applied AFTER
            # the fresh 4096-wide G. Never import narrower G/query tensors.
            same_message = False
        if same_message:
            if source['config'].get('message') != cfg.get('message'):
                raise ValueError('Message transfer requires the same message architecture')
            transferable = ('generator.wan.', 'conditioner.')
        copied = {n for n in current if n.startswith(transferable) and n in saved}
        required = {n for n in current if n.startswith(transferable if same_message else ('generator.wan.',))}
        if not required <= copied:
            raise ValueError('Message migration requires all inherited Wan adapter tensors')
    elif reasoning_migration:
        retired = lambda n: n == 'conditioner.kind.weight' or n.startswith('generator.interface.')
        copied = {n for n in saved if not retired(n)}
        fresh = {n for n in current if n.startswith('knowledge.reasoner.')}
        if alignment_migration: fresh |= {n for n in current if n.startswith('conditioner.alignment.')}
        if copied != set(current)-fresh or not fresh:
            raise ValueError('v14 migration differs outside removed source modulation and new reasoning workspace')
    elif alignment_migration:
        keep=lambda n:not n.startswith(('generator.interface.', 'conditioner.alignment.'))
        copied={n for n in saved if keep(n)}
        if copied!={n for n in current if keep(n)} or set(current)-copied!={
                'conditioner.alignment.gain', 'conditioner.alignment.bias'}:
            raise ValueError('Alignment migration differs outside the removed interface and new affine')
    elif migration:
        keep=lambda n:not n.startswith('generator.interface.') or n.startswith('generator.interface.intensity.')
        copied={n for n in saved if keep(n)}
        if copied!={n for n in current if keep(n)} or set(current)-copied!={'generator.interface.intensity_query'}:
            raise ValueError('Full-width migration differs outside the explicitly replaced C interface')
    else:
        copied=set(saved)
        if copied!=set(current):raise ValueError('Generator initialization parameter sets differ')
    if any(saved[n].shape!=current[n].shape for n in copied):
        raise ValueError('Generator initialization parameter shapes differ')
    # Validate every tensor before mutating any parameter.
    with torch.no_grad():
        for name in copied:current[name].copy_(saved[name].to(current[name]))
    discarded=sorted(set(saved)-copied);fresh=sorted(set(current)-copied)
    return dict(path=str(Path(path).resolve()),source_revision=source['revision'],source_step=source['step'],
                copied_tensors=len(copied),discarded_tensors=discarded,new_tensors=fresh,
                fullwidth_migration=migration,
                alignment_migration=alignment_migration,
                reasoning_migration=reasoning_migration,
                message_migration=message_migration,
                alignment=('New frozen-Qwen message workspace; retain compatible C affine and learn message parameters'
                           if message_migration else 'Initialize affine from native text context; learn semantic alignment with frozen Wan'
                           if alignment_migration else 'Native full-width reader requires adaptation'
                           if migration else 'Unchanged parameter architecture'))


def initialize_wan_adapter(model, cfg, path):
    """Read existing Wan-only attention weights; never import planner/optimizer state."""
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    source = state['config']
    if Path(source['paths']['wan_checkpoint']).resolve() != Path(cfg['paths']['wan_checkpoint']).resolve():
        raise ValueError('Wan adapter base checkpoint differs')
    if any(source['wan_lora'][k] != cfg['wan_lora'][k] for k in ('rank', 'alpha')):
        raise ValueError('Wan adapter rank or scale differs')
    expected = {name:p for name,p in model.generator.wan.named_parameters() if name.endswith(('lora_a','lora_b'))}
    saved = state['wan_lora']
    if set(saved) != set(expected) or any(saved[n].shape != p.shape for n,p in expected.items()):
        raise ValueError('Wan adapter names or shapes differ')
    with torch.no_grad():
        for name,p in expected.items(): p.copy_(saved[name].to(p))
    return dict(path=str(Path(path).resolve()), step=state['step'], tensors=len(saved),
                imported='Wan attention LoRA only; no planner, future predictor, or optimizer')


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
    if getattr(model, 'criterion_verifier', False):
        if 'verifier_state' not in state: raise ValueError('Missing verifier calibration provenance in checkpoint')
        model.verifier_state = dict(state['verifier_state'])
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
