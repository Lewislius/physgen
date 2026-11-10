from pathlib import Path
import json
import os
import re
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
OUTPUT_ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_v5_scaling_q256')
VARIANTS = ('v5_ce',)
REVISION = 'physical_v5_scaling_q256_s1000_20261029_r1'


def load_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['variant'] not in VARIANTS or cfg['revision'] != REVISION:
        raise ValueError('Unsupported independent Q256 scaling revision')
    if cfg['stages'] != {'interface': 128, 'joint': 872}:
        raise ValueError('Use 128 interface + 872 joint = 1000 optimizer updates')
    if cfg['model']['q_tokens'] != 256 or cfg['model']['rounds'] != 2:
        raise ValueError('Use 256 Q slots and exactly two shared enhancement passes')
    if cfg.get('initialize_generator') or cfg.get('verifier', {}).get('kind', 'none') != 'none':
        raise ValueError('Fresh official bases only; no project initializer or verifier')
    from .schedule import GROUPS
    total=sum(cfg['stages'].values())
    train,loss=cfg['train'],cfg['loss']
    if set(train['peak_lrs'])!=set(GROUPS) or any(not 0<v<1 for v in train['peak_lrs'].values()):
        raise ValueError('Need five explicit positive learning rates')
    if not (0<train['warmup']<=cfg['stages']['interface']<train['decay_start']<total):
        raise ValueError('Invalid learning-rate schedule')
    if not 0<train['wan_warmup']<cfg['stages']['joint'] or not 0<train['minimum_lr_ratio']<=1:
        raise ValueError('Invalid Wan warmup or learning-rate floor')
    if not (0<loss['warmup']<=loss['decay_start']<total and 0<loss['final_weight']<=loss['peak_weight']):
        raise ValueError('Invalid auxiliary schedule')
    if set(train['grad_clip'])!=set(GROUPS) or any(v<=0 for v in train['grad_clip'].values()):
        raise ValueError('Need a positive clipping threshold per optimizer role')
    for name in ['keep_steps','validation_steps']:
        if train[name]!=sorted(set(train[name])) or any(not 1<=s<=total for s in train[name]):
            raise ValueError(f'Invalid {name}')
    if cfg['stages']['interface'] not in train['keep_steps'] or total not in train['keep_steps']:
        raise ValueError('Preserve the stage-end and final checkpoints')
    for name in ['runs','supervision_cache']:
        if not Path(cfg['paths'][name]).resolve().is_relative_to(OUTPUT_ROOT):
            raise ValueError('Scaling outputs must stay separate from original runs/caches')
    return cfg


def is_allowed_gpu(name):
    name = re.sub(r'[^A-Z0-9]', '', name.upper())
    return 'H100' not in name and ('RTXPRO6000' in name or 'RTX6000BLACKWELL' in name)


def gpu_guard(run):
    """Run before importing/loading either foundation model. Exit 86 on mismatch."""
    import torch
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if count else 'NO CUDA DEVICE'
    accepted = count == 1 and is_allowed_gpu(name)
    record = dict(kind='gpu_policy', actual=name, visible_count=count, accepted=accepted,
                  resource_pool='blk-96g', experiment_id=os.getenv('DET_EXPERIMENT_ID'),
                  time=time.time())
    atomic_json(record, Path(run) / 'gpu.json')
    print(json.dumps(record), flush=True)
    if not accepted:
        raise SystemExit(86)
    return record


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temp, path)


def atomic_torch(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with temp.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def cpu_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


class Log:
    def __init__(self, run):
        self.path = Path(run) / 'train.jsonl'

    def __call__(self, kind, **values):
        record = dict(kind=kind, time=time.time(), **values)
        line = json.dumps(record, ensure_ascii=False)
        print(line, flush=True)
        with self.path.open('a') as stream:
            stream.write(line + '\n')


def audit(cfg):
    root = Path(cfg['paths']['cache'])
    rows = json.loads((root / 'base/manifest.json').read_text())['records']
    if len(rows) != 3857:
        raise ValueError('Existing WISA cache manifest differs')
    paths = [root / 'base/null_text.pt']
    for row in rows:
        paths += [root / f'base/{part}/{row["id"]}{suffix}'
                  for part, suffix in [('vae', '.pt'), ('text', '.pt'), ('first', '.png')]]
    for key, index in [('wan_checkpoint', 'diffusion_pytorch_model.safetensors.index.json'),
                       ('qwen_checkpoint', 'model.safetensors.index.json')]:
        directory = Path(cfg['paths'][key])
        mapping = json.loads((directory / index).read_text())['weight_map']
        paths += [directory / p for p in set(mapping.values())]
    paths += [Path(cfg['paths']['annotations'])]
    paths += [Path(cfg['paths']['wan_checkpoint']) / 'Wan2.2_VAE.pth']
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(str(missing[:10]))
    annotations=[json.loads(line) for line in Path(cfg['paths']['annotations']).read_text().splitlines()]
    by_id={r['sample_id']:r for r in annotations}
    if len(by_id)!=len(annotations) or not all(r['id'] in by_id for r in rows):
        raise ValueError('Existing WISA annotations must cover the full cache manifest')
    return dict(ready=True, rows=len(rows), train=sum(r['split']=='train' for r in rows),
                validation=sum(r['split']=='validation' for r in rows), variant=cfg['variant'],
                q_tokens=cfg['model']['q_tokens'],steps=sum(cfg['stages'].values()),
                sample_draws=sum(cfg['stages'].values())*cfg['train']['accumulation'],
                initialization='official foundations; fresh project modules and LoRA',
                teacher_cache='read-only reuse of unchanged r1 targets; new samples written only to scaling cache')
