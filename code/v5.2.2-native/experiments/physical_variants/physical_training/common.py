from pathlib import Path
import json
import os
import re
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
VARIANTS = ('v5_ce', 'v6_write', 'v6_assoc')


def load_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['variant'] not in VARIANTS or cfg['revision'] != 'physical_variants_20261029_r1':
        raise ValueError('Unsupported physical training revision')
    if cfg['stages'] != {'interface': 64, 'joint': 160}:
        raise ValueError('This initial recipe uses 64 interface + 160 joint steps')
    if cfg['model']['q_tokens'] != 128 or cfg['model']['rounds'] != 2:
        raise ValueError('Use 128 Q slots and exactly two shared enhancement passes')
    if cfg.get('initialize_generator') or cfg.get('verifier', {}).get('kind', 'none') != 'none':
        raise ValueError('Fresh official bases only; no project initializer or verifier')
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
    paths = [root / 'base/null_text.pt', Path(cfg['paths']['annotations'])]
    for row in rows:
        paths += [root / f'base/{part}/{row["id"]}{suffix}'
                  for part, suffix in [('vae', '.pt'), ('text', '.pt'), ('first', '.png')]]
    for key, index in [('wan_checkpoint', 'diffusion_pytorch_model.safetensors.index.json'),
                       ('qwen_checkpoint', 'model.safetensors.index.json')]:
        directory = Path(cfg['paths'][key])
        mapping = json.loads((directory / index).read_text())['weight_map']
        paths += [directory / p for p in set(mapping.values())]
    if cfg['variant'] == 'v6_write':
        paths += [Path(cfg['paths']['wan_checkpoint']) / 'models_t5_umt5-xxl-enc-bf16.pth']
    paths += [Path(cfg['paths']['wan_checkpoint']) / 'Wan2.2_VAE.pth']
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(str(missing[:10]))
    return dict(ready=True, rows=len(rows), train=sum(r['split']=='train' for r in rows),
                validation=sum(r['split']=='validation' for r in rows), variant=cfg['variant'],
                initialization='official foundations; fresh project modules and LoRA',
                teacher_cache='generated lazily from training data, before its gradient computation')
