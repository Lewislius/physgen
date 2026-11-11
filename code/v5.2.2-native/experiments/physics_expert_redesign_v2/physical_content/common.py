from pathlib import Path
import json
import yaml
from physical_scaling.common import gpu_guard as original_gpu_guard
from physical_scaling.common import atomic_json, atomic_torch, cpu_tree, Log
from physical_scaling.common import audit as original_audit

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physics_expert_redesign_v2')
REVISION = 'physics_expert_redesign_v2'


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get('revision') != REVISION:
        raise ValueError('Use the physics_expert_redesign_v2 revision')
    if cfg.get('variant') not in ('p1_request_key','p2_pre_self','p4_source_conditioned_experts'):
        raise ValueError('Unsupported physics expert variant')
    if sum(cfg['stages'].values()) != 400 or cfg['train']['accumulation'] != 4:
        raise ValueError('Keep 400 updates and 1600 samples')
    if cfg['stages'] != dict(interface=0, joint=400):
        raise ValueError('Use joint400 from update 1')
    if cfg['verifier']['kind'] != 'none' or cfg.get('initialize_generator'):
        raise ValueError('Fresh official foundations only')
    if not Path(cfg['paths']['runs']).resolve().is_relative_to(ROOT):
        raise ValueError('Keep new runs in this workspace')
    if cfg['inference'] != dict(steps=50,shift=5.,guidance=5.,sampler='shifted_euler'):
        raise ValueError('Use the fixed inference protocol')
    return cfg


def data_ready(cfg):
    cache = Path(cfg['paths']['cache'])
    for path in [cache/'base/manifest.json', cache/'base/null_text.pt', Path(cfg['paths']['interface'])]:
        if not path.is_file():
            raise FileNotFoundError(path)
    records = json.loads((cache/'base/manifest.json').read_text())['records']
    return dict(manifest_records=len(records), data_revision=cfg['data_revision'],
                source='existing original training cache', new_api_calls=0)


def gpu_guard(run):
    return original_gpu_guard(run)


def audit(cfg):
    result = original_audit(cfg)
    result.update(data_ready(cfg))
    result.update(variant=cfg['variant'], physics_sites=list(range(2,30,3)),
                 teacher='none', extra_loss=False, intervention_learning=False,
                 generation_objective=cfg['loss']['generation'],
                 route_target='raw_route_only', optimizer_updates=400,
                 initialization='fresh official foundations')
    result['quality_proven'] = False
    return result
