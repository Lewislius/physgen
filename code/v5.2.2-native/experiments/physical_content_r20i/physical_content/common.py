"""R20 configuration: separate task inputs, fresh400 and native-preserving physical execution."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
SCHEMA = 'R20-dual-physical-branch-v11'
REVISION = 'R20-dual-physical-branch-20261117'
ROUTE_GROUPS = ('wan_video_lora', 'enhancement', 'physics_read', 'physics_write', 'physics_video')

def validate_config(cfg):
    if cfg['schema_version'] != SCHEMA or cfg['revision'] != REVISION:
        raise ValueError('R20 configuration revision differs')
    if (cfg['plan'], cfg['task']) != ('IT', 't2v'):
        raise ValueError('Candidate and task disagree')
    if cfg['initialization']['mode'] != 'official_fresh' or cfg['initialization']['project_checkpoint'] is not None:
        raise ValueError('Fresh official foundations; no project checkpoint initialization')
    if not cfg['base']['freeze_text_qkvo']:
        raise ValueError('Native text projections stay frozen')
    if cfg['base']['model'] != 'Wan2.2-TI2V-5B':
        raise ValueError('R20IT uses the official Wan2.2-TI2V-5B base')
    if cfg['model']['width'] != 3072 or cfg['model']['qwen_width'] != 4096:
        raise ValueError('Full-width Wan/Qwen interfaces are required')
    if cfg['model']['conditioned_ffn_rank'] is not None:
        raise ValueError('R20IT has no low-dimensional conditioning bottleneck')
    if cfg['mechanism'] != 29 or cfg['model']['extra_video_blocks'] != 5:
        raise ValueError('R20IT requires five full-grid physical video blocks')
    if cfg['model']['injection_layers'] != [0, 7, 14, 21, 28]:
        raise ValueError('R20IT distributed physical placement differs from the declared design')
    if cfg['model']['physical_initialization_blocks'] != cfg['model']['injection_layers']:
        raise ValueError('Physical blocks must copy the corresponding pretrained depth')
    if cfg['model']['content_anchor'] != 'native_input_values_selected_by_reasoning_queries':
        raise ValueError('Native source values must be selected by learned queries')
    if cfg['model'].get('physical_execution') != 'source_C_global_prior_plus_complete_word_token_cross_attention':
        raise ValueError('R20IT execution declaration differs')
    if cfg['model'].get('word_memory') != 'complete_reasoning_words_frozen_kv_zero_init_output':
        raise ValueError('R20IT word-memory declaration differs')
    t=cfg['train']
    if (t['max_updates'],t['effective_global_batch'],t['microbatch_per_device'],t['gradient_accumulation'])!=(400,4,1,4):
        raise ValueError('Use400 updates and effective batch4')
    if set(t['peak_lrs']) != set(ROUTE_GROUPS):
        raise ValueError('Parameter roles differ')
    image=cfg['task']=='i2v'
    if bool(cfg['input']['external_image'])!=image or bool(cfg['inference']['clamp_first'])!=image:
        raise ValueError('Task image boundary differs')
    if cfg['input']['future_targets_in_condition']:
        raise ValueError('Future targets cannot enter inference conditions')
    return cfg

def load_config(path, *, require_paths=False):
    cfg=deepcopy(yaml.safe_load(Path(path).read_text()))
    validate_config(cfg)
    paths=cfg['paths']
    paths['cache']=paths['train_cache']
    paths['interface']=paths['interface_output']
    paths['runs']=str(Path(paths['run_dir']).parent)
    cfg['variant_id']='R20'+cfg['plan']
    interface=int(cfg['train'].get('interface_updates',0))
    cfg['stages']={'interface':interface,'joint':400-interface}
    cfg['train']['accumulation']=cfg['train']['gradient_accumulation']
    cfg['train']['text_dropout']=cfg['train']['condition_dropout']
    cfg['loss']['generation']={'tail_weight':cfg['loss']['tail'],'tail_fraction':cfg['loss']['tail_fraction']}
    return cfg

def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    ).hexdigest()


def design_hash(cfg):
    value = deepcopy(cfg)
    value.pop('paths', None)
    value.pop('revision', None)
    value.pop('variant_id', None)
    value.pop('stages', None)
    for key in ('accumulation', 'text_dropout'):
        value.get('train', {}).pop(key, None)
    value.get('loss', {}).pop('generation', None)
    return canonical_hash(value)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def atomic_torch(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def data_ready(cfg):
    cache = Path(cfg['paths']['cache'])
    interface = Path(cfg['paths']['interface'])
    required = [cache / 'base/manifest.json', cache / 'base/null_text.pt', interface]
    for root in (Path(cfg['paths']['wan_checkpoint']), Path(cfg['paths']['qwen_checkpoint'])):
        index = root / ('diffusion_pytorch_model.safetensors.index.json'
                        if root == Path(cfg['paths']['wan_checkpoint'])
                        else 'model.safetensors.index.json')
        required.append(index)
    required += [Path(cfg['paths']['wan_checkpoint']) / 'Wan2.2_VAE.pth',
                 Path(cfg['paths']['wan_checkpoint']) / 'models_t5_umt5-xxl-enc-bf16.pth']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    rows = json.loads((cache / 'base/manifest.json').read_text())['records']
    if len(rows) != 3857:
        raise ValueError('The audited WISA train cache must contain 3857 records')
    train = [row for row in rows if row['split'] == 'train']
    if not train:
        raise ValueError('Training split is empty')
    from .data import TrainingData
    available = TrainingData(cfg)
    return dict(
        manifest_records=len(rows), manifest_train_records=len(train),
        train_records=len(available), unavailable_train_records=len(train) - len(available),
        validation_records=len(rows) - len(train),
        interface=str(interface), interface_hash=sha256_file(interface),
        train_manifest_hash=sha256_file(cache / 'base/manifest.json'),
        wan_index_hash=sha256_file(Path(cfg['paths']['wan_checkpoint']) /
                                   'diffusion_pytorch_model.safetensors.index.json'),
        qwen_index_hash=sha256_file(Path(cfg['paths']['qwen_checkpoint']) /
                                    'model.safetensors.index.json'),
        data_revision=cfg['train'].get('data_revision'),
        source='existing original training cache and frozen input feature cache',
    )


def audit(cfg):
    result = data_ready(cfg)
    result.update(
        revision=REVISION, plan=cfg['plan'], design_plan=cfg['design_plan'],
        task=cfg['task'],
        mechanism=cfg['mechanism'], sites=cfg['model']['injection_layers'],
        max_updates=cfg['train']['max_updates'],
        effective_batch=cfg['train']['effective_global_batch'],
        initialization='official Wan/Qwen/T5/VAE only; fresh project weights',
        native_text_lora=False, old_project_checkpoint=False,
        local_semantic_loss=False, st_loss=False, post_training=False,
        quality_proven=False,
    )
    return result


def is_allowed_gpu(name, policy='rtx6000'):
    normalized=re.sub(r'[^A-Z0-9]','',name.upper())
    return policy=='rtx6000' and 'H100' not in normalized and (
        'RTXPRO6000' in normalized or 'RTX6000BLACKWELL' in normalized)

def gpu_guard(run):
    import torch
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if count else 'NO CUDA DEVICE'
    policy = 'rtx6000'
    accepted = count == 1 and is_allowed_gpu(name, policy)
    record = dict(kind='gpu_policy', actual=name, visible_count=count,
                  accepted=accepted, resource_pool=policy,
                  experiment_id=os.getenv('DET_EXPERIMENT_ID'))
    path = Path(run) / 'gpu.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    if not accepted:
        raise RuntimeError(
            f'R20IT requires actual RTX6000 on blk-96g, got {name}')
    return record
