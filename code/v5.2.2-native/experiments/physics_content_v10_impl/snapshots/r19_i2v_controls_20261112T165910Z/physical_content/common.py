"""Canonical R18E-H (NoChunks-v2 T1/I1/T2/I2) configuration and checks.

The v10 1000-update phase boundaries are compressed by the requested factor
400/1000 = 0.4 to integer update boundaries: video-LoRA warmup 51, physics
warmup 13, peak through 240, cosine/route decay 241..400.  This is a
recompressed 400-step schedule, not the step-400 truncation of the 1000-step
run.
"""
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
SCHEMA = 'NoChunks-v2-400'
REVISION = 'NoChunks-v2-20261111'
PLANS = ('E', 'F', 'G', 'H')
DESIGNS = {'E': 'T1', 'F': 'I1', 'G': 'T2', 'H': 'I2'}
VARIANTS = {
    'E': dict(design='T1', task='t2v', mechanism=1, external_image=False,
              scope='request_only', known=0, unknown_scope='all_valid_frames',
              clamp_first=False, conditioned_rank=None),
    'F': dict(design='I1', task='i2v', mechanism=1, external_image=True,
              scope='first_image_and_request_only', known=1,
              unknown_scope='valid_future_frames', clamp_first=True,
              conditioned_rank=None),
    'G': dict(design='T2', task='t2v', mechanism=2, external_image=False,
              scope='request_only', known=0, unknown_scope='all_valid_frames',
              clamp_first=False, conditioned_rank=64),
    'H': dict(design='I2', task='i2v', mechanism=2, external_image=True,
              scope='first_image_and_request_only', known=1,
              unknown_scope='valid_future_frames', clamp_first=True,
              conditioned_rank=64),
}
INJECTION_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
ROUTE_GROUPS = ('wan_video_lora', 'enhancement', 'physics_read',
                'physics_write', 'mechanism_adapter')
FEATURE_CACHE = (BASE / 'experiments/physical_content_v7/data_revision_qwen38/runs/'
                 '20261103_dataset/encoded/content')
I2V_INPUT_CACHE = FEATURE_CACHE / 'input'
I2V_WORDS_CACHE = FEATURE_CACHE / 'words'
I2V_NATIVE_CACHE = FEATURE_CACHE / 'native_words'


def _substitute(value, missing):
    if isinstance(value, dict):
        return {key: _substitute(item, missing) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, missing) for item in value]
    if not isinstance(value, str):
        return value
    pattern = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')

    def replace(match):
        name = match.group(1)
        if name not in os.environ:
            missing.add(name)
            return match.group(0)
        return os.environ[name]

    return pattern.sub(replace, value)


def validate_config(cfg):
    """Reject any drift from the NoChunks-v2 design contract."""
    if cfg.get('schema_version') != SCHEMA:
        raise ValueError(f'Use schema {SCHEMA}')
    plan = cfg.get('plan')
    if plan not in PLANS:
        raise ValueError('Plan must be one of T1, I1, T2, I2')
    expected = VARIANTS[plan]
    if cfg.get('design_plan') != expected['design']:
        raise ValueError('R18 plan/design label disagree')
    if (cfg.get('task'), cfg.get('mechanism')) != (expected['task'], expected['mechanism']):
        raise ValueError('Plan/task/mechanism disagree')
    if cfg.get('revision') != REVISION:
        raise ValueError('The NoChunks-v2 revision must remain fixed')
    if cfg.get('status') != 'unvalidated_candidate':
        raise ValueError('This is still an unvalidated candidate')
    if cfg.get('source', {}).get('reference_commit') is None:
        raise ValueError('Freeze the source reference commit')

    init = cfg['initialization']
    if (init['mode'] != 'official_fresh' or init['project_checkpoint'] is not None
            or init['new_adapter_seed'] != 42):
        raise ValueError('Fresh official initialization only')

    model = cfg['model']
    if (model['width'], model['ffn_width'], model['native_layers'],
            model['native_heads'], model['qwen_width']) != (3072, 14336, 30, 24, 4096):
        raise ValueError('Only the verified Wan2.2-TI2V-5B / Qwen4096 shapes are allowed')
    if (model['q_tokens'], model['q_heads'], model['grounded_heads'],
            model['content_rounds']) != (256, 32, 16, 2):
        raise ValueError('Keep the R10A 256-slot, 32-head, two-round content contract')
    if (model['physics_slots'], model['physics_width'], model['q_width'],
            model['q_rounds']) != (256, 3072, 4096, 2):
        raise ValueError('The v10 physics/Q shape aliases changed')
    if (model['physics_slots'] != model['q_tokens']
            or model['physics_width'] != model['width']
            or model['q_width'] != model['qwen_width']
            or model['q_rounds'] != model['content_rounds']):
        raise ValueError('v10 model aliases disagree with the implementation keys')
    if model['injection_layers'] != INJECTION_LAYERS:
        raise ValueError('Injection layers must be every third block, 0-based [2,5,...,29]')
    if model['extra_video_blocks'] != 0 or model['split_temporal_physics_memory']:
        raise ValueError('No extra video block and no temporal physics partition')
    if model['conditioned_ffn_rank'] != expected['conditioned_rank']:
        raise ValueError('Conditioned FFN rank belongs only to mechanism 2')

    base = cfg['base']
    if (base['model'] != 'Wan2.2-TI2V-5B' or base['project_checkpoint'] is not None
            or not base['freeze_text_qkvo'] or base['video_lora_from_update'] != 1):
        raise ValueError('Base contract changed')
    if (base['video_lora_rank'], base['video_lora_alpha']) != (32, 32):
        raise ValueError('Video LoRA must be rank32/alpha32')

    inp = cfg['input']
    if (bool(inp['external_image']) != expected['external_image']
            or inp['evidence_scope'] != expected['scope']
            or int(inp['known_latent_frames']) != expected['known']
            or inp['future_targets_in_condition']
            or inp['chunk_descriptions'] or inp['local_annotation_preprocessing']):
        raise ValueError('Task input/provenance contract changed')
    if bool(inp['external_image']) and inp['known_latent_frames'] != 1:
        raise ValueError('I2V has exactly one known latent frame')
    if not inp['external_image'] and inp['known_latent_frames'] != 0:
        raise ValueError('T2V has no known latent frame')

    train = cfg['train']
    if train['max_updates'] != 400:
        raise ValueError('This implementation runs the recompressed 400-step schedule')
    if (train['effective_global_batch'], train['world_size'],
            train['microbatch_per_device'], train['gradient_accumulation']) != (4, 1, 1, 4):
        raise ValueError('Use effective batch 4 as single-GPU micro1 x accum4')
    if (train['physics_warmup'], train['video_lora_warmup'],
            train['peak_end_update'], train['minimum_lr_ratio']) != (13, 51, 240, 0.2):
        raise ValueError('The compressed 400-step warmups/peak phase changed')
    if train['post_training'] or train['noise_strategy'] != 'unchanged_linear_FM':
        raise ValueError('No post-training and no new noise strategy')
    if (abs(train['sigma_min'] - 0.001) > 1e-12
            or abs(train['sigma_max'] - 0.999) > 1e-12):
        raise ValueError('Sigma sampling range changed')
    if train['condition_dropout'] != 0.1:
        raise ValueError('Condition dropout must stay 0.1')
    expected_lrs = {
        'wan_video_lora': 2e-5, 'enhancement': 5e-5, 'physics_read': 2e-5,
        'physics_write': 3e-5, 'mechanism_adapter': 2e-5,
    }
    if set(train['peak_lrs']) != set(expected_lrs):
        raise ValueError('Need exactly the five NoChunks parameter roles')
    for key, value in expected_lrs.items():
        if abs(float(train['peak_lrs'][key]) - value) > 1e-12:
            raise ValueError(f'Peak LR for {key} differs from the design')
    expected_keep = [100, 200, 300, 400]
    if (train['keep_updates'] != expected_keep or train['save_every'] != 100):
        raise ValueError('400-step checkpoint cadence must be 100/200/300/400')

    loss = cfg['loss']
    if (abs(loss['fm'] - 0.8) > 1e-12 or abs(loss['tail'] - 0.2) > 1e-12
            or loss['tail_fraction'] != 0.25):
        raise ValueError('Keep 0.8 FM + 0.2 tail with top 25%')
    if (loss['local_semantic_enabled'] or loss['st_enabled']
            or loss['supervise_known_latent']):
        raise ValueError('No local semantic loss, ST, or known-frame supervision')
    if (abs(loss['route_peak'] - 0.03) > 1e-12 or loss['route_warmup'] != 13
            or loss['route_decay_start'] != 240
            or abs(loss['route_final'] - 0.015) > 1e-12):
        raise ValueError('Route schedule must be the compressed 400-step plan')
    if loss['unknown_scope'] != expected['unknown_scope']:
        raise ValueError('Unknown-region scope differs from the task')

    inference = cfg['inference']
    if (inference['num_frames'], inference['width'], inference['height'],
            inference['steps'], inference['sampler'], inference['shift'],
            inference['cfg'], inference['seed'], inference['best_of_n']) != (
            81, 832, 480, 50, 'shifted_euler', 5.0, 5.0, 42, 1):
        raise ValueError('Fixed comparison sampler changed')
    if bool(inference['clamp_first']) != expected['clamp_first']:
        raise ValueError('I2V must clamp the first latent; T2V must not')
    return cfg


def load_config(path, *, require_paths=False):
    """Load one YAML and add implementation-only path aliases."""
    source = yaml.safe_load(Path(path).read_text(encoding='utf8'))
    validate_config(source)
    cfg = deepcopy(source)
    missing = set()
    cfg = _substitute(cfg, missing)
    if require_paths and missing:
        raise RuntimeError('Set runtime path variables: ' + ', '.join(sorted(missing)))
    paths = cfg['paths']
    if '${' not in str(paths.get('run_dir', '')):
        paths['runs'] = str(Path(paths['run_dir']).parent)
        if not Path(paths['runs']).resolve().is_relative_to(ROOT):
            raise ValueError('Run directories must stay inside this experiment workspace')
    if '${' not in str(paths.get('interface_output', '')):
        paths['interface'] = paths['interface_output']
    paths['cache'] = paths['train_cache']
    paths['input_cache'] = paths.get('input_cache', str(I2V_INPUT_CACHE))
    paths['words_cache'] = paths.get('words_cache', str(I2V_WORDS_CACHE))
    paths['native_words_cache'] = paths.get(
        'native_words_cache', str(I2V_NATIVE_CACHE))
    paths['eval_input_cache'] = paths.get('eval_input_cache', str(I2V_INPUT_CACHE))
    paths['inference_evidence'] = paths.get(
        'inference_evidence', str(ROOT / 'prepared_inputs'))
    cfg['stages'] = {'joint': cfg['train']['max_updates']}
    cfg['train']['accumulation'] = cfg['train']['gradient_accumulation']
    cfg['train']['text_dropout'] = cfg['train']['condition_dropout']
    cfg['loss']['generation'] = {
        'tail_weight': cfg['loss']['tail'], 'tail_fraction': cfg['loss']['tail_fraction']}
    cfg['variant_id'] = 'R18' + cfg['plan']
    cfg['revision'] = REVISION
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
    normalized = re.sub(r'[^A-Z0-9]', '', name.upper())
    if 'H100' in normalized:
        return False
    if policy == 'ada-48g':
        # The two T2V candidate jobs are explicitly approved for the 48 GB
        # L40S / RTX 4090 pool, never for H100.
        return 'L40S' in normalized or 'RTX4090' in normalized
    return 'RTXPRO6000' in normalized or 'RTX6000BLACKWELL' in normalized


def gpu_guard(run):
    import torch
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if count else 'NO CUDA DEVICE'
    policy = os.getenv('GPU_POLICY', 'rtx6000')
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
            f'NoChunks {policy} runs require an allowed 48/96 GB GPU, got {name}')
    return record
