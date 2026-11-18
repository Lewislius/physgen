"""Canonical R18 configuration, provenance and file checks.

The four YAML files keep the v9 architecture fields and replace only the
user-requested 400-update training/loss schedule.  Loading never changes a
model/training field.  Runtime only expands the documented ``${...}`` path
variables and adds file-location aliases used by the implementation.
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
SCHEMA = 'ABCD-official-fresh-lora-400-batch4-v5'
SCHEMA_1000 = 'ABCD-official-fresh-lora-1000-batch4-v4'
REVISION = 'physics_expert_abcd_fresh400_r1'
REVISION_1000 = 'physics_expert_abcd_fresh1000_r1'
VARIANTS = ('A', 'B', 'C', 'D')
EXPECTED_TARGETS = {
    'self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o', 'ffn.0', 'ffn.2',
}
SITES = {
    'A': [4, 9, 14, 19, 24, 29],
    'B': [2, 5, 8, 11, 14, 17, 20, 23, 26, 29],
    'C': [4, 9, 14, 19, 24, 29],
    'D': [4, 9, 14, 19, 24, 29],
}
FEATURE_CACHE = (BASE / 'experiments/physical_content_v7/data_revision_qwen38/runs/'
                 '20261103_dataset/encoded/content')
LEGACY_FEATURE_CACHE = BASE / 'experiments/physical_content_v7/cache'
SUPERVISION_CACHE = BASE / 'experiments/physical_variants/supervision_cache'
PERSISTED_NOISE = BASE / 'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
PROTOCOL_ROOT = BASE / 'experiments/physical_content_v7'
ROUTE_GROUPS = (
    'wan_video_lora', 'enhancement', 'physics_read', 'writer',
    'expert_lora', 'semantic_reader_lora', 'content_memory', 'semantic_guide',
)


def _substitute(value, missing):
    if isinstance(value, dict):
        return {key: _substitute(item, missing) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, missing) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name = match.group(1)
        if name not in os.environ:
            missing.add(name)
            return match.group(0)
        return os.environ[name]

    return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', replace, value)


def validate_config(cfg):
    """Reject any drift from the v9 design contract."""
    schema = cfg.get('schema_version')
    if schema not in (SCHEMA, SCHEMA_1000):
        raise ValueError(f'Use schema {SCHEMA} or {SCHEMA_1000}')
    max_updates = 1000 if schema == SCHEMA_1000 else 400
    variant = cfg.get('variant')
    if variant not in VARIANTS:
        raise ValueError('Variant must be one of A, B, C, D')
    init = cfg['initialization']
    if (init['mode'] != 'official_fresh' or init['project_checkpoint'] is not None
            or init['load_project_lora'] or init['load_project_optimizer']
            or init['interface_fit'] != 'recompute_train_only_closed_form'
            or not init['resume_same_run_only']):
        raise ValueError('R18 must start from official foundations only')

    model = cfg['model']
    if (model['width'], model['ffn_width'], model['native_layers']) != (3072, 14336, 30):
        raise ValueError('Only the verified Wan2.2-TI2V-5B shapes are allowed')
    if (model['qwen_width'], model['q_tokens'], model['q_heads'], model['grounded_heads']) != (4096, 256, 32, 16):
        raise ValueError('Keep the audited 256-slot Qwen content interface')
    if model['content_rounds'] != 2:
        raise ValueError('Exactly two shared content rounds')
    if model['native_base_trainable'] or model['native_text_lora'] or model['native_text_params_trainable']:
        raise ValueError('Native Wan base and native text attention must remain frozen')
    lora = model['native_video_lora']
    if (not lora['enabled'] or set(lora['targets']) != EXPECTED_TARGETS
            or lora['layers'] != list(range(30)) or lora['active_updates'] != [1, max_updates]
            or lora['dropout'] != 0.0):
        raise ValueError('Video LoRA must cover exactly the six allowlisted targets on all 30 layers for the full run')
    if model['sites_zero_based'] != SITES[variant]:
        raise ValueError('Injection sites differ from the selected plan')
    if model['sites_zero_based'] != [i for i in range(30) if (i + 1) % model['site_period'] == 0]:
        raise ValueError('Site period and site list disagree')
    if (model['expert_base_policy'] != 'shared_official_frozen'
            or not model['expert_lora_independent_from_native']):
        raise ValueError('Experts must share only frozen official bases and own independent adapters')
    if model['expert_norm_and_time_base_trainable']:
        raise ValueError('Expert norm/time bases are frozen')
    if model['physics_reader'] != {
            'k_v_trainable_dense': True,
            'query': 'official_frozen_native_cross_q',
            'value_source': 'raw_execute' if variant != 'C' else 'enhanced_execute',
            'output_coordinates': 'official_cross_o_no_bias',
            'reference_null_slot': False,
            'equal_group_prior': True,
            'old_state_value': False}:
        raise ValueError('Physical reader differs from the common contract')
    if model['write_head'] != {
            'width': 3072, 'bias': False, 'init': 'zeros', 'future_only': True,
            'inference_multiplier': 1.0}:
        raise ValueError('Final writer must be one zero-initialized full-width future-only map')
    if model['video_rope'] != 'native_grid_only':
        raise ValueError('Video RoPE is used only on the native spatio-temporal grid')
    if variant == 'A':
        if not model['semantic_reader'] or model['semantic_reader_lora_rank'] != 32:
            raise ValueError('A needs its independent semantic reader')
        if model['memory_enhancer']['enabled'] or model['semantic_guide']['enabled']:
            raise ValueError('A must not contain C memory or D guide')
        if model['expert_lora_rank'] != 32 or lora['rank'] != 32 or lora['rank64_layers']:
            raise ValueError('A uses rank32 main and expert adapters')
        if model['physical_value_executor_zero_preserving'] or model['ffn_conditional_mode'] != 'none':
            raise ValueError('A is a full joint executor, not a zero-preserving physical-value executor')
    elif variant == 'B':
        if not model['semantic_reader'] or model['semantic_reader_lora_rank'] != 32:
            raise ValueError('B needs its independent semantic reader')
        if model['memory_enhancer']['enabled'] or model['semantic_guide']['enabled']:
            raise ValueError('B must not contain C memory or D guide')
        if model['expert_lora_rank'] != 0 or model['expert_lora_alpha'] != 0:
            raise ValueError('B adds no separate video executor')
        if lora['rank'] != 32 or sorted(lora['rank64_layers']) != SITES['B']:
            raise ValueError('B rank64 belongs only to the ten injection layers')
        if model['physical_value_executor_zero_preserving'] or model['ffn_conditional_mode'] != 'none':
            raise ValueError('B executes through the native FFN, not a physical-value executor')
    elif variant == 'C':
        if model['semantic_reader'] or model['semantic_reader_lora_rank'] != 0:
            raise ValueError('C has no independent video semantic reader')
        mem = model['memory_enhancer']
        if (not mem['enabled'] or mem['source_layer_zero_based'] != 14
                or mem['rounds_shared'] != 2 or mem['lora_rank'] != 32
                or mem['route_projection_init'] != 'identity'):
            raise ValueError('C memory enhancer differs from the design')
        if model['semantic_guide']['enabled']:
            raise ValueError('C must not contain D guide')
        if model['expert_lora_rank'] != 32 or lora['rank'] != 32 or lora['rank64_layers']:
            raise ValueError('C uses rank32 main and expert adapters')
        if not model['physical_value_executor_zero_preserving']:
            raise ValueError('C physical value executor must be zero-preserving')
        if model['ffn_conditional_mode'] != 'none':
            raise ValueError('C FFN has no semantic channel guide')
    else:
        if model['semantic_reader'] or model['semantic_reader_lora_rank'] != 0:
            raise ValueError('D has no independent video semantic reader')
        guide = model['semantic_guide']
        if (not guide['enabled'] or guide['maps'] != ['q', 'k', 'ffn_channels']
                or guide['width'] != 3072 or guide['init'] != 'zeros'
                or guide['detach_native_text_update']):
            raise ValueError('D guide differs from the design')
        if model['memory_enhancer']['enabled']:
            raise ValueError('D must not contain C memory')
        if model['expert_lora_rank'] != 32 or lora['rank'] != 32 or lora['rank64_layers']:
            raise ValueError('D uses rank32 main and expert adapters')
        if not model['physical_value_executor_zero_preserving']:
            raise ValueError('D physical value executor must be zero-preserving')
        if model['ffn_conditional_mode'] != 'source_scale_only':
            raise ValueError('D FFN only source-scales channels; no additive time shift')

    train = cfg['train']
    if train['max_updates'] != max_updates or train['freeze_wan_lora_updates'] != 0:
        raise ValueError(f'Exactly {max_updates} updates, with video LoRA active from update 1')
    if train['effective_global_batch'] != 4:
        raise ValueError('Effective global batch must be 4')
    if train['world_size'] * train['microbatch_per_device'] * train['gradient_accumulation'] != 4:
        raise ValueError('world_size * microbatch * accumulation must equal 4')
    if train['world_size'] != 1 or train['microbatch_per_device'] != 1 or train['gradient_accumulation'] != 4:
        raise ValueError('This implementation uses the default single-GPU micro1 x accumulation4')
    if train['post_training'] or train['extra_training_views'] or train['condition_dropout'] != 0.1:
        raise ValueError('No post-training, extra views, or altered condition dropout')
    schedule = train['lr_schedule']
    expected_schedule = (
        {'warmup_updates': 32, 'decay_start_update': 128, 'decay_end_update': 320,
         'min_ratio': 0.2, 'final_phase_start_update': 321, 'final_ratio': 0.2}
        if max_updates == 400 else
        {'warmup_updates': 64, 'decay_start_update': 200, 'min_ratio': 0.2})
    if schedule != expected_schedule:
        raise ValueError(f'{max_updates}-update LR schedule differs from the contract')
    if set(train['peak_lrs']) != set(ROUTE_GROUPS):
        raise ValueError('Need exactly one peak LR per parameter role')
    expected_keep = ([32, 128, 200, 320, 400] if max_updates == 400
                     else [100, 250, 500, 750, 1000])
    if (train['keep_updates'] != expected_keep
            or train.get(f'resume_over_{max_updates}') != 'forbidden'
            or train['save_every'] != 100):
        raise ValueError('Checkpoint cadence differs from the contract')
    loss = cfg['loss']
    if (abs(loss['fm_weight'] - 0.8) > 1e-12 or abs(loss['tail_weight'] - 0.2) > 1e-12
            or loss['tail_fraction'] != 0.25 or not loss['exclude_known_first_frame']
            or not loss['apply_valid_mask']):
        raise ValueError('Generation loss must remain 0.8 FM + 0.2 tail on future valid tokens')
    if any(loss[key] != 0 for key in ('preference_weight', 'reward_weight', 'temporal_weight',
                                      'rgb_weight', 'teacher_feature_weight')):
        raise ValueError('No auxiliary objectives beyond the declared FM/tail/raw-route loss')
    route = loss['route']
    expected_route = (
        {'enabled': True, 'target': 'raw_route_only', 'peak': 0.03,
         'warmup_updates': 32, 'hold_end_update': 128,
         'decay_end_update': 320, 'final_weight': 0.01, 'after_final': 0.01}
        if max_updates == 400 else
        {'enabled': True, 'target': 'raw_route_only', 'peak': 0.03,
         'warmup_updates': 32, 'hold_end_update': 64,
         'zero_at_update': 320, 'after_zero': 0.0})
    if route != expected_route:
        raise ValueError(f'Route loss must use raw Cr with the {max_updates}-update schedule')
    inference = cfg['inference']
    if inference != {
            'task': 'historical_i2v', 'rgb_frames': 81, 'height': 480, 'width': 832,
            'steps': 50, 'shift': 5.0, 'cfg': 5.0, 'sampler': 'shifted_euler',
            'seed': 42, 'best_of_n': 1, 'external_guidance_optimizer': False}:
        raise ValueError('Fixed I2V inference protocol changed')
    return cfg


def load_config(path, *, require_paths=False):
    """Load one canonical YAML and add implementation-only path aliases."""
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
            raise ValueError('R18 run directories must stay inside this experiment workspace')
    if '${' not in str(paths.get('interface_output', '')):
        paths['interface'] = paths['interface_output']
    paths['cache'] = paths['train_cache']
    paths['feature_cache'] = str(FEATURE_CACHE)
    paths['supervision_cache'] = str(SUPERVISION_CACHE)
    paths['inference_evidence'] = str(ROOT / 'prepared_inputs')
    paths['source_snapshot'] = str(ROOT)
    paths['persisted_noise'] = str(PERSISTED_NOISE)
    cfg['stages'] = {'joint': cfg['train']['max_updates']}
    cfg['train']['accumulation'] = cfg['train']['gradient_accumulation']
    cfg['train']['text_dropout'] = cfg['train']['condition_dropout']
    cfg['train']['warmup'] = cfg['train']['lr_schedule']['warmup_updates']
    cfg['loss']['generation'] = {
        'tail_weight': cfg['loss']['tail_weight'],
        'tail_fraction': cfg['loss']['tail_fraction'],
    }
    cfg['loss']['semantic_target'] = 'raw_route_only'
    cfg['revision'] = (REVISION_1000 if cfg['schema_version'] == SCHEMA_1000
                       else REVISION)
    cfg['variant_id'] = 'R18' + cfg['variant']
    cfg['source_snapshot'] = str(ROOT)
    return cfg




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

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def design_hash(cfg):
    """Hash the design/training contract without runtime path expansion."""
    value = deepcopy(cfg)
    value.pop('paths', None)
    value.pop('source_snapshot', None)
    value.pop('revision', None)
    value.pop('variant_id', None)
    value.pop('stages', None)
    for key in ('accumulation', 'text_dropout', 'warmup'):
        value.get('train', {}).pop(key, None)
    value.get('loss', {}).pop('generation', None)
    value.get('loss', {}).pop('semantic_target', None)
    return canonical_hash(value)


def data_ready(cfg):
    cache = Path(cfg['paths']['cache'])
    interface = Path(cfg['paths']['interface'])
    required = [cache / 'base/manifest.json', cache / 'base/null_text.pt', interface]
    for variant_root in [Path(cfg['paths']['wan_checkpoint']), Path(cfg['paths']['qwen_checkpoint'])]:
        index = variant_root / ('diffusion_pytorch_model.safetensors.index.json'
                                if variant_root == Path(cfg['paths']['wan_checkpoint'])
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
    if len(available) < 1:
        raise ValueError('No training record has complete frozen input evidence')
    return dict(
        manifest_records=len(rows),
        manifest_train_records=len(train),
        train_records=len(available),
        unavailable_train_records=len(train) - len(available),
        validation_records=len(rows) - len(train),
        interface=str(interface),
        interface_hash=sha256_file(interface),
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
        revision=cfg['revision'],
        variant=cfg['variant'],
        variant_id=cfg['variant_id'],
        sites=cfg['model']['sites_zero_based'],
        max_updates=cfg['train']['max_updates'],
        effective_batch=cfg['train']['effective_global_batch'],
        initialization='official Wan/Qwen/T5/VAE only; fresh project weights',
        native_text_lora=False,
        old_project_checkpoint=False,
        route_target='raw_route_only',
        extra_losses=False,
        post_training=False,
        quality_proven=False,
    )
    return result




def is_allowed_gpu(name):
    normalized = re.sub(r'[^A-Z0-9]', '', name.upper())
    return 'H100' not in normalized and (
        'RTXPRO6000' in normalized or 'RTX6000BLACKWELL' in normalized)


def gpu_guard(run):
    """Formal R18 training/evaluation must use a 96 GB RTX PRO 6000 slot."""
    import torch
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if count else 'NO CUDA DEVICE'
    accepted = count == 1 and is_allowed_gpu(name)
    record = dict(
        kind='gpu_policy', actual=name, visible_count=count, accepted=accepted,
        resource_pool='blk-96g', experiment_id=os.getenv('DET_EXPERIMENT_ID'),
    )
    path = Path(run) / 'gpu.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    if not accepted:
        raise SystemExit(86)
    return record

def gpu_record(run):
    import torch
    record = dict(
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        count=torch.cuda.device_count(),
        experiment_id=os.getenv('DET_EXPERIMENT_ID'),
    )
    path = Path(run) / 'gpu.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + '\n')
    return record
