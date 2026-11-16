"""R20FT-on-Wan2.1 configuration, provenance and one-GPU policy.

The R20FT design is unchanged.  Every Wan2.2 dimension that was implicit in the
original implementation is replaced by an explicit Wan2.1-T2V-1.3B value:
width 1536, FFN 8960, 12 heads, 30 layers, VAE stride 4x8x8 and latent shape
(16, 21, 60, 104).  There is intentionally no accepted 3072/14336/48-channel
Wan2.2 configuration in this candidate.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import yaml

from .wan21_loader import checkpoint_hash, checkpoint_kind

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
SCHEMA = 'R20-grounded-physical-response-wan21-v1'
REVISION = 'R20-grounded-physical-response-wan21-20261116'
WAN21_MODEL = 'Wan2.1-T2V-1.3B'
WAN21_WIDTH = 1536
WAN21_FFN = 8960
WAN21_LAYERS = 30
WAN21_HEADS = 12
WAN21_LATENT = (16, 21, 60, 104)
ROUTE_GROUPS = ('wan_video_lora', 'enhancement', 'physics_read', 'physics_write', 'physics_video')


def validate_config(cfg):
    if cfg['schema_version'] != SCHEMA or cfg['revision'] != REVISION:
        raise ValueError('R20FT-Wan2.1 configuration revision differs')
    if (cfg['plan'], cfg['task']) != ('FT', 't2v'):
        raise ValueError('R20FT-Wan2.1 is the authorized text-to-video candidate')
    if cfg['initialization']['mode'] != 'official_fresh' or cfg['initialization']['project_checkpoint'] is not None:
        raise ValueError('Fresh official Wan2.1 foundations; no project checkpoint initialization')
    if not cfg['base']['freeze_text_qkvo']:
        raise ValueError('Native text projections stay frozen')
    if cfg['base']['model'] != WAN21_MODEL:
        raise ValueError('The backbone must be the official Wan2.1-T2V-1.3B')
    model = cfg['model']
    if int(model['width']) != WAN21_WIDTH or int(model['ffn_width']) != WAN21_FFN:
        raise ValueError('Wan2.1 width/FFN dimensions are required')
    if int(model['native_layers']) != WAN21_LAYERS or int(model['native_heads']) != WAN21_HEADS:
        raise ValueError('Wan2.1 depth/head configuration is required')
    if int(model['physics_width']) != WAN21_WIDTH:
        raise ValueError('Physical C must use the full Wan2.1 width (1536)')
    if int(model['qwen_width']) != 4096 or int(model['q_tokens']) != 256:
        raise ValueError('Qwen interface and 256 physical slots are required')
    if int(model['q_heads']) != 32 or int(model['grounded_heads']) != 16:
        raise ValueError('R20FT content attention head split differs')
    if int(model['content_rounds']) != 2:
        raise ValueError('R20FT uses two normalized content rounds')
    if model['conditioned_ffn_rank'] is not None:
        raise ValueError('R20FT has no low-dimensional conditioning bottleneck')
    if cfg['mechanism'] != 26 or int(model['extra_video_blocks']) != 5:
        raise ValueError('R20FT requires five depth-matched physical blocks')
    if model['injection_layers'] != [0, 7, 14, 21, 28]:
        raise ValueError('R20FT distributed physical placement differs')
    if model['physical_initialization_blocks'] != model['injection_layers']:
        raise ValueError('Physical blocks must copy the corresponding pretrained depth')
    if model['content_anchor'] != 'native_input_words_normalized_query_enhancement':
        raise ValueError('Native source values must be selected by learned queries')
    if list(cfg['data']['latent_shape']) != list(WAN21_LATENT):
        raise ValueError('Wan2.1 VAE latent shape must be [16, 21, 60, 104]')
    if not str(cfg['paths']['wan_code']).rstrip('/').endswith('Wan2.1'):
        raise ValueError('paths.wan_code must point at the Wan2.1 source tree')
    if Path(cfg['paths']['wan_checkpoint']).name != WAN21_MODEL:
        raise ValueError('paths.wan_checkpoint must be the Wan2.1-T2V-1.3B directory')
    train = cfg['train']
    if (train['max_updates'], train['effective_global_batch'],
            train['microbatch_per_device'], train['gradient_accumulation']) != (400, 4, 1, 4):
        raise ValueError('Use 400 updates and effective batch 4')
    if train.get('gpu_policy', 'any48') not in ('any48', 'ada48', 'rtx6000'):
        raise ValueError('Unknown single-GPU policy')
    if set(train['peak_lrs']) != set(ROUTE_GROUPS):
        raise ValueError('Parameter roles differ')
    if int(cfg['base']['video_lora_rank']) != 32 or int(cfg['base']['video_lora_alpha']) != 32:
        raise ValueError('Wan2.1 main video LoRA is rank32/alpha32')
    if bool(cfg['input']['external_image']) or cfg['input']['evidence_scope'] != 'request_only':
        raise ValueError('R20FT-Wan2.1 is request-only text-to-video')
    if int(cfg['input']['known_latent_frames']) != 0 or cfg['input']['future_targets_in_condition']:
        raise ValueError('No first image or future target may enter T2V conditions')
    inference = cfg['inference']
    if (int(inference['num_frames']), int(inference['width']), int(inference['height']),
            int(inference['steps']), float(inference['shift']), float(inference['cfg']),
            float(inference['fps']), int(inference['seed'])) != (81, 832, 480, 50, 5.0, 5.0, 24.0, 42):
        raise ValueError('Wan2.1 fixed VideoPhy2 inference protocol differs')
    if int(inference.get('best_of_n', 1)) != 1 or bool(inference.get('clamp_first', False)):
        raise ValueError('Use one sample per prompt and no first-frame clamp')
    return cfg


def load_config(path, *, require_paths=False):
    cfg = deepcopy(yaml.safe_load(Path(path).read_text()))
    validate_config(cfg)
    paths = cfg['paths']
    paths['cache'] = paths['train_cache']
    paths['interface'] = paths['interface_output']
    paths['runs'] = str(Path(paths['run_dir']).parent)
    cfg['variant_id'] = 'R20FT-Wan21'
    interface = int(cfg['train'].get('interface_updates', 0))
    cfg['stages'] = {'interface': interface, 'joint': 400 - interface}
    cfg['train']['accumulation'] = cfg['train']['gradient_accumulation']
    cfg['train']['text_dropout'] = cfg['train']['condition_dropout']
    cfg['loss']['generation'] = {'tail_weight': cfg['loss']['tail'],
                                 'tail_fraction': cfg['loss']['tail_fraction']}
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


def _require_official_checkpoint(root, *, wan):
    root = Path(root)
    index = root / 'diffusion_pytorch_model.safetensors.index.json'
    single = root / 'diffusion_pytorch_model.safetensors'
    if wan:
        if index.is_file() or single.is_file():
            return
        raise FileNotFoundError(index)
    if not index.is_file():
        raise FileNotFoundError(index)


def torch_load_latent(path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'latent' not in state:
        raise KeyError(f'{path}: no latent tensor')
    return state['latent']


def data_ready(cfg):
    cache = Path(cfg['paths']['cache'])
    interface = Path(cfg['paths']['interface'])
    wan = Path(cfg['paths']['wan_checkpoint'])
    qwen = Path(cfg['paths']['qwen_checkpoint'])
    required = [cache / 'base/manifest.json', cache / 'base/null_text.pt',
                cache / 'base/text', cache / 'base/vae', interface,
                wan / 'config.json', wan / 'Wan2.1_VAE.pth',
                wan / 'models_t5_umt5-xxl-enc-bf16.pth',
                qwen / 'model.safetensors.index.json']
    _require_official_checkpoint(wan, wan=True)
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    rows = json.loads((cache / 'base/manifest.json').read_text())['records']
    if len(rows) != 3857:
        raise ValueError('The audited WISA train cache must contain 3857 records')
    train = [row for row in rows if row['split'] == 'train']
    if not train:
        raise ValueError('Training split is empty')
    from .data import TrainingData
    available = TrainingData(cfg)
    if len(available) != 1600:
        raise ValueError(f'Expected 1600 fixed training inputs, got {len(available)}')
    missing_text = [row['id'] for row in available.rows
                    if not (cache / 'base/text' / f'{row["id"]}.pt').is_file()]
    if missing_text:
        raise FileNotFoundError(f'Missing UMT5 text cache: {missing_text[0]}')
    missing_vae = [row['id'] for row in available.rows
                   if not (cache / 'base/vae' / f'{row["id"]}.pt').is_file()]
    if missing_vae:
        raise FileNotFoundError(f'Missing Wan2.1 VAE latent: {missing_vae[0]}')
    sample_latent = torch_load_latent(cache / 'base/vae' / f'{available.rows[0]["id"]}.pt')
    if tuple(sample_latent.shape) != (1,) + tuple(cfg['data']['latent_shape']):
        raise ValueError('First Wan2.1 latent has the wrong shape')
    wan_hash = checkpoint_hash(wan)
    qwen_index = qwen / 'model.safetensors.index.json'
    return dict(
        manifest_records=len(rows), manifest_train_records=len(train),
        train_records=len(available), unavailable_train_records=len(train) - len(available),
        validation_records=len(rows) - len(train),
        interface=str(interface), interface_hash=sha256_file(interface),
        train_manifest_hash=sha256_file(cache / 'base/manifest.json'),
        wan_checkpoint_hash=wan_hash,
        wan_checkpoint_kind=checkpoint_kind(wan),
        qwen_index_hash=sha256_file(qwen_index),
        data_revision=cfg['train'].get('data_revision'),
        latent_shape=list(cfg['data']['latent_shape']),
        source='Wan2.1 VAE latents plus the frozen Qwen/T5 input caches',
    )


def audit(cfg):
    result = data_ready(cfg)
    result.update(
        revision=REVISION, plan=cfg['plan'], design_plan=cfg['design_plan'],
        task=cfg['task'], mechanism=cfg['mechanism'],
        sites=cfg['model']['injection_layers'],
        max_updates=cfg['train']['max_updates'],
        effective_batch=cfg['train']['effective_global_batch'],
        initialization='official Wan2.1-T2V-1.3B/Qwen3-VL/UMT5/VAE only; fresh project weights',
        native_text_lora=False, old_project_checkpoint=False,
        local_semantic_loss=False, st_loss=False, post_training=False,
        quality_proven=False,
    )
    return result


def is_allowed_gpu(name, policy='any48'):
    normalized = re.sub(r'[^A-Z0-9]', '', name.upper())
    if 'H100' in normalized:
        return False
    if policy == 'rtx6000':
        return 'RTXPRO6000' in normalized or 'RTX6000BLACKWELL' in normalized
    if policy == 'ada48':
        return 'L40S' in normalized or 'RTX4090' in normalized
    if policy == 'any48':
        return any(token in normalized for token in
                   ('RTXPRO6000', 'RTX6000BLACKWELL', 'L40S', 'RTX4090',
                    'A6000', 'A100', 'A800'))
    raise ValueError(policy)


def gpu_guard(run, policy='any48'):
    import torch
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if count else 'NO CUDA DEVICE'
    accepted = count == 1 and is_allowed_gpu(name, policy)
    record = dict(kind='gpu_policy', actual=name, visible_count=count,
                  accepted=accepted, policy=policy,
                  experiment_id=os.getenv('DET_EXPERIMENT_ID'))
    path = Path(run) / 'gpu.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    if not accepted:
        raise RuntimeError(
            f'R20FT-Wan2.1 requires one allowed 48GB+ GPU, got {name} (policy={policy})')
    return record
