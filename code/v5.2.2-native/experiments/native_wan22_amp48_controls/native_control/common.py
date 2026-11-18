"""Native Wan2.2 amp-48g controls: I/O, GPU policy and data audit."""
from pathlib import Path
import hashlib
import json
import os
import re
import time
import uuid


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def atomic_torch(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
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
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False,
                   separators=(',', ':')).encode()).hexdigest()


def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def validate_lora_config(cfg):
    if cfg.get('schema_version') != 'native-wan22-lora-amp48-v1':
        raise ValueError('Unexpected control schema')
    if cfg['base']['model'] != 'Wan2.2-TI2V-5B':
        raise ValueError('This control is only for Wan2.2-TI2V-5B')
    if (cfg['train']['max_updates'], cfg['train']['gradient_accumulation'],
            cfg['train']['effective_global_batch']) != (400, 4, 4):
        raise ValueError('Control requires 400 updates and effective batch 4')
    if cfg['base']['video_lora_rank'] != 32 or cfg['base']['video_lora_alpha'] != 32:
        raise ValueError('Video LoRA must be rank32/alpha32')
    if cfg['train']['condition_dropout'] != 0.1:
        raise ValueError('Condition dropout must match the fixed R20 draws')
    if cfg['base']['freeze_text_qkvo'] is not True:
        raise ValueError('Native text Q/K/V/O must remain frozen')
    return cfg


def data_audit(cfg):
    from .data import TrainingData
    cache = Path(cfg['paths']['train_cache'])
    rows = json.loads((cache / 'base/manifest.json').read_text())['records']
    if len(rows) != 3857:
        raise ValueError('Unexpected WISA manifest size')
    train = [row for row in rows if row['split'] == 'train']
    sequence = json.loads(Path(cfg['train']['sample_sequence']).read_text())
    if len(sequence['ids']) != 1600 or len(sequence['drop']) != 1600:
        raise ValueError('Fixed sample sequence is not 1600')
    data = TrainingData(cfg)
    if len(data) != 1600:
        raise ValueError(f'TrainingData has {len(data)} records, expected 1600')
    return dict(
        manifest_records=len(rows), manifest_train_records=len(train),
        train_records=len(data), sequence_records=len(sequence['ids']),
        train_manifest_hash=sha256_file(cache / 'base/manifest.json'),
        sample_sequence_hash=sha256_file(cfg['train']['sample_sequence']),
        wan_index_hash=sha256_file(Path(cfg['paths']['wan_checkpoint']) /
                                   'diffusion_pytorch_model.safetensors.index.json'),
        data_revision=cfg['train'].get('data_revision'),
        cache=str(cache), sample_sequence=str(cfg['train']['sample_sequence']),
    )


def gpu_guard_amp48(run):
    """One >=44 GiB accelerator in amp-48g; reject H100 and multi-GPU runs."""
    import torch
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if count else 'NO CUDA DEVICE'
    props = torch.cuda.get_device_properties(0) if count else None
    memory_gib = float(props.total_memory / 2**30) if props else 0.0
    uuid = str(props.uuid) if props else ''
    if uuid and not uuid.startswith('GPU-'):
        uuid = 'GPU-' + uuid
    normalized = re.sub(r'[^A-Z0-9]', '', name.upper())
    accepted = (count == 1 and 'H100' not in normalized
                and memory_gib >= 44.0)
    record = dict(kind='gpu_policy', actual=name, uuid=uuid,
                  visible_count=count, memory_gib=memory_gib,
                  accepted=accepted, resource_pool='amp-48g',
                  experiment_id=os.getenv('DET_EXPERIMENT_ID'), time=time.time())
    path = Path(run) / 'gpu.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    if not accepted:
        raise RuntimeError(
            f'Native Wan2.2 controls require one non-H100 GPU with >=44 GiB; '
            f'got {name}, {memory_gib:.1f} GiB on {count} devices')
    return record
