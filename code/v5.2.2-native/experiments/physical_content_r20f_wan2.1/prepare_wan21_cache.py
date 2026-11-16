"""Build the Wan2.1-T2V-1.3B training/evaluation cache for R20FT.

Stages
------
base
    Link the audited WISA manifest, UMT5 text features and null text from the
    existing Wan2.2 data cache.  The UMT5 checkpoint is byte-identical between
    Wan2.1 and Wan2.2, so reusing the 4096-wide text features is exact.
vae
    Decode the exact 81-frame WISA windows and encode them with the official
    Wan2.1 VAE (z=16, stride 4x8x8) into [1, 16, 21, 60, 104] latents.
native
    Project the established Qwen reasoning relations through the official
    Wan2.1 UMT5 + text_embedding into 1536-wide native word features.
inference
    Rebuild the 60 fixed VideoPhy2 inference arms with 1536-wide
    native_condition tensors while keeping the unchanged Qwen features.
all
    Run base, vae, native and inference in that order.

No API calls and no future-video target are used.  The Wan2.2 VAE cache and the
Wan2.2 3072-wide native words are never loaded into the Wan2.1 model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

import numpy as np
import torch

from physical_content.common import atomic_json, atomic_torch, load_config
from physical_content.wan21_runtime import install_wan21_namespace

SOURCE_CACHE = (bootstrap.BASE / 'experiments/physical_content_v7/data_revision_qwen38'
                / 'runs/20261103_dataset/encoded')
ENCODING = 'Wan21-R20FT-20261116'
NATIVE_WIDTH = 1536


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def _link_or_copy(source, target):
    source, target = Path(source), Path(target)
    if target.is_symlink():
        if Path(os.readlink(target)).resolve() != source.resolve():
            raise ValueError(f'Existing link points elsewhere: {target}')
        return
    if target.exists():
        if source.is_dir() and target.is_dir():
            return
        if source.is_file() and target.is_file() and _sha256(source) == _sha256(target):
            return
        raise ValueError(f'Existing cache file differs: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        os.symlink(source.resolve(), target, target_is_directory=True)
    else:
        os.symlink(source.resolve(), target)


def stage_base(cfg, output):
    source = Path(cfg['paths'].get('source_cache', SOURCE_CACHE))
    if not (source / 'base/manifest.json').is_file():
        raise FileNotFoundError(source / 'base/manifest.json')
    destination = output / 'base'
    destination.mkdir(parents=True, exist_ok=True)
    _link_or_copy(source / 'base/manifest.json', destination / 'manifest.json')
    _link_or_copy(source / 'base/null_text.pt', destination / 'null_text.pt')
    _link_or_copy(source / 'base/text', destination / 'text')
    (destination / 'vae').mkdir(parents=True, exist_ok=True)
    t5 = Path(cfg['paths']['wan_checkpoint']) / 'models_t5_umt5-xxl-enc-bf16.pth'
    vae = Path(cfg['paths']['wan_checkpoint']) / 'Wan2.1_VAE.pth'
    atomic_json(dict(
        revision=ENCODING,
        source_5b_cache=str(source),
        manifest='audited WISA records and unchanged 1600 order',
        text='reused 4096-wide UMT5 features; T5 checkpoint sha256=' + _sha256(t5),
        null_text='reused 4096-wide UMT5 null feature',
        vae='Wan2.1 VAE z=16 stride 4x8x8',
        vae_checkpoint=str(vae),
        native_width=NATIVE_WIDTH,
    ), destination / 'ready.json')
    print(json.dumps(dict(stage='base', output=str(destination))), flush=True)


def _decode_clip(row):
    """Decode the exact contiguous 81-frame WISA window to [3,81,480,832]."""
    geometry = row['geometry']
    indices = [int(value) for value in row['indices']]
    if len(indices) != 81 or indices != list(range(indices[0], indices[0] + 81)):
        raise ValueError(f'{row["id"]}: WISA window is not 81 contiguous frames')
    if (int(geometry['h']), int(geometry['w'])) != (480, 832):
        raise ValueError(f'{row["id"]}: unexpected target geometry')
    filters = (
        f"select='between(n\\,{indices[0]}\\,{indices[-1]})',"
        f"scale={geometry['resize_w']}:{geometry['resize_h']}:flags=bilinear,"
        f"crop={geometry['w']}:{geometry['h']}:{geometry['crop_left']}:"
        f"{geometry['crop_top']}")
    command = ['ffmpeg', '-v', 'error', '-i', row['video'], '-vf', filters,
               '-vsync', '0', '-frames:v', '81', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-']
    result = subprocess.run(command, check=True, capture_output=True)
    expected = 81 * int(geometry['h']) * int(geometry['w']) * 3
    if len(result.stdout) != expected:
        raise ValueError(f'{row["id"]}: decoded {len(result.stdout)} bytes, expected {expected}')
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        81, int(geometry['h']), int(geometry['w']), 3).astype(np.float32)
    video = torch.from_numpy(frames).permute(0, 3, 1, 2) / 127.5 - 1.0
    return video.permute(1, 0, 2, 3).contiguous()


def stage_vae(cfg, output, limit=None):
    source = Path(cfg['paths'].get('source_cache', SOURCE_CACHE))
    rows = json.loads((source / 'base/manifest.json').read_text())['records']
    by_id = {row['id']: row for row in rows}
    sequence = json.loads(Path(cfg['train']['sample_sequence']).read_text())['ids']
    ids = list(dict.fromkeys(sequence))
    if len(ids) != 1600:
        raise ValueError(f'Expected 1600 fixed training ids, got {len(ids)}')
    if limit is not None:
        ids = ids[:int(limit)]
    install_wan21_namespace(cfg['paths']['wan_code'])
    from wan.modules.vae import WanVAE

    vae = WanVAE(z_dim=16,
                 vae_pth=str(Path(cfg['paths']['wan_checkpoint']) / 'Wan2.1_VAE.pth'),
                 dtype=torch.bfloat16, device='cuda')
    destination = output / 'base/vae'
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for index, sample_id in enumerate(ids):
        target = destination / f'{sample_id}.pt'
        if target.is_file():
            saved = torch.load(target, map_location='cpu', weights_only=True)
            if tuple(saved['latent'].shape) != (1, 16, 21, 60, 104):
                raise ValueError(f'{sample_id}: existing Wan2.1 latent has wrong shape')
            continue
        row = by_id[sample_id]
        video = _decode_clip(row).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            latent = vae.encode([video])[0].float().cpu()[None]
        if tuple(latent.shape) != (1, 16, 21, 60, 104):
            raise ValueError(f'{sample_id}: Wan2.1 VAE produced {tuple(latent.shape)}')
        atomic_torch(dict(
            latent=latent,
            geometry=row['geometry'],
            source_video=row['video'],
            source_indices=row['indices'],
            revision=ENCODING,
            vae='official Wan2.1 VAE z=16 stride 4x8x8',
        ), target)
        if (index + 1) % 25 == 0:
            print(json.dumps(dict(stage='vae', done=index + 1, total=len(ids),
                                  seconds=round(time.monotonic() - started, 1))), flush=True)
    total = len(list(destination.glob('*.pt')))
    atomic_json(dict(status=('completed' if total == 1600 else 'partial'),
                     stage='vae', records=1600, files=total,
                     revision=ENCODING),
                output / 'base/vae/ready.json')
    print(json.dumps(dict(stage='vae', done=len(ids), files=total)), flush=True)


def _native_encoder(cfg):
    install_wan21_namespace(cfg['paths']['wan_code'])
    from torch import nn
    from safetensors import safe_open

    checkpoint = Path(cfg['paths']['wan_checkpoint'])
    model = nn.Sequential(nn.Linear(4096, NATIVE_WIDTH),
                          nn.GELU(approximate='tanh'),
                          nn.Linear(NATIVE_WIDTH, NATIVE_WIDTH))
    state = {}
    for key in model.state_dict():
        full = 'text_embedding.' + key
        path = checkpoint / 'diffusion_pytorch_model.safetensors'
        if not path.is_file():
            raise FileNotFoundError(path)
        with safe_open(str(path), framework='pt', device='cpu') as reader:
            state[key] = reader.get_tensor(full)
    model.load_state_dict(state)
    return model.to(device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)


def _t5_encoder(cfg):
    install_wan21_namespace(cfg['paths']['wan_code'])
    from wan.modules.t5 import T5EncoderModel

    checkpoint = Path(cfg['paths']['wan_checkpoint'])
    return T5EncoderModel(
        text_len=512, dtype=torch.bfloat16, device='cuda',
        checkpoint_path=str(checkpoint / 'models_t5_umt5-xxl-enc-bf16.pth'),
        tokenizer_path=str(checkpoint / 'google/umt5-xxl'))


def _encode_relations(encoder, native, relations, word_strings):
    if not any(str(value).strip() for value in relations):
        return torch.zeros((len(word_strings), NATIVE_WIDTH), dtype=torch.bfloat16)
    clean = [encoder.tokenizer._clean(text) for text in relations]
    tokens = encoder.tokenizer.tokenizer(
        clean, return_tensors='pt', return_offsets_mapping=True,
        add_special_tokens=True, padding=True, truncation=False)
    spans = tokens.pop('offset_mapping')
    if tokens['input_ids'].shape[1] > 512:
        raise ValueError('Native Wan2.1 relation text exceeds the 512-token budget')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = encoder.model(tokens['input_ids'].cuda(),
                               tokens['attention_mask'].cuda())
        target = native(hidden).float()
    features = []
    for group, text in enumerate(clean):
        for match in re.finditer(r'\S+', text):
            index = ((spans[group, :, 1] > match.start())
                     & (spans[group, :, 0] < match.end())).nonzero().flatten()
            if not len(index):
                raise ValueError('A native Wan2.1 word has no T5 tokens')
            features.append(target[group, index.cuda()].mean(0))
    if len(features) != len(word_strings):
        raise ValueError(
            f'Qwen/T5 Wan2.1 lexical word mapping differs: {len(features)} != {len(word_strings)}')
    return torch.stack(features).cpu().to(torch.bfloat16)


def stage_native(cfg, output, limit=None):
    words_root = Path(cfg['paths']['words_cache'])
    input_root = Path(cfg['paths']['input_cache'])
    names = sorted(path.stem for path in words_root.glob('*.pt'))
    if len(names) != 3320:
        raise ValueError(f'Expected 3320 train/inference word arms, got {len(names)}')
    if limit is not None:
        names = names[:int(limit)]
    encoder, native = _t5_encoder(cfg), _native_encoder(cfg)
    destination = output / 'native_words'
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for index, name in enumerate(names):
        target = destination / f'{name}.pt'
        if target.is_file():
            saved = torch.load(target, map_location='cpu', weights_only=True)
            if tuple(saved['native_words'].shape) != (len(saved['word_strings']), NATIVE_WIDTH):
                raise ValueError(f'{name}: existing Wan2.1 native words have wrong shape')
            continue
        words = torch.load(words_root / f'{name}.pt', map_location='cpu', weights_only=True)
        source = torch.load(input_root / f'{name}.pt', map_location='cpu', weights_only=True)
        relations = source.get('relations')
        strings = words['word_strings']
        if not isinstance(relations, list) or len(relations) == 0:
            raise ValueError(f'{name}: missing relations')
        if list(strings) != list(source.get('word_strings', [])):
            raise ValueError(f'{name}: Qwen words and input relations disagree')
        value = _encode_relations(encoder, native, relations, strings)
        atomic_torch(dict(
            native_words=value,
            word_strings=list(strings),
            relations=list(relations),
            caption=source.get('caption', ''),
            scope=words['scope'],
            sample=words['sample'],
            teacher='official frozen Wan2.1 UMT5 then official Wan2.1 text_embedding',
            revision=ENCODING,
        ), target)
        if (index + 1) % 100 == 0:
            print(json.dumps(dict(stage='native', done=index + 1, total=len(names),
                                  seconds=round(time.monotonic() - started, 1))), flush=True)
    total = len(list(destination.glob('*.pt')))
    atomic_json(dict(status=('completed' if total == 3320 else 'partial'),
                     stage='native', records=3320, width=NATIVE_WIDTH,
                     revision=ENCODING, files=total),
                output / 'native_words/ready.json')
    print(json.dumps(dict(stage='native', done=len(names), files=total)), flush=True)


def stage_inference(cfg, output, limit=None):
    source = Path(cfg['paths']['input_cache']).parent / 'inference'
    if not source.is_dir():
        raise FileNotFoundError(source)
    encoder, native = _t5_encoder(cfg), _native_encoder(cfg)
    destination = output / 'inference'
    folders = sorted(path for path in source.iterdir() if path.is_dir())
    if len(folders) != 60:
        raise ValueError(f'Expected 60 fixed inference arms, got {len(folders)}')
    if limit is not None:
        folders = folders[:int(limit)]
    started = time.monotonic()
    for index, folder in enumerate(folders):
        target = destination / folder.name
        target.mkdir(parents=True, exist_ok=True)
        ready = json.loads((folder / 'ready.json').read_text())
        ready.update(revision=ENCODING,
                     native_encoder='official Wan2.1 UMT5 + text_embedding (1536)')
        atomic_json(ready, target / 'ready.json')
        for arm in ('positive.pt', 'negative.pt'):
            value = torch.load(folder / arm, map_location='cpu', weights_only=True)
            relations = value['relations']
            strings = value['word_strings']
            condition = _encode_relations(encoder, native, relations, strings)
            value.update(native_condition=condition,
                         native_condition_source='official Wan2.1 UMT5 + text_embedding',
                         revision=ENCODING)
            atomic_torch(value, target / arm)
        if (index + 1) % 10 == 0:
            print(json.dumps(dict(stage='inference', done=index + 1, total=len(folders),
                                  seconds=round(time.monotonic() - started, 1))), flush=True)
    total = len(list(destination.glob('*/positive.pt')))
    atomic_json(dict(status=('completed' if total == 60 else 'partial'),
                     stage='inference', records=60, revision=ENCODING,
                     native_width=NATIVE_WIDTH, files=total),
                destination / 'ready.json')
    print(json.dumps(dict(stage='inference', done=len(folders), files=total)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('base', 'vae', 'native', 'inference', 'all'),
                        default='all')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg['task'] != 't2v':
        raise ValueError('R20FT-Wan2.1 cache is text-to-video only')
    output = args.output or Path(cfg['paths']['train_cache'])
    output.mkdir(parents=True, exist_ok=True)
    stages = ('base', 'vae', 'native', 'inference') if args.stage == 'all' else (args.stage,)
    for stage in stages:
        if stage == 'base':
            stage_base(cfg, output)
        elif stage == 'vae':
            stage_vae(cfg, output, args.limit)
        elif stage == 'native':
            stage_native(cfg, output, args.limit)
        elif stage == 'inference':
            stage_inference(cfg, output, args.limit)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps(dict(status='completed', stages=list(stages), output=str(output))), flush=True)


if __name__ == '__main__':
    main()
