"""Configuration and provenance checks for R15Bt (T2V)."""
from pathlib import Path
import json

import yaml

from physical_scaling.common import (Log, atomic_json, atomic_torch, cpu_tree,
                                     gpu_guard)
from physical_scaling.common import audit as original_audit

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/r15bt')
REVISION = 'physical_content_v7_request_bound_fusion_r15bt_t2v'
SOURCE_EVIDENCE_REVISION = 'NoChunks-v2-20261111'
DATA_REVISION = 'request_only_qwen38_wisa_reviewed_v1'


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get('revision') != REVISION or cfg.get('variant') != 'physical_content':
        raise ValueError('Use the R15Bt T2V revision')
    if cfg.get('task') != 't2v':
        raise ValueError('R15Bt is the T2V entry only')
    if not Path(cfg['source_snapshot']).is_dir():
        raise ValueError('The frozen R15Bt source snapshot is missing')
    if cfg['stages'] != dict(interface=128, joint=272):
        raise ValueError('Keep the R15B 128 interface + 272 joint updates')
    if cfg['train']['accumulation'] != 4:
        raise ValueError('Keep the matched 1600-sample budget')
    model = cfg['model']
    if (model['width'], model['qwen_width'], model['q_tokens'], model['q_heads'],
            model['rounds'], model['condition_every'], model['lora_rank'],
            model['grounded_read_heads']) != (3072, 4096, 256, 32, 2, 3, 32, 16):
        raise ValueError('R15B architecture constants changed')
    if model['native_text_adaptation'] != 'frozen':
        raise ValueError('Native text Q/K/V/O must stay frozen')
    if model['fusion_order'] != 'request_bound_physical_attention_parallel_to_frozen_text':
        raise ValueError('Keep the R15B request-bound joint read')
    if model['physical_execution'] != 'r10a_state_value_execution_after_request_binding':
        raise ValueError('Keep the R10A state-value execution')
    if cfg['loss']['semantic_target'] != 'routing_only':
        raise ValueError('R10A routing alignment is retained')
    if cfg['verifier']['kind'] != 'none' or cfg.get('initialize_generator'):
        raise ValueError('Fresh official foundations without a verifier')
    if cfg['data_revision'] != DATA_REVISION:
        raise ValueError('Use the audited request-only qwen38 T2V data revision')
    if cfg['input'] != dict(external_image=False, evidence_scope='request_only',
                            known_latent_frames=0, future_targets_in_condition=False,
                            chunk_descriptions=False,
                            local_annotation_preprocessing=False):
        raise ValueError('T2V must consume no image and no known latent frame')
    if cfg['data'] != dict(latent_shape=[1, 48, 21, 30, 52], frames=81, fps=24.0,
                           match_r18_t2v_loader=True):
        raise ValueError('The fixed T2V latent/geometry contract changed')
    if cfg['inference'] != dict(steps=50, shift=5., guidance=5., sampler='shifted_euler'):
        raise ValueError('Use the matched inference protocol')
    if cfg['train']['validation_steps']:
        raise ValueError('No validation T2V evidence exists; keep validation_steps empty')
    if cfg['train']['keep_steps'] != [128, 250, 400]:
        raise ValueError('Keep the R15B checkpoint cadence')
    if cfg['train']['peak_lrs'] != dict(enhancement=5e-5, fusion_read=2e-5,
                                        fusion_write=3e-5, wan_lora=2e-5):
        raise ValueError('R15B learning rates changed')
    if cfg['loss']['generation'] != dict(tail_weight=.2, tail_fraction=.25):
        raise ValueError('Generation loss weights changed')
    for name in ('runs', 'interface', 'inference_evidence', 'eval_native_inputs'):
        if not Path(cfg['paths'][name]).resolve().is_relative_to(ROOT):
            raise ValueError('R15Bt outputs must stay inside its own directory')
    return cfg


def interface_ready(cfg):
    path = Path(cfg['paths']['interface'])
    if not path.is_file():
        return False
    import torch
    value = torch.load(path, map_location='cpu', weights_only=True)
    return (value.get('revision') == REVISION and value.get('task') == 't2v'
            and value.get('data_revision') == DATA_REVISION
            and int(value.get('training_samples', -1)) == 1600)


def data_ready(cfg):
    cache = Path(cfg['paths']['cache'])
    input_root = Path(cfg['paths']['t2v_input_cache'])
    words_root = Path(cfg['paths']['t2v_words_cache'])
    native_root = Path(cfg['paths']['t2v_native_words_cache'])
    for path in (cache / 'base/manifest.json', cache / 'base/null_text.pt',
                 Path(cfg['paths']['interface']),
                 input_root.parent / 'ready.json',
                 native_root.parent / 'native_ready.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    ready = json.loads((input_root.parent / 'ready.json').read_text())
    if (ready.get('complete') is not True or ready.get('scope') != 'request_only'
            or ready.get('image_given_to_qwen') is not False
            or ready.get('revision') != SOURCE_EVIDENCE_REVISION
            or int(ready.get('records', -1)) != 1600 + 60):
        raise ValueError('Request-only Qwen cache provenance changed')
    native = json.loads((native_root.parent / 'native_ready.json').read_text())
    if (native.get('status') != 'complete'
            or native.get('revision', SOURCE_EVIDENCE_REVISION) != SOURCE_EVIDENCE_REVISION
            or int(native.get('count', -1)) != 2 * (1600 + 60)):
        raise ValueError('Native T5 target cache is incomplete')
    eval_texts = list(Path(cfg['paths']['t2v_eval_cache']).glob('*_text.pt'))
    if len(eval_texts) != 60:
        raise ValueError(f'Expected 60 evaluation evidence texts, got {len(eval_texts)}')
    native_inputs = list(Path(cfg['paths']['eval_native_inputs']).glob('*/conditions.pt'))
    if len(native_inputs) != 60:
        raise ValueError(f'Expected 60 T2V-only native inputs, got {len(native_inputs)}')
    if not interface_ready(cfg):
        raise ValueError('Interface belongs to another task/data revision')
    rows = json.loads((cache / 'base/manifest.json').read_text())['records']
    train = [row for row in rows if row['split'] == 'train']
    def complete(row):
        for suffix in ('_text', '_null'):
            name = f'{row["id"]}{suffix}'
            if not (input_root / f'{name}.pt').is_file():
                return False
            if not (words_root / f'{name}.pt').is_file():
                return False
            if not (native_root / f'{name}.pt').is_file():
                return False
        return True
    kept = [row for row in train if complete(row)]
    if len(kept) != 1600:
        raise ValueError(f'Expected 1600 R18-filtered train rows, got {len(kept)}')
    by_id = {row['id']: row for row in kept}
    from physgen_native.data import SampleOrder
    order = iter(SampleOrder(len(kept), 42, 0))
    draws = [kept[next(order)]['id'] for _ in range(400 * cfg['train']['accumulation'])]
    if len(set(draws)) != 400 * cfg['train']['accumulation']:
        raise ValueError('R18-filtered sample order is not one full epoch')
    import torch
    mismatched, native_bad = [], []
    for sample in draws:
        value = torch.load(input_root / f'{sample}_text.pt',
                           map_location='cpu', weights_only=True)
        native_value = torch.load(native_root / f'{sample}_text.pt',
                                  map_location='cpu', weights_only=True)
        row = by_id[sample]
        if (value.get('caption') != row['caption']
                or value.get('request_word_strings') != row['caption'].split()):
            mismatched.append(sample)
        if (native_value.get('scope') != 'request_only'
                or native_value.get('sample') != sample
                or list(native_value.get('word_strings', []))
                != list(value.get('word_strings', []))):
            native_bad.append(sample)
    if mismatched or native_bad:
        raise ValueError(
            'Evidence/native provenance differs from the training manifest for '
            f'captions={len(mismatched)} native={len(native_bad)}; '
            f'first={mismatched[:3] or native_bad[:3]}')
    return dict(manifest_records=len(rows), train_rows=len(train),
                filtered_train_rows=len(kept),
                matched_draw_samples=len(draws), unique_draw_samples=len(set(draws)),
                evidence_scope='request_only', image_given_to_qwen=False,
                first_frame_used=False, new_api_calls=0)


def audit(cfg):
    result = original_audit(cfg)
    result.update(data_ready(cfg),
                  task='t2v', first_frame='none', known_latent_frames=0,
                  physical_read='B400 joint direct and request-bound memory',
                  native_text_adaptation='frozen',
                  teacher_cache='audited request-only Qwen T2V cache; no image or future input',
                  physical_updates=sum(cfg['stages'].values()),
                  native_lora_updates=cfg['stages']['joint'],
                  initialization='official Wan2.2-TI2V-5B; fresh project modules')
    return result
