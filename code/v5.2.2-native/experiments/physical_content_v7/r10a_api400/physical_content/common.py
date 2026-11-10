from pathlib import Path
import json
import yaml
from physical_scaling.common import gpu_guard,atomic_json,atomic_torch,cpu_tree,Log
from physical_scaling.common import audit as original_audit

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
REVISION='physical_content_v7_r10a_api400'


def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if cfg['revision']!=REVISION or cfg['variant']!='physical_content':
        raise ValueError('Use the separate R11A rework revision')
    if cfg['stages']!={'interface':128,'joint':272} or cfg['train']['accumulation']!=4:
        raise ValueError('Retain R10A128 interface +272 joint updates on the same1600 inputs')
    if cfg['train'].get('gradient_ownership')!='r10a_interface_then_joint':
        raise ValueError('Use explicit R10A phases and one full-generation loss per sample')
    if cfg['model'].get('native_text_adaptation')!='frozen' or cfg['model'].get('fusion_order')!='r10a_parallel':
        raise ValueError('Preserve official native cross-attention and R10A fusion')
    if cfg['model']['q_tokens']!=256 or cfg['model']['rounds']!=2:
        raise ValueError('Use Q256/two shared refinement rounds')
    if cfg['model']['final_readout']!='semantic_routing_q_execution' or cfg['loss']['semantic_target']!='routing_only':
        raise ValueError('Retain R10A routing/execution readout and routing-only word auxiliary')
    if cfg['verifier']['kind']!='none' or cfg.get('initialize_generator'):
        raise ValueError('Official foundations only; no verifier/project initializer')
    if cfg.get('data_revision')!='qwen38_wisa_reviewed_v1':
        raise ValueError('This candidate explicitly requires revised API training data')
    if not Path(cfg['paths']['runs']).resolve().is_relative_to(ROOT/'r10a_api400'):
        raise ValueError('Do not overwrite previous R11A/R10A runs')
    if cfg['evaluation_budget']['control_modes']!=['no_c','reasoning_swapped','empty_reasoning','global_train_mean','anchor_routing']:
        raise ValueError('Use five declared fixed16 controls, not an expanded matrix')
    if cfg['inference']!=dict(steps=50,shift=5.,guidance=5.,sampler='shifted_euler'):
        raise ValueError('Keep matched inference settings')
    return cfg


def data_ready(cfg):
    root=Path(cfg['paths']['cache'])
    ready=json.loads((root/'readiness.json').read_text())
    if (not ready['training_ready'] or ready['revision']!=cfg['data_revision']
        or ready['unique_training_samples']!=1600 or ready['local_contexts']!=3200
        or not ready['original_sampling_order_preserved']):
        raise ValueError('Complete paired local features and fresh interface fitting are required')
    for key in ('interface','input_evidence','content_cache'):
        if not Path(cfg['paths'][key]).exists():raise FileNotFoundError(key)
    return ready


def audit(cfg):
    ready=data_ready(cfg)
    result=dict(ready=True,readiness_source=str(Path(cfg['paths']['cache'])/'readiness.json'),
        rows=ready['manifest_records'],steps=400,sample_draws=1600,
        initialization='official foundations; fresh project modules and LoRA',
        repeated_full_cache_audit=False)
    result.update(data_revision=ready['revision'],
        teacher_cache='API first-image/request text re-encoded with frozen local Qwen3-VL4096',
        generation_objective=cfg['loss']['generation'],
        native_text_adaptation='frozen official Q/K/V/O',fusion_order='R10A parallel',
        optimizer_updates=400,native_lora_updates=272,physical_updates=400,
        generator_forward_backward_views_per_sample=1,
        gradient_ownership=cfg['train']['gradient_ownership'],
        matched_budget='R10A128+272 phases,400 updates/1600 inputs, one full-generation pass per microbatch; revised API data/inference conditions')
    return result
