from pathlib import Path
import json
import yaml
from physical_scaling.common import gpu_guard,atomic_json,atomic_torch,cpu_tree,Log
from physical_scaling.common import audit as original_audit

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
REVISION='physical_content_v7_request_bound_fusion_r15b'


def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if cfg['revision']!=REVISION or cfg['variant']!='physical_content':
        raise ValueError('Use the request-bound R15B revision')
    if sum(cfg['stages'].values())!=400 or cfg['train']['accumulation']!=4:
        raise ValueError('Keep the matched400 update/1600 sample budget')
    if cfg['model']['native_text_adaptation']!='frozen':
        raise ValueError('This candidate retains frozen native text attention')
    if cfg['model']['fusion_order']!='request_bound_physical_attention_parallel_to_frozen_text':
        raise ValueError('Bind physical values to native request keys without changing native text attention')
    if cfg['stages']!=dict(interface=128,joint=272):
        raise ValueError('Use the matched R10A128 interface and272 joint updates for this candidate')
    if cfg['loss']['semantic_target']!='routing_only':
        raise ValueError('R10A routing alignment is retained; execution is trained by generation')
    if cfg['verifier']['kind']!='none' or cfg.get('initialize_generator'):
        raise ValueError('Use fresh official foundations without a training verifier')
    if cfg['data_revision']!='original_wisa4000_and_original_reasoning':
        raise ValueError('This candidate is paired with original-data R10A')
    if not Path(cfg['paths']['runs']).resolve().is_relative_to(ROOT/'request_bound_fusion_r15b'):
        raise ValueError('Preserve older runs in their own directories')
    if cfg['inference']!=dict(steps=50,shift=5.,guidance=5.,sampler='shifted_euler'):
        raise ValueError('Use the matched inference protocol')
    return cfg


def data_ready(cfg):
    cache=Path(cfg['paths']['cache'])
    for path in [cache/'base/manifest.json',cache/'base/null_text.pt',Path(cfg['paths']['interface'])]:
        if not path.is_file():raise FileNotFoundError(path)
    records=json.loads((cache/'base/manifest.json').read_text())['records']
    return dict(manifest_records=len(records),data_revision=cfg['data_revision'],
                source='existing original training cache used by R10A',new_api_calls=0)


def audit(cfg):
    result=original_audit(cfg)
    result.update(data_ready(cfg))
    result.update(generation_objective=cfg['loss']['generation'],
        teacher_cache='Original input-only Qwen request/reasoning and fitted interface',
        fusion_order=cfg['model']['fusion_order'],
        execution_interface='R10A enhancement and state-value execution. Native request keys bind physical C values, then native video queries read the bound values. Independent physical K/V/O remain trainable; original text Q/K/V/O remain frozen. No independent request output.',
        content_alignment_schedule=dict(total_peak=.03,warmup=32,routing_target_retained=True,execution_target=False),
        optimizer_updates=sum(cfg['stages'].values()),physical_updates=sum(cfg['stages'].values()),
        native_lora_updates=cfg['stages']['joint'],generator_forward_backward_views_per_sample=1,
        initialization='Fresh official foundations; no trained project initializer')
    return result
