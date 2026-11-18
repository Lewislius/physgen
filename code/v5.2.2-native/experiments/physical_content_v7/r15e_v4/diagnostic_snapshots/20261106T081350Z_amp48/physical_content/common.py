from pathlib import Path
import json
import yaml
from physical_scaling.common import atomic_json,atomic_torch,cpu_tree,Log
from physical_scaling.common import audit as original_audit

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
REVISION='physical_content_v7_state_response_fusion_r15e'


def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if cfg['revision']!=REVISION or cfg['variant']!='physical_content':
        raise ValueError('Use the state-response physical fusion R15E revision')
    if sum(cfg['stages'].values())!=400 or cfg['train']['accumulation']!=4:
        raise ValueError('Keep the matched400 update/1600 sample budget')
    if cfg['model']['native_text_adaptation']!='frozen':
        raise ValueError('This candidate retains frozen native text attention')
    if cfg['model']['fusion_order']!='direct_physical_write_and_conditioned_native_video_response':
        raise ValueError('Use direct physical execution with native video response')
    if cfg['stages']!=dict(interface=128,joint=272):
        raise ValueError('Use the matched R10A128 interface and272 joint updates for this candidate')
    if cfg['loss']['semantic_target']!='routing_only':
        raise ValueError('R10A routing alignment is retained; execution is trained by generation')
    if cfg['verifier']['kind']!='none' or cfg.get('initialize_generator'):
        raise ValueError('Use fresh official foundations without a training verifier')
    if cfg['data_revision']!='original_wisa4000_and_original_reasoning':
        raise ValueError('This candidate is paired with original-data R10A')
    if not Path(cfg['paths']['runs']).resolve().is_relative_to(ROOT/'state_response_fusion_r15e'):
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
        execution_interface='Exact R10A enhancement, independent physical K/V/O, direct C values and state-value interaction. Independent physical Q/K/V/O learn direct execution. The physical write also conditions a second native video attention query read with the same baseline keys/values. Original native text computation is preserved. Original FM/high-error loss retained; no new temporal loss.',
        content_alignment_schedule=dict(total_peak=.03,warmup=32,routing_target_retained=True,execution_target=False),
        optimizer_updates=sum(cfg['stages'].values()),physical_updates=sum(cfg['stages'].values()),
        native_lora_updates=cfg['stages']['joint'],generator_forward_backward_views_per_sample=1,
        initialization='Fresh official foundations; no trained project initializer')
    return result


def gpu_guard(run):
    """User-authorized amp-48g exception for this read-only16 job only."""
    import os,time,torch
    actual=torch.cuda.get_device_name(0)
    accepted=('A6000' in actual and torch.cuda.device_count()==1)
    record=dict(kind='gpu_policy',actual=actual,visible_count=torch.cuda.device_count(),
        accepted=accepted,resource_pool='amp-48g',experiment_id=os.getenv('DET_EXPERIMENT_ID'),
        time=time.time(),scope='Temporary read-only16 exception; later training/evaluation stays blk-96g RTX6000')
    atomic_json(record,run/'gpu.json');print(json.dumps(record),flush=True)
    if not accepted:raise RuntimeError('This temporary probe requires the allocated amp-48g A6000; no H100')
    return actual
