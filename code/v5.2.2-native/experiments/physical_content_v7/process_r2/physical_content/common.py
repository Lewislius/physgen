from pathlib import Path
import yaml
from physical_scaling.common import gpu_guard,atomic_json,atomic_torch,cpu_tree,Log
from physical_scaling.common import audit as original_audit

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')

def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if cfg['revision']!='physical_content_v7_process_r2' or cfg['variant'] not in ('physical_content','native_lora'):
        raise ValueError('Unsupported content revision')
    if cfg['model']['rounds']!=2 or cfg['model']['q_tokens']!=256:
        raise ValueError('Keep two shared updates and Q256 for this comparison')
    if cfg['verifier']['kind']!='none' or cfg.get('initialize_generator'):
        raise ValueError('Official foundations only, without a verifier')
    for key in ['runs','supervision_cache','content_cache','interface']:
        if not Path(cfg['paths'][key]).resolve().is_relative_to(ROOT):raise ValueError('New outputs must stay in this revision')
    if not Path(cfg['paths']['runs']).resolve().is_relative_to(ROOT/'process_r2'):
        raise ValueError('R2 checkpoints must not overwrite R1 runs')
    process=cfg['loss']['process_features']
    if process['kind']!='frozen_qwen_visual_observations' or process['pairs']!=3:
        raise ValueError('Use the reviewed three-pair observed-state objective')
    if not (0 < process['sigma_min'] < process['sigma_max'] < 1):
        raise ValueError('The direct clean estimate requires an interior noise interval')
    if process['weight']<=0 or process['warmup']<1:
        raise ValueError('Observed-state supervision must have positive weight and warmup')
    return cfg

def audit(cfg):
    result=original_audit(cfg)
    result['teacher_cache']='Input-only physical-word semantic calibration; no old weak association/write targets'
    result['generation_objective']=cfg['loss']['generation']
    result['observed_state_objective']=cfg['loss']['process_features']
    result['future_video_scope']='Training targets only, after the generation forward; no future input to C'
    if cfg['variant']=='physical_content' and not Path(cfg['paths']['interface']).is_file():
        raise FileNotFoundError('Fit the fresh semantic interface before candidate training')
    return result
