from pathlib import Path
import yaml
from physical_scaling.common import gpu_guard,atomic_json,atomic_torch,cpu_tree,Log
from physical_scaling.common import audit as original_audit

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')

def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if cfg['revision']!='physical_content_v7_grounded_execution_r10b' or cfg['variant'] not in ('physical_content','native_lora'):
        raise ValueError('Unsupported content revision')
    if cfg['model']['rounds']!=2 or cfg['model']['q_tokens']!=256:
        raise ValueError('Keep two shared updates and Q256 for this comparison')
    if cfg['verifier']['kind']!='none' or cfg.get('initialize_generator'):
        raise ValueError('Official foundations only, without a verifier')
    for key in ['runs','supervision_cache','content_cache','interface']:
        if not Path(cfg['paths'][key]).resolve().is_relative_to(ROOT):raise ValueError('New outputs must stay in this revision')
    if not Path(cfg['paths']['runs']).resolve().is_relative_to(ROOT/'grounded_execution_r10b'):
        raise ValueError('R10 runs need their own output directory')
    if cfg['model']['grounded_read_heads']!=cfg['model']['q_heads']//2:
        raise ValueError('Keep half the heads for relation-local reads')
    if cfg['model']['visual_read_heads']!=cfg['model']['q_heads']//4:
        raise ValueError('Keep a quarter of the heads for direct first-image visual reads')
    if cfg['model']['global_reader_input']!='visual_tokens_request_and_reasoning_words':
        raise ValueError('Use direct first-image, request and reasoning evidence')
    if cfg['model']['grounded_word_normalization']!='train_input_interface':
        raise ValueError('Use the shared training-input coordinates')
    if cfg['model']['final_readout']!='native_execution_core_content_routing' or not cfg['model']['shared_content_offset']:
        raise ValueError('Use a full-width native execution core with additional content routing')
    if cfg['loss'].get('semantic_target')!='execution_core':
        raise ValueError('Native-word alignment applies to the shared execution core')
    if 'process_features' in cfg['loss']:
        raise ValueError('Retain the original R1 objective for this readout comparison')
    return cfg

def audit(cfg):
    result=original_audit(cfg)
    result['teacher_cache']='Input-only physical-word semantic calibration; no old weak association/write targets'
    result['generation_objective']=cfg['loss']['generation']
    if cfg['variant']=='physical_content' and not Path(cfg['paths']['interface']).is_file():
        raise FileNotFoundError('Fit the fresh semantic interface before candidate training')
    return result
