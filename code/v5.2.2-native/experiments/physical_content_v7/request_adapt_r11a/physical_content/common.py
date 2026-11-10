from pathlib import Path
import yaml
from physical_scaling.common import gpu_guard,atomic_json,atomic_torch,cpu_tree,Log
from physical_scaling.common import audit as original_audit

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')

def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if cfg['revision']!='physical_content_v7_semantic_execution_r11a_v2' or cfg['variant']!='physical_content':
        raise ValueError('Unsupported content revision')
    if cfg['model']['rounds']!=2 or cfg['model']['q_tokens']!=256:
        raise ValueError('Keep two shared updates and Q256 for this comparison')
    if cfg['verifier']['kind']!='none' or cfg.get('initialize_generator'):
        raise ValueError('Official foundations only, without a verifier')
    for key in ['runs','supervision_cache','content_cache','interface']:
        if not Path(cfg['paths'][key]).resolve().is_relative_to(ROOT):raise ValueError('New outputs must stay in this revision')
    if not Path(cfg['paths']['runs']).resolve().is_relative_to(ROOT/'request_adapt_r11a'):
        raise ValueError('R11A runs need their own output directory')
    if cfg['model']['native_text_lora_rank']!=32 or cfg['model']['native_text_lora_targets']!=['q','k','v','o']:
        raise ValueError('Use native text Q/K/V/O adaptation at rank32')
    if cfg['model'].get('physical_query')!='frozen_native_q_on_post_text_state':
        raise ValueError('Physical execution uses the frozen base Q after native text')
    if cfg['model'].get('fusion_order')!='native_text_then_physical':
        raise ValueError('Use sequential native-text and physical execution')
    if cfg['model']['grounded_read_heads']!=cfg['model']['q_heads']//2:
        raise ValueError('Use equal relation-local and global content reader heads')
    if cfg['model']['global_reader_input']!='request_and_reasoning_words':
        raise ValueError('Use the declared literal-request and reasoning-word read')
    if cfg['model']['grounded_word_normalization']!='train_input_interface':
        raise ValueError('Use the shared training-input coordinates')
    if cfg['model']['final_readout']!='semantic_routing_q_execution' or not cfg['model']['shared_content_offset']:
        raise ValueError('Use full-width input-value routing and Q-composed execution')
    if cfg['loss'].get('semantic_target')!='routing_only':
        raise ValueError('Native-word alignment applies only to the routing memory')
    if 'process_features' in cfg['loss']:
        raise ValueError('Retain the original R1 objective for this readout comparison')
    if cfg['stages']!={'interface':128,'joint':272} or cfg['train']['accumulation']!=4:
        raise ValueError('Keep matched128+272=400 updates and accumulation4')
    groups={'enhancement','fusion_read','fusion_write','wan_lora','native_text'}
    if set(cfg['train']['peak_lrs'])!=groups or set(cfg['train']['grad_clip'])!=groups:
        raise ValueError('Each physical, backbone and native-text role needs its own optimizer settings')
    if cfg['train']['native_text_warmup']!=32 or cfg['train'].get('native_text_start_step')!=1:
        raise ValueError('Co-adapt native text from update1, with32 warmup updates')
    if cfg['evaluation_budget']['control_modes']!=['text_adapter_off','parallel_fusion','reasoning_swapped','empty_reasoning','global_train_mean','anchor_only','no_c']:
        raise ValueError('Retain the declared seven paired16 interventions')
    return cfg

def audit(cfg):
    result=original_audit(cfg)
    result['teacher_cache']='Input-only physical-word semantic calibration; no old weak association/write targets'
    result['generation_objective']=cfg['loss']['generation']
    result['semantic_adaptation']='Native cross-attention Q/K/V/O LoRA,400 updates; original self-attention/FFN LoRA272 updates'
    result['fusion_order']=cfg['model']['fusion_order']
    result['physical_query']=cfg['model']['physical_query']
    if cfg['variant']=='physical_content' and not Path(cfg['paths']['interface']).is_file():
        raise FileNotFoundError('Fit the fresh semantic interface before candidate training')
    return result
