from pathlib import Path
import json
import os
import time
import yaml
from physical_scaling.common import gpu_guard as original_gpu_guard
from physical_scaling.common import atomic_json, atomic_torch, cpu_tree, Log
from physical_scaling.common import audit as original_audit

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
REVISION = 'physical_content_v7_native_response_fusion_r16'
RESPONSE_MODES = {'global_request_memory', 'local_request_memory', 'late_joint_value'}


def gpu_guard(run):
    """RTX PRO6000/RTX6000 only; this shared base has no A100 exception."""
    return original_gpu_guard(run)


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['revision'] != REVISION or cfg['variant'] != 'physical_content':
        raise ValueError('Unsupported content revision')
    if cfg['model']['rounds'] != 2 or cfg['model']['q_tokens'] != 256:
        raise ValueError('Keep two shared updates and Q256 for this comparison')
    if cfg['verifier']['kind'] != 'none' or cfg.get('initialize_generator'):
        raise ValueError('Official foundations only, without a verifier')
    for key in ['runs', 'supervision_cache', 'content_cache', 'interface']:
        if not Path(cfg['paths'][key]).resolve().is_relative_to(ROOT):
            raise ValueError('New outputs must stay in this revision')
    if not Path(cfg['paths']['runs']).resolve().is_relative_to(ROOT / 'native_response_fusion_r16'):
        raise ValueError('Preserve historical runs in their original directories')
    if cfg['model']['grounded_read_heads'] != cfg['model']['q_heads'] // 2:
        raise ValueError('Use equal relation-local and global content reader heads')
    if cfg['model']['global_reader_input'] != 'original_contextual_request_and_physical_words':
        raise ValueError('Use original contextual input words, including the literal request')
    if cfg['model']['native_text_adaptation'] != 'frozen':
        raise ValueError('Preserve native text Q/K/V/O')
    if cfg['model'].get('response_mode') not in RESPONSE_MODES:
        raise ValueError('Use one declared response mode')
    declared = {
        'global_request_memory': ('R16G', 'response_memory_global_residual_on_local_QKV'),
        'local_request_memory': ('R16H', 'response_memory_condition_on_local_QKV'),
        'late_joint_value': ('R16I', 'late_joint_direct_request_value_on_local_QKV')}
    label, order = declared[cfg['model']['response_mode']]
    if cfg['model'].get('candidate_label') != label:
        raise ValueError('Candidate label does not match the declared response mode')
    if cfg['model']['fusion_order'] != order:
        raise ValueError('Use the declared response fusion order')
    if cfg['model'].get('response_memory_input') != 'literal_qwen_request_words':
        raise ValueError('Use the literal Qwen request words as response memory')
    if cfg['model']['grounded_word_normalization'] != 'train_fitted_whitened_words_and_request':
        raise ValueError('Use the existing training-fitted input normalization')
    if cfg['model']['final_readout'] != 'R10A_R15B_routing_values_and_local_QKV_execution' or not cfg['model']['shared_content_offset']:
        raise ValueError('Retain R10A/R15B routing and execution C with native anchors')
    if cfg['model']['physical_window'] != [5, 3, 3]:
        raise ValueError('Use the declared five-frame temporal by 3x3 spatial window')
    if cfg['model'].get('query_correction') != 'shared_full_width_zero_init_no_bottleneck':
        raise ValueError('Use the declared full-width shared query correction')
    if cfg['model'].get('value_correction') != 'shared_full_width_zero_init_request_bound_no_bottleneck':
        raise ValueError('Use the declared full-width shared request-bound value correction')
    if cfg['model']['response_mode'] == 'late_joint_value':
        start = cfg['model'].get('joint_start_step')
        if not isinstance(start, int) or not 0 < start < sum(cfg['stages'].values()):
            raise ValueError('Late joint mode requires a joint_start_step inside the 400-update budget')
    if cfg['stages'] != dict(interface=128, joint=272) or cfg['train']['accumulation'] != 4:
        raise ValueError('Use the matched 128+272 updates and 1600-sample budget')
    if cfg['loss'].get('semantic_target') != 'routing_C_only':
        raise ValueError('Word alignment supervises routing, never residual execution values')
    if 'process_features' in cfg['loss']:
        raise ValueError('Use the declared generation and routing-alignment objectives')
    if cfg['inference']['guidance'] != 5. or cfg['inference']['guidance_composition'] != 'joint_two_branch_cfg':
        raise ValueError('Record the exact chosen composition, retaining native gamma5')
    return cfg


def data_ready(cfg):
    cache = Path(cfg['paths']['cache'])
    for path in [cache / 'base/manifest.json', cache / 'base/null_text.pt', Path(cfg['paths']['interface'])]:
        if not path.is_file():
            raise FileNotFoundError(path)
    records = json.loads((cache / 'base/manifest.json').read_text())['records']
    return dict(manifest_records=len(records), data_revision=cfg['data_revision'],
                source='existing original training cache used by R10A', new_api_calls=0)


def audit(cfg):
    result = original_audit(cfg)
    result.update(data_ready(cfg))
    result['teacher_cache'] = 'Input-only physical-word semantic calibration; no old weak association/write targets'
    result['generation_objective'] = cfg['loss']['generation']
    result['architecture'] = (
        'R10A/R15B routing and execution C read two independent ways, conditioning Q/K/V of a '
        'local 5-frame x 3x3 video-state operator. Response mode: ' + cfg['model']['response_mode'] +
        '. Global request memory is a full-width literal-request residual; local request memory '
        'conditions local Q/K/V; late joint value uses the B400-style direct+request joint read '
        'after step ' + str(cfg['model'].get('joint_start_step', 'n/a')) + '. All new maps are '
        '3072->3072 or high-dimensional request-word maps; no low-dimensional bottleneck. Native '
        'text is frozen; same forward in training and normal inference; original400 data and gamma5.')
    result['quality_proven'] = False
    result.update(optimizer_updates=sum(cfg['stages'].values()),
                  physical_updates=sum(cfg['stages'].values()),
                  native_lora_updates=cfg['stages']['joint'],
                  generator_forward_backward_views_per_sample=1,
                  initialization='Fresh official foundations; no trained project initializer')
    if cfg['variant'] == 'physical_content' and not Path(cfg['paths']['interface']).is_file():
        raise FileNotFoundError('Fit the fresh semantic interface before candidate training')
    return result
