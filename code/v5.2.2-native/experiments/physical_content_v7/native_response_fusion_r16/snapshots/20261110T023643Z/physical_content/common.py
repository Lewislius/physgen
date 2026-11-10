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
RESPONSE_MODES = {'triple_global_pure_local', 'triple_global_joint_local',
                  'triple_global_full_local', 'direct_plus_request_safe',
                  'gated_direct_request', 'gated_triple', 'gated_union'}
DECLARED = {
    'triple_global_pure_local': dict(
        label='R16G', global_execution=['direct', 'request_bound', 'joint'],
        fusion_order='triple_global_pure_r15c_local',
        local_query_condition='direct_read_only',
        value_correction='none'),
    'triple_global_joint_local': dict(
        label='R16H', global_execution=['direct', 'request_bound', 'joint'],
        fusion_order='triple_global_joint_conditioned_r15c_local',
        local_query_condition='direct_plus_joint_zero_init_full_width',
        value_correction='none'),
    'triple_global_full_local': dict(
        label='R16I', global_execution=['direct', 'request_bound', 'joint'],
        fusion_order='triple_global_full_local_conditioning',
        local_query_condition='direct_plus_request_plus_joint_zero_init_full_width',
        value_correction='shared_full_width_zero_init_request_bound'),
    'direct_plus_request_safe': dict(
        label='R16G-SAFE', global_execution=['direct', 'request_bound'],
        fusion_order='state_gated_direct_and_request_bound_global_r15c_local',
        local_query_condition='direct_read_only',
        value_correction='none'),
    'gated_direct_request': dict(
        label='R16H', global_execution=['direct', 'request_bound'],
        fusion_order='gated_single_write_direct_request_plus_r15c_local',
        local_query_condition='direct_read_only',
        value_correction='none'),
    'gated_triple': dict(
        label='R16I', global_execution=['direct', 'request_bound', 'joint'],
        fusion_order='gated_single_write_direct_request_joint_plus_r15c_local',
        local_query_condition='direct_read_only',
        value_correction='none'),
    'gated_union': dict(
        label='R16J', global_execution=['direct', 'request_bound', 'joint', 'single'],
        fusion_order='gated_union_single_output_isolated_local_and_gated_value_correction',
        local_query_condition='direct_read_only',
        value_correction='gated_full_width_zero_init_request_bound_local'),
}


def gpu_guard(run):
    """RTX PRO6000/RTX6000 by default; one explicitly authorized A100 job."""
    if os.getenv('R16GHI_ALLOW_AMP80') == '1':
        import torch
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count else 'NO CUDA DEVICE'
        accepted = count == 1 and 'H100' not in name and ('A100' in name or 'A800' in name)
        record = dict(kind='gpu_policy', actual=name, visible_count=count, accepted=accepted,
                      resource_pool='amp-80g', experiment_id=os.getenv('DET_EXPERIMENT_ID'),
                      time=time.time())
        atomic_json(record, Path(run) / 'gpu.json')
        print(json.dumps(record), flush=True)
        if not accepted:
            raise SystemExit(86)
        return record
    if os.getenv('R16GSAFE_ALLOW_ADA48') == '1':
        import torch
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count else 'NO CUDA DEVICE'
        accepted = count == 1 and 'H100' not in name and (
            'L40S' in name or 'RTX 4090' in name)
        record = dict(kind='gpu_policy', actual=name, visible_count=count, accepted=accepted,
                      resource_pool='ada-48g', experiment_id=os.getenv('DET_EXPERIMENT_ID'),
                      time=time.time())
        atomic_json(record, Path(run) / 'gpu.json')
        print(json.dumps(record), flush=True)
        if not accepted:
            raise SystemExit(86)
        return record
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
    mode = cfg['model'].get('response_mode')
    if mode not in RESPONSE_MODES:
        raise ValueError('Use one declared response mode')
    declared = DECLARED[mode]
    if cfg['model'].get('candidate_label') != declared['label']:
        raise ValueError('Candidate label does not match the declared response mode')
    if cfg['model'].get('global_execution') != declared['global_execution']:
        raise ValueError('Global read set does not match the declared response mode')
    if cfg['model'].get('fusion_order') != declared['fusion_order']:
        raise ValueError('Use the declared response fusion order')
    if cfg['model'].get('local_query_condition') != declared['local_query_condition']:
        raise ValueError('Local query condition does not match the declared response mode')
    if cfg['model'].get('value_correction') != declared['value_correction']:
        raise ValueError('Local value correction does not match the declared response mode')
    if cfg['model'].get('local_execution_operator') != 'r15c_5x3x3_video_state_V':
        raise ValueError('Retain the proven local video-state operator with video V')
    if cfg['model']['grounded_word_normalization'] != 'train_fitted_whitened_words_and_request':
        raise ValueError('Use the existing training-fitted input normalization')
    if cfg['model'].get('physical_output_initialization') != 'zero_O_exact_native_start':
        raise ValueError('Retain exact native start through zero-initialized writes')
    if cfg['model'].get('global_execution_gate') != 'zero_init_full_width_state_value_no_bottleneck':
        raise ValueError('Use full-width zero-init state gates for every global read')
    if cfg['model']['final_readout'] != 'R10A_R15B_B400_reads_to_r15c_local_and_state_gated_global':
        raise ValueError('Declare the multi-read local/global final readout')
    if cfg['model']['physical_window'] != [5, 3, 3]:
        raise ValueError('Use the declared five-frame temporal by 3x3 spatial window')
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
    extra = ('R16I additionally uses zero-init full-width request/joint local-Q corrections and a '
             'shared zero-init full-width request-bound V correction.'
             if cfg['model']['response_mode'] == 'triple_global_full_local' else
             'R16H additionally uses a zero-init full-width joint local-Q correction.'
             if cfg['model']['response_mode'] == 'triple_global_joint_local' else
             'R16G keeps the local Q condition on the direct read alone.')
    if cfg['model']['response_mode'] == 'direct_plus_request_safe':
        extra = 'R16G-SAFE uses only direct and request-bound global reads, with no joint read.'
    if cfg['model']['response_mode'].startswith('gated_'):
        extra = ('Gated single-write fusion: full-width state-conditioned read gates combine the declared '
                 'physical reads before one zero-init output; full-width final local/global gates balance '
                 'the R15C local state operator against the fused global execution write.')
    result['architecture'] = (
        'R15C local 5x3x3 video-state operator (video V, direct zero-init full-width local Q condition) '
        'plus state-gated global execution writes from the declared clean reads: ' +
        ', '.join(cfg['model']['global_execution']) + '. Each read has its own zero-init '
        '3072->3072 output and zero-init 3072->3072 multiplicative state gate, matching R10A/R15B. ' +
        extra + ' Raw Qwen request words are never written globally. No low-dimensional bottleneck. '
        'Native text is frozen; same forward in training and normal inference; original400 data and gamma5.')
    result['quality_proven'] = False
    result.update(optimizer_updates=sum(cfg['stages'].values()),
                  physical_updates=sum(cfg['stages'].values()),
                  native_lora_updates=cfg['stages']['joint'],
                  generator_forward_backward_views_per_sample=1,
                  initialization='Fresh official foundations; no trained project initializer')
    if cfg['variant'] == 'physical_content' and not Path(cfg['paths']['interface']).is_file():
        raise FileNotFoundError('Fit the fresh semantic interface before candidate training')
    return result
