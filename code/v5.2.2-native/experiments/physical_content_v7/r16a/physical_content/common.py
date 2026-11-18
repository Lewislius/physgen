from pathlib import Path
import json
import yaml
from physical_scaling.common import gpu_guard,atomic_json,atomic_torch,cpu_tree,Log
from physical_scaling.common import audit as original_audit

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
REVISION='physical_content_v7_native_joint_read_r16a'

def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if cfg['revision']!=REVISION or cfg['variant']!='physical_content':
        raise ValueError('Unsupported content revision')
    if cfg['model']['rounds']!=2 or cfg['model']['q_tokens']!=256:
        raise ValueError('Keep two shared updates and Q256 for this comparison')
    if cfg['verifier']['kind']!='none' or cfg.get('initialize_generator'):
        raise ValueError('Official foundations only, without a verifier')
    for key in ['runs','supervision_cache','content_cache','interface']:
        if not Path(cfg['paths'][key]).resolve().is_relative_to(ROOT):raise ValueError('New outputs must stay in this revision')
    if not Path(cfg['paths']['runs']).resolve().is_relative_to(ROOT/'native_joint_read_r16a'):
        raise ValueError('Preserve historical runs in their original directories')
    if cfg['model']['grounded_read_heads']!=cfg['model']['q_heads']//2:
        raise ValueError('Use equal relation-local and global content reader heads')
    if cfg['model']['global_reader_input']!='original_contextual_request_and_physical_words':
        raise ValueError('Use original contextual input words, including the literal request')
    if cfg['model']['native_text_adaptation']!='frozen':
        raise ValueError('Preserve native text Q/K/V/O')
    if cfg['model']['fusion_order']!='parallel_native_text_and_request_prior_direct_C':
        raise ValueError('Use native request attention as a prior for the direct physical C read')
    if cfg['model']['grounded_word_normalization']!='train_fitted_whitened_words_and_request':
        raise ValueError('Use the existing training-fitted input normalization')
    if cfg['model']['final_readout']!='R10A_R15B_routing_values_and_execution_update' or not cfg['model']['shared_content_offset']:
        raise ValueError('Retain R10A/R15B routing and execution content with native anchors')
    if cfg['loss'].get('semantic_target')!='routing_C_only':
        raise ValueError('Word alignment supervises routing, never residual execution values')
    if 'process_features' in cfg['loss']:
        raise ValueError('Use the declared generation and routing-alignment objectives')
    if cfg['inference']['guidance']!=5. or cfg['inference']['guidance_composition']!='joint_two_branch_cfg':
        raise ValueError('Record the exact chosen composition, retaining native gamma5')
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
    result['teacher_cache']='Input-only physical-word semantic calibration; no old weak association/write targets'
    result['generation_objective']=cfg['loss']['generation']
    result['architecture']='R10A/R15B Q/C retained. Native request attention and request-to-C binding supply a normalized C prior of total mass R. Independent physical Q selects original C values using that prior, with one learned null. State interaction and frozen native text remain. No added parameters or loss versus I; identical training/inference reader.'
    result['quality_proven']=False
    result.update(optimizer_updates=sum(cfg['stages'].values()),
        physical_updates=sum(cfg['stages'].values()),native_lora_updates=cfg['stages']['joint'],
        generator_forward_backward_views_per_sample=1,
        initialization='Fresh official foundations; no trained project initializer')
    if cfg['variant']=='physical_content' and not Path(cfg['paths']['interface']).is_file():
        raise FileNotFoundError('Fit the fresh semantic interface before candidate training')
    return result
