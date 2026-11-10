from pathlib import Path
import yaml
from physical_scaling.common import gpu_guard,atomic_json,atomic_torch,cpu_tree,Log
from physical_scaling.common import audit as original_audit

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')

def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if cfg['revision']!='physical_content_v7_content_reader_r8' or cfg['variant'] not in ('physical_content','native_lora'):
        raise ValueError('Unsupported content revision')
    if cfg['model']['rounds']!=2 or cfg['model']['q_tokens']!=256:
        raise ValueError('Keep two shared updates and Q256 for this comparison')
    if cfg['verifier']['kind']!='none' or cfg.get('initialize_generator'):
        raise ValueError('Official foundations only, without a verifier')
    for key in ['runs','supervision_cache','content_cache','interface']:
        if not Path(cfg['paths'][key]).resolve().is_relative_to(ROOT):raise ValueError('New outputs must stay in this revision')
    if not Path(cfg['paths']['runs']).resolve().is_relative_to(ROOT/'content_reader_r8'):
        raise ValueError('R8 checkpoints must not overwrite earlier runs')
    if cfg['model']['native_output_init_scale']!=0.25:
        raise ValueError('This revision uses the declared quarter-scale pretrained Wan output initializer')
    if cfg['model']['grounded_read_heads']!=cfg['model']['q_heads']//2:
        raise ValueError('Use half relation-grounded and half full-context heads')
    if cfg['model']['grounded_word_normalization']!='train_input_interface':
        raise ValueError('Local words and query anchors must use the same fitted input coordinates')
    if cfg['model']['content_projection_init_scale']!=0.1:
        raise ValueError('Use the declared semantic residual initializer')
    if cfg['model']['content_projection_trainable'] is not False:
        raise ValueError('This comparison fixes the calibrated native coordinate map')
    if cfg['model']['content_update_mode']!='grounded_feature_modulation':
        raise ValueError('Q updates must act on the input physical content')
    if cfg['model']['global_reader_input']!='request_and_reasoning_words':
        raise ValueError('The R8 global reader uses contextual request and reasoning words')
    if cfg['evaluation_budget']['primary_cases']!=40 or cfg['evaluation_budget']['first_stop']!=128:
        raise ValueError('Use the requested full40 main comparison at the declared128-step stop')
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
    result['schedule_maximum_steps']=result['steps']
    result['steps']=cfg['evaluation_budget']['first_stop']
    result['sample_draws']=result['steps']*cfg['train']['accumulation']
    result['teacher_cache']='Input-only physical-word semantic calibration; no old weak association/write targets'
    result['generation_objective']=cfg['loss']['generation']
    result['observed_state_objective']=cfg['loss']['process_features']
    result['physical_output_initializer']='0.25 times corresponding official Wan cross-attention O; independent trainable physical K/V/O'
    result['early_evaluation']='Fresh128-update short experiment, with16/32/64/128 checkpoints; no automatic extension beyond128. The user authorized a few hundred updates for substantive design revisions.'
    result['physical_evidence_read']='16 same-relation physical-word heads and16 all-request/reasoning-word heads; both use the fitted input coordinates and shared Q/K/V/O. Frozen Qwen word states retain first-image context; direct image-prefix/template tokens are excluded.'
    result['input_preparation']='Recover literal-request word states from existing immutable Qwen memory with the local tokenizer only. No Qwen backbone pass, new generated reasoning, future condition, model download or shared-cache mutation.'
    result['content_projection_initializer']='Fixed0.1 times training-input P0, stored as a checkpoint constant; generation gradients still reach Q, all full-width reader and fusion parameters remain trainable'
    result['content_update']='C=P0(A)+0.1*P0(A*DeltaH), with native bias once; full-width conditional feature modulation, no free additive C correction. Raw DeltaH and effective content write must be diagnosed separately.'
    result['module_test']='Compare normal against a training-derived mean DeltaH while retaining each input anchor, plus anchor/off/unrelated controls; input-dependent algebra alone is not evidence of useful Q reasoning.'
    result['future_video_scope']='Training targets only, after the generation forward; no future input to C'
    if cfg['variant']=='physical_content' and not Path(cfg['paths']['interface']).is_file():
        raise FileNotFoundError('Fit the fresh semantic interface before candidate training')
    return result
