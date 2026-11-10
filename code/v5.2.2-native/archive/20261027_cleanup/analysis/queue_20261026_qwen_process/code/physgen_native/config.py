"""Versioned training objectives and progressive stages."""
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'v522_native_physical_dependency_v1'
TEMPORAL_REVISION = 'v522_native_temporal_distribution_v2'
ORDERED_REVISION = 'v522_native_ordered_process_v3'
CALIBRATED_REVISION = 'v522_native_calibrated_verifier_v4'
PAIRED_REVISION = 'v522_native_paired_verifier_v5'
OPERATOR_REVISION = 'v522_native_conditional_operator_v6'
REASONING_REVISION = 'v522_native_reasoning_operator_v7'
RESPONSE_REVISION = 'v522_native_causal_response_v8'
QWEN_REVISION = 'v522_native_qwen_process_v9'
OPERATOR_REVISIONS = (OPERATOR_REVISION, REASONING_REVISION, RESPONSE_REVISION, QWEN_REVISION)
REVISIONS = (REVISION,TEMPORAL_REVISION,ORDERED_REVISION,CALIBRATED_REVISION,PAIRED_REVISION,*OPERATOR_REVISIONS)


def read_config(path):
    def expand(value):
        if isinstance(value, str): return value.replace('${ROOT}', str(ROOT))
        if isinstance(value, dict): return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list): return [expand(v) for v in value]
        return value
    return validate(expand(yaml.safe_load(Path(path).read_text())))


def validate(cfg):
    if cfg['revision'] not in REVISIONS: raise ValueError('Use a native-process checkpoint revision')
    operator = cfg['revision'] in OPERATOR_REVISIONS
    rounds = cfg['knowledge'].get('latent_reasoning_steps', 0)
    source = cfg['knowledge'].get('condition_source', 'qwen')
    if source not in ('qwen', 'cached_t5'):
        raise ValueError('Condition source must be Qwen or the explicit cached-T5 ablation')
    if source == 'cached_t5' and (cfg['revision'] not in (RESPONSE_REVISION, QWEN_REVISION) or rounds):
        raise ValueError('The cached-T5 ablation uses v8/v9 without Qwen latent reasoning')
    if not isinstance(rounds, int) or rounds < 0:
        raise ValueError('Latent reasoning steps must be a nonnegative integer')
    if rounds:
        if cfg['revision'] not in (REASONING_REVISION, RESPONSE_REVISION, QWEN_REVISION):
            raise ValueError('Continuous latent refinement requires a reasoning checkpoint revision')
        if not 1 <= cfg['knowledge']['latent_reasoning_layers'] <= cfg['knowledge']['lora_last_layers']:
            raise ValueError('Reasoning must reuse knowledge-adapted Qwen layers')
    if cfg['knowledge'].get('condition_readout','queries')=='content' and cfg['revision']==REVISION:
        raise ValueError('Content readouts require v2 or later')
    if cfg['verifier'].get('kind')=='ordered_process' and cfg['revision']!=ORDERED_REVISION:
        raise ValueError('Ordered full-generation process targets require v3')
    if cfg['verifier'].get('kind')=='calibrated_process':
        method=cfg['verifier'].get('calibration','regression')
        required={'regression':CALIBRATED_REVISION,'paired':PAIRED_REVISION}
        if method not in required or cfg['revision']!=required[method]:
            raise ValueError('Verifier calibration and checkpoint revision must match: regression v4 / paired v5')
    if cfg['revision'] == REVISION and cfg['verifier'].get('kind') == 'temporal_distribution':
        raise ValueError('New critic parameters and targets require the v2 checkpoint revision')
    qwen_process = cfg['revision'] == QWEN_REVISION
    phases = (('calibration', 'interface', 'joint') if qwen_process else
              ('interface', 'calibration', 'joint')) if operator else ('interface', 'joint')
    if tuple(cfg['stages']) != phases or min(cfg['stages'].values()) < 1:
        raise ValueError('Use the ordered positive stage budgets for this revision')
    if operator:
        objectives = dict(interface={'fm', 'process', 'use', 'intensity'},
                          calibration={'rank', 'matching'}, joint={'fm', 'process', 'use', 'verifier'})
        if cfg['revision'] in (RESPONSE_REVISION, QWEN_REVISION):
            objectives = dict(interface={'fm', 'process', 'intensity'},
                              calibration={'rank', 'matching'}, joint={'fm', 'process', 'verifier'})
            if qwen_process:
                objectives = dict(calibration={'rank'}, interface={'fm', 'process', 'intensity', 'verifier'},
                                  joint={'fm', 'process', 'verifier'})
            if (cfg['knowledge'].get('latent_reasoning_attention') != 'joint' or
                (not qwen_process and not cfg['verifier'].get('response_only')) or
                cfg['condition_use'].get('mode') != 'positive_only'):
                raise ValueError('v8 requires joint C reasoning, response evidence and positive-only generator training')
        if set(cfg['loss']) != set(objectives) or any(set(cfg['loss'][p]) != names for p, names in objectives.items()):
            raise ValueError('Objective names must match this operator revision; extra auxiliaries are not supported')
        if any(v < 0 for losses in cfg['loss'].values() for v in losses.values()):
            raise ValueError('Objective weights must be nonnegative')
        expected_verifier = 'qwen_process' if qwen_process else 'conditional_process'
        if cfg['model'].get('injection') != 'conditional_operator' or cfg['verifier']['kind'] != expected_verifier:
            raise ValueError('Operator training requires the conditional state operator and process verifier')
        c = cfg['completion']
        if cfg['revision'] in (REASONING_REVISION, RESPONSE_REVISION, QWEN_REVISION) and c.get('feedback_to') != 'latent':
            raise ValueError('Reasoning verifier feedback must update C/knowledge')
        if not 0 < c['sigma_min'] <= c['sigma_max'] <= 1 or c['steps'] < 1 or c['every'] < 1:
            raise ValueError('Invalid complete conditional denoising window')
        if cfg['condition_use']['temperature'] <= 0 or cfg['verifier']['temperature'] <= 0:
            raise ValueError('Preference temperatures must be positive')
        if cfg['sequence']['state_weight'] != 0 or cfg['sequence']['perceptual_fraction'] != 0:
            raise ValueError('Operator process supervision is signed RGB change; no hidden level/perceptual auxiliary')
        if qwen_process:
            if any(k.startswith('video_evidence_') for k in cfg['paths']):
                raise ValueError('The Qwen-only revision must not configure a JEPA encoder or checkpoint')
            if not 3 <= cfg['verifier']['frames'] <= cfg['data']['frames']:
                raise ValueError('Qwen process evidence must cover the complete video interval')
            if not c.get('from_pure_noise') or c['sigma_min'] != 1 or c['sigma_max'] != 1:
                raise ValueError('Qwen process feedback uses a connected trajectory from pure noise')
    elif set(cfg['loss']) != {'fm', 'sequence', 'intensity', 'critic', 'physical', 'rollout'}:
        raise ValueError('Objective definitions differ from this revision')
    m = cfg['model']
    if m['attention_width'] % m['attention_heads']: raise ValueError('Invalid attention dimensions')
    if not operator:
        r = cfg['rollout']
        if not 1 <= r['gradient_steps'] <= r['steps']: raise ValueError('Invalid sampling gradient horizon')
    if any(x % 32 for key in ('image_size', 'video_size') for x in cfg['knowledge'][key]):
        raise ValueError('Qwen geometry must divide by patch times merge size')
    if cfg['train']['accumulation'] < 1: raise ValueError('Positive accumulation required')
    if not 0 <= cfg['train'].get('native_text_only_dropout',0.) <= 1:
        raise ValueError('Invalid independent native-text dropout probability')
    return cfg


def phase_at(cfg, step):
    end = 0
    for phase, count in cfg['stages'].items():
        end += count
        if step < end: return phase
    return 'joint'


def smoke_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg['run_name'] += '-smoke'
    cfg['stages'] = {phase: 1 for phase in cfg['stages']}
    cfg['train'].update(accumulation=1, save_every=1)
    cfg['data']['workers'] = 0
    cfg['verifier'].update(every=1, warmup=1)
    if cfg['revision'] in OPERATOR_REVISIONS: cfg['completion']['every'] = 1
    else: cfg['rollout'].update(every=1, start=0)
    cfg['validation'].update(initial=False, every=999999, sample_cases=0)
    cfg['smoke'] = True
    return cfg
