"""Versioned training objectives and progressive stages."""
import copy
import math
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
GROUNDED_QWEN_REVISION = 'v522_native_grounded_qwen_process_v10'
BASELINE_REVISION = 'v522_native_generation_baseline_v11'
FULLWIDTH_REVISION = 'v522_native_fullwidth_attention_v12'
ALIGNED_REVISION = 'v522_native_aligned_attention_v13'
MULTIROUND_REVISION = 'v522_native_multiround_reasoning_v14'
H_CONTENT_MESSAGE_REVISION = 'v522_native_h_content_message_v15'
Q_SEED128_MESSAGE_REVISION = 'v522_native_q_seed128_message_v16'
VERIFIER_MESSAGE_VARIANTS = {
    'v522_native_h_content_verifier_frozen_v17': ('h_content_message', False),
    'v522_native_h_content_verifier_lora_v18': ('h_content_message', True),
    'v522_native_q_seed128_verifier_frozen_v19': ('q_seed128_message', False),
    'v522_native_q_seed128_verifier_lora_v20': ('q_seed128_message', True),
}
NATIVE_GEOMETRY_VARIANTS = {
    'v522_native_h_content_verifier_frozen_geometry_v21': ('h_content_message', False),
    'v522_native_h_content_verifier_lora_geometry_v22': ('h_content_message', True),
    'v522_native_q_seed128_verifier_frozen_geometry_v23': ('q_seed128_message', False),
    'v522_native_q_seed128_verifier_lora_geometry_v24': ('q_seed128_message', True),
}
SIMPLE_MESSAGE_VARIANTS = {
    'v522_native_h_content_verifier_frozen_simple_v25': ('h_content_message', False),
    'v522_native_h_content_verifier_lora_simple_v26': ('h_content_message', True),
    'v522_native_q_seed128_verifier_frozen_simple_v27': ('q_seed128_message', False),
    'v522_native_q_seed128_verifier_lora_simple_v28': ('q_seed128_message', True),
}
# v25/v27 retain their historical fixed-judge feedback for checkpoint replay.
# The user's frozen ablation means the verifier is absent, not just frozen.
NO_VERIFIER_MESSAGE_VARIANTS = {
    'v522_native_h_content_no_verifier_simple_v29': 'h_content_message',
    'v522_native_q_seed128_no_verifier_simple_v30': 'q_seed128_message',
}
SIMPLE_ARCHITECTURE_REVISIONS = (*SIMPLE_MESSAGE_VARIANTS, *NO_VERIFIER_MESSAGE_VARIANTS)
CRITERION_MESSAGE_VARIANTS = {**VERIFIER_MESSAGE_VARIANTS, **NATIVE_GEOMETRY_VARIANTS,
                            **SIMPLE_MESSAGE_VARIANTS}
MESSAGE_MODES_BY_REVISION = {H_CONTENT_MESSAGE_REVISION: 'h_content_message',
                           Q_SEED128_MESSAGE_REVISION: 'q_seed128_message',
                           **NO_VERIFIER_MESSAGE_VARIANTS,
                           **{k: v[0] for k, v in CRITERION_MESSAGE_VARIANTS.items()}}
MESSAGE_REVISIONS = tuple(MESSAGE_MODES_BY_REVISION)
ALIGNED_REVISIONS = (ALIGNED_REVISION, MULTIROUND_REVISION, *MESSAGE_REVISIONS)
QWEN_REVISIONS = (QWEN_REVISION, GROUNDED_QWEN_REVISION, BASELINE_REVISION, FULLWIDTH_REVISION, *ALIGNED_REVISIONS)
OPERATOR_REVISIONS = (OPERATOR_REVISION, REASONING_REVISION, RESPONSE_REVISION, *QWEN_REVISIONS)
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
    multiround = cfg['revision'] == MULTIROUND_REVISION
    message = cfg['revision'] in MESSAGE_REVISIONS
    message_modes = MESSAGE_MODES_BY_REVISION
    criteria = cfg['revision'] in CRITERION_MESSAGE_VARIANTS
    verifier_lora = criteria and CRITERION_MESSAGE_VARIANTS[cfg['revision']][1]
    no_verifier = cfg['revision'] in NO_VERIFIER_MESSAGE_VARIANTS
    simple = cfg['revision'] in SIMPLE_ARCHITECTURE_REVISIONS
    native_geometry = cfg['revision'] in (*NATIVE_GEOMETRY_VARIANTS, *SIMPLE_ARCHITECTURE_REVISIONS)
    if no_verifier:
        if cfg['verifier'] != {'kind': 'none'} or 'completion' in cfg:
            raise ValueError('No-verifier recipes contain no judge or completion-feedback configuration')
        if any(k.startswith('verifier_') or k == 'critic_lr' for k in cfg['train']):
            raise ValueError('No-verifier recipes contain no verifier optimizer settings')
    if native_geometry != (cfg.get('geometry', {}).get('mode') == 'native'):
        raise ValueError('Native observation geometry requires its own checkpoint revision')
    if message and cfg['knowledge'].get('reasoning_mode') != message_modes[cfg['revision']]:
        raise ValueError('Message mode must match its checkpoint revision')
    if not message and cfg['knowledge'].get('reasoning_mode') in message_modes.values():
        raise ValueError('Message architecture requires its own checkpoint revision')
    if (cfg['knowledge'].get('reasoning_mode') == 'recurrent_prefix') != multiround:
        raise ValueError('Full-depth recurrent reasoning requires its own v14 checkpoint revision')
    source = cfg['knowledge'].get('condition_source', 'qwen')
    if source not in ('qwen', 'cached_t5'):
        raise ValueError('Condition source must be Qwen or the explicit cached-T5 ablation')
    if source == 'cached_t5' and (cfg['revision'] not in (RESPONSE_REVISION, *QWEN_REVISIONS) or rounds):
        raise ValueError('The cached-T5 ablation uses v8/v9 without Qwen latent reasoning')
    if not isinstance(rounds, int) or rounds < 0:
        raise ValueError('Latent reasoning steps must be a nonnegative integer')
    if rounds and not (multiround or message):
        if cfg['revision'] not in (REASONING_REVISION, RESPONSE_REVISION, *QWEN_REVISIONS):
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
    qwen_process = cfg['revision'] in QWEN_REVISIONS
    baseline = cfg['revision'] in (BASELINE_REVISION,FULLWIDTH_REVISION,*ALIGNED_REVISIONS)
    aligned = cfg['revision'] in ALIGNED_REVISIONS
    if message:
        k, m = cfg['knowledge'], cfg['message']
        if source != 'qwen' or k.get('condition_readout') != 'content' or not k.get('freeze_backbone'):
            raise ValueError('Message conditions require frozen Qwen and complete content')
        if any(x in k for x in ('max_condition_tokens', 'latent_reasoning_layers', 'lora_rank')):
            raise ValueError('Message conditions use no Qwen LoRA, pooled content or native layer replay')
        if not k.get('include_prompt_span') or k.get('latent_reasoning_attention') != 'bidirectional':
            raise ValueError('Message workspaces retain requirements and use bidirectional updates')
        if k['reasoning_mode'] == 'q_seed128_message' and k.get('relation_tokens') != 128:
            raise ValueError('Q_seed128_message has exactly 128 learned queries')
        if k['reasoning_mode'] == 'h_content_message' and 'relation_tokens' in k:
            raise ValueError('H_content_message derives its length from content')
        workspace_width = m.get('workspace_width', cfg['model']['width'])
        if (type(workspace_width) is not int or workspace_width < 1 or
                type(m['heads']) is not int or m['heads'] < 1 or workspace_width % m['heads'] or
                type(m['ffn_multiplier']) is not int or m['ffn_multiplier'] < 1 or m['init_std'] <= 0):
            raise ValueError('Invalid full-width message block dimensions')
        if simple:
            if m.get('projection') != 'output' or workspace_width != cfg['model']['qwen_width']:
                raise ValueError('Simple messages retain Qwen width until the output projection to Wan')
        elif 'workspace_width' in m or 'projection' in m:
            raise ValueError('Moving the message projection requires a new checkpoint revision')
        exits = k.get('train_reasoning_rounds', [])
        if not exits or len(set(exits)) != len(exits) or any(type(x) is not int or not 0 <= x <= rounds for x in exits):
            raise ValueError('Training exits must be distinct configured message rounds')
        if k.get('inference_reasoning_round') not in exits:
            raise ValueError('Default message inference exit must be trained')
        if not 0 <= k['reasoning_tokens'] <= k['reasoning_max_tokens'] <= k['max_text_tokens']:
            raise ValueError('Analysis budgets must fit the explicit text budget')
    if multiround:
        k = cfg['knowledge']
        if source != 'qwen' or k.get('condition_readout') != 'content':
            raise ValueError('v14 requires complete Qwen content readouts')
        if any(x in k for x in ('max_condition_tokens', 'latent_reasoning_layers')):
            raise ValueError('v14 has no pooled token budget or late-layer replay')
        if k.get('latent_reasoning_attention') != 'causal' or not k.get('include_prompt_span'):
            raise ValueError('v14 uses native causal working tokens and complete requirements')
        if not isinstance(k.get('relation_tokens'), int) or k['relation_tokens'] < 1:
            raise ValueError('A positive relation workspace is required')
        exits = k.get('train_reasoning_rounds', [])
        if not exits or len(set(exits)) != len(exits) or any(type(x) is not int or not 0 <= x <= rounds for x in exits):
            raise ValueError('Training exits must be distinct configured reasoning rounds')
        if k.get('inference_reasoning_round') not in exits:
            raise ValueError('Inference exit must be covered during training')
        if not 0 <= k['reasoning_tokens'] <= k['reasoning_max_tokens'] <= k['max_text_tokens']:
            raise ValueError('Analysis budgets must fit the explicit text budget')
    phases = (('calibration', 'interface', 'joint') if qwen_process and not baseline else
              ('interface', 'calibration', 'joint')) if operator else ('interface', 'joint')
    if message: phases = ('interface', 'joint')
    if verifier_lora: phases = ('interface', 'calibration', 'joint')
    if tuple(cfg['stages']) != phases or min(cfg['stages'].values()) < 1:
        raise ValueError('Use the ordered positive stage budgets for this revision')
    if operator:
        objectives = dict(interface={'fm', 'process', 'use', 'intensity'},
                          calibration={'rank', 'matching'}, joint={'fm', 'process', 'use', 'verifier'})
        if cfg['revision'] in (RESPONSE_REVISION, *QWEN_REVISIONS):
            objectives = dict(interface={'fm', 'process', 'intensity'},
                              calibration={'rank', 'matching'}, joint={'fm', 'process', 'verifier'})
            if qwen_process:
                objectives = dict(calibration={'rank'}, interface={'fm', 'process', 'intensity', 'verifier'},
                                  joint={'fm', 'process', 'verifier'})
            if baseline:
                objectives = dict(interface={'fm', 'flow_temporal', 'process', 'intensity'}, calibration={'rank'},
                                  joint={'fm', 'flow_temporal', 'process', 'verifier'})
                if aligned: objectives['interface'].remove('intensity')
            if message:
                objectives = dict(interface={'fm', 'flow_temporal', 'process'},
                                  joint={'fm', 'flow_temporal', 'process', 'verifier'})
                if simple:
                    objectives['interface'].remove('process')
                    objectives['joint'].remove('process')
                if no_verifier: objectives['joint'].remove('verifier')
                if verifier_lora: objectives['calibration'] = {'judge'}
            if (cfg['knowledge'].get('latent_reasoning_attention') != ('bidirectional' if message else 'causal' if multiround else 'joint') or
                (not qwen_process and not cfg['verifier'].get('response_only')) or
                cfg['condition_use'].get('mode') != 'positive_only'):
                raise ValueError('v8 requires joint C reasoning, response evidence and positive-only generator training')
        if set(cfg['loss']) != set(objectives) or any(set(cfg['loss'][p]) != names for p, names in objectives.items()):
            raise ValueError('Objective names must match this operator revision; extra auxiliaries are not supported')
        if any(v < 0 for losses in cfg['loss'].values() for v in losses.values()):
            raise ValueError('Objective weights must be nonnegative')
        expected_verifier = 'none' if no_verifier else 'qwen_criteria' if criteria else 'qwen_native' if message else 'qwen_process' if qwen_process else 'conditional_process'
        injection = ('aligned_native_cross_attention' if aligned else
                     'native_cross_attention' if cfg['revision']==FULLWIDTH_REVISION else 'conditional_operator')
        if cfg['model'].get('injection') != injection or cfg['verifier']['kind'] != expected_verifier:
            raise ValueError('The injection architecture and process verifier must match the checkpoint revision')
        c = cfg.get('completion', {})
        if not no_verifier and cfg['revision'] in (REASONING_REVISION, RESPONSE_REVISION, *QWEN_REVISIONS) and c.get('feedback_to') != 'latent':
            raise ValueError('Reasoning verifier feedback must update C/knowledge')
        if not no_verifier and (not 0 < c['sigma_min'] <= c['sigma_max'] <= 1 or c['steps'] < 1 or c['every'] < 1):
            raise ValueError('Invalid complete conditional denoising window')
        if cfg['condition_use']['temperature'] <= 0 or (not no_verifier and cfg['verifier']['temperature'] <= 0):
            raise ValueError('Preference temperatures must be positive')
        if not simple and (cfg['sequence']['state_weight'] != 0 or cfg['sequence']['perceptual_fraction'] != 0):
            raise ValueError('Operator process supervision is signed RGB change; no hidden level/perceptual auxiliary')
        if qwen_process:
            if any(k.startswith('video_evidence_') for k in cfg['paths']):
                raise ValueError('The Qwen-only revision must not configure a JEPA encoder or checkpoint')
            if not no_verifier and not 3 <= cfg['verifier']['frames'] <= cfg['data']['frames']:
                raise ValueError('Qwen process evidence must cover the complete video interval')
            if not no_verifier and (not c.get('from_pure_noise') or c['sigma_min'] != 1 or c['sigma_max'] != 1):
                raise ValueError('Qwen process feedback uses a connected trajectory from pure noise')
            if not no_verifier and cfg['revision'] in (GROUNDED_QWEN_REVISION, BASELINE_REVISION,FULLWIDTH_REVISION,*ALIGNED_REVISIONS) and (
                    cfg['knowledge'].get('video_observation_mode') != ('video' if simple else 'images') or
                    not cfg['knowledge'].get('video_observation_first') or
                    cfg['knowledge'].get('video_comparison') != 'symmetric_observations'):
                raise ValueError('Grounded Qwen uses timestamped frames, independent observations and symmetric comparison')
            if baseline:
                if (not simple and not 0 < cfg['sequence']['max_sigma'] < 1) or not cfg['flow_temporal']['lags']:
                    raise ValueError('The baseline separates low-noise RGB recovery from full-range temporal FM')
                if any(not isinstance(lag, int) or lag < 1 for lag in cfg['flow_temporal']['lags']):
                    raise ValueError('Temporal FM lags are positive latent-time offsets')
                if aligned and cfg['train'].get('wan_lora_in_interface', False):
                    raise ValueError('C alignment freezes all Wan parameters including LoRA until joint')
                if ((not aligned and not cfg['train'].get('wan_lora_in_interface')) or
                        (not message and not cfg['verifier'].get('calibrate_current'))):
                    raise ValueError('The baseline trains the inherited Wan adapter and calibrates actual conditioned generation')
    elif set(cfg['loss']) != {'fm', 'sequence', 'intensity', 'critic', 'physical', 'rollout'}:
        raise ValueError('Objective definitions differ from this revision')
    m = cfg['model']
    if m['attention_width'] % m['attention_heads']: raise ValueError('Invalid attention dimensions')
    if cfg['revision'] in (FULLWIDTH_REVISION,*ALIGNED_REVISIONS):
        if m['attention_width']!=m['width'] or 'interface_rank' in m:
            raise ValueError('Full-width native attention has no compact width or rank-bottleneck setting')
    if aligned:
        alignment = cfg.get('alignment', {})
        if (alignment.get('initialization') != 'native_text_context' or
                not isinstance(alignment.get('samples'), int) or alignment['samples'] < 1 or
                alignment.get('eps', 0) <= 0):
            raise ValueError('Full-width C alignment requires train-cache native context initialization')
    if not operator:
        r = cfg['rollout']
        if not 1 <= r['gradient_steps'] <= r['steps']: raise ValueError('Invalid sampling gradient horizon')
    if any(x % 32 for key in ('image_size', 'video_size') for x in cfg['knowledge'][key]):
        raise ValueError('Qwen geometry must divide by patch times merge size')
    if cfg['train']['accumulation'] < 1: raise ValueError('Positive accumulation required')
    if native_geometry:
        from .geometry import require_same_aspect
        size = cfg['geometry'].get('rgb_size')
        if not isinstance(size, list) or len(size) != 2 or any(type(x) is not int or x <= 0 or x % 32 for x in size):
            raise ValueError('Native HxW must align with Qwen spatial merging and Wan patches')
        if size != [16 * x for x in cfg['data']['latent_shape'][-2:]]:
            raise ValueError('Native observation geometry must match Wan2.2 VAE spatial stride 16')
        if cfg['data']['frames'] != 1 + 4 * (cfg['data']['latent_shape'][1] - 1):
            raise ValueError('Native RGB frame count must match the Wan VAE temporal geometry')
        views = [cfg['knowledge']['image_size'], cfg['knowledge']['video_size']]
        if not no_verifier: views.append(cfg['verifier']['size'])
        if any(value != size for value in views):
            raise ValueError('Reasoning, teacher and verifier must all use the native RGB geometry')
        require_same_aspect(size, cfg['sequence']['size'])
        if cfg['sequence']['size'] != size:
            raise ValueError('Native geometry requires native-resolution RGB supervision, without a reduced input view')
        if cfg['sequence']['scales'] != [1]:
            raise ValueError('Native RGB supervision uses scale 1 only; spatial pooling is disabled')
    if criteria:
        v = cfg['verifier']; adapter = v['adapter']
        if adapter.get('enabled') is not verifier_lora:
            raise ValueError('Verifier adapter mode must match the saved revision')
        if cfg['train']['qwen_lr'] != 0 or cfg['train']['critic_lr'] != 0:
            raise ValueError('Message generation Qwen and score head remain frozen')
        if not 0 <= v['teacher_min_confidence'] <= 1 or not 0 <= v['teacher_max_order_gap'] <= 1:
            raise ValueError('Invalid weak-teacher reliability thresholds')
        if type(v['observation_tokens']) is not int or not 0 < v['observation_tokens'] < cfg['knowledge']['max_text_tokens']:
            raise ValueError('Teacher observations need a bounded text budget')
        if v.get('protocol') != ('visible_process_v2' if simple else 'visible_process_v1') or v.get('label_orders') != 2:
            raise ValueError('Use four evidence criteria and two canonicalized label assignments')
        if not isinstance(v.get('gradient_checkpointing'), bool):
            raise ValueError('Explicit verifier checkpointing policy required')
        if verifier_lora:
            if (type(adapter.get('rank')) is not int or adapter['rank'] < 1 or
                    type(adapter.get('last_layers')) is not int or adapter['last_layers'] < 1 or
                    adapter.get('alpha', 0) <= 0 or cfg['train'].get('verifier_lr', 0) <= 0):
                raise ValueError('Verifier LoRA needs a positive rank, depth, scale and dedicated LR')
        elif set(adapter) != {'enabled'} or cfg['train'].get('verifier_lr', 0) != 0:
            raise ValueError('Frozen verifier installs no adapter or adapter optimizer')
    if not 0 <= cfg['train'].get('native_text_only_dropout',0.) <= 1:
        raise ValueError('Invalid independent native-text dropout probability')
    if simple and not no_verifier:
        k, t, c = cfg['knowledge'], cfg['train'], cfg['completion']
        if k['train_reasoning_rounds'] != [k['inference_reasoning_round']]:
            raise ValueError('The simple recipe trains its fixed deployment exit')
        stride = v.get('frame_stride', 0)
        if v.get('input_mode') != 'video' or type(stride) is not int or not 1 <= stride <= 3:
            raise ValueError('The simple verifier uses native video with frame stride at most 3')
        indices = list(range(0, cfg['data']['frames'], stride))
        if indices[-1] != cfg['data']['frames'] - 1: indices.append(cfg['data']['frames'] - 1)
        if v['frames'] != len(indices):
            raise ValueError('Verifier frame count must cover its stride and the final frame')
        if (c.get('guidance') != cfg['inference']['guidance'] or
                c['shift'] != cfg['inference']['shift'] or c['guidance'] <= 1 or
                cfg['inference'].get('latent_feedback', {}).get('enabled', False)):
            raise ValueError('Feedback and deployment must share CFG/shift without online C updates')
        if c.get('branch_checkpointing') is not True:
            raise ValueError('Single-GPU CFG feedback checkpoints each Wan branch')
    if simple:
        k, t = cfg['knowledge'], cfg['train']
        if k['train_reasoning_rounds'] != [k['inference_reasoning_round']]:
            raise ValueError('The simple recipe trains its fixed deployment exit')
        if no_verifier and cfg['inference'].get('latent_feedback', {}).get('enabled', False):
            raise ValueError('No-verifier inference has no online feedback')
        if t.get('optimizer_policy') != 'simple_message' or t.get('clip_by_owner') is not True:
            raise ValueError('The simple recipe needs phase-aware optimizer groups and separate clipping')
        if t.get('dropout_policy') != 'exclusive' or t['text_dropout'] + t['native_text_only_dropout'] > 1:
            raise ValueError('Text dropout modes must be disjoint probabilities')
        if not 0 <= t['requirements_dropout'] <= 1 or not 0 <= t['text_dropout'] <= 1:
            raise ValueError('Invalid condition dropout probability')
        rates = ('new_lr', 'joint_lr', 'alignment_lr', 'joint_alignment_lr', 'wan_lora_lr')
        if any(not math.isfinite(t.get(key, 0)) or t.get(key, 0) <= 0 for key in rates):
            raise ValueError('Simple optimizer learning rates must be finite and positive')
        if not 0 < t['minimum_lr_ratio'] <= 1 or t['warmup'] < 1 or (not no_verifier and t.get('verifier_warmup', 0) < 1):
            raise ValueError('Invalid simple learning-rate schedule')
        if t['alignment_lr'] > t['new_lr'] or t['joint_alignment_lr'] > t['joint_lr']:
            raise ValueError('Alignment scales must not learn faster than the C workspace')
        if any(not math.isfinite(weight) or weight <= 0 for loss in cfg['loss'].values() for weight in loss.values()):
            raise ValueError('All simple objectives need finite positive weights')
    return cfg


def phase_at(cfg, step):
    end = 0
    for phase, count in cfg['stages'].items():
        end += count
        if step < end: return phase
    return 'joint'


def select_reasoning_exit(cfg, *, without_analysis=False, without_updates=False, reasoning_round=None):
    """Inference intervention after checkpoint restoration; never changes weights."""
    k = cfg['knowledge']
    if without_updates and reasoning_round not in (None, 0):
        raise ValueError('Zero additional updates conflicts with the requested exit')
    if reasoning_round is not None and cfg['revision'] not in (MULTIROUND_REVISION, *MESSAGE_REVISIONS):
        raise ValueError('Independent reasoning exits require a v14 checkpoint')
    if cfg['revision'] in MESSAGE_REVISIONS:
        chosen = 0 if without_updates else reasoning_round
        if chosen is not None:
            if type(chosen) is not int or not 0 <= chosen <= k['latent_reasoning_steps']:
                raise ValueError('Message intervention exit is outside the configured range')
            k['inference_reasoning_round'] = chosen
    elif cfg['revision'] == MULTIROUND_REVISION:
        chosen = 0 if without_updates else reasoning_round
        if chosen is not None:
            if chosen not in k['train_reasoning_rounds']: raise ValueError('Select an exit covered by this checkpoint configuration')
            k['inference_reasoning_round'] = chosen
    elif without_updates: k['latent_reasoning_steps'] = 0
    if without_analysis: k['generation_reasoning'] = False


def smoke_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg['run_name'] += '-smoke'
    cfg['stages'] = {phase: 1 for phase in cfg['stages']}
    cfg['train'].update(accumulation=1, save_every=1)
    cfg['data']['workers'] = 0
    if cfg['revision'] not in NO_VERIFIER_MESSAGE_VARIANTS:
        cfg['verifier'].update(every=1, warmup=1)
        if cfg['revision'] in OPERATOR_REVISIONS: cfg['completion']['every'] = 1
        else: cfg['rollout'].update(every=1, start=0)
    cfg['validation'].update(initial=False, every=999999, sample_cases=0)
    cfg['smoke'] = True
    return cfg
