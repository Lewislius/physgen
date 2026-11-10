"""Independent native-process revision and two continuous training phases."""
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'v522_native_physical_dependency_v1'
TEMPORAL_REVISION = 'v522_native_temporal_distribution_v2'
ORDERED_REVISION = 'v522_native_ordered_process_v3'
CALIBRATED_REVISION = 'v522_native_calibrated_verifier_v4'
PAIRED_REVISION = 'v522_native_paired_verifier_v5'
REVISIONS = (REVISION,TEMPORAL_REVISION,ORDERED_REVISION,CALIBRATED_REVISION,PAIRED_REVISION)


def read_config(path):
    def expand(value):
        if isinstance(value, str): return value.replace('${ROOT}', str(ROOT))
        if isinstance(value, dict): return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list): return [expand(v) for v in value]
        return value
    return validate(expand(yaml.safe_load(Path(path).read_text())))


def validate(cfg):
    if cfg['revision'] not in REVISIONS: raise ValueError('Use a native-process checkpoint revision')
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
    if set(cfg['stages']) != {'interface', 'joint'} or min(cfg['stages'].values()) < 1:
        raise ValueError('Both generation-training phases require positive budgets')
    if set(cfg['loss']) != {'fm', 'sequence', 'intensity', 'critic', 'physical', 'rollout'}:
        raise ValueError('Objective definitions differ from this revision')
    m, r = cfg['model'], cfg['rollout']
    if m['attention_width'] % m['attention_heads']: raise ValueError('Invalid attention dimensions')
    if not 1 <= r['gradient_steps'] <= r['steps']: raise ValueError('Invalid sampling gradient horizon')
    if any(x % 32 for key in ('image_size', 'video_size') for x in cfg['knowledge'][key]):
        raise ValueError('Qwen geometry must divide by patch times merge size')
    if cfg['train']['accumulation'] < 1: raise ValueError('Positive accumulation required')
    if not 0 <= cfg['train'].get('native_text_only_dropout',0.) <= 1:
        raise ValueError('Invalid independent native-text dropout probability')
    return cfg


def phase_at(cfg, step):
    return 'interface' if step < cfg['stages']['interface'] else 'joint'


def smoke_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg['run_name'] += '-smoke'
    cfg['stages'] = dict(interface=1, joint=1)
    cfg['train'].update(accumulation=1, save_every=1)
    cfg['data']['workers'] = 0
    cfg['verifier'].update(every=1, warmup=1)
    cfg['rollout'].update(every=1, start=0)
    cfg['validation'].update(every=999999, sample_cases=0)
    cfg['smoke'] = True
    return cfg
