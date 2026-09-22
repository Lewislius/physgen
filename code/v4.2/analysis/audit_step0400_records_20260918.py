"""Summarize existing records only; no training or inference is performed."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/step0400_generation_failure_20260918'
V4 = ROOT.parent/'v4'
LOG4 = V4/'train/train_log/wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/20260912T151800Z-e189c27c'
LOG42 = ROOT/'train/train_log/v4_joint_20260918T034836Z_16183b'
GEN42 = ROOT/'inference_outputs/v4_joint_20260918T034836Z_16183b_step0400_ema_demo_seed42_r3_full'
GEN4 = V4/'inference_outputs/wisa_native_p_1xada48g/final_20260913-070408-373326450'


def rows(path):
    result = []
    with path.open() as stream:
        for line in stream:
            # Running training may be appending its last line.
            if not line.endswith('\n'):
                continue
            result.append(json.loads(line))
    return result


def avg(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    step4 = rows(LOG4/'steps.jsonl')
    update42 = rows(LOG42/'updates.jsonl')
    micro42 = rows(LOG42/'micro.jsonl')
    health42 = rows(LOG42/'health.jsonl')
    t4 = [v for v in step4 if v['split']=='train']
    t42 = [v for v in update42 if v['event']=='update' and v['step']<=600]
    report = dict(sources=dict(v4=str(LOG4),v42=str(LOG42),generation=str(GEN42)),
                  primary_checkpoint_step=400,extra_training_cutoff=600,
                  v4_config=json.loads((LOG4/'config.json').read_text()),
                  v42_config=json.loads((ROOT/'checkpoints/v4_joint_20260918T034836Z_16183b/step0400/config.json').read_text()),
                  v4_runtime=json.loads((LOG4/'runtime.json').read_text()),
                  windows=[],validation42=[v for v in update42 if v['event']=='validation' and v['step']<=600])
    for low,high in [(1,50),(151,200),(351,400),(451,500),(551,600)]:
        a=[v['metrics'] for v in t4 if low<=v['step']<=high]
        b=[v for v in t42 if low<=v['step']<=high]
        m=[v for v in micro42 if low<=v['step']<=high]
        entry=dict(steps=[low,high])
        if a:
            entry['v4']={k:avg(v[k] for v in a) for k in ['loss/fm','loss/struct','loss/prior',
                         'loss/weighted_struct','loss/weighted_prior','loss/total','learning_rate',
                         'grad/global_norm_before_clip','data/frames','prior/gt_conditioned']}
            entry['v4']['write_ratio']={str(n):avg(v[f'A@{n}/write_relative_rms'] for v in a) for n in [5,10,15,20,25,30]}
            entry['v4']['state_gate']={str(n):avg(v[f'A@{n}/state_gate_mean'] for v in a) for n in [5,10,15,20,25,30]}
        if b:
            entry['v42']=dict(loss={k:avg(v['loss_terms'][k]['raw_mean'] for v in b) for k in ['fm','struct','prior','write']},
                             global_grad=avg(v['global_grad_norm'] for v in b),
                             modes={mode:{k:avg(v[k] for v in m if v['mode']==mode) for k in ['fm','fm_base','fm_delta']}
                                    for mode in ['i2v','t2v']})
            entry['v42']['interventions']={g:{k:avg(v['interventions'][g].get(k) for v in m)
                for k in ['candidate_rms','actual_rms','compression','over_base','gate_mean']}
                for g in ['state5','state15','writer5','writer15','velocity']}
        report['windows'].append(entry)
    primary=[v for v in t42 if v['step']<=400]
    report['clipping42']=dict(count=len(primary),triggered=sum(v['global_grad_norm']>1 for v in primary),
                             max_norm=max(v['global_grad_norm'] for v in primary))
    diagnostics=[v['gradient_diagnostics']['groups'] for v in t42 if 101<=v['step']<=400 and v.get('gradient_diagnostics')]
    report['gradients42']=dict(scope='first micro only, every 25 steps, steps 101..400',count=len(diagnostics),
        groups={g:dict(fm_norm=avg(v[g]['fm_norm'] for v in diagnostics),
                       auxiliary_norm=avg(v[g]['auxiliary_norm'] for v in diagnostics),
                       cosine=avg(v[g]['cosine'] for v in diagnostics),
                       conflicts=sum(v[g]['conflict'] for v in diagnostics),
                       joint_opposes_fm=sum(v[g]['fm_dot_joint']<0 for v in diagnostics)) for g in diagnostics[0]})
    report['health400_by_case']=[dict(id=v['id'],mode=v['mode'],sigma=v['sigma'],fm=v['fm'],base=v['fm_base'],
                                    delta=v['fm_delta'],struct=v['struct']) for v in health42
                               if v['event']=='health' and v['step']==400 and v['weights']=='ema']
    report['generation42']={}
    for folder in sorted(GEN42.iterdir()):
        if not (folder/'metadata.json').exists():
            continue
        meta=json.loads((folder/'metadata.json').read_text())
        if meta['status']!='completed':
            continue
        trace=rows(folder/'trajectory.jsonl')
        summary={g:{k:avg(v['metrics'][g].get(k) for v in trace) for k in ['compression','over_base','gate_mean']}
                 for g in ['writer5','writer15','velocity']}
        align=[v['metrics'].get('cfg_alignment',{}) for v in trace]
        summary['cfg_alignment_keys']=list(align[0])
        summary['cfg_alignment']={k:avg(v.get(k) for v in align) for k in align[0]
                                  if isinstance(align[0][k],(int,float)) or align[0][k] is None}
        report['generation42'][folder.name]=summary
    report['generation4']={}
    for mode in ['i2v','t2v']:
        trace=rows(GEN4/f'demo20/P01/{mode}/seed42_full.steps.jsonl')
        report['generation4'][mode]={branch:{str(n):avg(v['metrics'][branch][f'A@{n}/write_relative_rms'] for v in trace)
                                             for n in [5,10,15,20,25,30]} for branch in ['cond','uncond']}
    report['caveats']=[
        'V4 and V4.2 losses use different samples, modes, video windows and teacher frame sampling; raw values are not paired comparisons.',
        'Gradient diagnostics are sparse first-micro observations, not accumulated optimizer update directions.',
        'Existing health checks are GT-noised conditional FM; they do not validate free-running CFG video quality.',
        'No new GPU inference was performed by this script.'
    ]
    (OUT/'record_evidence.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:report[k] for k in ['clipping42','gradients42','generation4']},indent=2))
    print('saved',OUT/'record_evidence.json')


if __name__=='__main__':
    main()
