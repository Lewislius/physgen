"""Read completed paired scores and matched training logs; no new model runs."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,statistics

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'r11_r12/reports/R11A_physical_content_67214'


def paired(actual,reference):
    a={v['id']:v for v in actual['cases']};b={v['id']:v for v in reference['cases']}
    if set(a)!=set(b):raise ValueError('Paired comparison needs identical cases')
    if any(a[k]['caption']!=b[k]['caption'] for k in a):raise ValueError('Paired captions differ')
    result={}
    for name in ('sa','pc'):
        changes=[a[k][name]-b[k][name] for k in a]
        result[name]=dict(mean_change=statistics.mean(changes),wins=sum(v>0 for v in changes),
            ties=sum(v==0 for v in changes),losses=sum(v<0 for v in changes),
            transitions=dict(Counter(f'{b[k][name]}->{a[k][name]}' for k in a)),
            pass_gains=[k for k in a if a[k][name]>=4 and b[k][name]<4],
            pass_losses=[k for k in a if a[k][name]<4 and b[k][name]>=4],
            changed_cases=[dict(id=k,caption=a[k]['caption'],before=b[k][name],after=a[k][name]) for k in a if a[k][name]!=b[k][name]])
    joint=lambda v:v['sa']>=4 and v['pc']>=4
    result['joint']=dict(gains=[k for k in a if joint(a[k]) and not joint(b[k])],
        losses=[k for k in a if joint(b[k]) and not joint(a[k])])
    return result


def logs(folder):
    records=[json.loads(s) for s in (folder/'train.jsonl').read_text().splitlines()]
    train=[v for v in records if v['kind']=='train' and v['step']<=400]
    micro=[v for v in records if v['kind']=='micro' and v['step']<=400]
    result=dict(source=str(folder/'train.jsonl'),steps=len(train),microbatches=len(micro),windows={})
    for name,low,high in [('first128',1,128),('129to250',129,250),('251to400',251,400),('last50',351,400)]:
        rows=[v for v in train if low<=v['step']<=high];examples=[v for v in micro if low<=v['step']<=high]
        values={k:statistics.mean(v[k] for v in rows if k in v) for k in ('fm','generation','tail','aux') if any(k in v for v in rows)}
        values.update({k:statistics.mean(v[k] for v in examples if k in v) for k in ('c_pair_cosine','c_centered_energy') if any(k in v for v in examples)})
        result['windows'][name]=values
    groups=sorted({k for v in train for k in v['gradients']})
    result['optimizer_groups']={k:dict(updates=sum(v['learning_rates'][k]>0 for v in train),
        nonzero_gradient_updates=sum(v['gradients'][k]>0 for v in train),
        clipped_updates=sum(v['clip_scales'][k]<.999999 for v in train)) for k in groups}
    if sorted(v['step'] for v in train)!=list(range(1,401)):
        raise ValueError('Stage analysis requires exactly one logged update per step1..400')
    peaks={k:max(v['learning_rates'][k] for v in train) for k in groups}
    result['stage_training']={}
    for name,low,high in [('steps1_128',1,128),('steps129_256',129,256),('steps257_400',257,400)]:
        rows=[v for v in train if low<=v['step']<=high]
        result['stage_training'][name]=dict(
            updates=len(rows),phase_counts=dict(Counter(v['phase'] for v in rows)),
            groups={k:dict(
                positive_lr_updates=sum(v['learning_rates'][k]>0 for v in rows),
                nonzero_gradient_updates=sum(v['gradients'][k]>0 for v in rows),
                median_preclip_gradient=statistics.median(v['gradients'][k] for v in rows),
                clipped_updates=sum(v['clip_scales'][k]<.999999 for v in rows),
                learning_rate_sum=sum(v['learning_rates'][k] for v in rows),
                peak_normalized_learning_rate_sum=(sum(v['learning_rates'][k] for v in rows)/peaks[k]
                    if peaks[k]>0 else None)) for k in groups})
    result['stage_scope']=(
        'These are actual logged gradient norms and learning-rate sums. Fusion has no direct word-alignment '
        'loss, so its nonzero gradients establish generation-loss backpropagation while Wan LoRA is frozen. '
        'Enhancement gradients combine generation and word losses and do not isolate their contributions. '
        'Learning-rate sums are schedule exposure, not equivalent Adam updates, compute or quality. '
        'This does not establish that128 interface steps are optimal or that the stage schedule causes SA loss.')
    result['scope']='Training diagnostics, not SA/PC proxies; only matched input windows are compared.'
    return result


def main():
    paths=json.loads((ROOT/'reports/r10_final_evidence.json').read_text())['source_paths']
    source={'Wan+LoRA400':paths['native400/original40'],'R1':paths['R1/original40'],
        'R10A':paths['R10-A/original40'],'R11A':str(ROOT/'request_adapt_r11a/inference_outputs/physical_content_67214/step0400_full40_controls16_physical5_seed42/videophy40/scores.json')}
    scores={k:json.loads(Path(p).read_text()) for k,p in source.items()}
    report=dict(utc=datetime.now(timezone.utc).isoformat(),protocol='same400/original40/I2V/seed42/gamma5',
        source_paths=source,summary={k:v['summary'] for k,v in scores.items()},
        paired={k:paired(scores['R11A'],v) for k,v in scores.items() if k!='R11A'},
        training={k:logs(ROOT/p) for k,p in dict(R1='runs/physical_content_66778',R10A='execution_value_r10/runs/physical_content_67003',
            R11A='request_adapt_r11a/runs/physical_content_67214',native400='runs/native_lora_66781').items()},
        interpretation=dict(reject_failed_combination=True,
            isolated_QKVO_causality_established=False,
            failures_not_explained_by_fm_convergence=True,
            no_new_training_or_inference_performed=True))
    full60_sources={k:paths[v+'/full60'] for k,v in
        {'Wan+LoRA400':'native400','R1':'R1','R10A':'R10-A'}.items()}
    full60_sources['R11A']=str(ROOT/'videophy60/inference_outputs/R11A_physical_content_67214/full60_gamma5_seed42/videophy60/scores.json')
    full60={k:json.loads(Path(p).read_text()) for k,p in full60_sources.items()}
    for row in full60.values():
        if len(row['cases'])!=60 or len({v['id'] for v in row['cases']})!=60:
            raise ValueError('Full60 requires 60 distinct cases')
        if any(v['generation_seed']!=42 for v in row['cases']):raise ValueError('Full60 seed differs')
    report['full60']=dict(source_paths=full60_sources,
        summary={k:v['summary'] for k,v in full60.items()},
        paired={k:paired(full60['R11A'],v) for k,v in full60.items() if k!='R11A'},
        scope='Original40 plus fixed extra20, not 60 independent additional cases; same I2V seed42 protocol.')
    (OUT/'failure_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(summary=report['summary'],output=str(OUT/'failure_analysis.json')),ensure_ascii=False))


if __name__=='__main__':main()
