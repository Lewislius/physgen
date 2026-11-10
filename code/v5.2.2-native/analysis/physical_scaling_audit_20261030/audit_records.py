"""Read-only analysis of immutable physical training and evaluation records."""
from pathlib import Path
import csv
import json
import statistics
from collections import Counter
import numpy as np

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP = BASE / 'experiments'


def readlog(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        if row['kind'] == 'micro' and 'aux_active' not in row:
            row['aux_active'] = not row.get('text_dropped', False)
    return rows


def selected_logs(parts):
    result = []
    for relative, lo, hi in parts:
        result.extend(r for r in readlog(EXP / relative / 'train.jsonl')
                      if 'step' not in r or lo <= r['step'] <= hi)
    return result


def stat(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p10=float(np.quantile(a, .1)), p90=float(np.quantile(a, .9)),
                minimum=float(a.min()), maximum=float(a.max())) if len(a) else {}


def training():
    jobs = {
        'assoc1000': [('physical_scaling_q256/runs/v6_assoc_66678', 1, 1000)],
        'write1000': [('physical_scaling_q256/runs/v6_write_66674', 1, 850),
                      ('physical_scaling_q256/runs/v6_write_resume850_66707', 851, 860),
                      ('physical_scaling_q256/runs/v6_write_resume860_s20_66710', 861, 1000)],
        'ce1000': [('physical_v5_scaling_q256/runs/v5_ce_66706', 1, 1000)],
        'assoc224': [('physical_variants/runs/v6_assoc_20261029T064043Z', 1, 224)],
        'write224': [('physical_variants/runs/v6_write_20261029T064043Z', 1, 224)],
        'ce224': [('physical_variants/runs/v5_ce_20261029T064043Z-retry1', 1, 224)],
    }
    results, all_logs, flat = {}, {}, []
    for name, parts in jobs.items():
        rows = selected_logs(parts)
        updates = [r for r in rows if r['kind'] == 'train']
        micro = [r for r in rows if r['kind'] == 'micro']
        assert len({r['step'] for r in updates}) == len(updates)
        assert len({(r['step'], r['micro']) for r in micro}) == len(micro)
        complete = max(r['step'] for r in updates)
        assert len(updates) == complete and len(micro) == complete * 4
        all_logs[name] = rows
        result = dict(updates=complete, micro=len(micro), unique_samples=len({r['sample'] for r in micro}),
                      dropout=sum(r.get('text_dropped', False) for r in micro),
                      auxiliary_active=sum(r.get('aux_active', False) for r in micro),
                      validation=[r for r in rows if r['kind'] == 'validation'], windows=[],
                      model_ready=next(r for r in rows if r['kind'] == 'model_ready'))
        windows = [(1,50),(51,128),(129,256),(257,500),(501,750),(751,950),(951,1000)] if complete==1000 else [(1,32),(33,64),(65,128),(129,192),(193,224)]
        for lo, hi in windows:
            u=[r for r in updates if lo<=r['step']<=hi]
            m=[r for r in micro if lo<=r['step']<=hi]
            active=[r for r in m if r.get('aux_active',False)]
            w=dict(start=lo,end=hi, fm=stat([r['fm'] for r in m]),
                   aux_all=stat([r['aux'] for r in m]),aux_active=stat([r['aux'] for r in active]),
                   weighted_aux=stat([r['weighted_aux'] for r in m]),
                   weighted_aux_to_fm=sum(r['weighted_aux'] for r in m)/sum(r['fm'] for r in m),
                   gradient={}, clipping={}, lrs={})
            for role in u[0].get('gradients',{}):
                w['gradient'][role]=stat([r['gradients'][role] for r in u])
                w['clipping'][role]=sum(r.get('clip_scales',{}).get(role,1)<.999999 for r in u)
                w['lrs'][role]=stat([r['learning_rates'][role] for r in u]) if isinstance(u[0].get('learning_rates'),dict) else {}
            result['windows'].append(w)
            flat.append(dict(job=name,start=lo,end=hi,fm=w['fm']['mean'],aux=w['aux_all']['mean'],
                             active_aux=w['aux_active'].get('mean'),weighted_aux=w['weighted_aux']['mean'],
                             aux_to_fm=w['weighted_aux_to_fm']))
        result['sigma_bins']={}
        for lo,hi in [(0,.25),(.25,.5),(.5,.75),(.75,1)]:
            ms=[r for r in micro if lo<=r['sigma']<hi and r.get('aux_active',False)]
            result['sigma_bins'][f'{lo}-{hi}']={k:stat([r[k] for r in ms]) for k in ['fm','aux','weighted_aux']}
        result['sample_frequency']=dict(Counter(Counter(r['sample'] for r in micro).values()))
        results[name]=result
    # Same deterministic sample, sigma and noise seeds make the scaled arms pairable.
    ref=[r for r in all_logs['assoc1000'] if r['kind']=='micro']
    paired={}
    for key in ['write1000','ce1000']:
        other=[r for r in all_logs[key] if r['kind']=='micro']
        assert [(r['sample'],r['sigma'],r['text_dropped']) for r in ref]==[(r['sample'],r['sigma'],r['text_dropped']) for r in other]
        paired[key+'_minus_assoc']={}
        for lo,hi in [(1,128),(129,500),(501,1000),(951,1000)]:
            vals=[b['fm']-a['fm'] for a,b in zip(ref,other) if lo<=a['step']<=hi]
            paired[key+'_minus_assoc'][f'{lo}-{hi}']=stat(vals)
    results['paired_fm']=paired
    (OUT/'training_statistics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    with (OUT/'training_windows.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    # Diagnostic curves: training windows mix samples, fixed validation uses four samples.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for i,name in enumerate(['assoc1000','write1000','ce1000']):
        u=[r for r in all_logs[name] if r['kind']=='train']
        steps=np.array([r['step'] for r in u])
        ax=axes[0,i]
        for key,label in [('fm','FM'),('aux','Auxiliary (includes dropout zeros)')]:
            values=np.array([r[key] for r in u]);smooth=np.convolve(values,np.ones(50)/50,mode='valid')
            ax.plot(steps[49:],smooth,label=label)
        ax.set_title(name);ax.set_xlabel('Optimizer update');ax.legend();ax.grid(alpha=.25)
        val=results[name]['validation'];ax=axes[1,i]
        ax.plot([r['step'] for r in val],[r['fm'] for r in val],'o-')
        ax.set_ylim(.1835,.187);ax.set_ylabel('Fixed validation FM');ax.set_xlabel('Optimizer update');ax.grid(alpha=.25)
    fig.suptitle('Recorded losses: auxiliary progress vs nearly flat fixed validation')
    fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    return results


def evaluation():
    root=EXP/'physical_scaling_q256/inference_outputs/v6_assoc_66678/final1000_demo21_videophy40_r1/videophy40'
    rows=list(csv.DictReader((root/'scores_by_case_40.csv').open(encoding='utf-8-sig')))
    new='physical_v6_assoc_q256_s1000'
    refs=['physical_v6_assoc_q128_s224','physical_v5_ce_q128_s224','physical_v6_write_q128_s224','independent_lora','joint_lora_only']
    result={}
    rng=np.random.default_rng(42)
    indices=rng.integers(0,len(rows),size=(50000,len(rows)))
    for ref in refs:
        d={};changed=[]
        for metric in ['sa','pc','mean','joint']:
            delta=np.array([float(r[f'{new}_{metric}'])-float(r[f'{ref}_{metric}']) for r in rows])
            ci=np.quantile(delta[indices].mean(1),[.025,.975])
            d[metric]=dict(delta=float(delta.mean()),wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum()),paired_bootstrap_95_ci=ci.tolist())
        for r in rows:
            ds=int(r[f'{new}_sa'])-int(r[f'{ref}_sa']);dp=int(r[f'{new}_pc'])-int(r[f'{ref}_pc'])
            if ds or dp:
                changed.append(dict(id=r['case_id'],caption=r['caption'],old=[int(r[f'{ref}_sa']),int(r[f'{ref}_pc'])],new=[int(r[f'{new}_sa']),int(r[f'{new}_pc'])],delta_sa=ds,delta_pc=dp))
        result[ref]=dict(metrics=d,changed_cases=changed,both_scores_identical=40-len(changed))
    (OUT/'evaluation_paired.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    t=training();e=evaluation()
    for k,v in t.items():
        if k=='paired_fm':continue
        print(k,{x:v[x] for x in ['updates','micro','unique_samples','dropout','auxiliary_active','sample_frequency']})
        print('windows',[(w['start'],w['end'],round(w['fm']['mean'],6),round(w['aux_all']['mean'],6),round(w['weighted_aux']['mean'],6),round(w['weighted_aux_to_fm']*100,3)) for w in v['windows']])
        print('validation',[(r['step'],r['fm']) for r in v['validation']])
    print('paired_evaluation',json.dumps(e['physical_v6_assoc_q128_s224'],ensure_ascii=False))
