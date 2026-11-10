"""Paired score comparisons with visible protocol checks and case-level results."""
from pathlib import Path
import argparse,json
import numpy as np

def effective_guidance(plan):
    if 'guidance' in plan:
        spec=plan['guidance']
        text_scale,physical_scale=spec['text_scale'],spec['physical_scale']
    else:
        text_scale=physical_scale=plan['sampling'].get('guidance',plan['sampling'].get('guidance_scale',5.))
    active=plan.get('variant')!='native_lora' and plan.get('mode')!='no_c'
    return dict(text_scale=text_scale,physical_scale=physical_scale if active else None,
                requested_physical_scale=physical_scale,physical_branch_active=active)

def compare(baseline,candidate,allow_guidance_change=False):
    bplan=json.loads((baseline/'manifest.json').read_text());cplan=json.loads((candidate/'manifest.json').read_text())
    bs=json.loads((baseline/'scores.json').read_text());cs=json.loads((candidate/'scores.json').read_text())
    b={r['id']:r for r in bs['cases']};c={r['id']:r for r in cs['cases']}
    if set(b)!=set(c):raise ValueError('Pair must contain exactly the same cases')
    bp={r['id']:r for r in bplan['cases']};cp={r['id']:r for r in cplan['cases']}
    for name in b:
        for key in ['caption','prepared_inputs','initial_noise','generation_seed']:
            if bp[name].get(key)!=cp[name].get(key):raise ValueError(f'{name}: mismatched {key}')
    if bplan['sampling']!=cplan['sampling'] or bplan['evaluation']!=cplan['evaluation']:
        raise ValueError('Sampling/evaluator mismatch')
    bg,cg=effective_guidance(bplan),effective_guidance(cplan)
    changed=(bg['text_scale']!=cg['text_scale'] or
             (bg['physical_branch_active'] and cg['physical_branch_active'] and bg['physical_scale']!=cg['physical_scale']))
    if changed and not allow_guidance_change:
        raise ValueError('Guidance differs; declare an explicit inference-guidance experiment')
    rows=[]
    for name in b:
        bv,cv=b[name],c[name]
        rows.append(dict(id=name,caption=bv['caption'],baseline={'sa':bv['sa'],'pc':bv['pc']},
            candidate={'sa':cv['sa'],'pc':cv['pc']},baseline_joint=int(bv['sa']>=4 and bv['pc']>=4),
            candidate_joint=int(cv['sa']>=4 and cv['pc']>=4)))
    rng=np.random.default_rng(47613);draws=rng.integers(0,len(rows),size=(20000,len(rows)))
    metrics={}
    for metric in ['sa','pc','mean','joint','sa_pass_rate','pc_pass_rate']:
        def value(row,which):
            if metric=='joint':return row[which+'_joint']
            if metric.endswith('_pass_rate'):return int(row[which][metric.split('_',1)[0]]>=4)
            if metric=='mean':return (row[which]['sa']+row[which]['pc'])/2
            return row[which][metric]
        bv=np.array([value(r,'baseline') for r in rows]);cv=np.array([value(r,'candidate') for r in rows]);delta=cv-bv
        metrics[metric]=dict(baseline=float(bv.mean()),candidate=float(cv.mean()),difference=float(delta.mean()),
            paired_bootstrap_95_interval=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
            wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum()))
    return dict(baseline=str(baseline),candidate=str(candidate),count=len(rows),metrics=metrics,cases=rows,
        inference_protocol_matched=not changed,scope=cplan['scope'],
        guidance=dict(baseline=bg,candidate=cg,intentionally_varied=changed,
            inactive_scale_note='Physical-scale arguments have no effect in native-only or no_c branches'),
        interpretation='Case-paired descriptive comparison at one seed; no claim that feature separation or a loss reduction proves physical improvement')

def main():
    p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-guidance-change',action='store_true');args=p.parse_args()
    result=compare(args.baseline,args.candidate,args.allow_guidance_change);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
