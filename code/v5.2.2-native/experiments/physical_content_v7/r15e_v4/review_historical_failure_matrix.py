"""Review complete paired40 and fixed16 coverage; never select per-case winners."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sources=json.loads((BASE/'reports/r10_final_evidence.json').read_text())['source_paths']
paths={name:Path(sources[key+'/original40']) for name,key in
       [('Wan+LoRA','native400'),('R1','R1'),('R10A','R10-A')]}
for name,folder,eid in [('R15B','request_bound_fusion_r15b',67829),('R15D','denoising_matched_fusion_r15d',67920)]:
    paths[name]=BASE/folder/'inference_outputs'/f'physical_content_{eid}'/'step0400_original_fixed40_controls16_seed42/videophy40/scores.json'
fixed={c['id'] for c in json.loads((ROOT/'subset16.json').read_text())['cases']}
data={name:{c['id']:c for c in json.loads(path.read_text())['cases']} for name,path in paths.items()}
expected={f'VP{i:03d}' for i in range(1,41)}
assert all(set(d)==expected for d in data.values())
assert all(c['generation_seed']==42 for d in data.values() for c in d.values())
assert all(c['caption']==data['R1'][key]['caption'] for d in data.values() for key,c in d.items())


def summarize(rows):
    n=len(rows)
    return dict(n=n,sa=sum(c['sa'] for c in rows)/n,pc=sum(c['pc'] for c in rows)/n,
                joint=sum(c['sa']>=4 and c['pc']>=4 for c in rows))


def deficits(rows):
    return dict(sa_integer_score_points_to_minimum=max(0,125-sum(c['sa'] for c in rows)),
        pc_integer_score_points_to_minimum=max(0,152-sum(c['pc'] for c in rows)),
        additional_joint_passes_to_minimum=max(0,12-sum(c['sa']>=4 and c['pc']>=4 for c in rows)))


results={}
for name,values in data.items():
    rows=list(values.values())
    classes={key:[] for key in ('both_pass','pc_pass_sa_fail','sa_pass_pc_fail','both_fail')}
    for cid,c in values.items():
        key=('both_pass' if c['sa']>=4 and c['pc']>=4 else
             'pc_pass_sa_fail' if c['pc']>=4 else 'sa_pass_pc_fail' if c['sa']>=4 else 'both_fail')
        classes[key].append(cid)
    results[name]=dict(full40=summarize(rows),fixed16=summarize([c for i,c in values.items() if i in fixed]),
        remaining24=summarize([c for i,c in values.items() if i not in fixed]),
        target_gap=deficits(rows),joint_failure_types=classes,
        versus_R1={metric:{label:sum((c[metric]-data['R1'][cid][metric])*sign>0 for cid,c in values.items())
                          for label,sign in [('wins',1),('losses',-1)]}
                   for metric in ['sa','pc']})
report=dict(utc=datetime.now(timezone.utc).isoformat(),sources={k:str(v) for k,v in paths.items()},models=results,
    cases=[dict(id=cid,caption=data['R1'][cid]['caption'],in_fixed16=cid in fixed,
                scores={name:{m:d[cid][m] for m in ['sa','pc']} for name,d in data.items()}) for cid in sorted(expected)],
    interpretation=[
        'Fixed16 is a mechanism set, not an estimate of the complete40 acceptance result. Remaining24 are a descriptive complement, not a held-out test.',
        'Target gaps count ordinal score points at the predeclared same40 thresholds; they are not a prediction of attainable gains or a training loss.',
        'Historical scores are used to identify recurrent versus version-induced failure patterns; no per-case mixture or benchmark-label training is produced.',
        'Independent physical and semantic means do not guarantee the two conditions pass in the same videos. Joint failure types must guide whole-model review.'
    ])
(ROOT/'review/historical_failure_matrix40.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({name:{k:v[k] for k in ['full40','remaining24','target_gap']} for name,v in results.items()},ensure_ascii=False))
