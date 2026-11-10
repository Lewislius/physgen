"""Paired, case-cluster bootstrap over independent evaluation runs/seeds."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(directory):
    p=Path(directory)/'scores.json';rows=json.loads(p.read_text())['cases']
    return {r['id']:r for r in rows}


def compare(baselines,candidates,draws=10000,seed=20261022):
    if len(baselines)!=len(candidates):raise ValueError('Pair every baseline seed with its candidate seed')
    ids=sorted(baselines[0])
    if not all(set(x)==set(ids) for x in baselines+candidates):raise ValueError('Case coverage differs')
    known_seeds=[]
    for a,b in zip(baselines,candidates):
        for case in ids:
            left,right=a[case].get('generation_seed'),b[case].get('generation_seed')
            if left != right:raise ValueError(f'Generation seeds do not pair for {case}: {left} != {right}')
            known_seeds.append(left is not None)
    functions=dict(sa=lambda r:r['sa'],pc=lambda r:r['pc'],joint=lambda r:float(r['sa']>=4 and r['pc']>=4))
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(ids),size=(draws,len(ids)))
    result=dict(cases=len(ids),seeds=len(baselines),generation_seed_metadata_verified=all(known_seeds),
                resampling_unit='case; all paired seeds retained within each case',metrics={})
    for name,f in functions.items():
        a=np.array([[f(run[i]) for i in ids] for run in baselines]);b=np.array([[f(run[i]) for i in ids] for run in candidates])
        differences=b-a;per_case=differences.mean(0);samples=per_case[indices].mean(1)
        result['metrics'][name]=dict(baseline=float(a.mean()),candidate=float(b.mean()),delta=float(differences.mean()),
            case_bootstrap_95=np.quantile(samples,[.025,.975]).tolist(),seed_deltas=differences.mean(1).tolist(),
            improved_cases=int((per_case>0).sum()),worsened_cases=int((per_case<0).sum()),tied_cases=int((per_case==0).sum()),
            case_deltas=dict(zip(ids,per_case.tolist())))
    result['interpretation']='Case bootstrap conditional on these seeds. An all-tie interval [0, 0] on a small discrete-score sample does not establish population equivalence. With few seeds, training-seed uncertainty remains unresolved; no SOTA claim follows from this report alone.'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',action='append',required=True)
    parser.add_argument('--candidate',action='append',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();result=compare([read(p) for p in args.baseline],[read(p) for p in args.candidate])
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
