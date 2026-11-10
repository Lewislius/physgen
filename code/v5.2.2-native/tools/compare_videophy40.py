"""Paired prompt-level comparison with archived references; no new baseline videos."""
import argparse
import json
from pathlib import Path
import random
import statistics


def paired_summary(candidate,reference,seed=261026,repeats=10000):
    if [r['id'] for r in candidate]!=[r['id'] for r in reference]:raise ValueError('Paired case IDs/order differ')
    n=len(candidate);rng=random.Random(seed);result={}
    for name,measure,scale in [('sa',lambda x:x['sa'],1),('pc',lambda x:x['pc'],1),
                               ('joint_pp',lambda x:int(x['sa']>=4 and x['pc']>=4),100)]:
        a=[measure(r) for r in candidate];b=[measure(r) for r in reference];diff=[(x-y)*scale for x,y in zip(a,b)]
        draws=sorted(sum(diff[rng.randrange(n)] for _ in range(n))/n for _ in range(repeats))
        result[name]=dict(candidate=statistics.mean(a)*scale,reference=statistics.mean(b)*scale,
            delta=statistics.mean(diff),paired_bootstrap95=[draws[int(repeats*.025)],draws[int(repeats*.975)]],
            wins=sum(d>0 for d in diff),ties=sum(d==0 for d in diff),losses=sum(d<0 for d in diff))
        if name in ('sa','pc'):
            result[name]['delta_on_0_100_scale']=25*statistics.mean(diff)
    return result


def compare(protocol_file,candidate_directory):
    protocol=json.loads(Path(protocol_file).read_text());root=Path(candidate_directory)
    generated=json.loads((root/'manifest.json').read_text())
    if generated['cases']!=protocol['cases'] or generated['settings']!=protocol['settings'] or generated['evaluation']!=protocol['evaluation']:
        raise ValueError('Candidate differs from the fixed cases, sampler, or independent evaluator')
    rows=[]
    for case in protocol['cases']:
        row=dict(id=case['id'],caption=case['caption'])
        for task in ('sa','pc'):
            score=json.loads((root/'cases'/case['id']/f'planned.{task}.json').read_text())
            if score['case_id']!=case['id'] or score['task']!=task:raise ValueError('Candidate score provenance differs')
            row[task]=score['score']
        rows.append(row)
    result=dict(count=len(rows),scope=protocol['scope'],primary_reference=protocol['primary_reference'],
        checkpoint=generated['checkpoint'],statistics_scope='Paired case bootstrap conditional on this fixed video seed; not evaluator-independent or repeated-training evidence',
        visual_quality_review='Required separately; scores do not establish artifact-free videos',
        comparisons={name:paired_summary(rows,arm['cases']) for name,arm in protocol['historical'].items()},cases=rows)
    (root/'paired_comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('protocol');p.add_argument('candidate');a=p.parse_args();compare(a.protocol,a.candidate)
