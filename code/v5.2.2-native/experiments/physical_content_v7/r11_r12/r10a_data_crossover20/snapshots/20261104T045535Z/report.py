"""Checkpoint-by-reasoning table; retain original16 and added4 as separate strata."""
from pathlib import Path
import json,statistics,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import WORK,OLD_OUTPUT,API_OUTPUT,ARMS,EXTRA,cases


def summary(values):
    n=len(values)
    return dict(count=n,sa_mean=statistics.mean(v['sa'] for v in values),pc_mean=statistics.mean(v['pc'] for v in values),
        sa_pass_rate=sum(v['sa']>=4 for v in values)/n,pc_pass_rate=sum(v['pc']>=4 for v in values)/n,
        joint_pass_rate=sum(v['sa']>=4 and v['pc']>=4 for v in values)/n)


def paired(after,before):
    lookup={v['id']:v for v in before};result={}
    for metric in ['sa','pc']:
        diffs=[v[metric]-lookup[v['id']][metric] for v in after]
        result[metric]=dict(mean_change=statistics.mean(diffs),wins=sum(x>0 for x in diffs),ties=sum(x==0 for x in diffs),losses=sum(x<0 for x in diffs))
    passed=lambda x:x['sa']>=4 and x['pc']>=4
    result['joint']=dict(gains=[v['id'] for v in after if passed(v) and not passed(lookup[v['id']])],losses=[v['id'] for v in after if not passed(v) and passed(lookup[v['id']])])
    return result


def main():
    _,selected=cases();ids={v['id'] for v in selected}
    sources={'old_checkpoint_old_reasoning':OLD_OUTPUT/'scores.json','api_checkpoint_api_reasoning':API_OUTPUT/'scores.json'}
    sources.update({key:WORK/'outputs'/key/'videophy40/scores.json' for key in ARMS})
    rows={};pending=[];strata={'original_fixed16':ids-set(EXTRA),'diagnostic4':set(EXTRA),'combined20':ids}
    for name,path in sources.items():
        if not path.exists():pending.append(name);continue
        values=[v for v in json.loads(path.read_text())['cases'] if v['id'] in ids]
        if len(values)!=20 or {v['id'] for v in values}!=ids:raise ValueError('Need all same20 cases')
        rows[name]=dict(source=str(path),cases=values,summary={key:summary([v for v in values if v['id'] in keep]) for key,keep in strata.items()})
    comparisons={}
    for key,trained in [('old_checkpoint_api_reasoning','old'),('api_checkpoint_old_reasoning','api')]:
        baseline=f'{trained}_checkpoint_{trained}_reasoning'
        if key not in rows:continue
        comparisons[key+'_minus_own_reasoning']={group:paired([v for v in rows[key]['cases'] if v['id'] in keep],rows[baseline]['cases']) for group,keep in strata.items()}
    result=dict(rows=rows,paired=comparisons,pending=pending,
        interpretation='Fixed checkpoint inference-evidence interventions. Extra4 are score-selected failures, reported separately; neither this20 nor existing16 is the full40 estimate. Training-side comparison includes revised captions, native text features, physical evidence and fitted alignment together. Single seed, no per-case winner selection. Preserve main40/60 results and review actual videos.')
    (WORK/'comparison_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(summary={k:v['summary'] for k,v in rows.items()},paired=comparisons,pending=pending),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
