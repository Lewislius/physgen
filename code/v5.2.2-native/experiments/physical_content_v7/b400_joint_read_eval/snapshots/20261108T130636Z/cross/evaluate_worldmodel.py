"""Run the unchanged local official WMB evaluator on the fixed70 manifest."""
from pathlib import Path
import argparse,json,os,runpy,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,BENCH,plan,save,gpu_guard

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True)
    p.add_argument('--resource-pool',choices=['ada-48g','amp-80g'],default=None,
                   help='Evaluation-only allocation; generation protocol and inputs stay unchanged')
    p.add_argument('--result-dir',type=Path,default=None,
                   help='Keep helper scoring isolated until all 70 official results are complete')
    a=p.parse_args()
    gpu_guard(a.model,a.resource_pool)
    cases=[c for c in plan()['cases'] if c['benchmark']=='worldmodelbench70']
    videos=ROOT/'outputs'/a.model/'seed42/worldmodelbench70'
    out=a.result_dir or ROOT/'outputs'/a.model/'seed42'
    expected={c['output_filename'] for c in cases}
    if {p.name for p in videos.glob('*.mp4')}!=expected:raise ValueError('Require exactly the fixed70 generated videos')
    context=out/'worldmodel_eval_context';context.mkdir(parents=True,exist_ok=True)
    save(context/'worldmodelbench.json',[c['official_record'] for c in cases])
    result=out/'worldmodelbench_official.json'
    sys.path.insert(0,str(BENCH/'WorldModelBench/VILA'))
    import torch
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    os.chdir(context)
    sys.argv=['evaluation.py','--judge',str(BENCH/'WorldModelBench/vila-ewm-qwen2-1.5b'),
        '--video_dir',str(videos),'--model_name',a.model,'--save_name',str(result.with_suffix(''))]
    runpy.run_path(str(BENCH/'WorldModelBench/evaluation.py'),run_name='__main__')
    raw=json.loads(result.read_text());preds=raw['preds'];acc=raw['accs']
    if set(preds)!={c['official_id'] for c in cases}:raise ValueError('Official scorer omitted cases')
    if [len(acc[k]) for k in ('instruction','physical_laws','common_sense')]!=[70,350,140]:raise ValueError('Missing official answers')
    rows=[]
    for i,c in enumerate(cases):
        rows.append(dict(id=c['official_id'],domain=c['domain'],subdomain=c['subdomain'],
            instruction=acc['instruction'][i],physical_laws=acc['physical_laws'][i*5:(i+1)*5],
            common_sense=acc['common_sense'][i*2:(i+1)*2]))
    def summarize(values):
        n=len(values);instruction=sum(v['instruction'] for v in values)/n
        physical=[sum(v['physical_laws'][i] for v in values)/n for i in range(5)]
        common=[sum(v['common_sense'][i] for v in values)/n for i in range(2)]
        return dict(count=n,instruction_mean=instruction,physical_sum=sum(physical),common_sense_sum=sum(common),
            total_score=instruction+sum(physical)+sum(common),
            physical=dict(zip(('newton','mass','fluid','penetration','gravity'),physical)),
            common_sense=dict(zip(('framewise_quality','temporal_consistency'),common)))
    save(out/'worldmodelbench_summary.json',dict(model=a.model,subset='fixed70_10_per_domain',
        judge=str(BENCH/'WorldModelBench/vila-ewm-qwen2-1.5b'),official_source=str(BENCH/'WorldModelBench/evaluation.py'),
        cot=False,scoring='Official default prompts/parsing; instruction0..3 + physical0..5 + commonsense0..2 = total0..10',
        overall=summarize(rows),domains={d:summarize([v for v in rows if v['domain']==d]) for d in sorted({v['domain'] for v in rows})},cases=rows,
        full_benchmark_score=False))

if __name__=='__main__':main()
