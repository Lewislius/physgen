"""Repair a fixed list of failed API samples without repeating completed stages."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from annotate import CACHE,Client,save,annotate_caption,annotate_reasoning
from review import review_one


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--ids',nargs='*')
    args=p.parse_args();output=args.output.resolve()
    selection=json.loads((output/'selection.json').read_text())
    if selection['scope']!='actual400_training_only':raise ValueError('Only actual400 preparation is authorized')
    sources={r['id']:r for r in json.loads((CACHE/'base/manifest.json').read_text())['records']}
    errors={p.stem:json.loads(p.read_text()) for p in (output/'errors').glob('*.json')}
    ids=args.ids or sorted(errors)
    if not set(ids)<=set(errors) or not set(ids)<=set(selection['ids']):
        raise ValueError('Repair only explicitly selected failed400 samples')
    save(args.report/'selection.json',dict(ids=ids,previous_errors={i:errors[i] for i in ids},
        utc=datetime.now(timezone.utc).isoformat(),completed_stages_reused=True,main_batch_unchanged=True))
    client=Client();results=[]
    def process(sample):
        row=sources[sample]
        try:
            if not (output/'records'/sample/'caption.json').exists():
                annotate_caption(row,output/'drafts',client)
                review_one(row,output/'drafts',output,client)
            annotate_reasoning(row,output,client)
            result=dict(id=sample,complete=True)
        except Exception as error:
            result=dict(id=sample,complete=False,error=str(error))
            if hasattr(error,'detail'):result['detail']=error.detail
            save(output/'errors'/f'{sample}.json',result)
        save(args.report/'records'/f'{sample}.json',result)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(process,i):i for i in ids}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='detail'}),flush=True)
    summary=dict(requested=len(ids),repaired=sum(r['complete'] for r in results),
        failed=[r['id'] for r in results if not r['complete']],results=results)
    save(args.report/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
