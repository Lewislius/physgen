"""Resumable new caption dataset; stream edits, factual review and input reasoning."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import json
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from annotate import CACHE,ROOT,Client,save,annotate_caption,annotate_reasoning,summarize,words
from review import review_one,REVISION


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-pilot',type=Path)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--scope',choices=['train400','all'],default='train400')
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    originals=json.loads((CACHE/'base/manifest.json').read_text())['records']
    train=[r for r in originals if r['split']=='train']
    priorities=[r['id'] for r in json.loads((ROOT/'r11_r12/data_audit/actual400_inputs.json').read_text())['records']]
    selected=set(priorities)
    selected_rows=[r for r in originals if args.scope=='all' or r['id'] in selected]
    ids=[r['id'] for r in selected_rows]
    if args.scope=='train400' and (len(ids)!=1600 or any(r['split']!='train' for r in selected_rows)):
        raise ValueError('Expected the exact 1600 unique samples from the matched 400-update schedule')
    selection=args.output/'selection.json'
    if selection.exists():
        previous=json.loads(selection.read_text())
        if previous['ids']!=ids:
            if not set(ids)<set(previous['ids']):raise ValueError('Selection changed beyond explicit narrowing')
            save(args.output/'selection_before_400_narrowing.json',previous)
    save(selection,dict(scope='actual400_training_only' if args.scope=='train400' else 'all_existing_records',
        ids=ids,revision=REVISION,cache=str(CACHE),training_videos_reannotated=sum(r['split']=='train' for r in selected_rows),
        validation_policy='No validation API calls in train400 scope; validation captions stay unchanged',
        processing_policy='Only actual 400-update samples' if args.scope=='train400' else 'actual400 first',
        original_sampling_order_preserved=True,maximum_training_updates=400 if args.scope=='train400' else None,
        schedule=str(ROOT/'r11_r12/data_audit/actual400_inputs.json')))
    ranks={name:i for i,name in enumerate(priorities)}
    todo=sorted(selected_rows,key=lambda r:(r['split']!='train',ranks.get(r['id'],len(ranks))))
    if args.limit:todo=todo[:args.limit]
    if args.reuse_pilot:
        review_file=args.reuse_pilot/'visual_review.json'
        if not review_file.exists() or not json.loads(review_file.read_text()).get('allow_batch'):
            raise ValueError('A completed visual pilot review is required before batch expansion')
        for row in selected_rows:
            source=args.reuse_pilot/'records'/row['id'];target=args.output/'records'/row['id']
            if source.exists() and not target.exists():shutil.copytree(source,target)
            for arm in ('text','null'):
                name=f'{row["id"]}_{arm}.json';source=args.reuse_pilot/'reasoning'/name
                target=args.output/'reasoning'/name
                if source.exists() and not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    client=Client();failures=[];completed=0
    def process(row):
        if row['split']=='train':
            final=args.output/'records'/row['id']/'caption.json'
            if not final.exists():
                annotate_caption(row,args.output/'drafts',client)
                review_one(row,args.output/'drafts',args.output,client)
        else:
            # Validation video is not inspected to tune or rewrite its text.
            path=args.output/'records'/row['id']/'caption.json'
            if not path.exists():
                original=row.get('original_caption') or row['caption']
                save(path,dict(id=row['id'],split=row['split'],revision=REVISION,old_caption=row['caption'],
                    caption=row['caption'],original_wisa_caption=original,original_words=words(original),
                    old_words=words(row['caption']),new_words=words(row['caption']),flags=[],
                    review_status='unchanged_validation_caption',trace=dict(usage={}),
                    source=dict(input_scope='original_validation_caption_passthrough')))
        annotate_reasoning(row,args.output,client)
        return row['id']
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(process,row):row['id'] for row in todo}
        for future in as_completed(futures):
            completed+=1
            try:sample=future.result();print(json.dumps(dict(completed=completed,total=len(todo),id=sample)),flush=True)
            except Exception as error:
                item=dict(id=futures[future],error=str(error));failures.append(item)
                if hasattr(error,'detail'):item['detail']=error.detail
                save(args.output/'errors'/f'{item["id"]}.json',item);print(json.dumps(item),flush=True)
            if completed%20==0 or completed==len(todo):
                save(args.output/'status.json',dict(completed_this_launch=completed,total_this_launch=len(todo),
                    failures=failures,stage='api_annotations',selected_size=len(selected_rows),
                    source_cache_unchanged=True,training_started=False))
    result=summarize(args.output,selected_rows);result.update(revision=REVISION,failures=failures,scope=args.scope)
    save(args.output/'summary.json',result)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
