"""One video-grounded factual edit pass; no training/generation-side verifier."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import json
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from annotate import (Client,CACHE,OFFSETS,read_frames,style_profile,save,words,caption_flags,
                      annotate_reasoning,summarize)

REVISION='qwen38_wisa_reviewed_v1'
REVIEW='''Audit this candidate caption against the actual chronological images, and correct factual errors.
This is a TEMPORALLY CROPPED EXCERPT from a longer video, not a complete event or a loop.
The first view may already be mid-action; the last view need not contain completion. Different
endpoints are normal, not physical errors. Do not add unseen causes, onset, stopping or outcomes.
An internal editing cut is not evidence of physical motion connecting the views.

Check identification before physical interpretation: a reflection is not a separate actor;
a powder/smoke/liquid outline or shadow is not automatically the boundary of a solid object;
an apparent change caused by occlusion/viewpoint is not automatically material deformation.
Do not claim contact, bending, compression, melting, disintegration or structural recovery unless
the appropriate material/surface itself provides visible evidence. Do not infer a force or earlier
collision just because the visible configuration would be compatible with it.
Compare several frames before asserting translation, rotation, direction, unchanged position or
camera tracking. A single visible texture/seam does not prove spin. Avoid unnecessary absolute
claims of stillness, no motion, no camera motion or 'exactly' synchronized changes.
Remove invented scene details and explanations, but keep clearly visible ongoing motion and
processes. Avoid deleting the real process merely because its cause lies outside the cropped window.

Retain the WISA English narrative style, detail density and paragraph structure. Aim for the
provided original word count within 85%-115%, using genuinely visible details and process.
Do not add repetition, watermark/logo descriptions or atmospheric filler to reach a word count.
Do not shorten a rich caption into a terse technical checklist. Avoid annotation phrases such as
'this mid-segment', 'the excerpt', 'the frame', 'these images', 'observed evidence' in the caption.
This is a direct video description, not an analysis note. No mandatory beginning/end formula.
Complete every sentence. If accuracy requires a substantial length difference, explain it outside
the caption; do not fill missing context by imagination. Optional synonym edits are not needed.

Return JSON {"decision":"keep|edit", "issues":[{"quote":"exact span in candidate caption",
"problem":"concrete visual mismatch or unsupported assertion"}], "caption":"complete final caption",
"length_note":"empty unless required"}. Keep means unchanged caption and no issues. An edit
must actually correct the cited issue. This is a fallible automatic review, not proof of correctness.
'''


def review_one(row,draft,output,client):
    folder=output/'records'/row['id'];path=folder/'caption.json'
    if path.exists():return json.loads(path.read_text())
    old=json.loads((draft/'records'/row['id']/'caption.json').read_text())
    images=read_frames(row)
    stamps=[row['pts'][i]-row['pts'][0] for i in OFFSETS]
    prompt=REVIEW+'\n'+json.dumps(dict(candidate_caption=old['caption'],
        wisa_style_profile=style_profile(old['original_wisa_caption'])),ensure_ascii=False)
    answer,trace=client.call(prompt,images,times=stamps,thinking=True,max_tokens=6144)
    if answer.get('decision') not in ('keep','edit') or not isinstance(answer.get('issues'),list):
        raise ValueError('Invalid caption review decision')
    final=answer.get('caption')
    if not isinstance(final,str) or not final.strip():raise ValueError('Empty reviewed caption')
    final=final.strip()
    if answer['decision']=='keep' and (final!=old['caption'] or answer['issues']):
        raise ValueError('Keep must preserve the caption without issues')
    if answer['decision']=='edit' and (final==old['caption'] or not answer['issues']):
        raise ValueError('Edit requires actual changes with stated issues')
    for issue in answer['issues']:
        if not issue.get('quote') or issue['quote'] not in old['caption'] or not issue.get('problem'):
            raise ValueError('Review must quote an actual candidate claim')
    result=copy.deepcopy(old)
    result.update(caption=final,new_words=words(final),length_ratio=words(final)/old['original_words'],
        flags=caption_flags(final,old['original_wisa_caption']),revision=REVISION,
        draft_caption=old['caption'],draft_annotation=str(draft/'records'/row['id']/'caption.json'),
        review_decision=answer['decision'],review_issues=answer['issues'],trace=trace,
        length_note=answer.get('length_note',''),review_status='automatic_video_review; pilot_visual_review_required')
    save(path,result)
    source_panel=draft/'records'/row['id']/'chronology.jpg'
    if source_panel.exists():shutil.copyfile(source_panel,folder/'chronology.jpg')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--draft',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--ids',nargs='*')
    p.add_argument('--reasoning',action='store_true')
    args=p.parse_args()
    selection=json.loads((args.draft/'selection.json').read_text());selection['revision']=REVISION
    save(args.output/'selection.json',selection)
    originals={r['id']:r for r in json.loads((CACHE/'base/manifest.json').read_text())['records']}
    ids=args.ids or selection['ids']
    if not set(ids)<=set(selection['ids']):raise ValueError('Unknown selected sample')
    rows=[originals[i] for i in ids];client=Client();failures=[]
    def process(row):
        result=review_one(row,args.draft,args.output,client)
        if args.reasoning:annotate_reasoning(row,args.output,client)
        return {key:result[key] for key in ('id','new_words','flags','review_decision')}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(process,row):row['id'] for row in rows}
        for future in as_completed(futures):
            try:print(json.dumps(future.result()),flush=True)
            except Exception as error:
                item=dict(id=futures[future],error=str(error));failures.append(item)
                save(args.output/'errors'/f'{item["id"]}.json',item);print(json.dumps(item),flush=True)
    result=summarize(args.output,[originals[i] for i in selection['ids']]);result.update(revision=REVISION,failures=failures)
    save(args.output/'summary.json',result);print(json.dumps(result),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
