"""Read actual400 input provenance/EOS flags and exact local-tokenizer lengths."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime,timezone
import json
import re
import sys
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
SOURCE=ROOT/'execution_value_r10/snapshots/20261101T034557Z'
sys.path.insert(0,str(SOURCE))
import bootstrap
import torch
from transformers import AutoTokenizer
import ftfy,html
from physical_content.evidence import find_input


def main():
    torch.set_num_threads(1)
    cache=Path('/home/liuzhirui/Project/physGen/data_cache/wisa_4000')
    records=json.loads((cache/'base/manifest.json').read_text())['records'];by_id={r['id']:r for r in records}
    logs=[json.loads(line) for line in (ROOT/'execution_value_r10/runs/physical_content_67003/train.jsonl').read_text().splitlines()]
    micros=[row for row in logs if row['kind']=='micro']
    t5=AutoTokenizer.from_pretrained('/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B/google/umt5-xxl',local_files_only=True)
    qwen=AutoTokenizer.from_pretrained('/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-8B-Instruct',local_files_only=True)
    captions=[' '.join(html.unescape(html.unescape(ftfy.fix_text(r['caption']))).split()) for r in records]
    t5lengths=[len(ids) for ids in t5(captions,add_special_tokens=True,truncation=False)['input_ids']]
    lengths=dict(zip([r['id'] for r in records],t5lengths))
    qlengths=[len(ids) for ids in qwen([r['caption'] for r in records],add_special_tokens=False,truncation=False)['input_ids']]
    def read(item):
        row=by_id[item['sample']];name=row['id']+('_null' if item['text_dropped'] else '_text')
        p=find_input(name);v=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        expected='' if item['text_dropped'] else row['caption']
        side=json.loads((cache/'base/text'/f'{row["id"]}.json').read_text())
        annotation=json.loads((cache/'work/annotations'/f'{row["id"]}.json').read_text())
        text=torch.load(cache/'base/text'/f'{row["id"]}.pt',map_location='cpu',weights_only=True,mmap=True)
        trace=annotation.get('trace',[])
        return dict(id=row['id'],step=item['step'],micro=item['micro'],drop=item['text_dropped'],
            input=str(p),qwen_caption_matches=v['caption']==expected,t5_caption_matches=side['caption']==row['caption'],
            cached_text_tokens=len(text),expected_text_tokens=min(lengths[row['id']],512),
            analysis_complete=v.get('analysis_complete'),relations=len(v.get('relations',[])),
            analysis=v.get('analysis',''),caption=row['caption'],
            annotation_matches=annotation.get('caption')==row['caption'],
            approved=annotation.get('approved'),review_version=annotation.get('review_version'),
            sampled_offsets=annotation.get('sampled_offsets'),
            recorded_review_images=[entry.get('image_count') for entry in trace if entry.get('kind')!='caption_resume'],
            last_review=trace[-1].get('parsed') if trace else None,
            clip_indices_count=len(row['indices']),clip_pts_count=len(row['pts']),
            clip_pts_increasing=all(a<b for a,b in zip(row['pts'],row['pts'][1:])),
            source_video=row['video'],source_indices=row['indices'],scope=v.get('scope'))
    rows=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(read,micros):
            rows.append(row)
            if len(rows)%200==0:print(json.dumps(dict(stage='cached_inputs',completed=len(rows),total=len(micros))),flush=True)
    false=lambda key:[r['id'] for r in rows if not r[key]]
    summary=dict(actual_micro_count=len(rows),unique_ids=len({r['id'] for r in rows}),
        t5_all_max=max(t5lengths),t5_over512=[records[i]['id'] for i,n in enumerate(t5lengths) if n>512],
        qwen_caption_max=max(qlengths),qwen_silent_input_truncation=False,
        qwen_input_policy='Actual _inputs tokenizes without truncation and raises if nonvisual text exceeds4096; cached caption literal equality checked.',
        qwen_caption_mismatches=false('qwen_caption_matches'),t5_caption_mismatches=false('t5_caption_matches'),
        t5_length_mismatches=[r['id'] for r in rows if r['cached_text_tokens']!=r['expected_text_tokens']],
        reasoning_eos=Counter(str(r['analysis_complete']) for r in rows),
        reasoning_eos_positive=Counter(str(r['analysis_complete']) for r in rows if not r['drop']),
        reasoning_eos_negative=Counter(str(r['analysis_complete']) for r in rows if r['drop']),
        caption_annotation_mismatches=false('annotation_matches'),caption_approval=Counter(str(r['approved']) for r in rows),
        review_images=Counter(str(n) for r in rows for n in r['recorded_review_images']),
        clip_count_issues=[r['id'] for r in rows if r['clip_indices_count']!=81 or r['clip_pts_count']!=81 or not r['clip_pts_increasing']],
        annotation_note='Prior model review is evidence of clip-based annotation, not a human-verified alignment accuracy estimate.')
    output=ROOT/'r11_r12/data_audit';output.mkdir(exist_ok=True)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),summary=summary,records=rows)
    (output/'actual400_inputs.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
