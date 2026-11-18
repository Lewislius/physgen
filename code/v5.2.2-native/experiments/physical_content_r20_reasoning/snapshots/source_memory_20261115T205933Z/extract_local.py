"""Source-grounded physical memory: select text spans, never invent new claims.

Short requests are already concise physical descriptions and are retained.
For long requests a local model selects source units covering the main visible
process. The subject-setting opening is always retained. Only source text is
encoded; selection-model explanations never enter Q/C.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time

BASE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
REVISION='R20-source-grounded-physical-memory-v5'
PROMPT='''Select the source units most useful for representing the visible
physical process in this video request: who/what participates, what interacts
or changes, and the observable response. Cover the main process across the
provided excerpt, including important later changes. Unit 0 will be kept to
retain the principal subject and scene. Select up to five additional units.
Return only JSON {"indices": [0, ...]} with valid integer indices in source
order. Do not rewrite anything, infer new events, or explain physical laws.
The clip can begin/end mid-process; no preceding or following story is needed.
Source units:
'''


def source_units(text):
    # Preserve exact source strings. Long sentences may be separated at their
    # existing punctuation; never add an actor, material, force or aftermath.
    units=[s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+',text) if s.strip()]
    result=[]
    for unit in units:
        if len(unit.split())<=55:
            result.append(unit);continue
        clauses=[s.strip() for s in re.split(r'(?<=[,;:])\s+',unit) if s.strip()]
        result.extend(clauses)
    if not result or any(s not in text for s in result):
        raise ValueError('Source units must be exact nonempty request spans')
    return result


def parse_indices(text,count):
    start=text.find('{')
    if start<0:raise ValueError('No selection JSON')
    value,_=json.JSONDecoder().raw_decode(text[start:])
    indices=value.get('indices')
    if (not isinstance(indices,list) or not indices or
        any(type(i)!=int or not 0<=i<count for i in indices)):
        raise ValueError('Invalid source indices')
    chosen=sorted(set([0,*indices]))
    return chosen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--model',default='/home/liuzhirui/model/Qwen3-30B-A3B-Instruct-2507')
    args=parser.parse_args();selection=json.loads(args.selection.read_text())
    if selection['scope']!='request_only':raise ValueError('Request-only T2V source required')
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();completed=[];pending=[];errors=[]
    def save(row,units,indices,method):
        relations=[units[i] for i in indices]
        if any(s not in row['caption'] for s in relations):raise ValueError('Invented source text')
        record=dict(row,relations=relations,analysis='\n'.join(relations),scope='request_only',
            revision=REVISION,model=(args.model if method=='local_selection' else 'exact_request'),
            source_units=units,source_indices=indices,method=method,
            analysis_complete=True,final_answer_complete=True,final_answer_only=True,
            thinking_end='not_applicable',native_request_unchanged=True,
            word_counts=[len(s.split()) for s in relations],invented_condition_text=False)
        (args.output/(row['id']+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        completed.append(row['id'])
    for row in selection['cases']:
        path=args.output/(row['id']+'.json')
        if path.exists():
            previous=json.loads(path.read_text())
            if previous['revision']!=REVISION or previous['caption']!=row['caption']:
                raise ValueError('Existing source memory has different provenance')
            completed.append(row['id']);continue
        units=source_units(row['caption'])
        if len(row['caption'].split())<=55:
            save(row,[row['caption']],[0],'exact_request')
        elif len(units)<=4:
            save(row,units,list(range(len(units))),'all_source_units')
        else:pending.append((row,units))
    gpu=None
    if pending:
        sys.path.insert(0,str(BASE/'tools'))
        from gpu_policy import require_rtx_pro_6000
        gpu=require_rtx_pro_6000()
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from transformers.utils import logging
        logging.disable_progress_bar();torch.set_num_threads(4)
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        tokenizer.padding_side='left'
        model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
            device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True,
            low_cpu_mem_usage=True).eval().requires_grad_(False)
        for start in range(0,len(pending),args.batch_size):
            batch=pending[start:start+args.batch_size]
            prompts=[PROMPT+'\n'.join(f'{i}: {s}' for i,s in enumerate(units)) for _,units in batch]
            rendered=[tokenizer.apply_chat_template([{'role':'user','content':p}],tokenize=False,
                        add_generation_prompt=True) for p in prompts]
            inputs=tokenizer(rendered,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                outputs=model.generate(**inputs,max_new_tokens=96,do_sample=False)
            for (row,units),ids in zip(batch,outputs[:,inputs['input_ids'].shape[1]:]):
                text=tokenizer.decode(ids,skip_special_tokens=True)
                try:indices=parse_indices(text,len(units))
                except (ValueError,TypeError) as exc:
                    errors.append(dict(id=row['id'],error=str(exc)))
                    (args.output/(row['id']+'.failed.txt')).write_text(text)
                    # The complete original request is a valid, lossless
                    # source memory. Never encode a malformed model response.
                    save(row,units,list(range(len(units))),'all_source_units_after_invalid_selection')
                    continue
                save(row,units,indices,'local_selection')
            status=dict(revision=REVISION,total=len(selection['cases']),completed=len(completed),
                selection_fallbacks=errors,seconds=time.monotonic()-started,gpu=gpu,api_calls=0,
                source_text_only=True)
            (args.output/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
            print(json.dumps(status),flush=True)
    status=dict(revision=REVISION,total=len(selection['cases']),completed=len(completed),
        selection_fallbacks=errors,seconds=time.monotonic()-started,gpu=gpu,api_calls=0,source_text_only=True)
    (args.output/'progress.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':main()
