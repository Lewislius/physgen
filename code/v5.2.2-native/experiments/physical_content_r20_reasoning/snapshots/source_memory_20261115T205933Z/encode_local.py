"""Rebuild compatible Qwen3-VL and native Wan input features, without generation."""
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
OLD = BASE/'experiments/physics_content_v10_impl'
QWEN = '/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-8B-Instruct'
WAN = Path('/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B')
REVISION = 'R20-source-grounded-physical-memory-v5'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--text-records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if selection['scope'] != 'request_only':
        raise ValueError('Only input-only T2V records belong here')
    rows = selection['cases']
    records = {}
    for row in rows:
        record = json.loads((args.text_records/(row['id']+'.json')).read_text())
        if (record['caption'] != row['caption'] or record['revision'] != REVISION
                or not record['final_answer_complete'] or not record['final_answer_only']):
            raise ValueError('Reasoning does not match the requested input')
        records[row['id']] = record
    sys.path.insert(0, str(BASE/'tools'))
    from gpu_policy import require_rtx_pro_6000
    require_rtx_pro_6000()
    import torch
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
    torch.set_num_threads(4)
    roots = {name:args.output/name for name in ('input','words','native_words','inference')}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()

    def progress(stage, completed):
        status = dict(stage=stage,completed=completed,total=len(rows),
                      seconds=time.monotonic()-began,revision=REVISION,api_calls=0)
        path=args.output/'progress.json'; temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(path)
        print(json.dumps(status),flush=True)

    def save(value, path):
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(value, temporary)
        temporary.replace(path)

    def word_features(text, offsets, hidden, begin, content):
        values, positions, strings = [], [], []
        for match in re.finditer(r'\S+', content):
            selected = ((offsets[:,1] > begin+match.start()) &
                        (offsets[:,0] < begin+match.end())).nonzero().flatten()
            if not len(selected):
                raise ValueError('A source word was truncated')
            values.append(hidden[selected.to(hidden.device)].float().mean(0).cpu())
            positions.append(int(selected[0]));strings.append(match.group())
        return values, positions, strings

    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(QWEN,
        dtype=torch.bfloat16, device_map={'':'cuda'}, attn_implementation='sdpa',
        local_files_only=True, low_cpu_mem_usage=True).eval().requires_grad_(False)
    with torch.inference_mode():
        for index,row in enumerate(rows):
            record = records[row['id']];relations = record['relations']
            ready = [roots[k]/(row['id']+suffix) for k,suffix in
                     [('input','_text.pt'),('words','_text.pt'),('input','_null.pt'),
                      ('words','_null.pt'),('native_words','_null.pt')]]
            if all(p.is_file() for p in ready):
                previous=torch.load(ready[0],map_location='cpu',weights_only=True)
                if previous.get('revision')!=REVISION or previous['relations']!=relations or previous['caption']!=row['caption']:
                    raise ValueError('Existing rebuilt input has a different description')
                if (index+1)%32==0 or index+1==len(rows):progress('qwen_features',index+1)
                continue
            header = (f"Request: {row['caption']}\nPresentation interval: {row['duration']:.3f} seconds.\n"
                      'Conditional physical analysis:\n')
            spans = [f'[RELATION {i+1}] {s}' for i,s in enumerate(relations)]
            text = header+'\n'.join(spans)
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                                return_tensors='pt', truncation=False)
            offsets = encoded.pop('offset_mapping')[0]
            hidden = model.model(**{k:v.to('cuda') for k,v in encoded.items()},
                                 use_cache=False, return_dict=True).last_hidden_state[0]
            values, positions, strings, groups = [], [], [], []
            cursor = len(header)
            for group,(span,relation) in enumerate(zip(spans,relations)):
                part,where,words = word_features(text,offsets,hidden,
                                                cursor+len(span)-len(relation),relation)
                values.extend(part);positions.extend(where);strings.extend(words)
                groups.extend([group]*len(part));cursor += len(span)+1
            request,_,request_strings = word_features(text,offsets,hidden,len('Request: '),row['caption'])
            value = dict(sample=row['id'],caption=row['caption'],duration=row['duration'],
                scope='request_only',relations=relations,analysis=record['analysis'],
                analysis_complete=record['analysis_complete'],final_answer_complete=True,
                thinking_end=record['thinking_end'],revision=REVISION,reasoning_model=record['model'],
                model='Qwen3-VL-8B-Instruct',words=torch.stack(values).to(torch.bfloat16),
                word_groups=torch.tensor(groups),word_positions=torch.tensor(positions),
                word_strings=strings,request_words=torch.stack(request).to(torch.bfloat16),
                request_word_strings=request_strings,reasoning_record=str(args.text_records/(row['id']+'.json')))
            save(value,roots['input']/(row['id']+'_text.pt'))
            save({k:value[k] for k in ('sample','scope','words','word_groups','word_positions','word_strings')},
                 roots['words']/(row['id']+'_text.pt'))
            source = Path(row['source'])
            negative_source = (source.with_name('negative.pt') if source.name=='positive.pt'
                               else source.with_name(source.name.replace('_text.pt','_null.pt')))
            negative = torch.load(negative_source,map_location='cpu',weights_only=True)
            if negative['sample']!=row['id'] or negative['caption']!='':
                raise ValueError('Original null input mismatch')
            save(negative,roots['input']/(row['id']+'_null.pt'))
            save({k:negative[k] for k in ('sample','scope','words','word_groups','word_positions','word_strings')},
                 roots['words']/(row['id']+'_null.pt'))
            if 'native_condition' in negative:
                null_native=dict(native_words=negative['native_condition'],word_strings=negative['word_strings'],
                                 sample=row['id'],scope='request_only')
            else:
                null_native=torch.load(source.parent.parent/'native_words'/negative_source.name,
                                       map_location='cpu',weights_only=True)
            save(null_native,roots['native_words']/(row['id']+'_null.pt'))
            if (index+1)%32==0 or index+1==len(rows):progress('qwen_features',index+1)
    del model,tokenizer
    # Last hidden/encoded may be absent when every Qwen record was resumed.
    if 'hidden' in locals():del hidden
    if 'encoded' in locals():del encoded
    gc.collect();torch.cuda.empty_cache()
    sys.path.insert(0,str(OLD))
    from prepare_native_t2v import native_embedding
    sys.path.insert(0,'/home/liuzhirui/model/Wan2.2')
    from wan.modules.t5 import T5EncoderModel
    encoder=T5EncoderModel(text_len=512,dtype=torch.bfloat16,device='cuda',
        checkpoint_path=str(WAN/'models_t5_umt5-xxl-enc-bf16.pth'),
        tokenizer_path=str(WAN/'google/umt5-xxl'))
    native=native_embedding('cuda')
    with torch.inference_mode():
        for index,row in enumerate(rows):
            value=torch.load(roots['input']/(row['id']+'_text.pt'),weights_only=True)
            previous_path=roots['native_words']/(row['id']+'_text.pt')
            inference_ready=roots['inference']/row['id']/'ready.json'
            if previous_path.is_file() and (not row['split'].endswith('eval') or inference_ready.is_file()):
                previous=torch.load(previous_path,map_location='cpu',weights_only=True)
                if previous.get('reasoning_revision')!=REVISION or previous['word_strings']!=value['word_strings']:
                    raise ValueError('Existing native features have different physical words')
                if (index+1)%32==0 or index+1==len(rows):progress('native_features',index+1)
                continue
            # Joint native encoding retains subjects and referents across
            # selected sentences, rather than encoding each sentence alone.
            cleaned=[encoder.tokenizer._clean(s) for s in value['relations']]
            clean=' '.join(cleaned)
            encoded=encoder.tokenizer.tokenizer(clean,return_tensors='pt',
                return_offsets_mapping=True,add_special_tokens=True,padding=False,truncation=False)
            offsets=encoded.pop('offset_mapping')[0]
            if encoded['input_ids'].shape[1]>512:
                raise ValueError('Physical memory exceeds the native text context; no truncation is allowed')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                hidden=encoder.model(encoded['input_ids'].to('cuda'),encoded['attention_mask'].to('cuda'))
                target=native(hidden).float()[0]
            features,_,words=word_features(clean,offsets,target,0,clean)
            if len(words)!=len(value['word_strings']):
                raise ValueError('Native and Qwen physical word alignment differs')
            native_value=dict(native_words=torch.stack(features).to(torch.bfloat16),
                word_strings=value['word_strings'],scope='request_only',sample=row['id'],
                reasoning_revision=REVISION,native_encoding='joint selected source context',
                teacher='official frozen T5 then official Wan text_embedding')
            save(native_value,roots['native_words']/(row['id']+'_text.pt'))
            if row['split'].endswith('eval'):
                folder=roots['inference']/row['id'];folder.mkdir(exist_ok=True)
                value['native_condition']=native_value['native_words'];save(value,folder/'positive.pt')
                negative=torch.load(roots['input']/(row['id']+'_null.pt'),weights_only=True)
                negative['native_condition']=torch.load(roots['native_words']/(row['id']+'_null.pt'),
                                                        weights_only=True)['native_words']
                save(negative,folder/'negative.pt')
                ready=dict(id=row['id'],caption=row['caption'],duration=row['duration'],scope='request_only',
                    revision='NoChunks-v2-20261111',reasoning_revision=REVISION,
                    feature_encoder=QWEN,native_wan_prompt_unchanged=True)
                (folder/'ready.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2)+'\n')
            if (index+1)%32==0 or index+1==len(rows):progress('native_features',index+1)
    (args.output/'ready.json').write_text(json.dumps(dict(count=len(rows),revision=REVISION,
        feature_encoder=QWEN,scope='request_only',api_calls=0,null_inputs='unchanged originals'),indent=2)+'\n')
    print(json.dumps(dict(status='encoded',count=len(rows),output=str(args.output))),flush=True)


if __name__=='__main__':
    main()
