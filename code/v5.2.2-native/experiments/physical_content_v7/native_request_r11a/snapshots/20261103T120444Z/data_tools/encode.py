"""Re-encode API text with the EXISTING local Qwen3-VL and native T5/Wan.

No API hidden state or external embedding is used. Preparation only, no training.
"""
from pathlib import Path
import argparse
import copy
import gc
import json
import os
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
SOURCE = HERE.parent
sys.path.insert(0, str(SOURCE))
import bootstrap
sys.path.insert(0, str(HERE))
from annotate import CACHE, save, validate_relations
import torch
from physical_scaling.common import atomic_torch, gpu_guard
from physical_content.evidence import recover_words, recover_request_words


def encode_input(knowledge, image, caption, duration, reasoning, *, sample, revision):
    """Identical local template and hidden space, with externally generated TEXT."""
    if reasoning['scope'] != 'first_image_and_request_only' or reasoning['caption'] != caption:
        raise ValueError('Reasoning scope/caption differs from the local encoder input')
    relations = validate_relations(reasoning)
    spans = [f'[RELATION {i+1}] {s}' for i, s in enumerate(relations)]
    body = (f'Request: {caption or "No text request."}\nPresentation interval: {duration:.3f} seconds.\n'
            'Conditional physical analysis:\n'+'\n'.join(spans))
    inputs, positions = knowledge._inputs([(image, image.new_zeros(1))], body, extract_spans=spans)
    hidden = knowledge.backbone.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
    if hidden.shape[-1] != 4096 or any(not indices for indices in positions):
        raise ValueError('The existing 4096-wide contextual relation interface must be preserved')
    vectors = torch.stack([hidden[:, indices].mean(1) for indices in positions], 1)
    evidence = dict(memory=hidden.cpu().to(torch.bfloat16),
        memory_mask=inputs['attention_mask'].bool().cpu(), relation_vectors=vectors.cpu().to(torch.bfloat16),
        relations=relations, caption=caption, analysis='\n'.join(relations), analysis_complete=True,
        scope='first_image_and_request_only', sample=sample, revision=revision,
        model=knowledge.cfg.get('encoder_checkpoint', str(Path('/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-8B-Instruct'))),
        text_producer=reasoning['producer'], producer_request_id=reasoning['trace']['request_id'],
        prompt=reasoning['trace']['prompt'])
    words = recover_words(evidence, duration, knowledge.tokenizer)
    evidence.update(recover_request_words(evidence, duration, knowledge.tokenizer))
    words.update(caption=caption, sample=sample, revision=revision)
    return evidence, words


def native_embedding(wan, device):
    from safetensors import safe_open
    from torch import nn
    index=json.loads((wan/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    model=nn.Sequential(nn.Linear(4096,3072),nn.GELU(approximate='tanh'),nn.Linear(3072,3072))
    state={}
    for key in model.state_dict():
        full='text_embedding.'+key
        with safe_open(wan/index[full],framework='pt',device='cpu') as f:state[key]=f.get_tensor(full)
    model.load_state_dict(state)
    return model.to(device=device,dtype=torch.bfloat16).eval().requires_grad_(False)


def manifest(output):
    selection=json.loads((output/'selection.json').read_text())
    originals=json.loads((CACHE/'base/manifest.json').read_text())
    lookup={r['id']:r for r in originals['records']}
    records=[]
    for sample in selection['ids']:
        row=copy.deepcopy(lookup[sample])
        annotation=json.loads((output/'records'/sample/'caption.json').read_text())
        if annotation['old_caption'] != row['caption']:
            raise ValueError('Caption source changed')
        row.update(caption=annotation['caption'], prior_caption=row['caption'],
                   caption_version=selection['revision'], caption_annotation=str(output/'records'/sample/'caption.json'))
        records.append(row)
    # Retain the original full order so seed42 selects exactly the same 1600
    # microbatches. Only those scheduled samples receive a new text/feature suite.
    by_id={row['id']:row for row in records}
    complete_rows=[by_id.get(row['id'],copy.deepcopy(row)) for row in originals['records']]
    result=dict(schema='qwen38_caption_encoding_candidate', revision=selection['revision'],
                source_manifest=str(CACHE/'base/manifest.json'), records=complete_rows,
                revised_sample_ids=selection['ids'], maximum_training_updates=selection.get('maximum_training_updates'),
                original_record_order=[r['id'] for r in originals['records']],
                training_ready=False, reason='Needs reviewed full training coverage and new interface fit')
    save(output/'encoded/base/manifest.json',result)
    base=output/'encoded/base'
    for name in ('vae','first'):
        target=base/name
        if not target.exists():target.symlink_to(CACHE/'base'/name, target_is_directory=True)
    if not (base/'null_text.pt').exists():(base/'null_text.pt').symlink_to(CACHE/'base/null_text.pt')
    (base/'text').mkdir(exist_ok=True)
    for row in complete_rows:
        if row['id'] not in by_id:
            target=base/'text'/f'{row["id"]}.pt'
            if not target.exists():target.symlink_to(CACHE/'base/text'/target.name)
    return records,selection


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['qwen','t5','all'],default='all')
    p.add_argument('--describe',action='store_true')
    args=p.parse_args()
    if args.describe:
        print(json.dumps(dict(encoder='existing local Qwen3-VL-8B-Instruct 4096 contextual features',
            text_encoder='existing local Wan UMT5', no_download=True, gpu='single RTX6000 Blackwell',
            source=str(SOURCE), output=str(args.output), future_video_in_generation=False)))
        return
    output=args.output.resolve()
    gpu_guard(output/'allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    torch.set_num_threads(4)
    import yaml
    from PIL import Image
    import numpy as np
    cfg=yaml.safe_load((SOURCE/'configs/physical_content.yaml').read_text())
    rows,selection=manifest(output)
    revision=selection['revision']
    cache=output/'encoded/content'
    inventory=[]
    for row in rows:
        for arm in ('text','null'):
            name=f'{row["id"]}_{arm}'
            inventory.append(dict(name=name,sample=row['id'],split=row['split'],
                duration=row['pts'][80]-row['pts'][0],path=str(cache/'input'/f'{name}.pt')))
    if args.stage in ('qwen','all'):
        from physgen_native.knowledge import QwenKnowledge
        pending=[item for item in inventory if not (cache/'words'/f'{item["name"]}.pt').exists()]
        if pending:
            knowledge=QwenKnowledge(cfg).eval()
            lookup={r['id']:r for r in rows}
            for item in pending:
                row=lookup[item['sample']]
                caption='' if item['name'].endswith('_null') else row['caption']
                reasoning=json.loads((output/'reasoning'/f'{item["name"]}.json').read_text())
                with Image.open(CACHE/'base/first'/f'{row["id"]}.png') as image:
                    rgb=torch.from_numpy(np.array(image.convert('RGB'),copy=True)).permute(2,0,1)[None].cuda().float()/127.5-1
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    evidence,words=encode_input(knowledge,rgb,caption,item['duration'],reasoning,
                        sample=row['id'],revision=revision)
                atomic_torch(evidence,cache/'input'/f'{item["name"]}.pt')
                atomic_torch(words,cache/'words'/f'{item["name"]}.pt')
                save(cache/'input'/f'{item["name"]}.json',dict(sample=row['id'],caption=caption,
                    revision=revision,relations=evidence['relations'],scope=evidence['scope'],
                    memory_shape=list(evidence['memory'].shape),request_words=len(evidence['request_word_strings']),
                    physical_words=len(words['word_strings']),producer_request_id=reasoning['trace']['request_id']))
                print(json.dumps(dict(stage='qwen',name=item['name'],memory_shape=list(evidence['memory'].shape),
                    words_shape=list(words['words'].shape),request_shape=list(evidence['request_words'].shape))),flush=True)
            del knowledge;gc.collect();torch.cuda.empty_cache()
    if args.stage in ('t5','all'):
        sys.path.insert(0,cfg['paths']['wan_code'])
        from wan.modules.t5 import T5EncoderModel
        wan=Path(cfg['paths']['wan_checkpoint'])
        encoder=T5EncoderModel(text_len=512,dtype=torch.bfloat16,device='cuda',
            checkpoint_path=str(wan/'models_t5_umt5-xxl-enc-bf16.pth'),tokenizer_path=str(wan/'google/umt5-xxl'))
        native=native_embedding(wan,'cuda')
        for row in rows:
            path=output/'encoded/base/text'/f'{row["id"]}.pt'
            if path.exists():continue
            clean=encoder.tokenizer._clean(row['caption'])
            ids=encoder.tokenizer.tokenizer(clean,add_special_tokens=True,truncation=False)['input_ids']
            if len(ids)>512:raise ValueError(f'Caption exceeds native T5 length: {row["id"]}')
            text=encoder([row['caption']],'cuda')[0].cpu()
            atomic_torch(text,path)
            save(path.with_suffix('.json'),dict(caption=row['caption'],tokens=len(ids),revision=revision))
        for item in inventory:
            path=cache/'native_words'/f'{item["name"]}.pt'
            if path.exists():continue
            source=torch.load(cache/'words'/f'{item["name"]}.pt',weights_only=True,map_location='cpu')
            clean=[encoder.tokenizer._clean(text) for text in source['relations']]
            tokens=encoder.tokenizer.tokenizer(clean,return_tensors='pt',return_offsets_mapping=True,
                add_special_tokens=True,padding=True,truncation=False)
            spans=tokens.pop('offset_mapping')
            if tokens['input_ids'].shape[1]>512:raise ValueError('Native relation text exceeds T5 budget')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                hidden=encoder.model(tokens['input_ids'].cuda(),tokens['attention_mask'].cuda())
                target=native(hidden).float()
            features=[];strings=[]
            for group,text in enumerate(clean):
                for match in re.finditer(r'\S+',text):
                    idx=((spans[group,:,1]>match.start())&(spans[group,:,0]<match.end())).nonzero().flatten()
                    if not len(idx):raise ValueError('Native word is missing tokens')
                    features.append(target[group,idx.cuda()].mean(0));strings.append(match.group())
            if len(strings)!=len(source['word_strings']):raise ValueError('Qwen/T5 lexical word mapping differs')
            atomic_torch(dict(native_words=torch.stack(features).cpu().to(torch.bfloat16),word_strings=strings,
                caption=source['caption'],relations=source['relations'],sample=item['sample'],revision=revision,
                scope='first_image_request_reasoning_only',teacher='official frozen T5 and Wan text_embedding'),path)
            print(json.dumps(dict(stage='native_words',name=item['name'],words=len(strings))),flush=True)
    save(cache/'inventory.json',inventory)
    completed=all((cache/k/f'{i["name"]}.pt').exists() for i in inventory for k in ('input','words','native_words'))
    completed=completed and all((output/'encoded/base/text'/f'{r["id"]}.pt').exists() for r in rows)
    save(output/'encoding_status.json',dict(stage=args.stage,complete=completed,rows=len(rows),revision=revision,
        api_hidden_states_used=False,local_feature_width=4096,native_word_width=3072,
        training_ready=False,reason='Caption review/full coverage and new train-input interface fit required'))


if __name__=='__main__':main()
