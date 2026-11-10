"""Recover actual physical word features from immutable first-image evidence."""
from pathlib import Path
import json,re
import torch
import torch.nn.functional as F

BASE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
INPUT_ROOTS=[
 BASE/'experiments/physical_scaling_q256/supervision_cache/r1/input',
 BASE/'experiments/physical_variants/supervision_cache/r1/input',
 BASE/'experiments/physical_v5_scaling_q256/supervision_cache/r1/input',
 BASE/'experiments/physical_variants/inference_cache/physical_r1_first_image_request/input',
]

def find_input(name):
    for root in INPUT_ROOTS:
        p=root/f'{name}.pt'
        if p.is_file():return p
    raise FileNotFoundError(f'No immutable first-image/request evidence for {name}')

def memory_text(evidence,duration,tokenizer):
    """Reconstruct the immutable Qwen input sequence without a backbone pass."""
    spans=[f'[RELATION {i+1}] {s}' for i,s in enumerate(evidence['relations'])]
    caption=evidence['caption']
    body=(f'Request: {caption or "No text request."}\nPresentation interval: {duration:.3f} seconds.\n'
          'Conditional physical analysis:\n'+'\n'.join(spans))
    def render(count):
        content='Initial observation:\n<|vision_start|>'+'<|image_pad|>'*count+'<|vision_end|>\n'+body
        return tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
    initial=tokenizer(render(1),add_special_tokens=False)['input_ids']
    count=evidence['memory'].shape[1]-len(initial)+1
    if count<1:raise ValueError('Input template does not match cached evidence')
    text=render(count)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    if len(encoded['input_ids'])!=evidence['memory'].shape[1]:raise ValueError('Cached memory length differs')
    offsets=torch.tensor(encoded['offset_mapping'])
    return text,offsets,spans


def recover_visual_tokens(evidence,duration,tokenizer):
    """Select actual first-image token states from the immutable input memory."""
    text,offsets,_=memory_text(evidence,duration,tokenizer)
    start=text.index('<|vision_start|>')+len('<|vision_start|>')
    end=text.index('<|vision_end|>',start)
    positions=((offsets[:,1]>start)&(offsets[:,0]<end)).nonzero().flatten()
    if 'memory_mask' in evidence:
        valid=evidence['memory_mask'][0].detach().cpu().index_select(0,positions).bool()
        positions=positions[valid]
    if not len(positions):raise ValueError('Cached first-image memory has no valid visual tokens')
    memory=evidence['memory']
    return dict(visual_memory=memory.index_select(1,positions.to(memory.device)),
                visual_token_positions=positions.to(memory.device))


def recover_request_words(evidence,duration,tokenizer):
    """Read only the literal request words from its original contextual states."""
    caption=evidence['caption']
    hidden=evidence['memory'][0]
    if not caption:
        return dict(request_words=hidden.new_empty((0,hidden.shape[-1])),request_word_strings=[])
    text,offsets,_=memory_text(evidence,duration,tokenizer)
    start=text.index('Request: ')+len('Request: ')
    if text[start:start+len(caption)]!=caption:
        raise ValueError('Request span differs from the immutable input caption')
    values=[];strings=[]
    for match in re.finditer(r'\S+',caption):
        a,b=start+match.start(),start+match.end()
        idx=((offsets[:,1]>a)&(offsets[:,0]<b)).nonzero().flatten()
        if not len(idx):raise ValueError('Request word has no cached Qwen tokens')
        values.append(hidden.index_select(0,idx.to(hidden.device)).float().mean(0))
        strings.append(match.group())
    return dict(request_words=torch.stack(values).to(hidden.dtype),request_word_strings=strings)


def recover_words(evidence,duration,tokenizer):
    """Recover spans without re-running Qwen, and verify against cached spans."""
    text,offsets,spans=memory_text(evidence,duration,tokenizer)
    hidden=evidence['memory'][0].float()
    values=[];groups=[];word_strings=[];word_positions=[];span_errors=[]
    for group,(span,relation) in enumerate(zip(spans,evidence['relations'])):
        start=text.index(span);end=start+len(span)
        idx=((offsets[:,1]>start)&(offsets[:,0]<end)).nonzero().flatten()
        original=evidence['relation_vectors'][0,group].float()
        relative=float((hidden[idx].mean(0)-original).norm()/original.norm().clamp_min(1e-8))
        span_errors.append(relative)
        if relative>.01:raise ValueError(f'Cached relation span mismatch: {relative}')
        relstart=start+len(span)-len(relation)
        for position,match in enumerate(re.finditer(r'\S+',relation)):
            a,b=relstart+match.start(),relstart+match.end()
            idx=((offsets[:,1]>a)&(offsets[:,0]<b)).nonzero().flatten()
            if not len(idx):raise ValueError('Physical word has no Qwen tokens')
            values.append(hidden[idx].mean(0));groups.append(group)
            word_strings.append(match.group());word_positions.append(position)
    return dict(words=torch.stack(values).to(torch.bfloat16),word_groups=torch.tensor(groups),
                word_positions=torch.tensor(word_positions),word_strings=word_strings,
                relations=evidence['relations'],scope=evidence['scope'],span_errors=span_errors)

def resample_words(words,word_groups,count=256):
    """Use the same lexical sequence for enhancement anchors and semantic targets."""
    if words.ndim==2:words=words[None]
    groups=torch.arange(count,device=words.device)*int(word_groups.max()+1)//count
    chunks=[];positions=[]
    for group in range(int(word_groups.max()+1)):
        x=words[:,word_groups==group]
        slots=int((groups==group).sum())
        chunks.append(F.interpolate(x.transpose(1,2).float(),size=slots,mode='linear',align_corners=True).transpose(1,2))
        positions.append(torch.linspace(0,len(x[0])-1,slots,device=words.device))
    return torch.cat(chunks,1),groups,torch.cat(positions)

def input_inventory():
    manifest=json.loads((Path('/home/liuzhirui/Project/physGen/data_cache/wisa_4000/base/manifest.json')).read_text())['records']
    rows={r['id']:r for r in manifest};paths={}
    for root in reversed(INPUT_ROOTS):
        paths.update({p.stem:p for p in root.glob('*.pt')})
    items=[]
    for name,path in sorted(paths.items()):
        sample=name.removesuffix('_text').removesuffix('_null')
        if sample in rows:
            row=rows[sample];duration=row['pts'][80]-row['pts'][0];split=row['split']
        elif sample.startswith(('videophy40_','demo21_')):
            duration=80/24;split='inference'
        else:continue
        items.append(dict(name=name,path=str(path),sample=sample,split=split,duration=duration))
    return items
