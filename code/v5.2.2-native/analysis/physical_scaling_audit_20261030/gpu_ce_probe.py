"""C-only decoder conditional-information audit; no optimizer or generation training."""
from pathlib import Path
import sys,json,os,time,argparse
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
sys.path.insert(0,str(BASE/'experiments/physical_variants/snapshots/20261029T064043Z'))
from physical_training.model import Enhancement
from physgen_native.knowledge import QwenKnowledge
from physical_training.common import gpu_guard


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--samples',type=int,default=32)
    args=parser.parse_args();device=args.device
    out=HERE/'ce_probe';out.mkdir(exist_ok=True)
    if device=='cuda':gpu_guard(out/'allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    torch.set_num_threads(16 if device=='cpu' else 4)
    path=BASE/'experiments/physical_v5_scaling_q256/runs/v5_ce_66706/step1000.pt'
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True);cfg=state['config']
    model=Enhancement(cfg).to(device).eval()
    model.load_state_dict({k.removeprefix('enhancement.'):v for k,v in state['model'].items() if k.startswith('enhancement.')})
    bridge=torch.nn.Linear(3072,4096,bias=False).to(device)
    bridge.weight.copy_(state['model']['bridge.weight']);del state
    knowledge=QwenKnowledge(cfg,device=device).eval()
    language=knowledge.backbone.model.language_model;tokenizer=knowledge.tokenizer
    prefix=tokenizer.apply_chat_template([{'role':'user','content':'Describe the relevant physical conditions, response, and allowed or preserved properties.'}],tokenize=False,add_generation_prompt=True)
    qids=tokenizer.encode(prefix,add_special_tokens=False)
    roots=[BASE/'experiments/physical_variants/supervision_cache/r1',BASE/'experiments/physical_scaling_q256/supervision_cache/r1',BASE/'experiments/physical_v5_scaling_q256/supervision_cache/r1']
    sem={p.stem:p for root in roots for p in (root/'semantic').glob('*.pt')}
    names=sorted(sem);names=[names[i] for i in np.linspace(0,len(names)-1,args.samples,dtype=int)]
    items=[]
    for name in names:
        p=next(root/'input'/f'{name}_text.pt' for root in roots if (root/'input'/f'{name}_text.pt').exists())
        e=torch.load(p,map_location='cpu',weights_only=True)
        e={k:v.to(device) if torch.is_tensor(v) else v for k,v in e.items()}
        with torch.autocast(device,dtype=torch.bfloat16):c,_=model(e)
        answer=torch.load(sem[name],map_location='cpu',weights_only=True)['answer']
        items.append((name,c,answer))
    results=[]
    for i,(name,c,answer) in enumerate(items):
        aids=tokenizer.encode(answer+tokenizer.eos_token,add_special_tokens=False)
        ids=torch.tensor([qids+aids],device=device)
        variants=dict(correct=c,unrelated=items[(i+13)%len(items)][1],token_mean=c.mean(1,keepdim=True).expand_as(c),zero=torch.zeros_like(c),no_c=c[:,:0])
        scores={}
        with torch.autocast(device,dtype=torch.bfloat16):
            for mode,tokens in variants.items():
                mapped=bridge(tokens)
                embedded=torch.cat((mapped.to(language.embed_tokens.weight.dtype),language.embed_tokens(ids)),1)
                hidden=language(inputs_embeds=embedded,use_cache=False,return_dict=True).last_hidden_state
                start=mapped.shape[1]+len(qids)
                logits=knowledge.backbone.lm_head(hidden[:,start-1:-1])
                per_token=F.cross_entropy(logits.float().flatten(0,1),ids[:,len(qids):].flatten(),reduction='none')
                scores[mode]=dict(ce=float(per_token.mean()),first8_ce=float(per_token[:8].mean()),answer_tokens=len(aids))
        row=dict(sample=name,answer=answer,scores=scores);results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    summary={mode:float(np.mean([r['scores'][mode]['ce'] for r in results])) for mode in variants}
    summary_first8={mode:float(np.mean([r['scores'][mode]['first8_ce'] for r in results])) for mode in variants}
    result=dict(checkpoint=str(path),samples=len(results),device=device,mean_ce=summary,mean_first8_ce=summary_first8,records=results)
    save(out/'results.json',result);print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
