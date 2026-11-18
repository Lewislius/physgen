"""One CPU audit of the explicit request path in the frozen R10A400 reader.

This does not remove request information contextualized into the reasoning words,
and is not a video-quality test. It locates information loss before Wan fusion.
"""
import json, sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
sys.path.insert(0, str(ROOT/'r11_r12'))
from audit_r10a400 import Enhancement, Teacher, SOURCE
import torch
from torch.nn import functional as F


def compare(actual, reference):
    a, b = actual.float(), reference.float()
    delta = a-b
    return dict(relative_l2=float(delta.norm()/b.norm().clamp_min(1e-12)),
                mean_token_cosine=float(F.cosine_similarity(a,b,dim=-1).mean()),
                delta_rms=float(delta.square().mean().sqrt()),
                reference_rms=float(b.square().mean().sqrt()))


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    ckpt=ROOT/'execution_value_r10/runs/physical_content_67003/step0400.pt'
    state=torch.load(ckpt,map_location='cpu',weights_only=False,mmap=True)
    model=Enhancement(state['config']).eval().requires_grad_(False)
    model.load_state_dict({k.removeprefix('enhancement.'):v for k,v in
        {**state['model'],**state['constants']}.items() if k.startswith('enhancement.')})
    model.checkpointing=False
    teacher=Teacher(state['config'],device='cpu')
    cases=json.loads((SOURCE/'subset16.json').read_text())['cases']
    evidence=[teacher.input(dict(id='videophy40_'+c['id'])) for c in cases]
    rows=[]
    for index,(case,own) in enumerate(zip(cases,evidence)):
        other=evidence[(index+1)%len(evidence)]
        conditions=dict(normal=own,
            explicit_request_removed=dict(own,request_words=own['request_words'][:0]),
            explicit_request_swapped=dict(own,request_words=other['request_words']))
        with torch.autocast('cpu',dtype=torch.bfloat16):
            output={name:model(item)[0] for name,item in conditions.items()}
        row=dict(id=case['id'],caption=case['caption'],swapped_request_from=cases[(index+1)%len(cases)]['id'],
            conditions={name:{key:compare(content[key],output['normal'][key])
                             for key in ('routing','execution')}
                        for name,content in output.items() if name!='normal'})
        rows.append(row)
    aggregate={arm:{key:{metric:sum(r['conditions'][arm][key][metric] for r in rows)/len(rows)
                         for metric in rows[0]['conditions'][arm][key]}
                    for key in ('routing','execution')}
               for arm in rows[0]['conditions']}
    result=dict(utc=datetime.now(timezone.utc).isoformat(),checkpoint=str(ckpt),count=len(rows),
        scope='Frozen R10A400; same16; only explicit contextual request-word memory replaced/removed. '
              'Reasoning words retain original request/image context. No generation, new GPU allocation, API or downloads. '
              'Relative norms quantify this path only, not causal importance to final videos.',
        aggregate=aggregate,cases=rows)
    target=ROOT/'state_response_fusion_r15e/review/r10a_literal_request_path16.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(target),aggregate=aggregate)),flush=True)


if __name__=='__main__':main()
