"""Measure existing R10A Q reads of literal requests on the already fixed16."""
from pathlib import Path
import json,sys
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_r10a400 import ROOT,SOURCE,Enhancement,Teacher
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    path=ROOT/'execution_value_r10/runs/physical_content_67003/step0400.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    model=Enhancement(saved['config']).eval().requires_grad_(False)
    model.load_state_dict({n.removeprefix('enhancement.'):v for n,v in
        {**saved['model'],**saved['constants']}.items() if n.startswith('enhancement.')})
    model.checkpointing=False;teacher=Teacher(saved['config'],device='cpu')
    rounds=[]
    def read(module,args):
        state,anchor,memory,mask,groups,words,word_groups=args
        a=module.cross;start=module.grounded_heads
        split=lambda v:v.unflatten(-1,(a.heads,a.head_dim)).transpose(1,2)
        q=split(a.q(module.query_norm(state)))[:,start:]
        k=split(a.k(module.memory_norm(memory)))[:,start:]
        count=memory.shape[1]-words.shape[1]
        logits=(q.float()@k.float().transpose(-1,-2))/(a.head_dim**.5)
        attention=logits.softmax(-1)
        fraction=attention[...,:count].sum(-1)
        rounds.append(dict(request_words=count,reasoning_words=words.shape[1],
            request_mass_global_heads=float(fraction.mean()),
            request_mass_all_heads=float(fraction.mean())*(a.heads-start)/a.heads,
            equal_logit_request_mass_global_heads=count/memory.shape[1],
            request_mass_each_global_head=fraction.mean((0,2)).tolist()))
    model.block.register_forward_pre_hook(read)
    cases=[]
    for case in json.loads((SOURCE/'subset16.json').read_text())['cases']:
        rounds.clear();evidence=teacher.input(dict(id='videophy40_'+case['id']))
        with torch.autocast('cpu',dtype=torch.bfloat16):model(evidence)
        cases.append(dict(id=case['id'],caption=case['caption'],rounds=list(rounds)))
        print(json.dumps(dict(completed=len(cases),total=16,id=case['id'])),flush=True)
    aggregate={str(i+1):{key:sum(c['rounds'][i][key] for c in cases)/16
        for key in ('request_mass_global_heads','request_mass_all_heads','equal_logit_request_mass_global_heads')}
        for i in range(2)}
    result=dict(utc=datetime.now(timezone.utc).isoformat(),checkpoint=str(path),cases=cases,aggregate=aggregate,
        scope='Actual trained Q/K projections on existing fixed16; no Wan/video/API calls. '
        'Float32 softmax of the same CPU-bf16 projected Q/K for read-weight diagnostics. '
        'Attention mass is not causal importance or generation quality; values/output mixing may change contribution.')
    out=ROOT/'r11_r12/reports/r10a_request_read16.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),aggregate=aggregate)),flush=True)


if __name__=='__main__':main()
