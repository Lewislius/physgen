"""Inspect the proposed request binding in actual frozen native coordinates."""
from pathlib import Path
import json
import math
import torch
from torch.nn import functional as F
from safetensors import safe_open

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[2]
OFFICIAL=Path('/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B')


def cosine(x):
    z=F.normalize(x.float(),dim=-1);n=len(z)
    return float((z.sum(0).square().sum()-n)/(n*(n-1)))


if __name__=='__main__':
    torch.set_num_threads(4)
    weights={}
    with safe_open(str(OFFICIAL/'diffusion_pytorch_model-00001-of-00003.safetensors'),
                   framework='pt',device='cpu') as f:
        for key in ('0.weight','0.bias','2.weight','2.bias'):
            weights[key]=f.get_tensor('text_embedding.'+key).float()
    protocol=json.loads((BASE/'experiments/physical_content_v7/videophy60/protocol.json').read_text())
    native_root=BASE/'experiments/physics_content_v10_impl/runs/t2v_cache/native_words'
    words_root=native_root.parent/'words'
    result=[]
    for row in protocol['cases']:
        x=torch.load(Path(row['prepared_inputs'])/'conditions.pt',weights_only=True,
                     map_location='cpu',mmap=True)['text'].float().reshape(-1,4096)
        native=F.linear(F.gelu(F.linear(x,weights['0.weight'],weights['0.bias']),
                              approximate='tanh'),weights['2.weight'],weights['2.bias'])
        suite='videophy40' if int(row['id'][2:])<=40 else 'videophy60'
        physical=torch.load(native_root/f'{suite}_{row["id"]}_text.pt',
                            weights_only=True,map_location='cpu')['native_words'].float()
        word_groups=torch.load(words_root/f'{suite}_{row["id"]}_text.pt',
                               weights_only=True,map_location='cpu')['word_groups']
        groups=torch.arange(256)*int(word_groups.max()+1)//256
        physical=torch.cat([F.interpolate(physical[word_groups==g].T[None],
                           size=int((groups==g).sum()),mode='linear',align_corners=True)[0].T
                           for g in range(int(word_groups.max()+1))])
        logits=F.normalize(physical,dim=-1)@F.normalize(native,dim=-1).T*math.sqrt(3072)
        binding=logits.softmax(-1);anchored=binding@native
        result.append(dict(id=row['id'],physics_slots=len(physical),request_tokens=len(native),
            physics_cosine=cosine(physical),request_bound_cosine=cosine(anchored),
            request_binding_entropy=float(-(binding*binding.clamp_min(1e-20).log()).sum(-1).mean()/math.log(len(native))),
            distinct_max_request_positions=int(binding.argmax(-1).unique().numel()),
            maximum_weight_mean=float(binding.max(-1).values.mean())))
    summary={key:sum(row[key] for row in result)/len(result) for key in
             ('physics_cosine','request_bound_cosine','request_binding_entropy','maximum_weight_mean')}
    out=dict(scope='all60 actual256-slot initialization: execution=physical, route=request binding; no optimization/video inference',
             summary=summary,records=result,quality_proven=False)
    (ROOT/'request_binding_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(summary))
