"""CPU audit of real cached source values, before any project optimization."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import torch
from torch.nn import functional as F
from safetensors import safe_open
from physical_content.physical_blocks import SourceContent
from physical_content.evidence import resample_words
from run_candidate import PAIRED

torch.set_num_threads(4)
root=Path(__file__).resolve().parents[1]
cache=root.parent/'physical_content_r19ct/prepared_inputs/grounded_ct'
wan=Path('/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B')
index=json.loads((wan/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
key='blocks.14.cross_attn.v.weight'
with safe_open(str(wan/index[key]),framework='pt',device='cpu') as f:
    weight=f.get_tensor(key).float()
records=[]
for case in PAIRED:
    ev=torch.load(cache/f'videophy40_{case}/positive.pt',map_location='cpu',weights_only=True)
    source=ev['native_condition'].float()
    anchor,groups,_=resample_words(source,ev['word_groups'],256)
    c,prob=SourceContent.bind(anchor,source[None],(groups[:,None]==ev['word_groups'][None,:])[None])
    value=F.linear(c,weight)
    def cosine(x):
        x=F.normalize(x[0],dim=-1);n=len(x)
        return float((x.sum(0).square().sum()-n)/(n*(n-1)))
    def rms(x):return float(x.square().mean().sqrt())
    records.append(dict(case=case,source_word_count=len(source),anchor_cos=cosine(anchor),selected_cos=cosine(c),
                        source_rms=rms(source),selected_rms=rms(c),condition_projection_rms=rms(value),
                        condition_silu_rms=rms(F.silu(value)),max_source_probability=float(prob.amax(-1).mean()),
                        selected_centered_energy=float((c-c.mean(1,keepdim=True)).square().sum()/c.square().sum())))
report=dict(scope='fixed16 existing frozen sources; initialization-only; not generated-video quality',cases=records,
            mean={k:sum(r[k] for r in records)/len(records) for k in records[0] if k!='case'})
(root/'reports/source_scale_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['mean']))
