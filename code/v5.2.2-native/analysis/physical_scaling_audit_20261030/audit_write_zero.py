"""Exact zero-write loss against real cached synthetic targets on final50 samples."""
from pathlib import Path
import sys,json,time
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from audit_targets import allfiles,mask,stats,BASE,OUT
from audit_representations import base_tensor
torch.set_num_threads(8)


def main():
    logs=[json.loads(s) for s in (BASE/'experiments/physical_scaling_q256/runs/v6_write_resume860_s20_66710/train.jsonl').read_text().splitlines()]
    active=[r for r in logs if r['kind']=='micro' and r['step']>=951 and r['aux_active']]
    manifest=json.loads(Path('/home/liuzhirui/Project/physGen/data_cache/wisa_4000/base/manifest.json').read_text())['records'];rows={r['id']:r for r in manifest}
    files=allfiles('v6_write');data=[]
    for r in active:
        d=torch.load(files[r['sample']],map_location='cpu',weights_only=True)
        data.append((r,d['support'][0].float(),d['text_anchors'][0].float(),mask(rows[r['sample']]).reshape(-1)))
    per=[dict(sample=r['sample'],step=r['step'],actual_aux=r['aux'],zero_losses=[],target_rms=[]) for r in active]
    start=time.monotonic()
    for site in range(2,30,3):
        v=base_tensor(f'blocks.{site}.cross_attn.v.weight');o=base_tensor(f'blocks.{site}.cross_attn.o.weight')
        for j,(row,support,anchor,w) in enumerate(data):
            desired=support@F.linear(F.linear(anchor,v),o)
            denom=w.sum()*desired.shape[-1]
            rms=((desired.square()*w[:,None]).sum()/denom).sqrt()
            z=desired/rms.clamp_min(1e-3)
            error=F.smooth_l1_loss(z,torch.zeros_like(z),reduction='none')
            loss=float((error*w[:,None]).sum()/denom)
            per[j]['zero_losses'].append(loss);per[j]['target_rms'].append(float(rms))
        print('write zero layer',site,'seconds',round(time.monotonic()-start,1),flush=True)
    for r in per:r['zero_aux']=sum(r['zero_losses'])/10
    result=dict(count=len(per),actual_aux=stats([r['actual_aux'] for r in per]),zero_aux=stats([r['zero_aux'] for r in per]),
                target_rms=stats([x for r in per for x in r['target_rms']]),
                fraction_rms_below_clamp=sum(x<1e-3 for r in per for x in r['target_rms'])/(10*len(per)),
                zero_better_than_model_samples=sum(r['zero_aux']<r['actual_aux'] for r in per),records=per,
                precision='FP32 analytical zero-write baseline; training targets use bf16-autocast V/O before float regression')
    (OUT/'write_zero_baseline.json').write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if k!='records'},flush=True)


if __name__=='__main__':main()
