"""Compare FP32 and bf16 native V/O target construction on eight real examples."""
from pathlib import Path
import sys,json
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from audit_targets import allfiles,mask,BASE
from audit_representations import base_tensor
torch.set_num_threads(8)

@torch.no_grad()
def main():
    old=json.loads((HERE/'write_zero_baseline.json').read_text())
    selected=old['records'][::max(1,len(old['records'])//8)][:8]
    records=json.loads(Path('/home/liuzhirui/Project/physGen/data_cache/wisa_4000/base/manifest.json').read_text())['records']
    manifest={r['id']:r for r in records};files=allfiles('v6_write');rows=[]
    for row in selected:
        d=torch.load(files[row['sample']],map_location='cpu',weights_only=True)
        support=d['support'][0].float();anchor=d['text_anchors'][0].float()
        w=mask(manifest[row['sample']]).reshape(-1);losses=[]
        for site in range(2,30,3):
            v=base_tensor(f'blocks.{site}.cross_attn.v.weight');o=base_tensor(f'blocks.{site}.cross_attn.o.weight')
            with torch.autocast('cpu',dtype=torch.bfloat16):
                projected=F.linear(F.linear(anchor,v),o)
            target=support@projected.float()
            denom=w.sum()*target.shape[-1]
            rms=((target.square()*w[:,None]).sum()/denom).sqrt().clamp_min(1e-3)
            error=F.smooth_l1_loss(target/rms,torch.zeros_like(target),reduction='none')
            losses.append(float((error*w[:,None]).sum()/denom))
        rows.append(dict(sample=row['sample'],zero_fp32=row['zero_aux'],zero_bf16_vo=sum(losses)/10))
    result=dict(count=len(rows),records=rows,max_absolute_difference=max(abs(r['zero_fp32']-r['zero_bf16_vo']) for r in rows),
                note='CPU bf16 V/O, FP32 support matmul; isolates dominant projection precision difference, not a bitwise CUDA reproduction.')
    (HERE/'write_zero_precision.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
