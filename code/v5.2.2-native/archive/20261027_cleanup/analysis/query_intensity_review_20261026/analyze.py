"""Recompute reported evidence from existing records/cache; no model training."""
from pathlib import Path
import json
import math
import sys
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from physgen_native.config import read_config
from physgen_native.data import manifest,valid_grid,load_tensor
from physgen_native.spatial import pool_grid
from physgen_native.sequence import weighted_mean

torch.set_num_threads(4)
cfg=read_config(ROOT/'configs/native_fullwidth_attention_wisa3857.yaml')
cache=Path(cfg['paths']['cache']);rows=manifest(cfg)
scale=json.loads((cache/'v52/train_scale.json').read_text())['scale']
records=[json.loads(line) for line in (ROOT/'analysis/queue_20261026_fullwidth_attention/results/metrics.jsonl').read_text().splitlines()]
read=next(r for r in records if r['kind']=='native_equivalence_and_read')
back=next(r for r in records if r['kind']=='generation_backward')
feedback=next(r for r in records if r['kind']=='complete_trajectory_feedback')
ratios=[dict(layer=x['layer'],C_message_rms=x['c_message_rms'],text_message_rms=x['text_message_rms'],
             ratio=x['c_message_rms']/x['text_message_rms']) for x in read['layers']]

def target(row):
    item=load_tensor(cache/'v52/change'/f'{row["id"]}.pt')
    valid=valid_grid(row,30,52)
    y,pv=pool_grid((item['intensity'].float()/scale).clamp(0,1)[None],valid,(15,26))
    iv,_=pool_grid(item['valid'].float()[None],valid,(15,26))
    return y,iv*pv

selected={}
for split in ('train','validation'):
    candidates=[r for r in rows if r['split']==split]
    ids=torch.linspace(0,len(candidates)-1,32).round().long().tolist()
    selected[split]=[candidates[i] for i in ids]
train=[target(row) for row in selected['train']]
constant=float(sum((y*w).sum() for y,w in train)/sum(w.sum() for _,w in train))
summary={}
for split,chosen in selected.items():
    cases=train if split=='train' else [target(row) for row in chosen]
    ys=torch.cat([y.flatten() for y,w in cases]);ws=torch.cat([w.flatten() for y,w in cases])
    summary[split]=dict(ids=[r['id'] for r in chosen],weighted_mean=float((ys*ws).sum()/ws.sum()),
                       fraction_below_half=float(((ys<.5)*ws).sum()/ws.sum()),
                       train_constant_smooth_l1=float((F.smooth_l1_loss(torch.full_like(ys,constant),ys,reduction='none')*ws).sum()/ws.sum()))
y,w=target(next(r for r in rows if r['id']==read['sample']))
single=dict(sample=read['sample'],reported_model_intensity_loss=back['losses']['intensity'],
            reported_weighted_rmse=math.sqrt(2*back['losses']['intensity']),
            target_mean=float(weighted_mean(y,w)),
            constant_losses={str(c):float(weighted_mean(F.smooth_l1_loss(torch.full_like(y,c),y,reduction='none'),w))
                             for c in (0.,constant,.5)},
            interpretation='One migrated unadapted model, one sample; constants use no model input. Not validation accuracy.')

torch.manual_seed(261026)
q=torch.randn(24,9,128,dtype=torch.float64);k=torch.randn(24,7,128,dtype=torch.float64)
r=torch.randn(24,128,dtype=torch.float64);a=torch.rand(9,dtype=torch.float64)-.5
delta=a[:,None]*r.reshape(1,-1)
actual=((q+a[None,:,None]*r[:,None,:])@k.transpose(-1,-2)-q@k.transpose(-1,-2))/128**.5
expected=a[None,:,None]*torch.einsum('hd,hnd->hn',r,k)[:,None,:]/128**.5
algebra=dict(delta_Q_rank=int(torch.linalg.matrix_rank(delta)),delta_Q_shape=list(delta.shape),
             attention_logit_identity_max_error=float((actual-expected).abs().max()),
             interpretation='Synthetic arithmetic check of exact rank-one formula, not physical performance evidence.')
result=dict(revision=cfg['revision'],native_equivalence_max_error=read['native_max_absolute_error'],
            sample=read['sample'],message_ratios=ratios,
            output_change_rms=read['on_minus_off_rms'],native_output_rms=read['off_velocity_rms'],
            output_change_over_native_rms=read['on_minus_off_rms']/read['off_velocity_rms'],
            generation_C_gradient_rms=back['C_gradient_rms'],verifier_C_gradient_rms=feedback['C_gradient_rms'],
            verifier_calibrated=False,existing_cache_intensity_scale=scale,train_constant=constant,
            target_summary=summary,single_sample_intensity=single,algebra=algebra)
(Path(__file__).parent/'evidence.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='target_summary'},indent=2))
