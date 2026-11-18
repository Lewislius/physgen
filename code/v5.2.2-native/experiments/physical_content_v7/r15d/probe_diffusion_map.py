from pathlib import Path
import sys,json,torch
r=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/denoising_matched_fusion_r15d');p=json.loads((r/'prepared.json').read_text());sys.path.insert(0,p['snapshot']);import bootstrap
from physgen_native.generator import sinusoidal_embedding_1d
s=torch.load(r/'runs/physical_content_67920/step0400.pt',map_location='cpu',weights_only=False,mmap=True);torch.set_num_threads(4)
u=torch.linspace(1,0,51)[:-1];sigmas=5*u/(1+4*u);rows={}
for name,w in s['model'].items():
 if not name.endswith('.denoising.weight'):continue
 features=sinusoidal_embedding_1d(w.shape[1],sigmas*1000).float();q,v=torch.nn.functional.linear(features,w.float()).chunk(2,-1);part={}
 for label,t in [('query_shift',q),('value_scale',v)]:
  norms=t.square().mean(-1).sqrt();part[label]=dict(rms=float(t.square().mean().sqrt()),rms_range=[float(norms.min()),float(norms.max())],time_varying_energy_fraction=float((t-t.mean(0)).square().sum()/t.square().sum().clamp_min(1e-20)))
 rows[name]=part
report=dict(step=400,scope='CPU float32 evaluation of only the learned diffusion map on the exact50-step inference sigma grid; no full-generator execution, new video, training or scoring. Query/output magnitude relative to native states is not measured here.',quality_proof=False,per_layer=rows,mean={k:{m:sum(v[k][m] for v in rows.values())/len(rows) for m in ['rms','time_varying_energy_fraction']} for k in ['query_shift','value_scale']})
(r/'reports/diffusion_map_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['mean']))
