"""Exercise production full/base sampling on a tiny zero-adapter case; not quality."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from inference.infer import RECIPE, sample_case
from physgen_v42.backbone import ProcessWan, load_wan
from physgen_v42.encoders import Decoder, load_vae
from physgen_v42.geometry import fit_geometry
from physgen_v42.model import Adapter
from physgen_v42.runtime import read_config, seed_all, save_tensor

cfg=read_config(ROOT/'configs/stability.yaml')
device=torch.device('cuda:0');torch.set_num_threads(4);seed_all(42)
stats=torch.load(Path(cfg['paths']['cache'])/'stats.pt',map_location='cpu',weights_only=True)
adapter=Adapter(cfg,stats).to(device).eval().requires_grad_(False)
model=ProcessWan(load_wan(cfg['paths']['wan_checkpoint'],device),adapter,recompute=False)
decoder=Decoder(load_vae(cfg['paths'],device),recompute=False,save_on_cpu=False)
source=ROOT/'inference_outputs/stability_20260917T113347Z_5d0494_step0100_ema_demo_seed42'
original=json.loads((source/'cases.json').read_text())
case=dict(key='single_t2v',id='contract',group='contract',mode='t2v',duration=5.,
          prompt=next(c['prompt'] for c in original['cases'] if c['key']=='P01_t2v'),
          geometry=dict(fit_geometry(64,64,64,64),frames=5))
condition=torch.load(source/'P01_t2v/condition.pt',map_location='cpu',weights_only=True)
condition.update(geometry=case['geometry'],first=None)
out=ROOT/'analysis/generation_failure_20260917/inference_contract_r2'
latents=[]
with torch.no_grad():
    for variant in ('full','base'):
        folder=out/variant
        save_tensor(condition,folder/case['key']/'condition.pt')
        plan=dict(config=cfg,checkpoint='zero_adapter_contract_only',checkpoint_step=0,ema_updates=0,
                  ema_file=dict(purpose='new zero-initialized adapter, no trained checkpoint'),
                  negative=original['negative'],negative_sha256=original['negative_sha256'],
                  recipe=dict(RECIPE,steps=4,variant=variant))
        handles=[]
        if variant=='base':
            def fail(*args): raise AssertionError('base variant called the adapter')
            handles=[c.register_forward_pre_hook(fail) for corr in adapter.correctors.values()
                     for c in (corr.core,corr.writer)]
        try:
            sample_case(model,decoder,case,plan,folder,device)
        finally:
            for h in handles: h.remove()
        latents.append(torch.load(folder/case['key']/'final_latent.pt',weights_only=True))
torch.testing.assert_close(latents[0],latents[1],rtol=0,atol=0)
result=dict(purpose='64x64,5 frames,4 solver steps; interface/zero-init check only',
            base_skips_adapter=True,full_base_max_abs=float((latents[0]-latents[1]).abs().max()),
            source_path=cfg['paths']['wan_code'],checkpoint_path=cfg['paths']['wan_checkpoint'])
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
