"""Validate that the additional intervention fixes only physical noise inputs."""
from pathlib import Path
import importlib.util,json,sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_denoising_fusion import DenoisingFusionTest
from physgen_native.generator import sinusoidal_embedding_1d
root=Path(__file__).resolve().parent
meta=json.loads((root/'time_mean_controls_helper_prepared.json').read_text())
spec=importlib.util.spec_from_file_location('physical_time_mean_control',Path(meta['source_snapshot'])/'time_mean.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
fixture=DenoisingFusionTest();fixture.setUp()
try:
 model=fixture.model();fixture.activate(model);model.cfg['inference']=dict(steps=50,shift=5.)
 fusion=model.fusions['1'];attention=model.wan.blocks[1].cross_attn
 c,groups=model.enhancement(fixture.evidence);x=torch.randn(1,6,16)
 args=(x,c,fixture.sample['times'],(3,1,2),attention,groups)
 low=sinusoidal_embedding_1d(8,torch.tensor([100.])).float();high=sinusoidal_embedding_1d(8,torch.tensor([900.])).float()
 with torch.no_grad():
  assert (fusion(*args,low)-fusion(*args,high)).abs().max()>1e-6
  u=torch.linspace(1,0,51)[:-1];mean=sinusoidal_embedding_1d(8,(5*u/(1+4*u))*1000).float().mean(0,keepdim=True)
  expected=fusion(*args,mean);handles,record=module.install_time_mean(model)
  torch.testing.assert_close(fusion(*args,low),expected,rtol=0,atol=0)
  torch.testing.assert_close(fusion(*args,high),expected,rtol=0,atol=0)
  native=[];handle=model.wan.time_embedding.register_forward_pre_hook(lambda _,a:native.append(a[0].detach().clone()))
  for sigma in [torch.tensor(.1),torch.tensor(.9)]:model.velocity(fixture.noise,fixture.sample['first'],fixture.text,sigma,c,fixture.sample['times'],groups)
  assert (native[0]-native[1]).abs().max()>1e-6
  handle.remove()
  for handle in handles:handle.remove()
  assert (fusion(*args,low)-fusion(*args,high)).abs().max()>1e-6
 print(json.dumps(dict(passed=True,physical_noise_fixed=True,native_timestep_still_varies=True,training_changes=0,definition=record)))
finally:fixture.doCleanups()
