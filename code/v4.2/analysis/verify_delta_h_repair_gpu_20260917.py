"""Two tiny real-Wan gradient checks, not a video-quality or capacity experiment."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from physgen_v42.backbone import ProcessWan, load_wan
from physgen_v42.geometry import fit_geometry
from physgen_v42.model import Adapter
from physgen_v42.optim import EMA, GradientMixer, optimizer_for, set_schedule
from physgen_v42.runtime import read_config, seed_all
from train.train import norm_group, train_micro_gated, health_check

torch.set_num_threads(4)
cfg = read_config(ROOT / 'configs/stability.yaml')
cache = Path(cfg['paths']['cache'])
device = torch.device('cuda:0')
seed_all(cfg['train']['seed'])
stats = torch.load(cache / 'stats.pt', map_location='cpu', weights_only=True)
adapter = Adapter(cfg, stats).float().to(device)
optimizer, ema = optimizer_for(adapter, cfg), EMA(adapter)
model = ProcessWan(load_wan(cfg['paths']['wan_checkpoint'], device), adapter)
key = 'source0000949'
def load(stage):
    return torch.load(cache / stage / (key + '.pt'), map_location=device, weights_only=True)
vae, text, target, anchor = load('vae'), load('text')[None], load('teacher'), load('anchors')
report = dict(purpose='real 5B model; 64x64, 5 frames, two one-micro updates; gradient check only', steps=[])
for step, mode in enumerate(('t2v', 'i2v'), 1):
    started = time.perf_counter()
    set_schedule(adapter, optimizer, cfg, step)
    optimizer.zero_grad(set_to_none=True)
    sample = dict(latent=vae['latent'][:, :, :2, :4, :4].clone(),
                  first=vae['first'][:, :, :, :4, :4].clone() if mode == 'i2v' else None,
                  text=text, target=target, anchor=anchor if mode == 'i2v' else None, instances=None,
                  record=dict(geometry=dict(fit_geometry(64, 64, 64, 64), frames=5), duration=5.))
    mixer = GradientMixer(adapter.parameters(), adapter.parameters(), adapter.core_parameters())
    values = train_micro_gated(model, sample, dict(mode=mode), step, mixer, 1, cfg['loss'], monitor=True, repair=dict(cfg['repair'],clean_probability=0.))
    fm_norms = {name: norm_group(group) for name, group in adapter.groups().items()}
    alignment = mixer.align_to_generation(adapter.groups())
    combined = mixer.combine(None)
    combined['alignment'] = alignment
    all_norms = {name: norm_group(group) for name, group in adapter.groups().items()}
    assert fm_norms['writer5'] > 0 and fm_norms['writer15'] > 0
    if step == 1:
        assert abs(values['fm_delta']) < 1e-7, values['fm_delta']
        assert all_norms['text_init'] > 0 and fm_norms['text_init'] == 0
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg['train']['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    ema.update(adapter)
    torch.cuda.synchronize()
    row = dict(step=step, mode=mode, fm=values['fm'], fm_base=values['fm_base'], fm_delta=values['fm_delta'],
               struct=values['struct'], struct_coefficient=values['struct_coefficient'],
               write=values['write'], write_supervision=values['write_supervision'], paired_repair=values['paired_repair'],
               write_credit={name:values['interventions'][name].get('fm_write_credit') for name in ('writer5','writer15')},
               correction_benefit=values['correction_benefit'], structure_components=values['structure_components'],
               fm_grad_groups=fm_norms, combined_grad_groups=all_norms, combined=combined,
               global_grad_norm=float(norm), seconds=time.perf_counter()-started,
               gate_means={name: values['interventions'][name]['gate_mean'] for name in ('writer5','writer15')})
    report['steps'].append(row)
    print(json.dumps(row), flush=True)
class TinyHealth:
    records=[dict(id='tiny_i2v'),dict(id='tiny_t2v')]
    manifest=dict(health_ids=['tiny_i2v','tiny_t2v'])
    def load(self,index,mode,device):
        return dict(latent=vae['latent'][:,:,:2,:4,:4].clone(),
                    first=vae['first'][:,:,:,:4,:4].clone() if mode=='i2v' else None,
                    text=text,target=target,anchor=anchor if mode=='i2v' else None,instances=None,
                    record=dict(id=self.records[index]['id'],geometry=dict(fit_geometry(64,64,64,64),frames=5),duration=5.))
health_path=ROOT/'analysis/delta_h_quality_20260917/health_contract.jsonl'
health_path.write_text('')
before_cpu,before_cuda=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
report['health']=health_check(model,TinyHealth(),2,health_path,console=False,return_summary=True,
                              ablation_cases=2,repair_settings=cfg['repair'])
assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state())
assert report['health']['writer_ablation']['cases']==2 and report['health']['paired_repair']['cases']==2
report['health_rng_restored']=True
report['parameters'] = sum(p.numel() for p in adapter.parameters())
report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
(ROOT / 'analysis/delta_h_quality_20260917/paired_write_training_gpu.json').write_text(json.dumps(report, indent=2)+'\n')
print('PASS: clean FM preserved, paired WRITE -> writers, STRUCT -> trajectory init, finite updates', flush=True)
