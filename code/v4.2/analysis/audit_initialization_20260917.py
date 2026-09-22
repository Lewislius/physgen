"""Measure the actual old checkpoint's deployable P0; never alter its weights."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from physgen_v42.model import Adapter
from physgen_v42.geometry import Geometry
from physgen_v42.losses import struct_loss
from physgen_v42.runtime import autocast

checkpoint = ROOT / 'checkpoints/stability_20260917T113347Z_5d0494/step0100'
cfg = json.loads((checkpoint / 'config.json').read_text())
cache = Path(cfg['paths']['cache'])
manifest = json.loads((cache / 'manifest.json').read_text())
records = {r['id']: r for r in manifest['records']}
device = torch.device('cuda:0')
report = {}
for weights in ('ema.pt', 'adapter.pt'):
    state = torch.load(checkpoint / weights, map_location='cpu', weights_only=True, mmap=True)
    stats = {k: state[k] for k in ('mu_first', 's_z', 's_v')}
    with torch.device('meta'):
        adapter = Adapter(cfg, stats)
    adapter.load_state_dict(state, assign=True, strict=True)
    adapter = adapter.eval().requires_grad_(False).to(device)
    results = []
    with torch.no_grad():
        for key in ('source0000949', 'source0001971'):
            record = records[key]
            geo = Geometry.build(record['geometry'], device)
            def load(stage):
                return torch.load(cache / stage / (key + '.pt'), map_location=device, weights_only=True)
            text, anchor, target = load('text')[None], load('anchors'), load('teacher')
            mean = adapter.mu_first[None, None].expand(1, len(geo.p_phase), -1, -1)
            with autocast(device):
                t2v = adapter.initialize(text, geo, 't2v')
                i2v = adapter.initialize(text, geo, 'i2v', anchor)
                shuffled = adapter.initialize(torch.flip(text, dims=[1]), geo, 't2v')
            results.append(dict(id=key, s_z=float(adapter.s_z),
                mean_to_target=float(struct_loss(mean, target, adapter.s_z, geo.p_weight)),
                t2v_to_target=float(struct_loss(t2v, target, adapter.s_z, geo.p_weight)),
                i2v_to_target=float(struct_loss(i2v, target, adapter.s_z, geo.p_weight)),
                t2v_delta_rms_over_sz=float((t2v-mean).square().mean().sqrt()/adapter.s_z),
                i2v_time_difference_max=float(i2v[:, 1:].sub(i2v[:, :1]).abs().max()),
                t2v_time_difference_max=float(t2v[:, 1:].sub(t2v[:, :1]).abs().max()),
                # Reordering contextual T5 tokens tests attention order invariance, not caption semantics.
                token_permutation_delta=float((t2v-shuffled).square().mean().sqrt())))
        outputs = []
        for key in ('P01_t2v', 'P02_t2v'):
            folder = ROOT / 'inference_outputs/stability_20260917T113347Z_5d0494_step0100_ema_demo_seed42' / key
            condition = torch.load(folder/'condition.pt', map_location=device, weights_only=True)
            geo = Geometry.build(condition['geometry'], device)
            with autocast(device):
                outputs.append(adapter.initialize(condition['text'][None], geo, 't2v')[:, :1])
        report[weights] = dict(samples=results,
                              different_prompt_p0_rms_over_sz=float((outputs[0]-outputs[1]).square().mean().sqrt()/adapter.s_z))
    del adapter
path = ROOT / 'analysis/generation_failure_20260917/initialization_comparison.json'
path.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
