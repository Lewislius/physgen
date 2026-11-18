"""Own400 fixed16: inspect route/execute geometry without loading Wan or a GPU."""
from pathlib import Path
import json
import sys
from datetime import datetime, timezone
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'snapshots/20261105T210320Z'
sys.path.insert(0, str(SOURCE))
import bootstrap
from physical_content.model import Enhancement
from physical_content.teacher import Teacher
from physical_content.evidence import resample_words


def stats(value):
    value = value.detach().float().reshape(-1, value.shape[-1])
    count = len(value)
    unit = F.normalize(value, dim=-1)
    return dict(token_cosine=float((unit.sum(0).square().sum()-count)/(count*(count-1))),
                rms=float(value.square().mean().sqrt()),
                common_token_energy_fraction=float(value.mean(0).square().sum()*count /
                                                   value.square().sum().clamp_min(1e-20)))


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    path = ROOT / 'runs/physical_content_67904/step0400.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert saved['step'] == 400 and Path(saved['config']['source_snapshot']) == SOURCE
    model = Enhancement(saved['config']).eval().requires_grad_(False)
    model.load_state_dict({k.removeprefix('enhancement.'): v
                           for k,v in {**saved['model'],**saved['constants']}.items()
                           if k.startswith('enhancement.')}, strict=True)
    model.checkpointing = False
    captured = {}
    model.execution_project.register_forward_pre_hook(lambda _, args: captured.update(update=args[0]))
    model.project.register_forward_pre_hook(lambda _, args: captured.update(final_read=args[0]))
    teacher = Teacher(saved['config'],device='cpu')
    cases = json.loads((SOURCE/'subset16.json').read_text())['cases']
    rows, means = [], {}
    for case in cases:
        evidence = teacher.input(dict(id='videophy40_'+case['id']))
        with torch.autocast('cpu',dtype=torch.bfloat16):
            content, _ = model(evidence)
            grounded=(evidence['words'].float()-model.anchor_mean)/model.anchor_scale
            anchor, _, _ = resample_words(grounded,evidence['word_groups'],model.seed.shape[1])
            native=F.linear(anchor,model.anchor_weight,model.anchor_bias)
        values = dict(query_update=captured['update'], final_value_read=captured['final_read'], anchor=native,
                      routing_correction=content['routing']-native,
                      execution_residual=content['execution']-native,
                      routing_C=content['routing'],execution_C=content['execution'])
        row=dict(id=case['id'],metrics={k:stats(v) for k,v in values.items()},
                 route_execution_cosine=float(F.cosine_similarity(content['routing'].float(),
                                                content['execution'].float(),dim=-1).mean()))
        rows.append(row)
        for k,v in values.items():means.setdefault(k,[]).append(v.float().mean(1))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),checkpoint=str(path),source=str(SOURCE),
                protocol='Own400 fixed16, CPU enhancement only, unchanged cached positive inputs; no training or generation. query_update is the execution_project input Q-anchor; final_value_read is the separate project input. execution_residual is executeC-native_anchor, not executeC-routeC.',
                limitations='Representation geometry does not measure actual Wan write magnitude, correctness or video quality.',
                aggregate={k:{metric:sum(row['metrics'][k][metric] for row in rows)/len(rows)
                              for metric in rows[0]['metrics'][k]} for k in values},
                route_execution_cosine=sum(r['route_execution_cosine'] for r in rows)/len(rows),
                cross_sample_means={k:stats(torch.cat(v,0)) for k,v in means.items()},cases=rows)
    (ROOT/'reports/own400_fixed16_representation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['aggregate','route_execution_cosine','cross_sample_means']}))


if __name__=='__main__':main()
