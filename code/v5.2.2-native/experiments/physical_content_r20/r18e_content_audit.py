"""Read-only CPU comparison of trained R18E content and its source anchors."""
import importlib
import json
from pathlib import Path
import statistics
import sys
import types

import torch
from torch.nn import functional as F

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
SOURCE = BASE/'experiments/physics_content_v10_impl'
sys.path.insert(0, str(SOURCE))
import bootstrap  # noqa: E402,F401
from physical_content.content import Enhancement  # noqa: E402
from physical_content.evidence import resample_words  # noqa: E402


def stats(value):
    value = value.reshape(-1, value.shape[-1]).float()
    normalized = F.normalize(value, dim=-1)
    count = len(value)
    return dict(cosine=float((normalized.sum(0).square().sum()-count)/(count*(count-1))),
                rms=float(value.square().mean().sqrt()),
                centered_energy=float((value-value.mean(0)).square().sum()/value.square().sum().clamp_min(1e-20)))


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    checkpoint = SOURCE/'runs/r18e_nchunks400/step0400.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    cfg = saved['config']
    cfg['train']['gradient_checkpointing'] = False
    interface = {key: saved['constants']['enhancement.anchor_'+key]
                 for key in ['mean', 'scale', 'weight', 'bias', 'target_scale']}
    with torch.device('meta'):
        model = Enhancement(cfg, interface)
    state = {key.removeprefix('enhancement.'): value for key, value in saved['model'].items()
             if key.startswith('enhancement.')}
    state.update({key.removeprefix('enhancement.'): value for key, value in saved['constants'].items()
                  if key.startswith('enhancement.')})
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    pairs = ['VP002','VP004','VP005','VP009','VP010','VP013','VP015','VP016',
             'VP017','VP018','VP024','VP025','VP026','VP027','VP034','VP038']
    rows = []
    for case in pairs:
        evidence = torch.load(Path(cfg['paths']['inference_evidence'])/f'videophy40_{case}/positive.pt',
                              map_location='cpu', weights_only=True)
        content, groups = model(evidence)
        anchor, _ = model(evidence, anchor_only=True)
        delta = content['execution']-anchor['execution']
        rows.append(dict(case=case, anchor=stats(anchor['execution']), route=stats(content['routing']),
                         execution=stats(content['execution']), delta=stats(delta),
                         delta_to_anchor_norm=float(delta.norm()/anchor['execution'].norm().clamp_min(1e-20))))
    aggregate = {name: {key: statistics.mean(row[name][key] for row in rows)
                        for key in ['cosine', 'rms', 'centered_energy']}
                 for name in ['anchor', 'route', 'execution', 'delta']}
    aggregate['delta_to_anchor_norm'] = statistics.mean(row['delta_to_anchor_norm'] for row in rows)
    output = BASE/'experiments/physical_content_r20/r18e_fixed16_content.json'
    output.write_text(json.dumps(dict(checkpoint=str(checkpoint), scope='CPU float32 fixed16 positive inputs; learned interface anchors, not exact native T5 sources',
                                     cases=pairs, rows=rows, aggregate=aggregate), indent=2)+'\n')
    print(json.dumps(dict(output=str(output), aggregate=aggregate)))


if __name__ == '__main__':
    main()
