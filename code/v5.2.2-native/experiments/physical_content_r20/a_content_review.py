"""CPU-only audit of A400 physical source versus learned increment on fixed16."""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401
from physical_content.native_residual import NativeAnchoredContent
from physical_content.r18e_reference.evidence import resample_words

ROOT = Path(__file__).resolve().parent


def stats(v):
    v = v.reshape(-1, v.shape[-1]).float()
    u = F.normalize(v, dim=-1)
    n = len(v)
    return dict(cosine=float((u.sum(0).square().sum()-n)/(n*(n-1))),
                rms=float(v.square().mean().sqrt()),
                centered_energy=float((v-v.mean(0)).square().sum()/v.square().sum().clamp_min(1e-20)))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['at', 'ai'], required=True)
    tag = p.parse_args().task
    torch.set_num_threads(4)
    checkpoint = ROOT/'runs'/f'r20{tag}_native_physical400/step0400.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    cfg, constants = saved['config'], saved['constants']
    spec = cfg['model']
    with torch.device('meta'):
        model = NativeAnchoredContent(spec['qwen_width'], 3072, spec['q_heads'],
                                      spec['grounded_heads'], spec['q_tokens'],
                                      spec['content_rounds'], checkpointing=False)
    weights = {k.removeprefix('enhancement.'):v for k,v in saved['model'].items() if k.startswith('enhancement.')}
    weights.update({k.removeprefix('enhancement.'):v for k,v in constants.items() if k.startswith('enhancement.')})
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval().requires_grad_(False)
    normalizer = (constants['condition_mean'], constants['condition_scale'])
    output = ROOT/'inference_outputs'/f'r20{tag}_native_physical400/step0400_seed42'
    cases = json.loads((output/'paired_summary.json').read_text())['cases']
    rows, pooled = [], {k:[] for k in ['native','route','execute','delta','query_delta']}
    for case in cases:
        evidence = torch.load(Path(cfg['paths']['inference_evidence'])/f'videophy40_{case}/positive.pt',
                              map_location='cpu', weights_only=True)
        words = (evidence['words'].float()-normalizer[0])/normalizer[1]
        request = (evidence['request_words'].float()-normalizer[0])/normalizer[1]
        raw = model(words, request, evidence['native_condition'].float(), evidence['word_groups'])
        anchor_q, _, _ = resample_words(words, evidence['word_groups'], spec['q_tokens'])
        values = {k:raw[k] for k in ['native','route','execute']}
        values.update(delta=raw['execute']-raw['native'],query_delta=raw['query_state']-anchor_q)
        rows.append(dict(case=case, **{k:stats(v) for k,v in values.items()},
                         delta_to_native_norm=float(values['delta'].norm()/raw['native'].norm().clamp_min(1e-20))))
        for k,v in values.items():
            pooled[k].append(v.reshape(-1,v.shape[-1]).mean(0))
        print(json.dumps(dict(task=tag, case=case, delta_cosine=rows[-1]['delta']['cosine'])), flush=True)
    aggregate = {k:{s:statistics.mean(x[k][s] for x in rows) for s in ['cosine','rms','centered_energy']} for k in pooled}
    aggregate['delta_to_native_norm'] = statistics.mean(x['delta_to_native_norm'] for x in rows)
    result = dict(checkpoint=str(checkpoint), task=tag, cases=cases, rows=rows, aggregate=aggregate,
                  pooled_across_samples={k:stats(torch.stack(v)) for k,v in pooled.items()},
                  scope='CPU float32 fixed16 positive inputs. Full C/source and learned increment are measured separately; this is not a generation quality test.')
    path = ROOT/'reports'/f'{tag}_fixed16_content_review.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(task=tag, output=str(path), aggregate=aggregate)), flush=True)


if __name__ == '__main__':
    main()
