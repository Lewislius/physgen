"""Read-only C audit on the exact fixed16 inputs; no generator/GPU allocation."""
from pathlib import Path
import argparse
import json
import sys

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.source))
    import bootstrap  # noqa: F401
    from physical_content.native_residual import NativeAnchoredContent

    saved = torch.load(args.checkpoint, mmap=True, map_location='cpu', weights_only=False)
    cfg = saved['config']
    spec = cfg['model']
    constants = saved['constants']
    module = NativeAnchoredContent(spec['qwen_width'], spec['width'], spec['q_heads'],
        spec['grounded_heads'], spec['q_tokens'], spec['content_rounds'],
        source_scale=constants['enhancement.source_scale']).eval()
    state = {n.removeprefix('enhancement.'): p for n, p in saved['model'].items()
             if n.startswith('enhancement.')}
    state['source_scale'] = constants['enhancement.source_scale']
    module.load_state_dict(state, strict=True)
    base = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
    ft = json.loads((base/'experiments/physical_content_r20f/reports/c_correct_vs_wrong16.json').read_text())
    pairs = [(r['case'], r['donor']) for r in ft['cases']]
    inputs = {}
    for case in {x for pair in pairs for x in pair}:
        inputs[case] = torch.load(Path(cfg['paths']['inference_evidence'])/
            f'videophy40_{case}'/'positive.pt', map_location='cpu', weights_only=True)
    mean, scale = constants['condition_mean'], constants['condition_scale']
    rows, means = [], {n: [] for n in ('native','route','execute')}
    with torch.inference_mode():
        for case, donor in pairs:
            own, other = inputs[case], inputs[donor]
            def run(source):
                return module((source['words'].float()-mean)/scale,
                    (own['request_words'].float()-mean)/scale,
                    source['native_condition'].float(), source['word_groups'])
            normal, wrong = run(own), run(other)
            row = dict(case=case, donor=donor)
            for name in means:
                x, y = normal[name].float()[0], wrong[name].float()[0]
                means[name].append(x.mean(0))
                row[name+'_relative_change'] = float((x-y).norm()/x.norm().clamp_min(1e-12))
                row[name+'_mean_cosine'] = float(F.cosine_similarity(x.mean(0),y.mean(0),dim=0))
                z = F.normalize(x,dim=-1); n = len(z)
                row[name+'_token_cosine'] = float((z.sum(0).square().sum()-n)/(n*(n-1)))
            rows.append(row)
    summary = {}
    for name, values in means.items():
        z = F.normalize(torch.stack(values),dim=-1); n = len(z)
        summary[name+'_cross_sample_mean_cosine'] = float((z.sum(0).square().sum()-n)/(n*(n-1)))
        for metric in ('relative_change','mean_cosine','token_cosine'):
            summary[name+'_wrong_'+metric+'_mean' if metric != 'token_cosine' else name+'_token_cosine_mean'] = sum(r[name+'_'+metric] for r in rows)/len(rows)
    result = dict(checkpoint_step=saved['step'], revision=saved['revision'],
        source=str(args.source), checkpoint=str(args.checkpoint), mean_statistics=summary,
        cases=rows, interpretation='Representation diagnostics only; not PC/SA or proof of physical benefit.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(checkpoint_step=saved['step'], mean_statistics=summary)))


if __name__ == '__main__':
    main()
