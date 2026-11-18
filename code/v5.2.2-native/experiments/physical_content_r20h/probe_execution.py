"""One read-only GT400 execution audit. No gradients, generation or training."""
from pathlib import Path
import argparse
import json
import os
import sys

import torch


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import bootstrap  # noqa: F401
    from physical_content.common import gpu_guard
    from physical_content.model import PhysicalModel
    from physical_content.data import TrainingData
    from physical_content.teacher import Teacher
    from physical_content.interventions import physics_evidence
    from physical_content.supervision import flow_matching
    import physical_content.architecture as architecture

    gpu_guard(args.output.parent/'jobs'/os.getenv('DET_EXPERIMENT_ID','local'))
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, mmap=True, map_location='cpu', weights_only=False)
    assert saved['step'] == 400, 'Do not reserve a GPU while waiting for training'
    cfg = saved['config']
    model = PhysicalModel(cfg).eval().requires_grad_(False)
    assert list(model.injection_layers) == saved['physical_source_layers']
    named = dict(model.named_parameters())
    assert set(saved['model']) == set(model.learned_names)
    for name, value in saved['model'].items():
        named[name].copy_(value)
    buffers = dict(model.named_buffers())
    for name, value in saved['constants'].items():
        buffers[name].copy_(value)
    del saved
    data, teacher = TrainingData(cfg), Teacher(cfg)
    sequence = json.loads(Path(cfg['train']['sample_sequence']).read_text())
    positive = [i for i, drop in enumerate(sequence['drop']) if not drop]
    indices = [positive[i] for i in (0, 100, 350, 700)]
    samples = [data.get(i) for i in indices]
    evidence = [teacher.input(sample) for sample in samples]
    active = False
    records, current, last_features, native_rms = [], {}, {}, {}
    hooks = []
    sites = {id(model.wan.blocks[site]): site for site in model.injection_layers}
    original_native = architecture.native_block_with_read_state

    def native_observer(block, *a, **kw):
        result = original_native(block, *a, **kw)
        if active:
            native_rms[sites[id(block)]] = result[0].float().square().mean().sqrt()
        return result

    architecture.native_block_with_read_state = native_observer
    for site, block in zip(model.injection_layers, model.physical_blocks):
        def read_hook(module, a, kw, read, site=site):
            if not active:
                return
            r = read.float()
            current[str(site)] = dict(read_rms=float(r.square().mean().sqrt()),
                read_token_variation_fraction=float((r-r.mean(1,keepdim=True)).square().sum()
                    /r.square().sum().clamp_min(1e-20)))
        def feature_hook(module, a, features, site=site):
            if active:
                last_features[site] = features
        def gain_hook(module, a, gain, site=site):
            if not active:
                return
            f = last_features.pop(site).float()
            d = gain.float()
            current[str(site)].update(
                gain_rms=float(d.square().mean().sqrt()),
                feature_change_relative_rms=float((f*d).square().sum().sqrt()/f.square().sum().sqrt().clamp_min(1e-20)),
                gain_token_variation_fraction=float((d-d.mean(1,keepdim=True)).square().sum()/d.square().sum().clamp_min(1e-20)),
                flipped_feature_fraction=float((d < -1).float().mean()),
                gain_min=float(d.min()), gain_max=float(d.max()))
        def write_hook(module, a, kw, hint, site=site):
            if active:
                current[str(site)]['hint_over_native_rms'] = float(hint.float().square().mean().sqrt()/native_rms[site].clamp_min(1e-20))
        hooks.extend([block.conditioning.register_forward_hook(read_hook,with_kwargs=True),
            block.native.ffn[1].register_forward_hook(feature_hook),
            block.feature_update.register_forward_hook(gain_hook),
            block.register_forward_hook(write_hook,with_kwargs=True)])
    maximum_observer_difference = 0.
    for i, sample in enumerate(samples):
        other = physics_evidence(evidence[i], evidence[(i+1)%len(samples)])
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            content, groups = model.prepare_content(evidence[i],sample['text'])
            swapped, wrong_groups = model.prepare_content(other,sample['text'])
            noise = torch.randn(sample['latent'].shape,device='cuda',
                generator=torch.Generator(device='cuda').manual_seed(8317+indices[i]))
            for level in (.25,.75):
                sigma = torch.tensor(level,device='cuda')
                noisy = (1-sigma)*sample['latent']+sigma*noise
                predictions, losses = {}, {}
                layer_stats = None
                for mode in ('normal','off','feature_off','wrong'):
                    active = mode == 'normal'
                    current, last_features, native_rms = {}, {}, {}
                    c,g = (swapped,wrong_groups) if mode == 'wrong' else (content,groups)
                    prediction,_ = model.velocity(noisy,sample.get('first'),sample['text'],sigma,
                        c,g,sample['times'],physical_mode='normal' if mode=='wrong' else mode)
                    predictions[mode]=prediction
                    losses[mode]=float(flow_matching(prediction,noise,sample['latent'],sample['valid'],model.task))
                    if active:
                        layer_stats=current
                    active=False
                if i==0 and level==.25:
                    plain,_=model.velocity(noisy,sample.get('first'),sample['text'],sigma,content,groups,sample['times'])
                    maximum_observer_difference=float((plain-predictions['normal']).abs().max())
                    assert maximum_observer_difference==0, 'Observer changed the prediction'
                n=predictions['normal']
                row=dict(sample=sample['id'],sequence_index=indices[i],sigma=level,
                    fm=losses,layers=layer_stats,
                    prediction_relative_changes={mode:float((n-p).norm()/n.norm().clamp_min(1e-20))
                        for mode,p in predictions.items() if mode!='normal'})
                records.append(row)
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(dict(status='running',records=records),indent=2)+'\n')
                print(json.dumps(dict(sample=sample['id'],sigma=level,fm=losses)),flush=True)
    for hook in hooks:
        hook.remove()
    architecture.native_block_with_read_state=original_native
    result=dict(status='complete',checkpoint=str(args.checkpoint),step=400,source=str(args.source),
        observer_output_max_difference=maximum_observer_difference,records=records,
        interpretation='Paired denoising and actual-state execution diagnostics on four training examples, not generalization/PC or video-quality scores.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
