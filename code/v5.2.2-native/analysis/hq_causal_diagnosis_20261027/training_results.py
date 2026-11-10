"""Re-evaluate actual seen training examples, with matched module interventions.

This is a final-weight diagnostic, not saved historical training outputs.
No optimizer is constructed and no source checkpoint/cache is modified.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import torch
from physgen_native.data import VideoDataset, load_tensor, move
from physgen_native.generator import load_wan
from physgen_native.runtime import restore, seed_all
from physgen_native.sequence import weighted_mean, temporal_flow_error
from physgen_native.system import PhysGenNative
from train.train import export_video

RUNS = {'H': 'v522-h-content-message-verifier-frozen-simple_66356',
        'Q': 'v522-q-seed128-message-verifier-frozen-simple_66353'}
TRAIN = ['wisa0003617', 'wisa0005325', 'source0001608', 'wisa0005723',
         'source0000527', 'source0002333', 'source0002013', 'wisa0030684']
VAL = ['source0000042', 'source0000086', 'source0000104', 'source0000107']


def write_video(model, sample, latent, path, **metadata):
    export_video(model, latent, sample, path, 224, 42)
    info = json.loads(path.with_suffix('.json').read_text())
    info.update(metadata, evaluation_scope='Final weights on known training samples; no optimizer update')
    path.with_suffix('.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', choices=RUNS, required=True)
    args = parser.parse_args(); seed_all(42); torch.cuda.set_device(0)
    out = HERE/'training_results'/args.run; out.mkdir(parents=True, exist_ok=True)
    source = ROOT/'checkpoints'/RUNS[args.run]/'latest.pt'
    checkpoint = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    cfg = checkpoint['config']; assert checkpoint['step'] == 224
    model = PhysGenNative(cfg, load_wan(cfg), device='cuda', training_observers=False)
    model.conditioner.cuda(); restore(model, None, None, checkpoint, cfg)
    model.eval().requires_grad_(False)
    current_lora = {n: v.clone() for n,v in checkpoint['model'].items() if n.startswith('generator.wan.')}
    initial = torch.load(ROOT/'checkpoints/initialization/generator.pt', map_location='cpu', weights_only=False, mmap=True)
    inherited_lora = {n: v.clone() for n,v in initial['model'].items() if n.startswith('generator.wan.')}
    del checkpoint, initial
    cfg['inference']['latent_feedback']['enabled'] = False
    actual = json.loads((HERE/'training_content_audit.json').read_text())['actual_training_ids']
    assert all(cid in actual for cid in TRAIN)
    datasets = {'train': VideoDataset(cfg), 'validation': VideoDataset(cfg, 'validation')}
    samples = []
    for split, ids in [('train', TRAIN), ('validation', VAL)]:
        data = datasets[split]; indices = {r['id']:i for i,r in enumerate(data.rows)}
        for cid in ids:
            sample = data[indices[cid]]; sample['split'] = split
            sample['original_microbatch'] = actual.index(cid)+1 if split == 'train' else None
            samples.append(sample)
    null = load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
    encoded = {}
    records = []
    # Qwen and Wan take turns on the GPU, matching established demo inference.
    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        for raw in samples:
            sample = move(raw, 'cuda'); condition = model.conditions(sample)
            first = model.decoder(sample['first'], [0]); positive = model.encode_condition_rounds(condition, first)
            negative = model.encode_condition_rounds(dict(condition, caption='', clauses=[]), first)
            chosen = cfg['knowledge']['inference_reasoning_round']
            encoded[sample['id']] = dict(full=(positive['conditions'][chosen], negative['conditions'][chosen]),
                round0=(positive['conditions'][0], negative['conditions'][0]), analysis=positive['reasoning'])
            records.append(dict(id=sample['id'], split=sample['split'], original_microbatch=sample['original_microbatch'],
                original_step=(sample['original_microbatch']+3)//4 if sample['original_microbatch'] else None,
                caption=sample['caption'], analysis=positive['reasoning'], tokens=positive['conditions'][chosen].shape[1],
                full_vs_round0_relative_rms=float((positive['conditions'][chosen]-positive['conditions'][0]).square().mean().sqrt()
                    /positive['conditions'][chosen].square().mean().sqrt().clamp_min(1e-12))))
            print(json.dumps(dict(encoded=args.run, sample=sample['id'])), flush=True)
            del sample, positive, negative, first
    (out/'inputs.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    model.knowledge.cpu(); model.conditioner.cpu(); gc.collect(); torch.cuda.empty_cache(); model.generator.cuda()
    named = dict(model.named_parameters())

    def lora(which):
        values = inherited_lora if which == 'inherited' else current_lora
        for name, value in values.items():
            if which == 'zero' and name.endswith('lora_b'): named[name].zero_()
            else: named[name].copy_(value)

    variants = ['full', 'round0', 'C_off', 'C_swapped', 'inherited_lora_C_off', 'base_Wan']
    done = set()
    metrics_path = out/'paired_metrics.jsonl'
    if metrics_path.exists():
        done = {(r['id'], r['variant'], r['sigma']) for r in map(json.loads, metrics_path.read_text().splitlines())}
    with metrics_path.open('a', buffering=1) as stream, torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        for variant in variants:
            lora('inherited' if variant == 'inherited_lora_C_off' else 'zero' if variant == 'base_Wan' else 'current')
            for index, raw in enumerate(samples):
                sample = move(raw, 'cuda'); z = sample['latent']; cid = sample['id']
                p, n = encoded[cid]['round0' if variant == 'round0' else 'full']
                if variant == 'C_swapped': p = encoded[samples[(index+1)%len(samples)]['id']]['full'][0]
                strength = 0. if variant in ('C_off','inherited_lora_C_off','base_Wan') else 1.
                noise = torch.randn(z.shape, device=z.device, generator=torch.Generator(device=z.device).manual_seed(4242+index))
                if variant == 'full' and cid in TRAIN[:4]:
                    path = out/f'{cid}_target.mp4'
                    if not path.exists(): write_video(model,sample,z,path,intervention='cached_training_target',contains_reference_future=True)
                for level in (.25,.65,.95):
                    if (cid,variant,level) in done: continue
                    sigma=z.new_tensor(level); noisy=torch.cat((sample['first'],((1-sigma)*z+sigma*noise)[:,:,1:]),2)
                    model.generator.interface.capture = variant in ('full','round0')
                    model.generator.interface.observations=[]
                    velocity=model.generator(noisy,sample['first'],sample['text'],sigma,p,sample['times'],strength)
                    fm=weighted_mean((velocity[:,:,1:]-(noise-z)[:,:,1:]).square().mean(1),sample['valid'])
                    temporal=temporal_flow_error(velocity,noise-z,sample['valid'],cfg['flow_temporal']['lags'])
                    record=dict(run=args.run,id=cid,split=sample['split'],variant=variant,sigma=level,
                        fm=float(fm),temporal_fm=float(temporal),objective=float(fm+.1*temporal),
                        attention=model.generator.interface.observations,
                        noise_seed=4242+index,checkpoint_step=224)
                    stream.write(json.dumps(record)+'\n')
                    if cid in TRAIN[:2] and ((variant=='full') or (variant=='C_off' and level==.65)):
                        clean=torch.cat((sample['first'],(noisy-sigma*velocity)[:,:,1:]),2)
                        write_video(model,sample,clean,out/f'{cid}_paired_s{level}_{variant}.mp4',
                            intervention=variant,contains_reference_future=True,paired_sigma=level,noise_seed=4242+index)
                    del velocity,noisy
                print(json.dumps(dict(paired=args.run,sample=cid,variant=variant)),flush=True)
                del sample
    model.generator.interface.capture=False
    (out/'paired_complete.json').write_text(json.dumps(dict(status='complete',cases=len(samples),variants=variants)))
    lora('current')
    with (out/'generation.jsonl').open('a', buffering=1) as stream, torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        plans=[(cid,variant,16) for cid in TRAIN[:4] for variant in ('full','C_off','round0')]
        plans += [(cid,'full',50) for cid in TRAIN[:2]]
        if args.run=='H': plans += [(cid,'base_Wan',16) for cid in TRAIN[:2]]
        for cid,variant,steps in plans:
            path=out/f'{cid}_pure{steps}_{variant}.mp4'
            if path.exists() and path.with_suffix('.json').exists(): continue
            started=time.monotonic();sample=move(next(s for s in samples if s['id']==cid),'cuda')
            lora('zero' if variant=='base_Wan' else 'current')
            positive,negative=encoded[cid]['round0' if variant=='round0' else 'full']
            strength=0. if variant in ('C_off','base_Wan') else 1.
            latent=model.sample(model.conditions(sample),sample['text'],null,seed=42,steps=steps,strength=strength,
                token_override=positive,negative_token_override=negative)
            model.last_sample_analysis=encoded[cid]['analysis']
            write_video(model,sample,latent,path,intervention=variant,contains_reference_future=False,sampling_steps=steps,
                        sampler=dict(cfg['inference'],steps=steps,condition_strength=strength))
            record=dict(run=args.run,id=cid,variant=variant,steps=steps,seconds=time.monotonic()-started,video=str(path))
            stream.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
            del sample,latent
    (out/'complete.json').write_text(json.dumps(dict(status='complete',run=args.run)))


if __name__=='__main__': main()
