"""Paired full-sampling C interventions. Never overwrite trained checkpoints."""
import argparse
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
from physgen_native.system import PhysGenNative
from train.train import export_video
from inference.infer_message_demo21 import load_inputs

RUNS = {'H': 'v522-h-content-message-verifier-frozen-simple_66356',
        'Q': 'v522-q-seed128-message-verifier-frozen-simple_66353'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', choices=RUNS, required=True); args = parser.parse_args()
    torch.cuda.set_device(0); seed_all(42)
    source = ROOT / 'checkpoints' / RUNS[args.run] / 'latest.pt'
    state = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    cfg = state['config']; assert state['step'] == 224
    out = HERE / 'controls' / args.run; out.mkdir(parents=True, exist_ok=True)
    model = PhysGenNative(cfg, load_wan(cfg), device='cuda', training_observers=False).cuda()
    restore(model, None, None, state, cfg)
    model.eval().requires_grad_(False)
    del state
    cfg['inference']['latent_feedback']['enabled'] = False
    data = VideoDataset(cfg, 'validation', targets=False)
    rows = {r['id']: i for i,r in enumerate(data.rows)}
    null = load_tensor(Path(cfg['paths']['cache']) / 'base/null_text.pt').float()[None].cuda()
    inputs = []
    for cid in ['source0000042', 'source0000086']:
        sample = move(data[rows[cid]], 'cuda'); sample['null_text'] = null
        sample['seeds'] = [42, 314159] if cid.endswith('42') else [43,314160]
        inputs.append(sample)
    inventory = json.loads((HERE / 'inventory.json').read_text())
    for cid in ['P01','P02','P03']:
        row = next(r for r in inventory if r['run'] == args.run and r['case_id'] == cid)
        prepared = Path(row['prepared_inputs'])
        sample = move(load_inputs(prepared, cfg, 24), 'cuda')
        requirements = json.loads((prepared / 'requirements.json').read_text())['requirements']
        sample.update(id=cid, caption=row['caption'], clauses=[dict(text=t,basis='both',phase='evolution') for t in requirements], seeds=[42])
        inputs.append(sample)
    encoded = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        for sample in inputs:
            condition = model.conditions(sample)
            first = model.decoder(sample['first'], [0])
            positive = model.encode_condition(condition, first)
            analysis = model.last_condition_analysis
            negative = model.encode_condition(dict(condition, caption='', clauses=[]), first)
            encoded[sample['id']] = (positive, negative, analysis)
    with (out / 'records.jsonl').open('a', buffering=1) as log:
        for sample in inputs:
            positive, negative, analysis = encoded[sample['id']]
            variants = ['correct', 'C_off']
            if sample['id'].startswith('source'):
                variants.append('C_swapped')
            for variant in variants:
                p, n = positive, negative
                strength = 0. if variant == 'C_off' else 1.
                if variant == 'C_swapped':
                    other = 'source0000086' if sample['id'] == 'source0000042' else 'source0000042'
                    p, n, _ = encoded[other]
                for seed in sample['seeds']:
                    path = out / f'{sample["id"]}_seed{seed}_{variant}.mp4'
                    if path.exists() and path.with_suffix('.json').exists(): continue
                    start = time.monotonic()
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        latent = model.sample(model.conditions(sample), sample['text'], sample['null_text'], seed=seed,
                            token_override=p, negative_token_override=n, strength=strength)
                        model.last_sample_analysis = analysis
                        export_video(model, latent, sample, path, 224, seed)
                    meta = json.loads(path.with_suffix('.json').read_text())
                    meta.update(intervention=variant, source_checkpoint=str(source), controlled_diagnostic=True)
                    meta['sampler']['condition_strength'] = strength
                    path.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
                    record = dict(run=args.run, case_id=sample['id'], seed=seed, variant=variant, video=str(path), seconds=time.monotonic()-start)
                    log.write(json.dumps(record)+'\n'); print(json.dumps(record), flush=True)
                    del latent
        if args.run == 'H':
            # Explicit base-model diagnostic: bypass C and zero only LoRA B.
            # This remains the same I2V sampler/CFG/geometry and does not alter source files.
            with torch.no_grad():
                for name, parameter in model.generator.wan.named_parameters():
                    if name.endswith('lora_b'): parameter.zero_()
            for sample in inputs:
                p,n,_ = encoded[sample['id']]
                for seed in sample['seeds']:
                    path = out / f'{sample["id"]}_seed{seed}_base_Wan.mp4'
                    if path.exists() and path.with_suffix('.json').exists(): continue
                    start = time.monotonic()
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        latent = model.sample(model.conditions(sample), sample['text'], sample['null_text'], seed=seed,
                            token_override=p, negative_token_override=n, strength=0.)
                        export_video(model, latent, sample, path, 224, seed)
                    meta = json.loads(path.with_suffix('.json').read_text())
                    meta.update(intervention='base_Wan', C_enabled=False, wan_lora_enabled=False, source_checkpoint=str(source), controlled_diagnostic=True)
                    meta['sampler']['condition_strength'] = 0.
                    path.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
                    record = dict(run='base',case_id=sample['id'],seed=seed,variant='base_Wan',video=str(path),seconds=time.monotonic()-start)
                    log.write(json.dumps(record)+'\n'); print(json.dumps(record),flush=True)
                    del latent
    (out / 'complete.json').write_text(json.dumps(dict(status='complete',run=args.run)))
    sys.path.insert(0, str(HERE))
    from probe_gradient_directions import run_probes
    run_probes(model, source, HERE / 'gradient_probes' / args.run)


if __name__ == '__main__': main()
