"""Test query-residual scale in the frozen, current v30 checkpoint.

The only intervention replaces query_seed for one functional forward call.
No source model, checkpoint, optimizer, cached training data or running job is
changed. One-step reconstruction and token spread are not video-quality scores.
"""
import gc
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import torch
from torch.func import functional_call
from probe import IDS, RUNS, representation, rms, write
from physgen_native.data import VideoDataset, move
from physgen_native.generator import load_wan
from physgen_native.runtime import restore, seed_all
from physgen_native.sequence import weighted_mean
from physgen_native.system import PhysGenNative


def measure_message(conditioner, evidence, seed, *, trace):
    """Observe actual branch outputs, retaining native residual arithmetic."""
    branches = []
    handles = []
    cursor = {}

    def start(_module, inputs):
        cursor['state'] = inputs[0].detach()
        branches.append(dict(input=representation(cursor['state'])))

    def record(name):
        def hook(_module, _inputs, output):
            previous = cursor['state']
            cursor['state'] = previous + output.detach()
            branches[-1][name] = dict(update=representation(output),
                update_over_state_rms=rms(output)/max(rms(previous), 1e-12),
                after=representation(cursor['state']))
        return hook

    if trace:
        handles.append(conditioner.block.register_forward_pre_hook(start))
        for name, module in [('cross', conditioner.block.cross),
                             ('self_attention', conditioner.block.self_attention),
                             ('ffn', conditioner.block.ffn)]:
            handles.append(module.register_forward_hook(record(name)))
    try:
        rounds = functional_call(conditioner, {'query_seed': seed}, (evidence,),
                                 dict(all_rounds=True, reasoning_round=2))
    finally:
        for handle in handles:
            handle.remove()
    stages = []
    for index, value in enumerate(rounds['readouts']):
        projected = conditioner.project(value)
        stages.append(dict(round=index, state=representation(value),
            projected=representation(projected), C=representation(rounds['conditions'][index])))
    return rounds['conditions'][2].detach().cpu(), dict(branches=branches, stages=stages)


@torch.no_grad()
def main():
    seed_all(42)
    torch.cuda.set_device(0)
    out = HERE/'Q'/'query_scale'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        print('Completed query-scale diagnostic already exists; preserving it.', flush=True)
        return
    checkpoint = ROOT/'checkpoints'/RUNS['Q']/'latest.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = saved['config']
    assert saved['step'] == 224 and cfg['train']['seed'] == 42
    assert cfg['verifier']['kind'] == 'none'
    assert cfg['knowledge']['inference_reasoning_round'] == 2
    model = PhysGenNative(cfg, load_wan(cfg), device='cuda', training_observers=False)
    model.conditioner.cuda()
    restore(model, None, None, saved, cfg)
    model.eval().requires_grad_(False)
    del saved
    original_seed = model.conditioner.query_seed.detach().clone()
    unit_seed = original_seed.float()/original_seed.float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
    seeds = {'original': original_seed, 'rms0p2': unit_seed*.2, 'rms1p0': unit_seed}
    write(out/'protocol.json', dict(checkpoint=str(checkpoint), training_seed=42,
        noise=str(HERE/'Q'/'noise_seed42.pt'), native_text='present', sigma=.95,
        cases=IDS, interventions=list(seeds), rounds='R0 grounding + R1 + R2',
        checkpoint_updated=False, original_source_modified=False,
        only_intervention='Per-query residual RMS before R0; all message/projector/Wan weights fixed',
        purpose='Test whether small initial query residuals cause loss of slot identity',
        limitations='This is an out-of-training-distribution intervention, not a trained candidate or quality claim',
        gpu=torch.cuda.get_device_name(0)))
    dataset = VideoDataset(cfg, 'validation')
    indices = {row['id']: i for i, row in enumerate(dataset.rows)}
    samples = [dataset[indices[cid]] for cid in IDS]
    encoded = {}
    for index, raw in enumerate(samples):
        sample = move(raw, 'cuda')
        correct = model.conditions(sample)
        other = samples[(index+1) % len(samples)]
        wrong = model._alternative_condition(dict(sample, alternative={
            'id': other['id'], 'caption': other['caption'], 'clauses': other['clauses']}))
        encoded[raw['id']] = {}
        details = dict(id=raw['id'], caption=correct['caption'], wrong_caption=wrong['caption'])
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            first_rgb = model.decoder(sample['first'], [0])
            for text_name, condition in [('correct', correct), ('wrong', wrong)]:
                evidence = model.knowledge.encode(first_rgb, condition['caption'], condition['clauses'],
                    duration=float(condition['times'][-1]-condition['times'][0]))
                details[text_name] = dict(reasoning=evidence.get('reasoning', ''), variants={})
                for name, seed in seeds.items():
                    tokens, stats = measure_message(model.conditioner, evidence, seed, trace=text_name=='correct')
                    encoded[raw['id']][f'{name}/{text_name}'] = tokens
                    details[text_name]['variants'][name] = stats
                del evidence
        assert torch.equal(model.conditioner.query_seed, original_seed), 'Functional intervention leaked into model'
        torch.save(encoded[raw['id']], out/f'{raw["id"]}_C.pt')
        write(out/f'{raw["id"]}_representations.json', details)
        print(json.dumps(dict(kind='query_scale_encoded', id=raw['id'])), flush=True)
        del sample, first_rgb, tokens
    model.knowledge.cpu()
    model.conditioner.cpu()
    model.decoder.cpu()
    del original_seed, unit_seed, seeds
    gc.collect()
    torch.cuda.empty_cache()
    model.generator.cuda()
    noise = torch.load(HERE/'Q'/'noise_seed42.pt', map_location='cuda', weights_only=True)
    rows = []
    with (out/'metrics.jsonl').open('w', buffering=1) as log:
        for index, raw in enumerate(samples):
            sample = move(raw, 'cuda')
            z = sample['latent']
            sigma = z.new_tensor(.95)
            noisy = torch.cat((sample['first'], ((1-sigma)*z+sigma*noise)[:, :, 1:]), 2)
            target = (noise-z)[:, :, 1:]
            other_id = samples[(index+1) % len(samples)]['id']
            velocity = {}
            variants = [('C_off', encoded[raw['id']]['original/correct'], 0.)]
            for name in ('original', 'rms0p2', 'rms1p0'):
                c = encoded[raw['id']][f'{name}/correct']
                variants += [(f'{name}/correct', c, 1.),
                    (f'{name}/wrong_caption', encoded[raw['id']][f'{name}/wrong'], 1.),
                    (f'{name}/mean_slots', c.mean(1, keepdim=True).expand_as(c), 1.),
                    (f'{name}/other_case_C', encoded[other_id][f'{name}/correct'], 1.)]
            for name, c, strength in variants:
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    v = model.generator(noisy, sample['first'], sample['text'], sigma,
                                        c.cuda(), sample['times'], strength)[:, :, 1:].float()
                if not bool(torch.isfinite(v).all()):
                    raise FloatingPointError(f'Nonfinite velocity: {raw["id"]} {name}')
                velocity[name] = v
            old = velocity['original/correct']
            for name, v in velocity.items():
                family = name.split('/')[0] if name != 'C_off' else 'original'
                correct_v = velocity[f'{family}/correct']
                effect = rms(correct_v-velocity['C_off'])
                row = dict(id=raw['id'], variant=name, sigma=.95,
                    fm=float(weighted_mean((v-target).square().mean(1), sample['valid'])),
                    delta_vs_original=rms(v-old), family_C_effect_rms=effect,
                    delta_vs_family_correct=rms(v-correct_v),
                    difference_over_family_C_effect=rms(v-correct_v)/max(effect, 1e-12))
                rows.append(row)
                log.write(json.dumps(row)+'\n')
            print(json.dumps(dict(kind='query_scale_measured', id=raw['id'])), flush=True)
            del sample, z, noisy, target, velocity, old, correct_v, v
    write(out/'complete.json', dict(status='complete', cases=len(samples), records=len(rows),
        scope='Frozen-checkpoint causal diagnostic only; no generation-quality claim'))


if __name__ == '__main__':
    main()
