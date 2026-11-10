"""Matched-noise interventions on the trained condition, with late-time scores.

Turning C off, replacing its content, collapsing its slots, and removing explicit
analysis answer different necessity questions. None alone identifies causality.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from physgen_native.data import VideoDataset, load_tensor, move
from physgen_native.generator import load_wan
from physgen_native.runtime import Logger, load_checkpoint, restore
from physgen_native.sequence import sequence_error, sequence_metrics, perceptual_process_error, weighted_mean
from physgen_native.system import PhysGenNative
from train.train import export_video


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cases', type=int, default=4)
    parser.add_argument('--case-ids', nargs='+', help='Predeclared existing validation cases in this order')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 314159])
    parser.add_argument('--full-samples', type=int, default=0)
    parser.add_argument('--only-sampling', action='store_true')
    parser.add_argument('--sampling-seeds',type=int,nargs='+',help='Optional smaller seed set for complete video export')
    parser.add_argument('--sampling-variants',nargs='+',
                        help='Predeclared interventions to export; default is correct/off for operator revisions')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    state = load_checkpoint(args.checkpoint); cfg = state['config']
    model = PhysGenNative(cfg, load_wan(cfg), device='cuda', training_observers=False).cuda()
    completed, _ = restore(model, None, None, state, cfg); del state
    model.eval().requires_grad_(False)
    if args.case_ids:
        cfg['data']['validation_ids'] = args.case_ids
        args.cases = len(args.case_ids)
    dataset = VideoDataset(cfg, 'validation')
    if not 2 <= args.cases <= len(dataset): raise ValueError('Condition comparisons require at least two available cases')
    out = Path(args.output); log = Logger(out/'metrics.jsonl')
    null = load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
    tokens, no_analysis, alternative_tokens, no_latent = [], [], [], []
    for index in range(args.cases):
        sample = move(dataset[index], 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            c = model.encode_condition(model.conditions(sample))
            if not model.operator_training:
                budget = cfg['knowledge']['reasoning_tokens']; cfg['knowledge']['reasoning_tokens'] = 0
                no_analysis.append(model.encode_condition(model.conditions(sample)))
                cfg['knowledge']['reasoning_tokens'] = budget
            if model.operator_training:
                sample['alternative'] = dataset.alternative_condition(sample,index)
                alternative_tokens.append(model.encode_condition(model._alternative_condition(sample)))
                rounds = cfg['knowledge'].get('latent_reasoning_steps',0)
                cfg['knowledge']['latent_reasoning_steps'] = 0
                no_latent.append(model.encode_condition(model.conditions(sample)))
                cfg['knowledge']['latent_reasoning_steps'] = rounds
                log.write('condition_text_control',completed,sample=sample['id'],
                          alternate_caption_source=sample['alternative']['id'],same_initial_image=True,
                          same_native_text=True,uses_reference_future=False)
        normalized = torch.nn.functional.normalize(c.float(), dim=-1)
        similarities = normalized[0]@normalized[0].T
        off_diagonal = similarities[~torch.eye(len(similarities), device='cuda', dtype=torch.bool)]
        log.write('condition_geometry', completed, sample=sample['id'], tokens=c.shape[1],
                  off_diagonal_cosine=off_diagonal.mean(),
                  relative_slot_variation=(c-c.mean(1, keepdim=True)).square().mean().sqrt()/c.square().mean().sqrt())
        tokens.append(c)
    for index in range(args.cases):
        sample = move(dataset[index], 'cuda'); c = tokens[index]
        if model.operator_training:
            choices = dict(correct=(c,1.),off=(c,0.),alternate_text=(alternative_tokens[index],1.),
                           no_latent_reasoning=(no_latent[index],1.))
        else:
            choices = dict(correct=(c, 1.), off=(c, 0.), other_case=(tokens[(index+1)%args.cases], 1.),
                           mean_slots=(c.mean(1, keepdim=True).expand_as(c), 1.), no_analysis=(no_analysis[index], 1.))
        sampling_variants = args.sampling_variants or (
            ['correct', 'off'] if model.operator_training else ['correct', 'off', 'other_case'])
        if any(variant not in choices for variant in sampling_variants):
            raise ValueError('Sampling intervention is unavailable for this checkpoint revision')
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            truth = model.decoder(sample['latent'], resize=cfg['knowledge']['video_size']) if not args.only_sampling else None
            target_features = model.knowledge.process_features(truth) if truth is not None and not model.operator_training else None
            for seed in ([] if args.only_sampling else args.seeds):
                noise = torch.randn(sample['latent'].shape, device='cuda',
                                    generator=torch.Generator(device='cuda').manual_seed(seed+index))
                for level in (.65, .95):
                    sigma = torch.tensor(level, device='cuda')
                    noisy = (1-sigma)*sample['latent']+sigma*noise
                    noisy[:, :, :1] = sample['first']
                    for variant, (condition, strength) in choices.items():
                        velocity = model.generator(noisy, sample['first'], sample['text'], sigma,
                                                   condition, sample['times'], strength)
                        clean = torch.cat((sample['first'], (noisy-sigma*velocity)[:, :, 1:]), 2)
                        frames = model.decoder(clean, resize=cfg['knowledge']['video_size'])
                        rgb = sequence_error(frames, truth, sample['valid'], cfg['sequence'])[0]
                        visual = (perceptual_process_error(model.knowledge.process_features(frames), target_features,
                                                          sample['valid'], cfg['sequence'])[0] if target_features is not None else None)
                        metrics = sequence_metrics(frames, truth, sample['valid'])
                        if visual is not None: metrics['visual'] = visual
                        log.write('condition_intervention', completed, sample=sample['id'], seed=seed+index,
                                  sigma=level, variant=variant, process=rgb if visual is None else .5*(rgb+visual), rgb=rgb,
                                  fm=weighted_mean((velocity[:,:,1:]-(noise-sample['latent'])[:,:,1:]).square().mean(1),sample['valid']), **metrics)
            if index < args.full_samples:
                for seed in args.sampling_seeds or args.seeds:
                    for variant in sampling_variants:
                        condition, strength = choices[variant]
                        path = out/'videos'/f'{sample["id"]}_seed{seed+index}_{variant}.mp4'
                        rounds = cfg['knowledge'].get('latent_reasoning_steps', 0)
                        if variant == 'no_latent_reasoning': cfg['knowledge']['latent_reasoning_steps'] = 0
                        try:
                            # Remove reasoning from BOTH CFG conditions for a
                            # full-module ablation. The positive C above was
                            # encoded with the same intervention already.
                            latent = model.sample(model.conditions(sample), sample['text'], null, seed=seed+index,
                                                  token_override=condition, strength=strength)
                            export_video(model, latent, sample, path, completed, seed+index)
                        finally:
                            if variant == 'no_latent_reasoning': cfg['knowledge']['latent_reasoning_steps'] = rounds
                        metadata = json.loads(path.with_suffix('.json').read_text())
                        metadata['sampler']['condition_strength'] = strength
                        metadata['condition_intervention'] = variant
                        if variant == 'no_latent_reasoning': metadata['latent_reasoning_steps'] = 0
                        if variant == 'other_case': metadata['condition_source'] = dataset.rows[(index+1)%args.cases]['id']
                        path.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
                        log.write('full_sample',completed,sample=sample['id'],seed=seed+index,variant=variant,file=str(path))
        del sample, truth, target_features
    log.write('complete', completed, cases=args.cases, seeds=args.seeds,
              sampling_variants=sampling_variants,
              scope='same-checkpoint condition interventions; paired reconstruction is not free-video physics accuracy',
              peak_gib=torch.cuda.max_memory_allocated()/2**30)
    log.close()


if __name__ == '__main__': main()
