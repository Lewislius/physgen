"""Paired full generation with current H/Q C enabled or disabled.

Preserve checkpoint LoRA, both CFG text arms, native image, saved seed42 noise,
Euler50/shift5/CFG5 and two trained message refinements. Evaluation runs in the
existing independent VideoPhy2 environment, after the generation process exits.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import torch
from probe import RUNS
from inference.infer_message_demo21 import final_checkpoint, load_inputs, load_initial_noise, save_json, verify_video
from physgen_native.data import move
from physgen_native.generator import load_wan
from physgen_native.runtime import restore, seed_all
from physgen_native.system import PhysGenNative
from train.train import export_video

CASE_IDS = ['VP001', 'VP006', 'VP026', 'VP031']
OUTPUT = ROOT/'inference_outputs'/'v522_native_C_use4_seed42_20261027'


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['H', 'Q'])
    parser.add_argument('--shard', required=True, type=int, choices=[0, 1])
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--generated-first', action='store_true',
                        help='Separate two-case follow-up with existing reviewed generated first images')
    args = parser.parse_args()
    checkpoint = ROOT/'checkpoints'/RUNS[args.mode]/'latest.pt'
    mode = 'h_content_message' if args.mode == 'H' else 'q_seed128_message'
    state = final_checkpoint(checkpoint, mode, no_verifier_only=True)
    cfg = state['config']
    assert cfg['train']['seed'] == 42
    original = json.loads((ROOT/'inference_outputs'/RUNS['H']/
        'videophy40_final224_round2_shared_noise'/'manifest.json').read_text())
    by_id = {row['id']: row for row in original['cases']}
    case_ids = ['VP001', 'VP006'] if args.generated_first else CASE_IDS
    cases = [dict(by_id[cid]) for cid in case_ids[args.shard::2]]
    output = OUTPUT
    if args.generated_first:
        output = ROOT/'inference_outputs'/'v522_native_C_use2_generated_first_seed42_20261027'
        prepared = ROOT/'inference_outputs'/'generated_first_frame_diagnostic_seed42_20261027_inputs'/'cases'
        for case in cases:
            folder = prepared/case['id']
            old = load_inputs(Path(case['prepared_inputs']), cfg, 24.)
            new = load_inputs(folder, cfg, 24.)
            if any(not torch.equal(old[k], new[k]) for k in ('text', 'null_text', 'times', 'valid')):
                raise ValueError('Generated-first follow-up must retain all non-image input tensors')
            if torch.equal(old['first'], new['first']):
                raise ValueError('Generated-first follow-up did not change the first image')
            if not torch.equal(load_initial_noise(case, cfg), torch.load(folder/'noise.pt', weights_only=True)):
                raise ValueError('Generated-first follow-up must replay identical saved noise')
            case.update(prepared_inputs=str(folder), canvas=str(folder/'first.png'),
                image=str(ROOT/'analysis'/'fast_baseline_no_rl_20261027'/'generated_first_frames'/f'{case["id"]}.png'),
                initial_noise=str(folder/'noise.pt'),
                image_source='Existing built-in image-generation asset; previous first-image diagnostic',
                clauses_policy='Retain original process requirements; regenerate Qwen analysis with the changed first image')
    root = output/args.mode/f'shard{args.shard}'
    for arm, strength in [('full', 1.), ('C_off', 0.)]:
        manifest = dict(cases=cases, evaluation=original['evaluation'],
            scope=('Two targeted first-image follow-up cases; not representative benchmark results' if args.generated_first else
                   'Four-case diagnostic subset of shared-first-image VideoPhy2 I2V; not a benchmark-wide score'),
            checkpoint=str(checkpoint), checkpoint_step=224, seed=42, training_seed=42,
            arm=arm, condition_strength=strength, inference_round=2,
            sampling=dict(cfg['inference'], condition_strength=strength),
            input_scope='First image, caption and original first/caption requirements only',
            paired_control='Same checkpoint including trained Wan LoRA; only C strength changes in both CFG arms',
            selection=('Previously inspected VP001 and VP006 with existing generated first images' if args.generated_first else
                       'Fixed before this ablation: jump, wet-canvas twisting, elastic-cord stretching, parachute unfolding'))
        path = root/arm/'manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError(f'Different existing protocol: {path}')
        save_json(path, manifest)
    inputs = {}
    for case in cases:
        stored = load_inputs(Path(case['prepared_inputs']), cfg, 24.)
        load_initial_noise(case, cfg)
        inputs[case['id']] = dict(stored, id=case['id'], caption=case['caption'], clauses=case['clauses'])
    if args.check_only:
        print(json.dumps(dict(status='checked', mode=args.mode, shard=args.shard, cases=[c['id'] for c in cases])))
        return
    seed_all(42)
    torch.cuda.set_device(0)
    save_json(root/'runtime.json', dict(experiment=os.environ.get('DET_EXPERIMENT_ID'),
        gpu=torch.cuda.get_device_name(0), original_design_modified=False, optimizer_constructed=False))
    model = PhysGenNative(cfg, load_wan(cfg), device='cuda', training_observers=False)
    model.conditioner.cuda()
    restore(model, None, None, state, cfg)
    del state
    model.eval().requires_grad_(False)
    encoded = {}
    for case in cases:
        sample = move(inputs[case['id']], 'cuda')
        condition = model.conditions(sample)
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            first_rgb = model.decoder(sample['first'], [0])
            positive = model.encode_condition(condition, first_rgb).detach().cpu()
            analysis = model.last_condition_analysis
            negative = model.encode_condition(dict(condition, caption='', clauses=[]), first_rgb).detach().cpu()
        encoded[case['id']] = dict(positive=positive, negative=negative, analysis=analysis)
        torch.save(dict(positive=positive, negative=negative), root/f'{case["id"]}_C.pt')
        del sample, first_rgb, positive, negative
    model.knowledge.cpu()
    model.conditioner.cpu()
    model.decoder.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    model.generator.cuda()
    for case in cases:
        sample = move(inputs[case['id']], 'cuda')
        e = encoded[case['id']]
        noise = load_initial_noise(case, cfg)
        for arm, strength in [('full', 1.), ('C_off', 0.)]:
            folder = root/arm/'cases'/case['id']
            folder.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                latent = model.sample(model.conditions(sample), sample['text'], sample['null_text'],
                    seed=42, strength=strength, initial_noise=noise,
                    token_override=e['positive'].cuda(), negative_token_override=e['negative'].cuda())
            if not bool(torch.isfinite(latent).all()):
                raise FloatingPointError('Nonfinite video latent')
            torch.save(latent.cpu(), folder/'latent.pt')
            print(json.dumps(dict(kind='sampled', mode=args.mode, arm=arm, id=case['id'], seconds=time.monotonic()-start)), flush=True)
            del latent
        del sample, noise
    model.generator.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    model.decoder.cuda()
    for case in cases:
        sample = move(inputs[case['id']], 'cuda')
        e = encoded[case['id']]
        for arm, strength in [('full', 1.), ('C_off', 0.)]:
            folder = root/arm/'cases'/case['id']
            latent = torch.load(folder/'latent.pt', map_location='cuda', weights_only=True)
            model.last_sample_analysis = e['analysis'] if strength else 'C disabled in both CFG arms; trained Wan LoRA retained.'
            model.last_sample_reasoning_round = 2
            model.last_sample_condition_tokens = e['positive'].shape[1] if strength else 0
            export_video(model, latent, sample, folder/'planned.mp4', 224, 42)
            metadata = json.loads((folder/'planned.json').read_text())
            metadata.update(condition_strength=strength, checkpoint=str(checkpoint),
                saved_initial_noise=case['initial_noise'], arm=arm,
                encoded_video=verify_video(folder/'planned.mp4', cfg, 24.))
            metadata['sampler']['condition_strength'] = strength
            save_json(folder/'planned.json', metadata)
            del latent
        del sample
    save_json(root/'generation_complete.json', dict(status='complete', videos=2*len(cases)))
    sys.stdout.flush()
    evaluator = '/home/liuzhirui/miniconda3/envs/physvid/bin/python'
    os.execv(evaluator, [evaluator, '-I', '-B', str(ROOT/'tools'/'evaluate_videophy2.py'),
                        str(root/'full'), str(root/'C_off')])


if __name__ == '__main__':
    main()
