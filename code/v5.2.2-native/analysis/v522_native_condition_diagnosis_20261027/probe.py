"""Read-only causal diagnostics of current v29/v30 H/Q checkpoints.

Preserve current LoRA, message rounds, native first-frame geometry and seed42.
Future training-cache latents are used only as labelled reconstruction targets.
No optimizer, external model, new dataset, or design-source edit is involved.
The outputs diagnose conditional use; they do not establish video quality gains.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import torch
from physgen_native.data import VideoDataset, load_tensor, move
from physgen_native.generator import load_wan
from physgen_native.runtime import restore, seed_all
from physgen_native.sequence import weighted_mean
from physgen_native.system import PhysGenNative

RUNS = dict(H='v522-h-content-message-no-verifier-simple_66438',
            Q='v522-q-seed128-message-no-verifier-simple_66443')
IDS = ['source0000042', 'source0001596', 'wisa0013758', 'wisa0032606']


def rms(value):
    return float(value.float().square().mean().sqrt())


def representation(value):
    value = value.float()
    mean = value.mean(1, keepdim=True)
    centered = value-mean
    norm = value.norm(dim=-1).clamp_min(1e-12)
    cosine = ((value*mean).sum(-1)/(norm*mean.norm(dim=-1).clamp_min(1e-12))).mean()
    return dict(shape=list(value.shape), rms=rms(value), mean_rms=rms(mean),
        token_centered_rms=rms(centered), token_centered_fraction=rms(centered)/max(rms(value),1e-12),
        mean_token_cosine=float(cosine))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['H','Q'], required=True)
    parser.add_argument('--representations-only', action='store_true')
    args = parser.parse_args()
    seed_all(42); torch.cuda.set_device(0)
    out = HERE/args.mode
    if args.representations_only:
        out = out/'representations'
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ROOT/'checkpoints'/RUNS[args.mode]/'latest.pt'
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    cfg = state['config']
    if state['step'] != 224 or cfg['train']['seed'] != 42 or cfg['verifier']['kind'] != 'none':
        raise ValueError('Expected the current no-verifier final224 seed42 checkpoint')
    model = PhysGenNative(cfg, load_wan(cfg), device='cuda', training_observers=False)
    model.conditioner.cuda()
    restore(model, None, None, state, cfg)
    model.eval().requires_grad_(False)
    del state
    dataset = VideoDataset(cfg, 'validation')
    indices = {row['id']:i for i,row in enumerate(dataset.rows)}
    samples = [dataset[indices[cid]] for cid in IDS]
    write(out/'protocol.json', dict(checkpoint=str(checkpoint_path), revision=cfg['revision'], step=224,
        training_seed=42, diagnostic_noise_seed=42, same_saved_noise_for_all_variants=True,
        diagnostic_noise_device='CPU, for matching H/Q and different allocated GPU families',
        inference_round=cfg['knowledge']['inference_reasoning_round'], cases=IDS,
        sigmas=[.65,.95,.98], native_text_modes=['present','absent'],
        variants=['correct','wrong_caption_same_first','mean_slots','C_off','round0','gain025'],
        scope='Paired one-step denoising and condition sensitivity only; not generation-quality evidence',
        wrong_caption_policy='Keep the same first image and image-derived requirements; replace caption and prompt-derived requirements',
        existing_design_modified=False, optimizer_constructed=False, gpu=torch.cuda.get_device_name(0)))
    encoded = {}
    for index, raw in enumerate(samples):
        sample = move(raw, 'cuda')
        condition = model.conditions(sample)
        other = samples[(index+1)%len(samples)]
        alternative = model._alternative_condition(dict(sample, alternative={
            'id':other['id'],'caption':other['caption'],'clauses':other['clauses']}))
        started = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            first_rgb = model.decoder(sample['first'], [0])
            rounds = model.encode_condition_rounds(condition, first_rgb)
            selected = rounds['conditions'][cfg['knowledge']['inference_reasoning_round']].detach().cpu()
            round0 = rounds['conditions'][0].detach().cpu()
            correct_analysis = rounds['reasoning']
            if args.representations_only:
                layers = dict(content=representation(rounds['content']))
                if hasattr(model.conditioner, 'query_seed'):
                    layers['query_seed'] = representation(model.conditioner.query_seed)
                for round_id, value in enumerate(rounds['readouts']):
                    projected = model.conditioner.project(value)
                    normalized = model.conditioner.alignment.norm(projected.float())
                    affine_fp32 = normalized*model.conditioner.alignment.gain.float()+model.conditioner.alignment.bias.float()
                    layers[f'round{round_id}'] = dict(message_state=representation(value),
                        projected=representation(projected), layernorm=representation(normalized),
                        affine_fp32=representation(affine_fp32), final_C=representation(rounds['conditions'][round_id]))
                write(out/f'{raw["id"]}.json', dict(id=raw['id'], layers=layers))
                print(json.dumps(dict(kind='representation_audit',mode=args.mode,id=raw['id'])),flush=True)
                del sample,rounds,selected,round0,first_rgb,projected,normalized,affine_fp32,value
                continue
            wrong = model.encode_condition(alternative, first_rgb).detach().cpu()
            wrong_analysis = model.last_condition_analysis
        encoded[raw['id']] = dict(correct=selected,wrong=wrong,round0=round0)
        torch.save(encoded[raw['id']], out/f'{raw["id"]}_C.pt')
        write(out/f'{raw["id"]}_conditions.json',dict(id=raw['id'],caption=raw['caption'],
            wrong_caption=alternative['caption'], correct_analysis=correct_analysis, wrong_analysis=wrong_analysis,
            full_rms=rms(selected), token_mean_rms=rms(selected.mean(1,keepdim=True)),
            centered_rms=rms(selected-selected.mean(1,keepdim=True)),
            correct_token_count=selected.shape[1],wrong_token_count=wrong.shape[1],
            round0_relative_change=rms(selected-round0)/max(rms(selected),1e-12),
            encoder_seconds=time.monotonic()-started))
        print(json.dumps(dict(kind='encoded',mode=args.mode,id=raw['id'],seconds=time.monotonic()-started)),flush=True)
        del sample,rounds,selected,round0,wrong,first_rgb
    if args.representations_only:
        write(out/'complete.json', dict(status='complete',cases=4,mode=args.mode))
        return
    model.knowledge.cpu(); model.conditioner.cpu(); model.decoder.cpu()
    gc.collect();torch.cuda.empty_cache()
    model.generator.cuda()
    null = load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
    noise = torch.randn(samples[0]['latent'].shape, generator=torch.Generator(device='cpu').manual_seed(42))
    torch.save(noise, out/'noise_seed42.pt')
    noise = noise.cuda()
    records = []
    with (out/'metrics.jsonl').open('w',buffering=1) as log:
        for raw in samples:
            sample = move(raw,'cuda'); z=sample['latent']
            c = {k:v.cuda() for k,v in encoded[raw['id']].items()}
            variants = [('correct',c['correct'],1.),('wrong_caption_same_first',c['wrong'],1.),
                        ('mean_slots',c['correct'].mean(1,keepdim=True).expand_as(c['correct']),1.),
                        ('C_off',c['correct'],0.),('round0',c['round0'],1.),('gain025',c['correct'],.25)]
            for level in [.65,.95,.98]:
                sigma = z.new_tensor(level)
                noisy = torch.cat((sample['first'],((1-sigma)*z+sigma*noise)[:,:,1:]),2)
                target = (noise-z)[:,:,1:]
                for text_mode, text in [('present',sample['text']),('absent',null)]:
                    velocities = {}
                    base_rows = []
                    for name,tokens,strength in variants:
                        model.generator.interface.capture = name=='correct'
                        model.generator.interface.observations=[]
                        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                            velocity=model.generator(noisy,sample['first'],text,sigma,tokens,sample['times'],strength)
                        if not torch.isfinite(velocity).all():
                            raise FloatingPointError('Nonfinite denoising output')
                        velocity=velocity[:,:,1:].float()
                        fm=weighted_mean((velocity-target).square().mean(1),sample['valid'])
                        velocities[name]=velocity
                        base_rows.append(dict(mode=args.mode,id=raw['id'],sigma=level,native_text=text_mode,
                            variant=name,fm=float(fm),velocity_rms=rms(velocity),
                            attention=model.generator.interface.observations))
                    effect = rms(velocities['correct']-velocities['C_off'])
                    for row in base_rows:
                        delta=rms(velocities[row['variant']]-velocities['correct'])
                        row.update(output_rms_difference_from_correct=delta,
                            correct_C_effect_rms=effect, difference_over_full_C_effect=delta/max(effect,1e-12))
                        log.write(json.dumps(row)+'\n');records.append(row)
                    del velocities,base_rows
            print(json.dumps(dict(kind='measured',mode=args.mode,id=raw['id'])),flush=True)
            del sample,z,c,noisy,target
    write(out/'complete.json',dict(status='complete',records=len(records),mode=args.mode,
        scope='Mechanism diagnostics only; meaningful video improvement remains to be tested'))


if __name__=='__main__':
    main()
