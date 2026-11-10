"""Compare C's caption response with the same Wan's native T5 response.

This uses cached legal C from the completed diagnostic, not new encoders or
teachers. A noisy validation latent is only a labelled diagnostic query state.
Native T5 response is a reference direction, not physical ground truth.
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
from probe import IDS, RUNS, rms, write
from physgen_native.data import VideoDataset, load_tensor, move
from physgen_native.generator import ConditionedWan, load_wan
from physgen_native.runtime import seed_all


def compare(delta, reference):
    x, y = delta.flatten().double(), reference.flatten().double()
    return dict(rms=rms(delta), native_response_rms=rms(reference),
        norm_ratio=float(x.norm()/y.norm().clamp_min(1e-12)),
        cosine=float(x.dot(y)/(x.norm()*y.norm()).clamp_min(1e-12)),
        signed_projection_gain=float(x.dot(y)/y.dot(y).clamp_min(1e-12)),
        relative_error=float((x-y).norm()/y.norm().clamp_min(1e-12)))


@torch.no_grad()
def main():
    seed_all(42)
    torch.cuda.set_device(0)
    out = HERE/'native_response'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for mode in ('H', 'Q'):
        state = torch.load(ROOT/'checkpoints'/RUNS[mode]/'latest.pt',
                           map_location='cpu', weights_only=False, mmap=True)
        cfg = state['config']
        generator = ConditionedWan(load_wan(cfg), cfg)
        parameters = dict(generator.named_parameters())
        for name, value in state['model'].items():
            if name.startswith('generator.'):
                key = name.removeprefix('generator.')
                parameters[key].copy_(value)
        generator.eval().requires_grad_(False).cuda()
        del state
        data = VideoDataset(cfg, 'validation')
        indices = {r['id']: i for i, r in enumerate(data.rows)}
        samples = [data[indices[cid]] for cid in IDS]
        noise = torch.load(HERE/mode/'noise_seed42.pt', map_location='cuda', weights_only=True)
        null = load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
        for i, raw in enumerate(samples):
            sample = move(raw, 'cuda')
            alternate_text = samples[(i+1) % len(samples)]['text'].cuda()
            c = {k: v.cuda() for k, v in torch.load(HERE/mode/f'{raw["id"]}_C.pt',
                   map_location='cpu', weights_only=True).items()}
            sigma = sample['latent'].new_tensor(.95)
            z = sample['latent']
            noisy = torch.cat((sample['first'], ((1-sigma)*z+sigma*noise)[:, :, 1:]), 2)

            def forward(text, tokens, strength):
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    return generator(noisy, sample['first'], text, sigma, tokens,
                                     sample['times'], strength)[:, :, 1:].float()

            native = forward(sample['text'], c['correct'], 0.)-forward(alternate_text, c['correct'], 0.)
            for text_name, text in [('present', sample['text']), ('absent', null)]:
                positive = forward(text, c['correct'], 1.)
                repeated = forward(text, c['correct'], 1.)
                wrong = forward(text, c['wrong'], 1.)
                row = dict(mode=mode, id=raw['id'], native_text=text_name,
                    repeat_output_rms=rms(positive-repeated),
                    **compare(positive-wrong, native))
                records.append(row)
                print(json.dumps(row), flush=True)
            del sample, c, noisy, z, native, positive, repeated, wrong, alternate_text
        del generator, parameters, samples, data, noise, null
        gc.collect()
        torch.cuda.empty_cache()
    write(out/'results.json', dict(records=records,
        scope='Semantic-response diagnostic; native T5 is a reference, not a physics label or video-quality result',
        original_design_modified=False, existing_checkpoint_LoRA_retained=True))


if __name__ == '__main__':
    main()
