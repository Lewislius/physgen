"""Compare a one-step clean estimate with actual multistep denoising.

The same noised reference is used throughout. This tests the process target's
location in the computation, not free-generation quality or generalization.
"""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from physgen_native.config import read_config
from physgen_native.data import VideoDataset, load_tensor, move
from physgen_native.generator import load_wan
from physgen_native.runtime import Logger, seed_all
from physgen_native.sequence import sequence_error, sequence_metrics, perceptual_process_error
from physgen_native.system import PhysGenNative, shifted_sigma


def integrate(model, sample, noisy, positive, negative, null_text, sigma, steps, guidance):
    # Truncate the native shifted schedule at the requested noise level.
    shift = model.cfg['inference']['shift']
    source = sigma/(shift-(shift-1)*sigma)
    sigmas = shifted_sigma(torch.linspace(float(source), 0, steps+1, device=noisy.device), shift)
    state = noisy
    for current, following in zip(sigmas[:-1], sigmas[1:]):
        v = model.generator(state, sample['first'], sample['text'], current, positive, sample['times'])
        if guidance != 1:
            u = model.generator(state, sample['first'], null_text, current, negative, sample['times'])
            v = u+guidance*(v-u)
        state = torch.cat((sample['first'], (state+(following-current)*v)[:, :, 1:]), 2)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'analysis/denoising_process'))
    args = parser.parse_args()
    cfg = read_config(ROOT/'configs/native_physical_wisa3857.yaml')
    seed_all(42); torch.cuda.set_device(0)
    log = Logger(Path(args.output)/'metrics.jsonl')
    model = PhysGenNative(cfg, load_wan(cfg), device='cuda').cuda().eval().requires_grad_(False)
    data = VideoDataset(cfg, 'validation')
    null = load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
    for index in (0, 1, 3, 4):
        sample = move(data[index], 'cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            c = model.encode_condition(model.conditions(sample))
            u = model.encode_condition(model.conditions(sample, dropped=True))
            truth = model.decoder(sample['latent'], resize=cfg['knowledge']['video_size'])
            target_features = model.knowledge.process_features(truth)
            noise = torch.randn(sample['latent'].shape, device='cuda',
                                generator=torch.Generator(device='cuda').manual_seed(20261022+index))
            for value in (.65, .85):
                sigma = torch.tensor(value, device='cuda')
                noisy = (1-sigma)*sample['latent']+sigma*noise
                noisy[:, :, :1] = sample['first']
                for steps, guidance in ((1, 1), (4, 1), (4, 5)):
                    estimate = integrate(model, sample, noisy, c, u, null, sigma, steps, guidance)
                    frames = model.decoder(estimate, resize=cfg['knowledge']['video_size'])
                    rgb = sequence_error(frames, truth, sample['valid'], cfg['sequence'])[0]
                    visual = perceptual_process_error(model.knowledge.process_features(frames), target_features,
                                                       sample['valid'], cfg['sequence'])[0]
                    metrics = sequence_metrics(frames, truth, sample['valid'])
                    log.write('denoising_process', index, sample=sample['id'], sigma=value, steps=steps,
                              guidance=guidance, process=.5*(rgb+visual), rgb=rgb, visual=visual, **metrics)
    log.write('complete', 0, peak_gib=torch.cuda.max_memory_allocated()/2**30)
    log.close()


if __name__ == '__main__': main()
