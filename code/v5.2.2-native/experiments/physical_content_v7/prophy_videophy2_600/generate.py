"""Generate the 600 ProPhy/VideoPhy2 captions with R15B + B400 joint read.

This is the I2V approximation: the official VideoPhy2 prompt list is used, while
each prompt reuses the independently generated shared first image from the
existing physvid_i2v_599 protocol. It is not a T2V reproduction of ProPhy.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

import imageio.v2 as imageio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SHARED, activate, case_seed, gpu_guard, plan, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int)
    p.add_argument('--checkpoint', type=Path)
    args = p.parse_args()
    protocol = plan()
    spec = protocol['model']
    gpu_guard(ROOT / 'jobs' / os.getenv('DET_EXPERIMENT_ID', 'local'))
    activate(spec['source'])
    from physical_content.model import Fusion, PhysicalModel
    from joint_read import fusion_forward
    from physgen_native.decoder import VideoDecoder

    Fusion.forward = fusion_forward
    state = torch.load(args.checkpoint or spec['checkpoint'], map_location='cpu',
                       weights_only=False, mmap=True)
    if state['step'] != spec['step'] or state['config']['source_snapshot'] != spec['source']:
        raise ValueError('Wrong R15B400 checkpoint/source snapshot')
    cfg = state['config']
    model = PhysicalModel(cfg)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    if set(state['model']) != set(model.learned_names):
        raise ValueError('Checkpoint ownership mismatch')
    with torch.no_grad():
        for name, value in state['model'].items():
            parameters[name].copy_(value)
        for name, value in state['constants'].items():
            buffers[name].copy_(value)
    del state, parameters, buffers
    model.eval().requires_grad_(False)
    decoder = VideoDecoder(cfg).eval()

    root = ROOT / 'outputs' / spec['name']
    cases = protocol['cases'][:args.limit] if args.limit else protocol['cases']
    sampling = protocol['sampling']
    root.mkdir(parents=True, exist_ok=True)
    save(root / 'manifest.json', dict(model=spec, sampling=sampling,
                                      source_prompts=protocol['dataset'],
                                      scope=protocol['scope'], cases=protocol['cases']))
    pending = [c for c in cases if not (root / 'cases' / c['id'] / 'video.json').exists()]
    for case in pending:
        begin = time.monotonic()
        folder = SHARED / case['source_id']
        inputs = torch.load(folder / 'conditions.pt', map_location='cpu', weights_only=True)
        inputs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs.items()}
        times = inputs['times'][None]
        positive = torch.load(folder / 'positive.pt', map_location='cpu', weights_only=True)
        negative = torch.load(folder / 'negative.pt', map_location='cpu', weights_only=True)
        positive = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in positive.items()}
        negative = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in negative.items()}
        destination = root / 'cases' / case['id']
        destination.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            c, groups = model.enhancement(positive)
            nc, ngroups = model.enhancement(negative)
            actual_seed = case_seed(sampling['seed'], case['index'])
            x = torch.randn((1, 48, 21, 30, 52),
                            generator=torch.Generator(device='cpu').manual_seed(actual_seed),
                            dtype=torch.float32).cuda()
            x[:, :, :1] = inputs['first']
            grid = torch.linspace(1, 0, sampling['steps'] + 1, device='cuda')
            grid = sampling['shift'] * grid / (1 + (sampling['shift'] - 1) * grid)
            for index in range(sampling['steps']):
                pos, _ = model.velocity(x, inputs['first'], inputs['text'], grid[index],
                                        c, times, groups)
                neg, _ = model.velocity(x, inputs['first'], inputs['null_text'], grid[index],
                                        nc, times, ngroups)
                velocity = neg + sampling['guidance'] * (pos - neg)
                x = x + (grid[index + 1] - grid[index]) * velocity
                x[:, :, :1] = inputs['first']
                if (index + 1) % 10 == 0:
                    print(json.dumps(dict(stage='sampling', case=case['id'],
                                          step=index + 1)), flush=True)
            frames = decoder(x)
        rgb = ((frames.permute(0, 2, 3, 1).clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
        if rgb.shape != (81, 480, 832, 3):
            raise ValueError(f'Unexpected decoded video shape: {rgb.shape}')
        path = destination / 'video.partial.mp4'
        imageio.mimwrite(path, rgb, fps=sampling['fps'], codec='libx264',
                         macro_block_size=None,
                         ffmpeg_params=['-crf', '18', '-pix_fmt', 'yuv420p'])
        path.replace(destination / 'video.mp4')
        record = dict(case=case['id'], source_case=case['source_id'],
                      caption=case['caption'], is_hard=case['is_hard'],
                      model=spec['name'], checkpoint=spec['checkpoint'],
                      seed=sampling['seed'], noise_seed=actual_seed,
                      first_image=str(folder / 'first.png'),
                      sampling=sampling, seconds=time.monotonic() - begin,
                      scope=protocol['scope'])
        save(destination / 'video.json', record)
        save(root / 'generation_status.json', dict(status='running', complete=len(cases)-len(pending),
                                                   requested=len(cases), model=spec['name']))
        print(json.dumps(dict(stage='video_complete', **record)), flush=True)
        del inputs, positive, negative, c, nc, x, pos, neg, velocity, frames, rgb
    save(root / 'generation_status.json', dict(status='partial' if args.limit else 'completed',
                                               complete=len(cases), requested=len(cases),
                                               model=spec['name']))


if __name__ == '__main__':
    main()
