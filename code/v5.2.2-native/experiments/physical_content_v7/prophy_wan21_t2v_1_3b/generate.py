"""Generate the 600 ProPhy/VideoPhy2 captions with official Wan2.1-T2V-1.3B.

Text-to-video, native Wan2.1 checkpoint and T5/VAE.  No first image, no project
LoRA, no prompt extension.  Sampling follows the established local ProPhy replay
protocol: 50 steps, shift5, CFG5, 81 frames, 832x480, 16 fps, seed 42+index.
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
from common import ROOT, case_seed, gpu_guard, plan, save


def write_video(frames, path, fps):
    path = Path(path)
    temporary = path.with_suffix('.partial.mp4')
    try:
        with imageio.get_writer(
                str(temporary), fps=fps, codec='libx264', pixelformat='yuv420p',
                macro_block_size=None,
                ffmpeg_params=['-crf', '18', '-pix_fmt', 'yuv420p']) as stream:
            for frame in frames:
                stream.append_data(frame)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--no-offload', action='store_true',
                        help='Keep the 1.3B model and T5 on GPU between videos')
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError('shard must be in [0, shards)')
    protocol = plan()
    spec = protocol['model']
    sampling = protocol['sampling']
    gpu_guard(ROOT / 'jobs' / os.getenv('DET_EXPERIMENT_ID', 'local'))
    sys.path.insert(0, spec['source'])
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V

    torch.set_num_threads(4)
    model = WanT2V(
        WAN_CONFIGS['t2v-1.3B'], spec['checkpoint'], device_id=0,
        t5_cpu=False)
    root = ROOT / 'outputs' / spec['name']
    root.mkdir(parents=True, exist_ok=True)
    save(root / 'manifest.json', dict(
        model=spec, sampling=sampling, source_prompts=protocol['dataset'],
        scope=protocol['scope'], cases=protocol['cases']))
    cases = protocol['cases'][:args.limit] if args.limit else protocol['cases']
    # Shards are disjoint stride partitions of the same declared case list.
    cases = [case for index, case in enumerate(cases)
             if index % args.shards == args.shard]
    pending = [case for case in cases
               if not (root / 'cases' / case['id'] / 'video.json').is_file()]
    for index, case in enumerate(pending):
        begin = time.monotonic()
        seed = case_seed(sampling['seed'], case['index'])
        video = model.generate(
            case['caption'].strip(),
            size=(int(sampling['width']), int(sampling['height'])),
            frame_num=int(sampling['frames']),
            shift=float(sampling['shift']),
            sample_solver='unipc',
            sampling_steps=int(sampling['steps']),
            guide_scale=float(sampling['guidance']),
            seed=seed,
            offload_model=not args.no_offload)
        if tuple(video.shape) != (3, 81, 480, 832):
            raise ValueError(f'Unexpected Wan2.1 output shape {tuple(video.shape)}')
        rgb = ((video.float().clamp(-1, 1).permute(1, 2, 3, 0) + 1) * 127.5
               ).round().byte().cpu().numpy()
        destination = root / 'cases' / case['id']
        destination.mkdir(parents=True, exist_ok=True)
        write_video(rgb, destination / 'video.mp4', sampling['fps'])
        save(destination / 'video.json', dict(
            case=case['id'], caption=case['caption'], is_hard=case['is_hard'],
            source_case=case['source_id'], seed=seed,
            model=spec, sampling=sampling, task='t2v', first_frame_used=False,
            seconds=time.monotonic() - begin, scope=protocol['scope']))
        save(root / f'generation_status_shard{args.shard}.json', dict(
            status='running', shard=args.shard, shards=args.shards,
            complete=len(cases) - len(pending) + index + 1,
            requested=len(cases), model=spec['name']))
        print(json.dumps(dict(stage='video_complete', case=case['id'],
                              seconds=time.monotonic() - begin)), flush=True)
        del video, rgb
        torch.cuda.empty_cache()
    save(root / f'generation_status_shard{args.shard}.json', dict(
        status='partial' if args.limit else 'completed', shard=args.shard,
        shards=args.shards, complete=len(cases), requested=len(cases),
        model=spec['name']))


if __name__ == '__main__':
    main()
