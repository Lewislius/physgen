"""One failing case: retain latent and separate correction from decoding."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from inference import infer_native_p as inference
from physgen_v4.encoders import DifferentiableDecoder

OUT = ROOT / 'analysis/fullwidth3_step0300_flicker_20260918/p03_probe'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / 'inference_outputs/v42_fullwidth3_write_20260918T081555Z_1x96g/demo20/P03/i2v/seed42_full.json'
args = SimpleNamespace(**json.loads(SOURCE.read_text())['arguments'])
args.suite = 'single'
args.output_dir = None
args.case = dict(case_id='P03', mode='i2v', purpose='flicker_diagnosis')
torch.set_num_threads(4)
report = dict(source=str(SOURCE), probes={})
original_sample = inference.sample_latents


def capture(model, first, conditional, unconditional, coords, request, on_step=None):
    latent, counters = original_sample(model, first, conditional, unconditional, coords, request, on_step)
    torch.save(latent.cpu(), Path(request.output).with_suffix('.latent.pt'))
    report['probes'][request.variant] = dict(
        latent_rms_by_frame=latent.square().mean((0, 1, 3, 4)).sqrt().cpu().tolist(),
        latent_channel_means=latent.mean((0, 3, 4)).cpu().tolist(),
        all_finite=bool(torch.isfinite(latent).all()))
    return latent, counters


inference.sample_latents = capture


def measure(path):
    with imageio.get_reader(path, 'ffmpeg') as reader:
        x = np.stack([f for f in reader]).astype(np.float32) / 255
    jump = np.abs(np.diff(x.mean((1, 2)), axis=0)).mean(1)
    return dict(color_jump_mean=float(jump[4:].mean()), color_jump_p95=float(np.quantile(jump[4:], .95)))


with torch.no_grad(), inference.InferenceRuntime(args) as runtime:
    runtime.prepare_texts([args])
    request = copy.copy(args)
    request.output = str(OUT / 'full_default.mp4')
    inference.generate(request, runtime)
    report['probes']['full']['video'] = measure(request.output)
    latent = torch.load(OUT / 'full_default.latent.pt', map_location=runtime.device, weights_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    with torch.autocast('cuda', enabled=False):
        video = DifferentiableDecoder(runtime.image_vae(), recompute=False)(latent)
    rgb = ((video[0].permute(1, 2, 3, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    imageio.mimwrite(OUT / 'full_same_latent_strict_fp32.mp4', rgb, fps=24, codec='libx264', quality=8, macro_block_size=1)
    report['probes']['full']['strict_decode'] = measure(OUT / 'full_same_latent_strict_fp32.mp4')
    del video, rgb, latent
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('FULL REPORT', json.dumps(report['probes']['full']['video']),
          json.dumps(report['probes']['full']['strict_decode']), flush=True)
    # Continue only when the actual failure persists under a fresh decode.
    if report['probes']['full']['strict_decode']['color_jump_mean'] > .02:
        torch.backends.cudnn.allow_tf32 = True
        request = copy.copy(args)
        request.variant = 'wan'
        request.output = str(OUT / 'wan_same_seed.mp4')
        inference.generate(request, runtime)
        report['probes']['wan']['video'] = measure(request.output)
        print('WAN REPORT', json.dumps(report['probes']['wan']['video']), flush=True)
        (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
paths = [SOURCE.with_suffix('.mp4'), OUT / 'full_default.mp4', OUT / 'full_same_latent_strict_fp32.mp4']
if (OUT / 'wan_same_seed.mp4').exists():
    paths.append(OUT / 'wan_same_seed.mp4')
sheet = Image.new('RGB', (192 * 9, 128 * len(paths)), 'white')
for row, path in enumerate(paths):
    with imageio.get_reader(path, 'ffmpeg') as reader:
        for col, index in enumerate((0, 1, 4, 8, 16, 32, 64, 96, 120)):
            tile = Image.fromarray(reader.get_data(index)).resize((192, 108))
            sheet.paste(tile, (col * 192, row * 128 + 20))
            ImageDraw.Draw(sheet).text((col * 192 + 3, row * 128 + 3), f'{path.stem} f{index}', fill='black')
sheet.save(OUT / 'frames.jpg', quality=92)
print('Completed P03 latent isolation', flush=True)
