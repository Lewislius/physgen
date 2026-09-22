"""Decode an existing, visually stable LoRA latent with the V4.2 VAE path.

No diffusion sampling or training. This isolates VAE precision from correctors.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from physgen_v4.encoders import load_vae, DifferentiableDecoder
from physgen_v4.data import letterbox
from physgen_v4.runtime import read_config

OUT = ROOT / 'analysis/fullwidth3_step0300_flicker_20260918/vae_precision'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT.parent / 'v4-static-lora/inference_outputs/lora_20260917T140727Z_dae2a0_step0800_ema_demo_seed42/P03_i2v'
config = read_config(ROOT / 'checkpoints/v42_v4_fullwidth3_without_prior_A1_1x96g_20260918T071253Z-9945c9d7/checkpoint-0000300/config.yaml')
sys.path.insert(0, config['paths']['wan_code'])
device = torch.device('cuda:0')
torch.cuda.set_device(device)
torch.set_num_threads(4)
report = dict(torch=str(torch.__version__), gpu=torch.cuda.get_device_name(device),
              source=str(SOURCE), default_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
              default_cudnn_tf32=torch.backends.cudnn.allow_tf32, variants={})
with torch.no_grad():
    latent = torch.load(SOURCE / 'final_latent.pt', map_location=device, weights_only=True)
    condition = torch.load(SOURCE / 'condition.pt', map_location='cpu', weights_only=True)
    with imageio.get_reader(SOURCE / 'video.mp4', 'ffmpeg') as reader:
        original = np.stack([f for f in reader]).astype(np.float32) / 255
    with Image.open(ROOT.parent / 'v1/demo/P03/P03-i0-1280x704.png') as image:
        pixels = torch.from_numpy(np.array(image.convert('RGB'))).unsqueeze(0)
    reference, _ = letterbox(pixels, latent.shape[-2] * 16, latent.shape[-1] * 16)
    vae = load_vae(config['paths'], device, dtype=torch.float32)
    thumbnails = []
    for label, tf32 in [('v42_default', report['default_cudnn_tf32']), ('strict_fp32', False)]:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = tf32
        torch.set_float32_matmul_precision('highest')
        start = time.perf_counter()
        with torch.autocast('cuda', enabled=False):
            first = vae.model.encode(reference.unsqueeze(0).to(device), vae.scale).float()
            video = DifferentiableDecoder(vae, recompute=False)(latent)
        rgb = ((video[0].permute(1, 2, 3, 0).float().cpu().numpy() + 1) * .5).clip(0, 1)
        means = rgb.mean(axis=(1, 2))
        colors = np.abs(np.diff(means, axis=0)).mean(axis=1)
        expected = condition['first'].to(device)
        report['variants'][label] = dict(cudnn_tf32=tf32, seconds=time.perf_counter() - start,
            first_mae_vs_saved=float((first - expected).abs().mean()),
            first_max_vs_saved=float((first - expected).abs().max()),
            first_rms=float(first.square().mean().sqrt()),
            decoded_mae_vs_existing_mp4=float(np.abs(rgb - original).mean()),
            color_jump_mean=float(colors[4:].mean()),
            color_jump_p95=float(np.quantile(colors[4:], .95)),
            clipping_fraction=float(((rgb < .02) | (rgb > .98)).mean()),
            all_finite=bool(torch.isfinite(video).all()))
        imageio.mimwrite(OUT / f'{label}.mp4', (rgb * 255).round().astype(np.uint8), fps=24,
                        codec='libx264', quality=8, macro_block_size=1)
        thumbnails.append((label, rgb))
        print(json.dumps({label: report['variants'][label]}), flush=True)
        (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    sheet = Image.new('RGB', (192 * 9, 128 * 3), 'white')
    for row, (label, rgb) in enumerate([('existing_lora_mp4', original), *thumbnails]):
        for col, index in enumerate((0, 1, 4, 8, 16, 32, 64, 96, 120)):
            tile = Image.fromarray((rgb[index] * 255).round().astype(np.uint8)).resize((192, 108))
            sheet.paste(tile, (col * 192, row * 128 + 20))
            ImageDraw.Draw(sheet).text((col * 192 + 3, row * 128 + 3), f'{label} f{index}', fill='black')
    sheet.save(OUT / 'same_latent_precision.jpg', quality=92)
print('Completed existing-latent precision verification', flush=True)
