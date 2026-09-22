"""Decode existing latents without sampling or changing production files."""
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image, ImageDraw
import imageio.v2 as imageio
from physgen_v42.encoders import Decoder, load_vae

SOURCE = ROOT / 'inference_outputs/stability_20260917T181001Z_a83129_step1200_ema_demo_seed42_r3_full'
OUT = ROOT / 'analysis/step1200_generation_failure_20260918'
OUT.mkdir(exist_ok=True)
plan = json.loads((SOURCE / 'cases.json').read_text())
sys.path.insert(0, plan['config']['paths']['wan_code'])
device = torch.device('cuda:0')
torch.set_num_threads(4)
old = ROOT / 'analysis/generation_failure_20260917'
sources = {
    'step1200_P01_t2v': SOURCE / 'P01_t2v/final_latent.pt',
    'old_base_512': old / 'P01_t2v_base_only_latent.pt',
    'old_base_1280': old / 'P01_t2v_base_only_native_resolution_latent.pt',
}
report = {}
with torch.no_grad():
    for precision in ('fp32', 'bf16'):
        vae = load_vae(plan['config']['paths'], device)
        if precision == 'bf16':
            vae.model.bfloat16()
            vae.scale = [s.bfloat16() for s in vae.scale]
        from wan.modules.vae2_2 import count_conv3d, unpatchify
        for name, source in sources.items():
            latent = torch.load(source, map_location=device, weights_only=True)[:, :, :4]
            frames = []
            print(f'Decoding {name} {precision} shape={tuple(latent.shape)}', flush=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16', cache_enabled=False):
                z = latent / vae.scale[1].view(1, 48, 1, 1, 1) + vae.scale[0].view(1, 48, 1, 1, 1)
                z = vae.model.conv2(z)
                cache = [None] * count_conv3d(vae.model.decoder)
                for i in range(z.shape[2]):
                    raw = vae.model.decoder(z[:, :, i:i+1], feat_cache=cache, feat_idx=[0], first_chunk=i == 0)
                    rgb = (unpatchify(raw, patch_size=2).float() + 1) * .5
                    frames.extend(rgb[0].transpose(0, 1).cpu().unbind())
                    del raw, rgb
            rgb = torch.stack(frames)
            label = name + '_' + precision
            report[label] = dict(frames=len(rgb), outside_01=float(((rgb < 0) | (rgb > 1)).float().mean()),
                minimum=float(rgb.min()), maximum=float(rgb.max()), latent_rms=float(latent.square().mean().sqrt()))
            pixels = (rgb.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()
            imageio.mimwrite(OUT / (label + '.mp4'), pixels, fps=24, quality=8, macro_block_size=16)
            sheet = Image.new('RGB', (1280, 210), '#eeeeee'); draw = ImageDraw.Draw(sheet)
            for j, i in enumerate((0, 4, 8, 12)):
                im = Image.fromarray(pixels[i]); im.thumbnail((320, 180))
                sheet.paste(im, (j*320, 25)); draw.text((j*320+5, 5), label + f' f{i}', fill='black')
            sheet.save(OUT / (label + '.jpg'))
            print(json.dumps({label: report[label]}), flush=True)
            (OUT / 'decode_report.json').write_text(json.dumps(report, indent=2) + '\n')
            del latent, z, cache, frames, rgb, pixels
            gc.collect(); torch.cuda.empty_cache()
        del vae
        gc.collect(); torch.cuda.empty_cache()
