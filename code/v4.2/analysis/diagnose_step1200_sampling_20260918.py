"""Controlled P01 sampling comparisons. Writes only analysis artifacts."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'v4'))
import torch
from PIL import Image, ImageDraw
import imageio.v2 as imageio
from physgen_v42.backbone import ProcessWan, fix_first, load_wan
from physgen_v42.encoders import Decoder, load_vae
from physgen_v42.geometry import Geometry
from physgen_v42.runtime import autocast


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=('v4_base', 'v42_base_v4_recipe', 'v42_full_v4_recipe', 'v42_base', 'v42_full',
        'v42_full_unipc101', 'v42_full_euler121', 'v42_full_empty'))
    args = parser.parse_args()
    source = ROOT / 'inference_outputs/stability_20260917T181001Z_a83129_step1200_ema_demo_seed42_r3_full'
    output = ROOT / 'analysis/step1200_generation_failure_20260918' / args.variant
    output.mkdir(exist_ok=True)
    plan = json.loads((source / 'cases.json').read_text())
    paths = plan['config']['paths']
    sys.path.insert(0, paths['wan_code'])
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    device = torch.device('cuda:0'); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    condition = torch.load(source / 'P01_t2v/condition.pt', map_location=device, weights_only=True)
    v4_recipe = args.variant in ('v4_base', 'v42_base_v4_recipe', 'v42_full_v4_recipe')
    frames = 101 if v4_recipe or args.variant == 'v42_full_unipc101' else 121
    euler = v4_recipe or args.variant == 'v42_full_euler121'
    empty = v4_recipe or args.variant == 'v42_full_empty'
    duration = (frames-1)/24
    geo_dict = dict(condition['geometry'], frames=frames)
    geo = Geometry.build(geo_dict, device)
    text = condition['text'][None]
    negative = (torch.load(ROOT / 'analysis/generation_failure_20260917/empty_fresh.pt',
                           map_location=device, weights_only=True)[None]
                if empty else condition['negative'][None])
    print(f'Loading {args.variant}', flush=True)
    if args.variant == 'v4_base':
        from physgen_v4.backbone import load_wan as load_v4
        wan = load_v4(paths['wan_checkpoint'], device, sharded=False, recompute=False)
    else:
        wan = load_wan(paths['wan_checkpoint'], device)
    adapter = None
    if 'full' in args.variant:
        # Avoid the v4 inference namespace which is also on sys.path.
        sys.path.insert(0, str(ROOT))
        from inference.infer import load_adapter
        adapter = load_adapter(plan, device)
    model = ProcessWan(wan, adapter, recompute=False).eval()
    with autocast(device):
        p0 = adapter.initialize(text, geo, 't2v', duration=duration) if adapter is not None else None
    generator = torch.Generator(device=device).manual_seed(42)
    x = torch.randn((1,48,1+(frames-1)//4,18,32), generator=generator, device=device)
    if euler:
        raw = torch.linspace(1,0,51,device=device)
        sigmas = 5 * raw / (1+4*raw)
        sampler = None
        timesteps = sigmas[:-1]*1000
    else:
        sampler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        sampler.set_timesteps(50, device=device, shift=5)
        timesteps, sigmas = sampler.timesteps, sampler.sigmas.to(device)
    trajectory = []
    start = time.perf_counter()
    for i, timestep in enumerate(timesteps):
        sigma = sigmas[i].float()
        with autocast(device):
            if adapter is not None:
                result = model(x,None,text,sigma,duration,geo,p0,negative)
                velocity = result['guided']; del result
            else:
                positive = model.negative(x,None,text,sigma)
                uncond = model.negative(x,None,negative,sigma)
                velocity = uncond + 5*(positive-uncond)
        x = (x + (sigmas[i+1]-sigmas[i])*velocity if sampler is None else
             sampler.step(velocity,timestep,x,return_dict=False,generator=generator)[0])
        assert bool(torch.isfinite(x).all())
        record = dict(step=i+1,sigma=float(sigma),latent_rms=float(x.square().mean().sqrt()),seconds=time.perf_counter()-start)
        trajectory.append(record)
        if i%10==0 or i==49: print(json.dumps(record),flush=True)
    torch.save(x.cpu(), output / 'final_latent.pt')
    (output / 'trajectory.json').write_text(json.dumps(trajectory,indent=2)+'\n')
    del model, wan, adapter, p0, text, negative, condition, velocity
    if 'positive' in locals(): del positive, uncond
    gc.collect(); torch.cuda.empty_cache()
    vae = load_vae(paths, device)
    decoder = Decoder(vae,recompute=False,save_on_cpu=False)
    samples = {}
    outside = total = 0
    with imageio.get_writer(output / 'video.mp4',fps=24,codec='libx264',quality=8,macro_block_size=16) as writer:
        for i, frame in enumerate(decoder.frames(x)):
            outside += int(((frame<0)|(frame>1)).sum()); total += frame.numel()
            pixels = (frame.clamp(0,1)*255).round().byte().permute(1,2,0).cpu().numpy()
            writer.append_data(pixels)
            if i in (0,frames//4,frames//2,3*frames//4,frames-1): samples[i]=pixels
    sheet = Image.new('RGB',(1600,210),'#eeeeee'); draw=ImageDraw.Draw(sheet)
    for j,(i,pixels) in enumerate(samples.items()):
        im=Image.fromarray(pixels);im.thumbnail((320,180));sheet.paste(im,(j*320,25))
        draw.text((j*320+5,5),args.variant+f' f{i}',fill='black')
    sheet.save(output/'frames.jpg')
    report=dict(variant=args.variant,case='P01_t2v',frames=frames,seed=42,steps=50,
                negative='empty' if empty else 'native',solver='euler' if euler else 'unipc',
                wan_loader='v4' if args.variant=='v4_base' else 'v4.2',
                decoder='v4.2 FP32',outside_01=outside/total,gpu=torch.cuda.get_device_name(device),
                seconds=time.perf_counter()-start)
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
