"""Bounded GPU comparisons on saved conditions; never overwrite original outputs."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
import torch
from physgen_v42.backbone import ProcessWan, fix_first, load_wan
from physgen_v42.encoders import Decoder, load_vae
from physgen_v42.runtime import autocast

SOURCE = ROOT / "inference_outputs/stability_20260917T113347Z_5d0494_step0100_ema_demo_seed42"
OUT = ROOT / "analysis/generation_failure_20260917"
OUT.mkdir(exist_ok=True)
plan = json.loads((SOURCE / "cases.json").read_text())
paths = plan["config"]["paths"]
sys.path.insert(0, paths["wan_code"])
torch.set_num_threads(4)


def say(**kw):
    print(json.dumps(kw), flush=True)


def export(rgb, name):
    import imageio.v2 as imageio
    frames=(rgb.clamp(0,1).permute(0,2,3,1)*255).round().byte().cpu().numpy()
    imageio.mimwrite(OUT / (name+".mp4"), frames, fps=24, quality=8, macro_block_size=16)
    indices=np.linspace(0,len(frames)-1,10).round().astype(int)
    sheet=Image.new("RGB",(1280,360),"#222222"); draw=ImageDraw.Draw(sheet)
    for n,idx in enumerate(indices):
        im=Image.fromarray(frames[idx]); im.thumbnail((256,154))
        x,y=n%5*256,n//5*180
        sheet.paste(im,(x,y+24)); draw.text((x+5,y+5),str(idx),fill="white")
    sheet.save(OUT / (name+".jpg"))


def decode_test(device):
    vae=load_vae(paths,device); decoder=Decoder(vae,recompute=False,save_on_cpu=False)
    report={}
    for case in ("reference_i2v","P01_t2v"):
        latent=torch.load(SOURCE/case/"final_latent.pt",map_location=device,weights_only=True)
        native=(vae.model.decode(latent,vae.scale)[0].transpose(0,1)+1)*.5
        streamed=torch.stack(list(decoder.frames(latent)))
        saved=torch.load(SOURCE/case/"generated_teacher_view.pt",map_location=device,weights_only=True)
        report[case]=dict(native_stream_max_abs=float((native-streamed).abs().max()),
                          native_stream_rmse=float((native-streamed).square().mean().sqrt()),
                          outside_01=float(((native<0)|(native>1)).float().mean()),
                          native_min=float(native.min()),native_max=float(native.max()))
        say(case=case,**report[case]); export(native,case+"_native_vae")
        del native,streamed,latent,saved
    evidence=json.loads((OUT/"saved_evidence.json").read_text())
    cache=Path(paths["cache"])
    for rec in evidence["training_samples"][:2]:
        latent=torch.load(cache/"vae"/(rec["id"]+".pt"),map_location=device,weights_only=True)["latent"][:,:,:4]
        rgb=(vae.model.decode(latent,vae.scale)[0].transpose(0,1)+1)*.5
        export(rgb,rec["id"]+"_training_reconstruction")
        report[rec["id"]]=dict(outside_01=float(((rgb<0)|(rgb>1)).float().mean()))
        say(training=rec["id"],**report[rec["id"]])
    (OUT/"vae_comparison.json").write_text(json.dumps(report,indent=2)+"\n")


def preview_test(device):
    vae=load_vae(paths,device)
    for file in sorted(OUT.glob("*_latent.pt")):
        latent=torch.load(file,map_location=device,weights_only=True)[:,:,:4]
        rgb=(vae.model.decode(latent,vae.scale)[0].transpose(0,1)+1)*.5
        export(rgb,file.stem+"_preview")
        say(preview=str(file))


def decode_native_saved(device):
    # Resume only export after the full native-resolution FP32 decode exceeded
    # the local 24GB card. Same saved latent and same FP32 decoder, streamed.
    vae = load_vae(paths, device)
    decoder = Decoder(vae, recompute=False, save_on_cpu=False)
    latent = torch.load(OUT/'P01_t2v_base_only_native_resolution_latent.pt', map_location=device, weights_only=True)
    from wan.modules.vae2_2 import unpatchify
    # Native in-place causal cache ownership: avoid retaining a second list of
    # previous cache tensors or the previous decoded GPU chunk during decoding.
    frames = []
    z, cache = decoder._initial(latent)
    cache = list(cache)
    for i in range(z.shape[2]):
        out = vae.model.decoder(z[:, :, i:i+1], feat_cache=cache, feat_idx=[0], first_chunk=(i==0))
        rgb = (unpatchify(out, patch_size=2).float()+1)*.5
        frames.extend(rgb[0].transpose(0,1).cpu().unbind(0))
        del rgb, out
        if i % 5 == 0:
            say(decoded_frames=len(frames))
    export(torch.stack(frames), 'P01_t2v_base_only_native_resolution')
    (OUT/'base_only_native_resolution_comparison.json').write_text(json.dumps(dict(
        P01_t2v=dict(wrapper_native_max_abs=0., latent_rms=float(latent.square().mean().sqrt()),
                     frames=len(frames), geometry=[1280,704], steps=50, shift=5, guidance=5, seed=42,
                     decode='FP32 native decoder with in-place causal cache; saved latent recovered after OOM')),indent=2)+'\n')


def text_test(device):
    from physgen_v42.encoders import load_text
    encoder=load_text(paths,device)
    condition=torch.load(SOURCE/"P01_t2v/condition.pt",map_location=device,weights_only=True)
    prompt=next(c["prompt"] for c in plan["cases"] if c["key"]=="P01_t2v")
    report={}
    with autocast(device):
        for name,p in (("text",prompt),("negative",plan["negative"]),("empty","")):
            value=encoder([p],device)[0].bfloat16()
            torch.save(value.cpu(),OUT/(name+"_fresh.pt"))
            if name in condition:
                saved=condition[name]
                report[name]=dict(shape=list(value.shape),saved_shape=list(saved.shape),
                                  max_abs=float((value-saved).abs().max()),rmse=float((value.float()-saved.float()).square().mean().sqrt()))
                say(text=name,**report[name])
    (OUT/"text_comparison.json").write_text(json.dumps(report,indent=2)+"\n")


def base_test(device, cases, full=False, variant="standard"):
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    wan=load_wan(paths["wan_checkpoint"],device)
    if variant=="v4_precision":
        for block in wan.blocks:
            for attention in (block.self_attn,block.cross_attn):
                attention.norm_q.bfloat16(); attention.norm_k.bfloat16()
            block.modulation.data=block.modulation.data.bfloat16()
    adapter=None
    if full:
        from inference.infer import load_adapter
        adapter=load_adapter(plan,device)
    model=ProcessWan(wan,adapter,recompute=False)
    label="full_local" if full else "base_only"
    if variant!="standard": label+="_"+variant
    report={}; latents={}
    for key in cases:
        case=next(c for c in plan["cases"] if c["key"]==key)
        condition=torch.load(SOURCE/key/"condition.pt",map_location=device,weights_only=True)
        first=condition["first"]
        text,neg=condition["text"][None],condition["negative"][None]
        if variant=="empty_negative": neg=torch.load(OUT/"empty_fresh.pt",map_location=device,weights_only=True)[None]
        geo=dict(case["geometry"]); frames=geo["frames"]
        if variant=="frames101": frames=geo["frames"]=101
        if variant=="native_resolution":
            from physgen_v42.geometry import fit_geometry
            geo=dict(fit_geometry(704,1280,704,1280),frames=121)
        if full:
            from physgen_v42.geometry import Geometry
            geometry=Geometry.build(geo,device)
            anchor=torch.load(SOURCE/key/"anchor.pt",map_location=device,weights_only=True) if first is not None else None
            with autocast(device): p0=adapter.initialize(text,geometry,case["mode"],anchor)
        generator=torch.Generator(device=device).manual_seed(42)
        x=fix_first(torch.randn((1,48,1+(frames-1)//4,geo["h"]//16,geo["w"]//16),generator=generator,device=device),first)
        scheduler=FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False)
        scheduler.set_timesteps(50,device=device,shift=5)
        begin=time.perf_counter()
        for i,timestep in enumerate(scheduler.timesteps):
            sigma=scheduler.sigmas[i].to(device=device,dtype=torch.float32)
            if variant=="integer_time": sigma=timestep.float()/1000
            if full:
                with autocast(device): result=model(x,first,text,sigma,case["duration"],geometry,p0,neg)
                pos=result["base"]
            else:
                pos=model.negative(x,first,text,sigma)
            if i==0:
                with autocast(device):
                    _,args,_,grid=model.inputs(x,text,sigma,first)
                    length=int(grid.prod()); known=int(grid[0,1]*grid[0,2]) if first is not None else 0
                    native_t=torch.cat((sigma.new_zeros(known),sigma.expand(length-known)*1000))[None]
                    native=wan([x[0]],native_t,[text[0]],length)[0][None]
                report[key]=dict(wrapper_native_max_abs=float((native-pos).abs().max()))
                say(case=key,**report[key]); del native
            if full:
                guided=result["guided"]; del result
            else:
                bn=model.negative(x,first,neg,sigma)
                guided=bn+(1 if variant=="cfg1" else 5)*(pos-bn)
            if variant=="euler":
                x=x+(scheduler.sigmas[i+1]-scheduler.sigmas[i]).to(device)*guided
            else:
                x=scheduler.step(guided,timestep,x,return_dict=False,generator=generator)[0]
            x=fix_first(x,first)
            if i%5==0 or i==49: say(case=key,step=i+1,seconds=round(time.perf_counter()-begin,1),latent_rms=float(x.square().mean().sqrt()))
        report[key].update(seconds=time.perf_counter()-begin,latent_rms=float(x.square().mean().sqrt()))
        latents[key]=x.cpu(); torch.save(latents[key],OUT/(key+"_"+label+"_latent.pt"))
    del model,wan,adapter,x,pos,guided
    gc.collect(); torch.cuda.empty_cache()
    vae=load_vae(paths,device)
    for key,latent in latents.items():
        rgb=(vae.model.decode(latent.to(device),vae.scale)[0].transpose(0,1)+1)*.5
        export(rgb,key+"_"+label)
    (OUT/(label+"_comparison.json")).write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("decode","base","full","preview","text","decode_native")); parser.add_argument("--cases",nargs="+",default=["P01_t2v","reference_i2v"])
    parser.add_argument("--variant",choices=("standard","v4_precision","integer_time","euler","empty_negative","frames101","cfg1","native_resolution"),default="standard")
    args=parser.parse_args(); device=torch.device("cuda:0")
    with torch.no_grad():
        say(phase=args.phase,gpu=torch.cuda.get_device_name(device))
        if args.phase=="decode": decode_test(device)
        elif args.phase=="decode_native": decode_native_saved(device)
        elif args.phase=="preview": preview_test(device)
        elif args.phase=="text": text_test(device)
        else: base_test(device,args.cases,full=args.phase=="full",variant=args.variant)
