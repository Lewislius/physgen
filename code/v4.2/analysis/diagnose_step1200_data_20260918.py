"""Compare v4/v4.2 preprocessing and actual training cache against source RGB."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'v4'))
import torch
from PIL import Image,ImageDraw
from physgen_v42.data import read_window
from physgen_v42.encoders import load_vae,encode_vae,Decoder
from physgen_v4.data import read_video


@torch.no_grad()
def main():
    cfg=json.loads((ROOT/'checkpoints/stability_20260917T181001Z_a83129/step1200/config.json').read_text())
    sys.path.insert(0,cfg['paths']['wan_code'])
    cache=Path(cfg['paths']['cache']);manifest=json.loads((cache/'manifest.json').read_text())
    records={r['index']:r for r in manifest['records']}
    out=ROOT/'analysis/step1200_generation_failure_20260918/data';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);device=torch.device('cuda:0')
    vae=load_vae(cfg['paths'],device);decoder=Decoder(vae,recompute=False,save_on_cpu=False)
    report={}
    for index in (0,1,2,949,1971,2399):
        record=records[index];print('Checking '+record['id'],flush=True)
        short=dict(record,indices=record['indices'][:13],pts=record['pts'][:13])
        pixels=read_window(short)
        legacy,_,_=read_video(record['video'],dict(max_frames=121,max_long_side=512,max_area=147456,
            decode_chunk_frames=8,teacher_frames=16),start_frame=record['indices'][0],requested_frames=13)
        value=torch.load(cache/'vae'/(record['id']+'.pt'),map_location='cpu',weights_only=True)
        latent=value['latent'][:,:,:4].to(device)
        fresh=encode_vae(vae,pixels.to(device))
        first=encode_vae(vae,pixels[:,:,:1].to(device))
        assert bool(torch.isfinite(fresh).all())
        decoded=torch.stack([frame.cpu() for frame in decoder.frames(latent)])
        target=((pixels[0].transpose(0,1)+1)*.5).float()
        rmse=(decoded.clamp(0,1)-target).square().mean().sqrt()
        report[record['id']]=dict(source_video=record['video'],source_indices=short['indices'],
            caption_prefix=record['caption'][:100],source_rgb_range=[float(pixels.min()),float(pixels.max())],
            v4_v42_preprocessing_max_abs=float((legacy-pixels[0]).abs().max()),
            cache_fresh_latent_rmse=float((latent-fresh).square().mean().sqrt()),
            cache_fresh_latent_max_abs=float((latent-fresh).abs().max()),
            cache_fresh_first_max_abs=float((value['first'].to(device)-first).abs().max()),
            cache_first_vs_full_first_max_abs=float((value['first']-value['latent'][:,:,:1]).abs().max()),
            cache_dtype=str(value['latent'].dtype),cache_shape=list(value['latent'].shape),
            reconstruction_rgb_rmse=float(rmse),reconstruction_psnr=float(-20*rmse.log10()),
            latent_rms=float(latent.square().mean().sqrt()))
        sheet=Image.new('RGB',(1280,420),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for row,rgb in enumerate((target,decoded.clamp(0,1))):
            for col,i in enumerate((0,4,8,12)):
                data=(rgb[i]*255).round().byte().permute(1,2,0).numpy()
                im=Image.fromarray(data);im.thumbnail((320,180));sheet.paste(im,(col*320,row*210+25))
                draw.text((col*320+5,row*210+5),record['id']+(' source' if row==0 else ' cached VAE')+f' f{i}',fill='black')
        sheet.save(out/(record['id']+'.jpg'))
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({record['id']:report[record['id']]}),flush=True)


if __name__=='__main__':main()
