"""Compare pretrained whole-video and isolated-pair evidence on identical videos."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import imageio.v2 as imageio
from physgen_native.data import VideoDataset,move
from physgen_native.decoder import VideoDecoder
from physgen_native.process_encoder import VideoProcessEncoder
from physgen_native.runtime import load_checkpoint,atomic_save,Logger
from train.train import export_video


def read_video(path,device):
    import numpy as np
    reader=imageio.get_reader(path)
    frames=np.stack([frame for frame in reader]);reader.close()
    return torch.from_numpy(frames).to(device).permute(0,3,1,2).float()/127.5-1


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Reuse complete full/pair feature records; recreate incomplete study exports')
    args=parser.parse_args();torch.cuda.set_device(0)
    state=load_checkpoint(args.checkpoint);cfg=state['config'];step=state['step'];del state
    cfg['data'].pop('train_ids',None)
    records={}
    for split in ['train','validation']:
        data=VideoDataset(cfg,split)
        records.update({row['id']:(data,index) for index,row in enumerate(data.rows)})
    decoder=VideoDecoder(cfg,'cuda').cuda().eval()
    encoder=VideoProcessEncoder('/home/liuzhirui/model/vjepa2',
                               '/home/liuzhirui/model/vjepa2/checkpoint/vjepa2_1_vitG_384.pt',
                               checkpointing=False).cuda().eval()
    class Exporter:pass
    exporter=Exporter();exporter.decoder=decoder;exporter.cfg=cfg
    log=Logger(args.output/'encode.jsonl')
    for path in sorted((args.source/'features').glob('*.pt')):
        if args.resume and all((args.output/mode/'features'/path.name).is_file() for mode in ['full','pairs']):
            continue
        original=torch.load(path,map_location='cpu',weights_only=True)
        name=original['id'];split=original['split'];seed=original['seed']
        data,index=records[name];sample=move(data[index],'cuda')
        real_path=args.output/'reference_videos'/f'{name}.mp4'
        # Both classes receive the SAME export codec and subsequent preprocessing.
        # This study isolates encoder context, rather than uncompressed/compressed input classes.
        # A prior allocation lost its mounted storage during MP4 export. Rebuild
        # unfinished study outputs from the unchanged cache, not from that file.
        export_video(exporter,sample['latent'],sample,real_path,step,seed)
        real=read_video(real_path,'cuda')
        generated=read_video(args.source/'videos'/f'{split}_{name}_seed{seed}.mp4','cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            for mode in ['full','pairs']:
                target=args.output/mode/'features'/path.name
                a=encoder(real,context=mode);b=encoder(generated,context=mode)
                record=dict(original,real=a.cpu().half(),generated=b.cpu().half(),
                            evidence_encoder='V-JEPA 2.1 ViT-g',evidence_context=mode,
                            input_protocol='Both real and generated clips exported with identical H264 CRF18, then identically resized')
                atomic_save(record,target)
        log.write('encoded',step,sample=name,split=split,feature_shape=list(a.shape))
        del sample,real,generated,original,record,a,b
    log.write('complete',step,peak_gib=torch.cuda.max_memory_allocated()/2**30);log.close()


if __name__=='__main__':main()
