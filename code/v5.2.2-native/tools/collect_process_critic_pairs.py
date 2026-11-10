"""Real/full-generation feature pairs from existing train/held-out caches.

These are bounded training computations, not a new curated video dataset. No
edited video is labeled a physical failure. Held-out records never train D.
"""
import argparse
import json
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from physgen_native.data import VideoDataset,move,load_tensor
from physgen_native.generator import load_wan
from physgen_native.runtime import load_checkpoint,restore,Logger,atomic_save,seed_all
from physgen_native.system import PhysGenNative
from train.train import export_video


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--shards',type=int,default=2)
    parser.add_argument('--train-cases',type=int,default=16)
    parser.add_argument('--validation-cases',type=int,default=8)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0);seed_all(20261022)
    state = load_checkpoint(args.checkpoint);cfg = state['config']
    model = PhysGenNative(cfg,load_wan(cfg),device='cuda').cuda()
    completed,_ = restore(model,None,None,state,cfg);del state
    model.eval().requires_grad_(False)
    null = load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
    # The generator checkpoint's eight-case study selection is NOT a restriction
    # on training the critic. Select only from the unchanged original train split.
    cfg['data'].pop('train_ids',None)
    args.output.mkdir(parents=True,exist_ok=True)
    log = Logger(args.output/f'collect_shard{args.shard}.jsonl')
    for split,count in [('train',args.train_cases),('validation',args.validation_cases)]:
        dataset = VideoDataset(cfg,split)
        indices = list(range(len(dataset)))
        if split == 'train':random.Random(20261022).shuffle(indices)
        indices = indices[:count]
        for order,index in enumerate(indices):
            if order % args.shards != args.shard:continue
            sample = move(dataset[index],'cuda')
            path = args.output/'features'/f'{split}_{sample["id"]}.pt'
            if path.exists():continue
            seed = 20261022+order
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                first = model.decoder(sample['first'],[0])
                context = model.knowledge.encode(first,sample['caption'],sample['clauses'],adapt=False,readout='queries')['contextual']
                latent = model.sample(model.conditions(sample),sample['text'],null,seed=seed)
                real = model.decoder(sample['latent'],resize=cfg['knowledge']['video_size'])
                generated = model.decoder(latent,resize=cfg['knowledge']['video_size'])
                real_features = model.knowledge.process_features(real)
                generated_features = model.knowledge.process_features(generated)
                record = dict(id=sample['id'],split=split,seed=seed,caption=sample['caption'],
                              checkpoint=str(Path(args.checkpoint).resolve()),optimizer_step=completed,
                              real=real_features.cpu().half(),generated=generated_features.cpu().half(),
                              condition=context.cpu().half(),times=sample['times'].cpu(),valid=sample['valid'].cpu())
                atomic_save(record,path)
                export_video(model,latent,sample,args.output/'videos'/f'{split}_{sample["id"]}_seed{seed}.mp4',completed,seed)
            log.write('pair',completed,sample=sample['id'],split=split,seed=seed,
                      feature_shape=list(real_features.shape),file=str(path))
            del sample,real,generated,real_features,generated_features,record,latent,context
    log.write('complete',completed,shard=args.shard,peak_gib=torch.cuda.max_memory_allocated()/2**30)
    log.close()


if __name__=='__main__':main()
