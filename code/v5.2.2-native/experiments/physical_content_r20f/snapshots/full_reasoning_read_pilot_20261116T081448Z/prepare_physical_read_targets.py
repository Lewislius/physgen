"""Frozen-Wan whole-video physical-read distributions, for training only.

No boxes, segmentation, tracks, video chunks, external API, or generated test
video is used. This is a candidate semantic localization teacher; the pilot
must be inspected before its maps can be treated as useful supervision.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import numpy as np
from PIL import Image, ImageDraw
import torch

from physical_content.common import atomic_json, atomic_torch, gpu_guard, load_config
from physical_content.data import TrainingData
from physical_content.teacher import Teacher
from physical_content.wan_loader import ensure_wan_namespace
from physical_content.r18e_reference.evidence import resample_words
from physgen_native.context_alignment import native_text_context
from physgen_native.generator import load_wan, sinusoidal_embedding_1d

REVISION = 'FT-native-whole-video-read-teacher-v1'


def distribution(attention, state, condition, groups):
    heads, dim = attention.num_heads, attention.head_dim
    q = attention.norm_q(attention.q(state)).reshape(1,-1,heads,dim).transpose(1,2).float()
    # Match the independent physical reader: pretrained K weight, no caption bias.
    k = torch.nn.functional.linear(condition,attention.k.weight,None)
    k = attention.norm_k(k).reshape(1,-1,heads,dim).transpose(1,2).float()
    score = (q @ k.transpose(-2,-1)) / math.sqrt(dim)
    probability = score.transpose(-2,-1).softmax(-1).mean(1)[0]
    return torch.stack([probability[groups==group].mean(0)
                        for group in range(int(groups.max())+1)])


def render(row, targets, relations, output):
    # Only visualization of actual training observations, not model conditions.
    spec=importlib.util.spec_from_file_location('_training_frames',
        bootstrap.BASE.parent/'v5.2/physgen_v52/preparation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frames=module.read_frames(row,[0,20,40,60,80])
    maps=targets.float().mean(0)  # mean of the five actual-depth distributions
    canvas=Image.new('RGB',(1600,210*(len(relations)+1)),'#202020')
    draw=ImageDraw.Draw(canvas)
    for col,frame in enumerate(frames):
        canvas.paste(frame.resize((320,184)),(col*320,26))
        draw.text((col*320+4,5),f"real training frame {20*col}",fill='white')
    for group,relation in enumerate(relations):
        low=float(maps[group].min());high=float(maps[group].max())
        for col,index in enumerate([0,5,10,15,20]):
            value=maps[group,index].numpy()
            # Shared clip-wide scale: do not brighten each flat frame separately.
            normalized=(value-low)/max(high-low,1e-12)
            mask=Image.fromarray(np.uint8(np.clip(normalized,0,1)*255)).resize((320,184),Image.Resampling.BILINEAR)
            rgb=np.asarray(frames[col].resize((320,184))).astype(float)
            opacity=np.asarray(mask,dtype=float)[...,None]/255*.55
            overlay=rgb*(1-opacity)+np.array([255,35,0])*opacity
            canvas.paste(Image.fromarray(np.uint8(np.clip(overlay,0,255))),(col*320,210*(group+1)+26))
        draw.text((4,210*(group+1)+5),f"physical relation {group+1}: {relation[:175]}",fill='white')
    canvas.save(output,quality=92)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cfg=load_config(args.config);gpu_guard(args.output)
    selection=json.loads(args.selection.read_text())
    if selection['scope']!='training_only_actual_full_clip' or selection.get('chunk_conditioning',False):
        raise ValueError('Whole-video training evidence only')
    data=TrainingData(cfg);inputs=Teacher(cfg);by_id={r['id']:i for i,r in enumerate(data.rows)}
    requested=[r['id'] for r in selection['cases']]
    if not set(requested)<=set(by_id):raise ValueError('Non-training input in teacher selection')
    ensure_wan_namespace(cfg);torch.set_num_threads(4)
    wan=load_wan(cfg,'cuda');wan.freqs=wan.freqs.cuda()
    sites=cfg['model']['injection_layers'];began=time.monotonic();records=[]
    for sid in requested:
        dest=args.output/sid;dest.mkdir(exist_ok=True)
        sample=data.get(by_id[sid]);evidence=inputs.input(sample,drop=False)
        source,groups,_=resample_words(evidence['native_condition'],evidence['word_groups'],256)
        sigma=.1
        seed=int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)
        noise=torch.randn(sample['latent'].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
        latent=(1-sigma)*sample['latent']+sigma*noise
        values=[]
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            x=wan.patch_embedding(latent);geometry=tuple(x.shape[2:]);x=x.flatten(2).transpose(1,2)
            context=native_text_context(wan,sample['text'])
            with torch.autocast('cuda',enabled=False):
                e=wan.time_embedding(sinusoidal_embedding_1d(wan.freq_dim,
                    torch.full((x.shape[1],),sigma*1000,device='cuda')).float()[None])
                mod=wan.time_projection(e).unflatten(-1,(6,wan.dim))
            grid=torch.tensor([geometry],dtype=torch.long);lengths=torch.tensor([x.shape[1]],dtype=torch.long)
            for index,block in enumerate(wan.blocks):
                x=block(x,mod,lengths,grid,wan.freqs,context,None)
                if index in sites:
                    value=distribution(block.cross_attn,block.norm3(x),source,groups)
                    values.append(value.reshape(len(evidence['relations']),*geometry).cpu())
        target=torch.stack(values)
        probabilities=target.flatten(2).clamp_min(1e-20)
        entropy=-(probabilities*probabilities.log()).sum(-1)/math.log(probabilities.shape[-1])
        record=dict(id=sid,revision=REVISION,sites=sites,geometry=geometry,
            relations=evidence['relations'],caption=sample['caption'],source_word_strings=evidence['word_strings'],
            actual_training_frame_indices=sample['row']['indices'],sigma=sigma,seed=seed,
            teacher='official frozen Wan without LoRA or physical modules',
            distribution='per-C-slot spatial softmax, then mean heads and source-relation slots',
            normalized_entropy_by_site_and_relation=entropy.tolist(),
            max_over_uniform=(probabilities.max(-1).values*probabilities.shape[-1]).tolist(),
            training_only=True,bounding_boxes=False,chunk_conditioning=False,
            approved_for_training=False,video_quality_proven=False)
        atomic_torch(dict(**record,distribution_targets=target.to(torch.bfloat16)),dest/'target.pt')
        atomic_json(record,dest/'record.json')
        render(sample['row'],target,evidence['relations'],dest/'panel.jpg')
        records.append(record)
        atomic_json(dict(completed=len(records),total=len(requested),seconds=time.monotonic()-began),args.output/'progress.json')
        print(json.dumps(dict(id=sid,completed=len(records),mean_normalized_entropy=float(entropy.mean()),
                              max_over_uniform=float((probabilities.max(-1).values*probabilities.shape[-1]).max()))),flush=True)
    atomic_json(dict(complete=True,revision=REVISION,count=len(records),
        seconds=time.monotonic()-began,approved_for_training=False,records=records),args.output/'complete.json')


if __name__=='__main__':main()
