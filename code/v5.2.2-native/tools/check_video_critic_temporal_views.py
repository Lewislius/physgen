"""Test actual RGB omissions, not post-encoder edits of contextual features.

All views, including the unedited control, receive the same extra codec pass.
An edit is a temporal-evidence intervention, not a universal physics label.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import imageio.v2 as imageio
import numpy as np
import torch
from physgen_native.process_encoder import VideoProcessEncoder
from physgen_native.temporal_critic import TemporalProcessCritic
from physgen_native.runtime import Logger


def read(path):
    reader=imageio.get_reader(path)
    try:return np.stack(list(reader.iter_data())),reader.get_meta_data()['fps']
    finally:reader.close()


def write(frames,path,fps):
    path.parent.mkdir(parents=True,exist_ok=True)
    writer=imageio.get_writer(path,fps=fps,codec='libx264',quality=None,macro_block_size=None,
        ffmpeg_params=['-crf','18','-profile:v','baseline','-pix_fmt','yuv420p'])
    try:
        for frame in frames:writer.append_data(frame)
    finally:writer.close()


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.cuda.set_device(0)
    encoder=VideoProcessEncoder('/home/liuzhirui/model/vjepa2',
        '/home/liuzhirui/model/vjepa2/checkpoint/vjepa2_1_vitG_384.pt',checkpointing=False).eval()
    critics={}
    for context in ['full','pairs']:
        for seed in [42,314159,271828]:
            state=torch.load(args.directory/context/f'critic_temporal_seed{seed}.pt',map_location='cpu',weights_only=True)
            model=TemporalProcessCritic(state['feature_width'],state['condition_width'],state['width']).cuda().eval()
            model.load_state_dict(state['model']);critics[context,seed]=model
    template=json.loads((ROOT/'archive/20261027_cleanup/analysis/independent_video_evaluation_v2/v522-diagnostic-direct_fm-8cases32updates-seed42_65835/step0000_repeat0/manifest.json').read_text())
    manifests={view:dict(cases=[],evaluation=template['evaluation'],scope='Temporal evidence sensitivity on existing WISA validation videos; not a benchmark score')
               for view in ['original','freeze','late_freeze']}
    log=Logger(args.output/'metrics.jsonl')
    for path in sorted((args.directory/'full/features').glob('validation_*.pt')):
        record=torch.load(path,map_location='cpu',weights_only=True)
        rgb,fps=read(args.directory/'reference_videos'/f'{record["id"]}.mp4')
        if len(rgb)!=81:raise ValueError('The declared temporal intervention assumes 81 RGB frames')
        late=rgb.copy();late[49:]=rgb[48]
        views=dict(original=rgb,freeze=np.repeat(rgb[:1],len(rgb),axis=0),late_freeze=late)
        condition,times,valid=[record[k].float().cuda() for k in ['condition','times','valid']]
        for view,array in views.items():
            video=args.output/view/'cases'/record['id']/'planned.mp4';write(array,video,fps)
            observed,_=read(video)
            frames=torch.from_numpy(observed).cuda().permute(0,3,1,2).float()/127.5-1
            manifests[view]['cases'].append(dict(id=record['id'],caption=record['caption']))
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                for context in ['full','pairs']:
                    features=encoder(frames,context=context)
                    for seed in [42,314159,271828]:
                        scores=critics[context,seed](features,condition,times,valid)
                        log.write('view',0,sample=record['id'],view=view,context=context,seed=seed,
                                  score=torch.stack([s.mean() for s in scores.values()]).mean(),
                                  lag_scores={str(h):float(s.mean()) for h,s in scores.items()})
    for view,manifest in manifests.items():
        (args.output/view/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    log.write('complete',0);log.close()


if __name__=='__main__':main()
