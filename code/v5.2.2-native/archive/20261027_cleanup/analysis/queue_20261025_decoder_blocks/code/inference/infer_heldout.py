"""Normal CFG/Euler generation from held-out first frames and existing text."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from physgen_native.data import VideoDataset, load_tensor, move
from physgen_native.generator import load_wan
from physgen_native.system import PhysGenNative
from physgen_native.runtime import load_checkpoint, restore
from train.train import export_video


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output')
    parser.add_argument('--samples',type=int,default=1,help='Number of distinct inputs, one video for each input')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--guidance',type=float)
    parser.add_argument('--condition-strength',type=float,default=1.)
    parser.add_argument('--without-reasoning',action='store_true')
    parser.add_argument('--without-latent-reasoning',action='store_true')
    parser.add_argument('--refine-c',action='store_true',help='Verifier feedback within the single sampling trajectory')
    parser.add_argument('--c-step-size',type=float,default=.1)
    parser.add_argument('--c-feedback-sigmas',type=float,nargs='+',default=[.85,.65])
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('Inference requires an allocated GPU')
    torch.cuda.set_device(0)
    state=load_checkpoint(args.checkpoint); cfg=state['config']
    model=PhysGenNative(cfg,load_wan(cfg),device='cuda',training_observers=args.refine_c).cuda()
    completed,_=restore(model,None,None,state,cfg)
    del state
    model.eval().requires_grad_(False)
    cfg['inference']['latent_feedback'] = dict(enabled=args.refine_c,step_size=args.c_step_size,sigmas=args.c_feedback_sigmas)
    if args.steps is not None: cfg['inference']['steps']=args.steps
    if args.guidance is not None: cfg['inference']['guidance']=args.guidance
    cfg['inference']['condition_strength']=args.condition_strength
    if args.without_reasoning: cfg['knowledge']['generation_reasoning']=False
    if args.without_latent_reasoning: cfg['knowledge']['latent_reasoning_steps']=0
    data=VideoDataset(cfg,'validation',targets=False)
    null=load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
    output=Path(args.output) if args.output else Path(cfg['paths']['outputs'])/Path(args.checkpoint).parent.name/'heldout'
    for index in range(min(args.samples,len(data))):
        sample=move(data[index],'cuda')
        path=output/f'{sample["id"]}.mp4'
        if path.exists(): raise FileExistsError(path)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            prediction=model.sample(model.conditions(sample),sample['text'],null,seed=args.seed+index)
        export_video(model,prediction,sample,path,completed,args.seed+index)
        print(path,flush=True)


if __name__=='__main__': main()
