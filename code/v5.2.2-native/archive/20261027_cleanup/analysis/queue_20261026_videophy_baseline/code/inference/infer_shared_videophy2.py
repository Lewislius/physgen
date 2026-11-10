"""Replay the existing 60 VideoPhy2 inputs and exact saved video noise.

This is the project's controlled I2V comparison, not the official full T2V leaderboard.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from physgen_native.data import load_tensor,move
from physgen_native.generator import load_wan
from physgen_native.system import PhysGenNative
from physgen_native.runtime import load_checkpoint,restore
from train.train import export_video


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--source',default=str(ROOT.parent/'v5.2/inference_outputs/videophy2_lora_run64543_joint400_60_seed42'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--condition-strength',type=float,default=1.)
    parser.add_argument('--without-reasoning',action='store_true')
    parser.add_argument('--without-latent-reasoning',action='store_true')
    parser.add_argument('--refine-c',action='store_true',help='Verifier feedback within the single sampling trajectory')
    parser.add_argument('--c-step-size',type=float,default=.1)
    parser.add_argument('--c-feedback-sigmas',type=float,nargs='+',default=[.85,.65])
    parser.add_argument('--noise-seed',type=int,help='New paired evaluation seed; default reuses the exact historical noise')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--case-ids-file',help='Ordered JSON case IDs fixed before candidate scoring')
    parser.add_argument('--score-at-end',action='store_true',help='Replace this process with the existing independent evaluation interpreter')
    parser.add_argument('--paired-protocol',help='Compare completed candidate scores with the fixed historical references')
    args=parser.parse_args(); source=Path(args.source); out=Path(args.output)
    original=json.loads((source/'manifest.json').read_text()); settings=original['settings']
    state=load_checkpoint(args.checkpoint); cfg=state['config']; torch.cuda.set_device(0)
    model=PhysGenNative(cfg,load_wan(cfg),device='cuda',training_observers=args.refine_c).cuda()
    completed,_=restore(model,None,None,state,cfg);del state
    model.eval().requires_grad_(False)
    cfg['inference']['latent_feedback'] = dict(cfg['inference'].get('latent_feedback',{}),
        enabled=args.refine_c,step_size=args.c_step_size,sigmas=args.c_feedback_sigmas)
    cfg['inference'].update(steps=settings['steps'],shift=settings['shift'],guidance=settings['guidance'],condition_strength=args.condition_strength)
    if args.without_reasoning: cfg['knowledge']['generation_reasoning']=False
    if args.without_latent_reasoning: cfg['knowledge']['latent_reasoning_steps']=0
    cases=[dict(c) for c in original['cases'][:args.limit]]
    if args.case_ids_file:
        if args.limit is not None: raise ValueError('Use either a fixed case list or a prefix limit')
        ids=json.loads(Path(args.case_ids_file).read_text());available={c['id']:c for c in cases}
        if len(set(ids))!=len(ids) or not set(ids)<=set(available):raise ValueError('Invalid fixed case selection')
        cases=[available[i] for i in ids]
    if args.noise_seed is not None:
        for index,case in enumerate(cases):case['seed']=args.noise_seed+index
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(cases=cases,settings=settings,evaluation=original['evaluation'],checkpoint=str(Path(args.checkpoint).resolve()),
                  source=str(source),source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
                  scope='existing shared I2V subset; not official full T2V leaderboard',
                  condition_strength=args.condition_strength,reasoning=not args.without_reasoning,
                  latent_reasoning_steps=cfg['knowledge'].get('latent_reasoning_steps',0),
                  latent_feedback=cfg['inference']['latent_feedback'],
                  evaluation_noise_seed=args.noise_seed,
                  initial_observation='decode of the same saved first latent',training_cache_unchanged=True)
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=manifest:raise ValueError('Output belongs to a different evaluation')
    else:(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    for case in cases:
        output=out/'cases'/case['id']/'planned.mp4'
        if output.exists() and output.with_suffix('.json').exists():continue
        folder=source/'cases'/case['id']; tensors=move(load_tensor(folder/'conditions.pt'),'cuda')
        clauses=json.loads((folder/'requirements.json').read_text())['requirements']
        conditions={k:tensors[k] for k in ('first','times','valid')}
        conditions.update(caption=case['caption'],clauses=[dict(text=x,basis='both',phase='evolution') for x in clauses])
        noise=load_tensor(folder/'noise.pt').cuda()
        if args.noise_seed is not None:
            noise=torch.randn(noise.shape,device='cuda',dtype=torch.float32,
                              generator=torch.Generator(device='cuda').manual_seed(case['seed']))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            latent=model.sample(conditions,tensors['text'],tensors['null_text'],seed=case['seed'],initial_noise=noise)
        export_video(model,latent,dict(conditions,id=case['id']),output,completed,case['seed'])
        print(case['id'],flush=True)
    if args.score_at_end:
        # exec replaces the generation process; no extra GPU worker or rank.
        import os
        python='/home/liuzhirui/miniconda3/envs/physvid/bin/python'
        command=[python,'-I','-B',str(ROOT/'tools/evaluate_videophy2.py'),str(out)]
        if args.paired_protocol:command+=['--paired-protocol',args.paired_protocol]
        os.execv(python,command)


if __name__=='__main__':main()
