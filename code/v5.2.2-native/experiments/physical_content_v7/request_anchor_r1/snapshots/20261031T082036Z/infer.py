"""Paired generation with exact saved inputs/noise and positive-C interventions."""
from pathlib import Path
import sys,argparse,json,os,time,copy,random
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch,numpy as np
from guidance import combine,description

PINNED=bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
CONTROL_CASES=['VP004','VP010','VP015','VP034']
DONORS={'VP004':'VP010','VP010':'VP034','VP015':'VP004','VP034':'VP015'}

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--suite',choices=['videophy40','demo21','heldout8'],default='videophy40')
    parser.add_argument('--modes',nargs='+',default=['real']);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--physical-guidance',type=float,required=True)
    parser.add_argument('--global-condition',type=Path)
    parser.add_argument('--request-evidence',type=Path)
    parser.add_argument('--cases',nargs='*');args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    from physical_scaling.common import gpu_guard,atomic_json
    gpu_guard(root/'allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=state['config'];step=state['step'];variant=cfg['variant']
    if not np.isfinite(args.physical_guidance) or args.physical_guidance<0:
        raise ValueError('Physical guidance must be a finite nonnegative scale')
    guidance=description(cfg['inference']['guidance'],args.physical_guidance)
    sys.path.insert(0,cfg['source_snapshot'])
    from physical_content.model import PhysicalModel
    from physical_content.teacher import Teacher
    from physgen_native.decoder import VideoDecoder
    sys.path.insert(0,str(PINNED))
    from protocol import manifest_for,load_inputs,load_noise,verify_video
    from generate import write_video
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    def log(kind,**kw):
        row=dict(kind=kind,time=time.time(),**kw);print(json.dumps(row,ensure_ascii=False),flush=True)
        with (root/'generation.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    log('loading',checkpoint=args.checkpoint)
    model=PhysicalModel(cfg)
    named=dict(model.named_parameters());buffers=dict(model.named_buffers())
    if set(state['model'])!=set(model.learned_names):raise ValueError('Checkpoint parameter ownership mismatch')
    for name,value in state['model'].items():named[name].copy_(value)
    for name,value in state['constants'].items():buffers[name].copy_(value)
    del state
    model.eval().requires_grad_(False);teacher=Teacher(cfg);decoder=VideoDecoder(cfg).eval()
    global_condition=None
    if 'global_train_mean' in args.modes:
        if variant!='physical_content' or args.global_condition is None:
            raise ValueError('global_train_mean requires a physical checkpoint and --global-condition')
        global_condition=torch.load(args.global_condition,map_location='cpu',weights_only=True)
        if global_condition['checkpoint']!=str(Path(args.checkpoint).resolve()) or global_condition['split']!='train':
            raise ValueError('Global condition must come from this checkpoint and training inputs')
        global_condition={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in global_condition.items()}
    reference=(json.loads((bootstrap.BASE/'experiments/physical_content_v7/heldout8/manifest.json').read_text())
               if args.suite=='heldout8' else manifest_for('v6_assoc',args.suite))
    saved_outputs={key:module.output.weight.detach().clone() for key,module in model.fusions.items()}
    original_project=model.enhancement.project.weight.detach().clone() if variant=='physical_content' else None
    for mode in args.modes:
        if mode not in ['real','swapped','mean','image_only','no_c','anchor_only','reasoning_swapped','empty_reasoning','global_train_mean','request_anchored']:raise ValueError(mode)
        if mode=='request_anchored' and args.request_evidence is None:
            raise ValueError('request_anchored requires its own prepared input cache')
        directory=root/args.suite if mode=='real' else root/'controls'/mode/args.suite
        directory.mkdir(parents=True,exist_ok=True)
        selected=args.cases or ([case['id'] for case in reference['cases']] if mode=='real' else CONTROL_CASES)
        plan=copy.deepcopy(reference);plan['cases']=[c for c in plan['cases'] if c['id'] in selected]
        if len(plan['cases'])!=len(reference['cases']):
            plan['source_scope']=reference['scope']
            plan['scope']=f'Fixed{len(plan["cases"])} diagnostic subset of the existing {args.suite} shared-first-image protocol'
        plan.update(checkpoint=args.checkpoint,source_checkpoint=args.checkpoint,checkpoint_step=step,
            revision=cfg['revision'],variant=variant,training_source=cfg['source_snapshot'],mode=mode,
            training_experiment_id=Path(args.checkpoint).parent.name.split('_')[-1],generation_seed=args.seed,
            guidance=guidance,
            intervention_scope='positive C only for swapped/mean/image_only; native T5 and own negative C fixed',
            conditioning='Input-grounded Q/C with fresh native semantic interface' if variant=='physical_content' else 'Native text/image with full-budget Wan LoRA',
            architecture=dict(c_tokens=256 if variant=='physical_content' else 0,enhancement_width=4096,wan_width=3072,
                              fusion_layers=list(range(3,31,3)) if variant=='physical_content' else []))
        for case in plan['cases']:case['generation_seed']=args.seed
        if mode=='global_train_mean':
            plan['global_condition']=str(args.global_condition.resolve())
            plan['intervention_scope']='Both positive and negative C fixed to training means; no sample-specific Qwen information; native T5/first image retained'
        if mode=='request_anchored':
            plan['request_evidence']=str(args.request_evidence.resolve())
            plan['intervention_scope']='All original reasoning retained in positive Qwen memory; the fourth C anchor uses the verbatim requested future process. Four C groups, Q256, original negative condition, native T5 and first image retained.'
        atomic_json(plan,directory/'manifest.json')
        for key,module in model.fusions.items():module.output.weight.copy_(saved_outputs[key] if mode!='no_c' else torch.zeros_like(saved_outputs[key]))
        if original_project is not None:
            model.enhancement.project.weight.copy_(original_project if mode!='anchor_only' else torch.zeros_like(original_project))
        completed=[]
        for case in plan['cases']:
            output=directory/'cases'/case['id']/'planned.mp4';output.parent.mkdir(parents=True,exist_ok=True)
            provenance=dict(checkpoint=args.checkpoint,step=step,mode=mode,case=case['id'],seed=args.seed,
                guidance=guidance)
            if mode=='global_train_mean':provenance['global_condition']=str(args.global_condition.resolve())
            if mode=='request_anchored':provenance['request_evidence']=str(args.request_evidence.resolve())
            if output.is_file() and output.with_suffix('.json').is_file():
                previous=json.loads(output.with_suffix('.json').read_text())
                if any(previous.get(k)!=v for k,v in provenance.items()):raise ValueError('Output provenance differs')
                completed.append(case['id']);continue
            begin=time.monotonic()
            if args.suite=='heldout8':
                folder=Path(case['prepared_inputs']);stored=torch.load(folder/'conditions.pt',map_location='cpu',weights_only=True)
                inputs={key:stored[key] for key in ['first','text','null_text','times','valid']};canvas=Path(case['canvas'])
            else:inputs,canvas=load_inputs(case)
            sample={key:value.cuda() for key,value in inputs.items()};sample['times']=sample['times'][None]
            sample.update(id=case['id'] if args.suite=='heldout8' else f'{args.suite}_{case["id"]}',caption=case['caption'])
            if mode=='global_train_mean':
                positive=dict(analysis=None,relations=None);negative=dict(analysis=None,relations=None)
            elif args.suite=='heldout8' and variant=='physical_content':
                def read_evidence(name):
                    value=torch.load(folder/name,map_location='cpu',weights_only=True)
                    if value['sample']!=sample['id'] or value['scope']!='first_image_and_request_only':raise ValueError('Invalid held-out condition')
                    return {k:(v.cuda() if torch.is_tensor(v) else v) for k,v in value.items()}
                positive=read_evidence('positive.pt');negative=read_evidence('negative.pt')
            else:positive=teacher.input(sample);negative=teacher.input(sample,drop=True)
            if mode=='swapped':
                donor=copy.copy(sample);donor['id']=f'{args.suite}_{DONORS[case["id"]]}'
                positive=teacher.input(donor)
            if mode=='image_only':positive=negative
            if mode in ['reasoning_swapped','empty_reasoning']:
                source=bootstrap.BASE/'experiments/physical_content_v7/cache/reasoning_controls'/mode/f'{case["id"]}.pt'
                positive=torch.load(source,map_location='cpu',weights_only=True)
                if positive['caption']!=case['caption'] or positive['sample']!=sample['id']:
                    raise ValueError('Reasoning intervention must retain the own image/request')
                positive={key:(value.cuda() if torch.is_tensor(value) else value) for key,value in positive.items()}
            if mode=='request_anchored':
                source=args.request_evidence/f'{case["id"]}.pt'
                positive=torch.load(source,map_location='cpu',weights_only=True)
                if (positive['caption']!=case['caption'] or positive['sample']!=sample['id']
                        or positive['scope']!='first_image_and_request_only'
                        or positive['intervention']!='request_anchored'
                        or sorted(positive['word_groups'].unique().tolist())!=[0,1,2,3]):
                    raise ValueError('Request anchor must retain the own input and four content groups')
                positive={key:(value.cuda() if torch.is_tensor(value) else value) for key,value in positive.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                if mode=='global_train_mean':
                    c=global_condition['positive'];nc=global_condition['negative']
                    groups=ngroups=global_condition['groups']
                else:c,groups=model.enhancement(positive);nc,ngroups=model.enhancement(negative)
                if mode=='mean':c=c.mean(1,keepdim=True).expand_as(c)
                noise=load_noise(case) if args.seed==42 else None
                x=noise.cuda().clone() if noise is not None else torch.randn((1,48,21,30,52),device='cuda',generator=torch.Generator(device='cuda').manual_seed(args.seed))
                x[:,:,:1]=sample['first'];policy=cfg['inference']
                grid=torch.linspace(1,0,policy['steps']+1,device='cuda');grid=policy['shift']*grid/(1+(policy['shift']-1)*grid)
                for index in range(policy['steps']):
                    if args.physical_guidance==policy['guidance'] or variant=='native_lora' or mode=='no_c':
                        pos,_=model.velocity(x,sample['first'],sample['text'],grid[index],c,sample['times'],groups)
                        neg,_=model.velocity(x,sample['first'],sample['null_text'],grid[index],nc,sample['times'],ngroups)
                        velocity=combine(pos,neg,None,policy['guidance'],policy['guidance'])
                    else:
                        neg,_=model.velocity(x,sample['first'],sample['null_text'],grid[index],nc,sample['times'],ngroups)
                        neutral,_=model.velocity(x,sample['first'],sample['text'],grid[index],nc,sample['times'],ngroups)
                        if args.physical_guidance:
                            pos,_=model.velocity(x,sample['first'],sample['text'],grid[index],c,sample['times'],groups)
                        else:pos=neutral
                        velocity=combine(pos,neg,neutral,policy['guidance'],args.physical_guidance)
                        del neutral
                    x=x+(grid[index+1]-grid[index])*velocity;x[:,:,:1]=sample['first']
                    if (index+1)%10==0:log('sampling',case=case['id'],mode=mode,step=index+1)
                frames=decoder(x)
            frames=((frames.permute(0,2,3,1).clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
            if args.suite=='heldout8':
                import imageio.v2 as imageio
                imageio.mimwrite(output,frames,fps=case['fps'],codec='libx264',macro_block_size=None,ffmpeg_params=['-crf','18'])
                if len(frames)!=81:raise ValueError('Expected complete held-out video')
            else:write_video(frames,output);verify_video(output)
            atomic_json(dict(**provenance,caption=case['caption'],canvas=str(canvas),sampler=cfg['inference'],
                positive_analysis=positive.get('analysis'),positive_relations=positive.get('relations'),
                negative_analysis=negative.get('analysis'),donor=DONORS.get(case['id']) if mode=='swapped' else positive.get('donor'),
                scope=plan['intervention_scope'],
                seconds=time.monotonic()-begin),output.with_suffix('.json'))
            completed.append(case['id']);log('video_complete',case=case['id'],mode=mode,seconds=time.monotonic()-begin)
            atomic_json(dict(status='running',mode=mode,completed_cases=len(completed)),directory/'summary.json')
            del sample,positive,negative,c,nc,frames,x,pos,neg,velocity
        atomic_json(dict(status='completed',mode=mode,completed_cases=len(completed)),directory/'summary.json')
    atomic_json(dict(status='completed',checkpoint=args.checkpoint,modes=args.modes),root/'generation_status.json')

if __name__=='__main__':main()
