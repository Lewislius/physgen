"""Matched I2V evaluation of direct physics and native-request response."""
from pathlib import Path
import argparse,copy,json,os,random,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import numpy as np
import torch
from benchmark_inputs import reference,case_input,load_noise,verify_video
from condition_controls import InferenceCache,mean_with_own_prior
from guidance import description,guided_velocity
from completed_suite import completion


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',choices=['videophy40','videophy60'],required=True)
    p.add_argument('--modes',nargs='+',default=['real'])
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--global-condition',type=Path)
    args=p.parse_args()
    from physical_content.common import gpu_guard,atomic_json,REVISION
    from physical_content.model import PhysicalModel
    from physgen_native.decoder import VideoDecoder
    from generate import write_video
    root=args.output;gpu_guard(root/'allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=saved['config'];step=saved['step']
    if saved['revision']!=REVISION or step!=400:raise ValueError('Use this candidate own400 weights')
    pending=[]
    for mode in args.modes:
        directory=root/args.suite if mode=='real' else root/'controls'/mode/args.suite
        generated,_=completion(directory,args.checkpoint,cfg['source_snapshot'],args.cases,mode)
        if generated:print(json.dumps(dict(status='reused_complete_generation',mode=mode,cases=len(args.cases))),flush=True)
        else:pending.append(mode)
    if not pending:return
    args.modes=pending
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    model=PhysicalModel(cfg);parameters=dict(model.named_parameters());buffers=dict(model.named_buffers())
    if set(saved['model'])!=set(model.learned_names):raise ValueError('Checkpoint ownership mismatch')
    for name,value in saved['model'].items():parameters[name].copy_(value)
    for name,value in saved['constants'].items():buffers[name].copy_(value)
    del saved
    model.eval().requires_grad_(False);decoder=VideoDecoder(cfg).eval()
    reader=InferenceCache(cfg['paths']['inference_evidence'])
    mean=None
    if 'global_train_mean' in args.modes:
        if args.global_condition is None:raise ValueError('A fixed train mean is required')
        mean=torch.load(args.global_condition,map_location='cpu',weights_only=True)
        if (mean['checkpoint']!=str(args.checkpoint.resolve()) or mean['split']!='train'
            or mean['kind']!='fixed_mean_c_own_relation_prior'):raise ValueError('Wrong training mean')
    base=reference()
    selected=[case for case in base['cases'] if case['id'] in args.cases]
    if len(selected)!=len(set(args.cases)):raise ValueError('Case selection differs')
    scopes=dict(real='Own first image, original request and original Qwen reasoning; the declared direct/request-bound/joint global execution writes and the R15C local video-state operator are all active',
        no_c='Native generation branch: the complete physical local and global operator is bypassed in both CFG arms',
        condition_off='Local video operator stays active but every C-derived local and global path is zeroed',
        global_off='Retain the full local video-state operator and its direct C query condition; remove every state-gated global execution write',
        local_off='Retain every declared state-gated global execution write; remove only the local video-state operator output',
        global_direct_off='Remove only the direct-C global execution write',
        global_request_off='Remove only the request-bound-C evidence bank from the single physical read',
        global_literal_off='Remove only the native-T5 literal evidence bank from the single physical read',
        iterative_off='Disable only the second evidence-read query refinement',
        execution_condition_off='Remove only the full-width T5 request-conditioned execution-C residual',
        local_t5_condition_off='Remove only the zero-init full-width T5 local-Q condition',
        global_joint_off='Remove only the joint direct+request-C global execution write',
        global_single_off='Remove only the R1-style single-C global execution read',
        local_request_condition_off='Remove only the zero-init full-width request local-Q correction',
        local_joint_condition_off='Remove only the zero-init full-width joint local-Q correction',
        value_correction_off='Remove only the zero-init full-width request-bound local V correction',
        local_value_correction_off='Remove only the gated zero-init full-width request-bound local V correction',
        temporal_off='Remove only the local operator temporal edges; retain spatial local attention and all global writes',
        reasoning_swapped='Same-count unrelated reasoning with own image/request; own negative condition',
        empty_reasoning='Empty reasoning delimiters with own image/request and own relation count',
        global_train_mean='Fixed train-mean routing/execution C; own relation priors and video states remain input-specific')
    def log(kind,**values):
        row=dict(kind=kind,time=time.time(),**values);print(json.dumps(row),flush=True)
        with (root/'generation.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    for mode in args.modes:
        if mode not in scopes:raise ValueError(mode)
        composition=cfg['inference']['guidance_composition']
        directory=root/args.suite if mode=='real' else root/'controls'/mode/args.suite
        directory.mkdir(parents=True,exist_ok=True)
        plan=copy.deepcopy(base)
        plan.update(cases=copy.deepcopy(selected),suite=args.suite,checkpoint=str(args.checkpoint),source_checkpoint=str(args.checkpoint),
            checkpoint_step=step,revision=cfg['revision'],variant='physical_content',training_source=cfg['source_snapshot'],
            training_experiment_id=args.checkpoint.parent.name.split('_')[-1],mode=mode,generation_seed=42,
            guidance=description(composition,5.),intervention_scope=scopes[mode],
            data_revision=cfg['data_revision'],inference_evidence=cfg['paths']['inference_evidence'],
            conditioning='Original frozen-Qwen contextual physical and literal-request words; original native T5 request',
            fusion_order=cfg['model']['fusion_order'],
            global_execution=cfg['model']['global_execution'],
            architecture=dict(c_tokens=256,enhancement_width=4096,wan_width=3072,
                fusion_layers=list(range(3,31,3)),
                local_execution_operator=cfg['model']['local_execution_operator'],
                global_execution_gate=cfg['model']['global_execution_gate']))
        if mode=='global_train_mean':plan['global_condition']=str(args.global_condition.resolve())
        for case in plan['cases']:case['generation_seed']=42
        atomic_json(plan,directory/'manifest.json')
        completed=[]
        for case in plan['cases']:
            output=directory/'cases'/case['id']/'planned.mp4';output.parent.mkdir(parents=True,exist_ok=True)
            provenance=dict(checkpoint=str(args.checkpoint),step=step,mode=mode,case=case['id'],seed=42,
                inference_source=str(Path(__file__).resolve().parent),inference_evidence=cfg['paths']['inference_evidence'],
                guidance=description(composition,5.))
            if mode=='global_train_mean':provenance['global_condition']=str(args.global_condition.resolve())
            if output.is_file() and output.with_suffix('.json').is_file():
                previous=json.loads(output.with_suffix('.json').read_text())
                if any(previous.get(k)!=v for k,v in provenance.items()):raise ValueError('Different output provenance')
                completed.append(case['id']);continue
            begin=time.monotonic();inputs,item=case_input(case)
            sample={key:value.cuda() for key,value in inputs.items()};sample['times']=sample['times'][None]
            positive=reader.input(item,image=item['image'],duration=item['duration'],mode=mode)
            negative=reader.input(item,image=item['image'],duration=item['duration'],drop=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                c,groups=model.enhancement(positive)
                nc,ngroups=model.enhancement(negative)
                if mode=='global_train_mean':
                    literal=c.get('literal_request'); nliteral=nc.get('literal_request')
                    c,groups,nc,ngroups=mean_with_own_prior(mean,groups,ngroups,'cuda')
                    if literal is not None:c['literal_request']=literal
                    if nliteral is not None:nc['literal_request']=nliteral
                noise=load_noise(case)
                if noise is None:raise ValueError('Use the persisted matched seed42 noise')
                x=noise.cuda().clone();x[:,:,:1]=sample['first']
                policy=cfg['inference'];grid=torch.linspace(1,0,policy['steps']+1,device='cuda')
                grid=policy['shift']*grid/(1+(policy['shift']-1)*grid)
                for index in range(policy['steps']):
                    velocity=guided_velocity(model,x,sample['first'],sample['text'],sample['null_text'],grid[index],
                        c,nc,sample['times'],groups,ngroups,composition=composition,guidance=5.,mode=mode)
                    x=x+(grid[index+1]-grid[index])*velocity;x[:,:,:1]=sample['first']
                    if (index+1)%10==0:log('sampling',case=case['id'],mode=mode,step=index+1)
                frames=decoder(x)
            frames=((frames.permute(0,2,3,1).clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
            write_video(frames,output);verify_video(output)
            atomic_json(dict(**provenance,caption=case['caption'],canvas=item['image'],sampler=cfg['inference'],
                positive_analysis=positive.get('analysis'),positive_relations=positive.get('relations'),
                negative_analysis=negative.get('analysis'),donor=positive.get('donor'),scope=scopes[mode],
                seconds=time.monotonic()-begin),output.with_suffix('.json'))
            completed.append(case['id']);log('video_complete',case=case['id'],mode=mode,seconds=time.monotonic()-begin)
            atomic_json(dict(status='running',mode=mode,completed_cases=len(completed)),directory/'summary.json')
            del sample,positive,negative,c,nc,x,velocity,frames
        atomic_json(dict(status='completed',mode=mode,completed_cases=len(completed)),directory/'summary.json')
    atomic_json(dict(status='completed',checkpoint=str(args.checkpoint),modes=args.modes),root/'generation_status.json')


if __name__=='__main__':main()
