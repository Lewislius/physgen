"""Read-only R15B400 joint-memory probe, not trained R15I generation."""
from pathlib import Path
import argparse,copy,json,os,random,sys,time
PROBE_SOURCE=Path(__file__).resolve().parent
B_SOURCE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/request_bound_fusion_r15b/snapshots/20261105T152223Z')
sys.path.insert(0,str(PROBE_SOURCE))
sys.path.insert(0,str(B_SOURCE))
import bootstrap
import numpy as np
import torch
from benchmark_inputs import reference,case_input,load_noise,verify_video
from condition_controls import InferenceCache,mean_with_own_prior
from guidance import description


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',choices=['videophy40'],required=True)
    p.add_argument('--modes',nargs='+',choices=['real'],default=['real'])
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--global-condition',type=Path)
    args=p.parse_args()
    from physical_content.common import gpu_guard,atomic_json,REVISION
    from physical_content.model import PhysicalModel,Fusion
    from joint_read import fusion_forward
    Fusion.forward=fusion_forward
    from physgen_native.decoder import VideoDecoder
    from generate import write_video
    root=args.output;gpu_guard(root/'allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=saved['config'];step=saved['step']
    if saved['revision']!=REVISION or step!=400 or Path(cfg['source_snapshot'])!=B_SOURCE:
        raise ValueError('Use only the immutable R15B400 checkpoint for this read-only probe')
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
    expected={x['id'] for x in json.loads((PROBE_SOURCE/'subset16.json').read_text())['cases']}
    if set(args.cases)!=expected:raise ValueError('Use exactly the fixed16, no case selection')
    scopes=dict(real='R15B400 read-only joint direct-C and request-associated-C memory; original own image/request/reasoning; native text and all learned weights unchanged',
        no_c='Native generation branch: physical fusion bypassed in both CFG arms',
        direct_physical_read='Bypass request binding and restore the R10A direct C read using this checkpoint weights in both CFG arms',
        state_value_off='Bypass only the state-value interaction after the request-bound physical read in both CFG arms',
        reasoning_swapped='Same-count unrelated reasoning with own image/request; own negative condition',
        empty_reasoning='Empty reasoning delimiters with own image/request and own relation count',
        global_train_mean='Fixed train-mean routing/execution C; own relation priors and native request binding remain input-specific')
    def log(kind,**values):
        row=dict(kind=kind,time=time.time(),**values);print(json.dumps(row),flush=True)
        with (root/'generation.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    for mode in args.modes:
        if mode not in scopes:raise ValueError(mode)
        directory=root/args.suite if mode=='real' else root/'controls'/mode/args.suite
        directory.mkdir(parents=True,exist_ok=True)
        plan=copy.deepcopy(base)
        plan.update(cases=copy.deepcopy(selected),suite=args.suite,checkpoint=str(args.checkpoint),source_checkpoint=str(args.checkpoint),
            checkpoint_step=step,revision=cfg['revision'],variant='physical_content',training_source=cfg['source_snapshot'],
            training_experiment_id=args.checkpoint.parent.name.split('_')[-1],mode=mode,generation_seed=42,
            guidance=description(5.,5.),intervention_scope=scopes[mode],
            data_revision=cfg['data_revision'],inference_evidence=cfg['paths']['inference_evidence'],
            conditioning='Original input-only reasoning and frozen local Qwen3-VL4096; original native T5',
            fusion_order=cfg['model']['fusion_order'],
            architecture=dict(c_tokens=256,enhancement_width=4096,wan_width=3072,fusion_layers=list(range(3,31,3))))
        plan.update(inference_revision='r15i_design_joint_memory_probe_v1',probe_source=str(PROBE_SOURCE),training_updates=0,physical_query='unchanged native R15B query',not_trained_R15I=True)
        if mode=='global_train_mean':plan['global_condition']=str(args.global_condition.resolve())
        for case in plan['cases']:case['generation_seed']=42
        atomic_json(plan,directory/'manifest.json')
        completed=[]
        for case in plan['cases']:
            output=directory/'cases'/case['id']/'planned.mp4';output.parent.mkdir(parents=True,exist_ok=True)
            provenance=dict(checkpoint=str(args.checkpoint),step=step,mode=mode,case=case['id'],seed=42,
                inference_revision='r15i_design_joint_memory_probe_v1',probe_source=str(PROBE_SOURCE),training_updates=0,not_trained_R15I=True,
                inference_source=str(Path(__file__).resolve().parent),inference_evidence=cfg['paths']['inference_evidence'],
                guidance=description(5.,5.))
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
                if mode=='global_train_mean':c,groups,nc,ngroups=mean_with_own_prior(mean,groups,ngroups,'cuda')
                noise=load_noise(case)
                if noise is None:raise ValueError('Use the persisted matched seed42 noise')
                x=noise.cuda().clone();x[:,:,:1]=sample['first']
                policy=cfg['inference'];grid=torch.linspace(1,0,policy['steps']+1,device='cuda')
                grid=policy['shift']*grid/(1+(policy['shift']-1)*grid)
                execution=dict(physical_enabled=mode!='no_c',direct_physical_read=mode=='direct_physical_read',state_value_enabled=mode!='state_value_off')
                for index in range(policy['steps']):
                    pos,_=model.velocity(x,sample['first'],sample['text'],grid[index],c,sample['times'],groups,**execution)
                    neg,_=model.velocity(x,sample['first'],sample['null_text'],grid[index],nc,sample['times'],ngroups,**execution)
                    velocity=neg+5.*(pos-neg)
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
            del sample,positive,negative,c,nc,x,pos,neg,velocity,frames
        atomic_json(dict(status='completed',mode=mode,completed_cases=len(completed)),directory/'summary.json')
    atomic_json(dict(status='completed',checkpoint=str(args.checkpoint),modes=args.modes),root/'generation_status.json')


if __name__=='__main__':main()
