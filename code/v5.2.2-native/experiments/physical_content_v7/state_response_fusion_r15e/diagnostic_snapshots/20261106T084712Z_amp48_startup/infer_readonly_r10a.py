"""Read-only R10A checkpoint intervention before deciding the R15E training design."""
from pathlib import Path
import argparse,copy,json,os,random,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
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
    print(json.dumps(dict(kind='load_stage',stage='checkpoint_metadata',time=time.time())),flush=True)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=saved['config'];step=saved['step']
    if saved['revision']!='physical_content_v7_execution_value_r10' or step!=400:raise ValueError('Use the immutable original R10A400 checkpoint')
    cfg=copy.deepcopy(cfg)
    cfg['paths']['inference_evidence']=str(bootstrap.BASE/'experiments/physical_content_v7/denoising_matched_fusion_r15d/prepared_inputs')
    cfg['source_snapshot']=str(Path(__file__).resolve().parent)
    cfg['data_revision']='original_wisa4000_and_original_reasoning'
    cfg['model']['fusion_order']='readonly_R10A_direct_write_plus_native_video_response'
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    print(json.dumps(dict(kind='load_stage',stage='official_wan_and_physical_model',time=time.time())),flush=True)
    model=PhysicalModel(cfg);parameters=dict(model.named_parameters());buffers=dict(model.named_buffers())
    print(json.dumps(dict(kind='load_stage',stage='copy_readonly_r10a_weights',time=time.time())),flush=True)
    missing=set(model.learned_names)-set(saved['model'])
    expected={n for n in model.learned_names if n.startswith('fusions.') and ('.query.' in n or '.query_norm.' in n)}
    if missing!=expected or set(saved['model'])-set(model.learned_names):raise ValueError('Unexpected diagnostic weight mapping')
    for name,value in saved['model'].items():parameters[name].copy_(value)
    for name,value in saved['constants'].items():buffers[name].copy_(value)
    del saved
    model.eval().requires_grad_(False)
    print(json.dumps(dict(kind='load_stage',stage='vae_decoder',time=time.time())),flush=True)
    decoder=VideoDecoder(cfg).eval()
    print(json.dumps(dict(kind='load_stage',stage='generation_ready',time=time.time())),flush=True)
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
    scopes=dict(real='Read-only immutable R10A400 with native video response added; independent physical Q copied from frozen native Q; no parameter update',
        no_c='Native generation branch: physical fusion bypassed in both CFG arms',
        video_response_off='Return to original R10A physical/text computation with copied native physical Q',
        reasoning_swapped='Same-count unrelated reasoning with own image/request; own negative condition',
        empty_reasoning='Empty reasoning delimiters with own image/request and own relation count',
        global_train_mean='Fixed train-mean routing/execution C; own relation priors and video states remain input-specific')
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
            diagnostic_only=True,training_updates=0,candidate='R10A_readonly_video_response_probe_NOT_R15E_trained_result',
            data_revision=cfg['data_revision'],inference_evidence=cfg['paths']['inference_evidence'],
            conditioning='Original input-only reasoning and frozen local Qwen3-VL4096; original native T5',
            fusion_order=cfg['model']['fusion_order'],
            architecture=dict(c_tokens=256,enhancement_width=4096,wan_width=3072,fusion_layers=list(range(3,31,3))))
        if mode=='global_train_mean':plan['global_condition']=str(args.global_condition.resolve())
        for case in plan['cases']:case['generation_seed']=42
        atomic_json(plan,directory/'manifest.json')
        completed=[]
        for case in plan['cases']:
            output=directory/'cases'/case['id']/'planned.mp4';output.parent.mkdir(parents=True,exist_ok=True)
            provenance=dict(checkpoint=str(args.checkpoint),step=step,mode=mode,case=case['id'],seed=42,
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
                execution=dict(physical_enabled=mode!='no_c',video_response=mode!='video_response_off')
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
