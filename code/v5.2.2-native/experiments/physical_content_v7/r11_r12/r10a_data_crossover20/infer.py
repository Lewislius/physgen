"""Change inference evidence only; retain the checkpoint's complete trained state."""
from pathlib import Path
import argparse,json,os,random,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import WORK,OLD,API,CHECKPOINTS,ARMS,cases,setup
import numpy as np
import torch

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=list(ARMS),required=True);args=parser.parse_args()
    trained,evidence_name=ARMS[args.arm];checkpoint=CHECKPOINTS[trained]
    setup(OLD if trained=='old' else API)
    from physical_scaling.common import gpu_guard,atomic_json
    from physical_content.model import PhysicalModel
    from physgen_native.decoder import VideoDecoder
    from protocol import load_inputs,load_noise,verify_video
    from generate import write_video
    root=WORK/'outputs'/args.arm;root.mkdir(parents=True,exist_ok=True)
    gpu_guard(root/'allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);cfg=saved['config']
    if saved['step']!=400:raise ValueError('Use complete matched400 checkpoints')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    model=PhysicalModel(cfg);params=dict(model.named_parameters());buffers=dict(model.named_buffers())
    if set(saved['model'])!=set(model.learned_names):raise ValueError('All original trained parameters must be restored')
    for name,value in saved['model'].items():params[name].copy_(value)
    for name,value in saved['constants'].items():buffers[name].copy_(value)
    del saved
    model.eval().requires_grad_(False);decoder=VideoDecoder(cfg).eval()
    plan,selected=cases();folder=root/'videophy40';folder.mkdir(exist_ok=True)
    plan.update(cases=selected,checkpoint=str(checkpoint),source_checkpoint=str(checkpoint),checkpoint_step=400,
        revision=cfg['revision'],training_source=cfg['source_snapshot'],training_data=trained,inference_evidence=evidence_name,
        training_experiment_id=checkpoint.parent.name.split('_')[-1],
        mode=args.arm,generation_seed=42,scope='Existing fixed16 plus four score-selected process failures;20 paired checkpoint-by-inference-reasoning cases, original I2V inputs, no retraining.',
        intervention_scope='Both positive and negative Qwen reasoning inputs changed; checkpoint alignment buffers, all weights, native T5, image, noise and sampler retained.')
    atomic_json(plan,folder/'manifest.json')
    policy=cfg['inference']
    if policy!=dict(steps=50,shift=5.,guidance=5.,sampler='shifted_euler'):raise ValueError('Keep gamma5 and original sampler')
    for case in selected:
        output=folder/'cases'/case['id']/'planned.mp4';output.parent.mkdir(parents=True,exist_ok=True)
        provenance=dict(checkpoint=str(checkpoint),step=400,mode=args.arm,case=case['id'],seed=42,
            inference_source=str(Path(__file__).resolve().parent),inference_evidence=evidence_name,guidance=plan['guidance'])
        if output.exists() and output.with_suffix('.json').exists():
            old=json.loads(output.with_suffix('.json').read_text())
            if any(old.get(k)!=v for k,v in provenance.items()):raise ValueError('Different generation provenance')
            continue
        begin=time.monotonic();inputs,canvas=load_inputs(case)
        sample={k:v.cuda() for k,v in inputs.items()};sample['times']=sample['times'][None]
        def read(arm):
            value=torch.load(WORK/'inputs'/case['id']/f'{evidence_name}_{arm}.pt',map_location='cpu',weights_only=True)
            if (value['first_image']!=str(canvas.resolve()) or value['sample']!='videophy40_'+case['id'] or value['encoding']!=evidence_name
                or value['caption']!=('' if arm=='negative' else case['caption']) or value['scope']!='first_image_and_request_only'):
                raise ValueError('Wrong evidence identity')
            return {k:v.cuda() if torch.is_tensor(v) else v for k,v in value.items()}
        positive,negative=read('positive'),read('negative')
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            c,groups=model.enhancement(positive);nc,ngroups=model.enhancement(negative)
            noise=load_noise(case)
            if noise is None:raise ValueError('Use persisted paired seed42 noise')
            x=noise.cuda().clone();x[:,:,:1]=sample['first']
            grid=torch.linspace(1,0,policy['steps']+1,device='cuda');grid=policy['shift']*grid/(1+(policy['shift']-1)*grid)
            for index in range(policy['steps']):
                pos,_=model.velocity(x,sample['first'],sample['text'],grid[index],c,sample['times'],groups)
                neg,_=model.velocity(x,sample['first'],sample['null_text'],grid[index],nc,sample['times'],ngroups)
                x=x+(grid[index+1]-grid[index])*(neg+5.*(pos-neg));x[:,:,:1]=sample['first']
            frames=decoder(x)
        frames=((frames.permute(0,2,3,1).clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
        write_video(frames,output);verify_video(output)
        atomic_json(dict(**provenance,caption=case['caption'],canvas=str(canvas),sampler=policy,
            positive_analysis=positive.get('analysis'),positive_relations=positive['relations'],negative_analysis=negative.get('analysis'),
            positive_evidence=positive['source'],negative_evidence=negative['source'],seconds=time.monotonic()-begin),output.with_suffix('.json'))
        print(json.dumps(dict(case=case['id'],arm=args.arm,seconds=time.monotonic()-begin)),flush=True)
        del sample,positive,negative,c,nc,noise,x,pos,neg,frames
    atomic_json(dict(status='completed',cases=20,checkpoint=str(checkpoint),inference_evidence=evidence_name),root/'status.json')

if __name__=='__main__':main()
