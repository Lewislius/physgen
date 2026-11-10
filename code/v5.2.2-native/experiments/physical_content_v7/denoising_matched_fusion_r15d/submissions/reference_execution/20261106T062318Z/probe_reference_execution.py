"""Read-only matched two-case execution diagnostics for useful historical checkpoints."""
from pathlib import Path
import argparse,json,os,sys,time
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
CURRENT=ROOT/'denoising_matched_fusion_r15d'
REFERENCES={
 'R1':(ROOT/'snapshots/20261030T141432Z',ROOT/'runs/physical_content_66778/step0400.pt'),
 'R10A':(ROOT/'execution_value_r10/snapshots/20261101T034557Z',ROOT/'execution_value_r10/runs/physical_content_67003/step0400.pt'),
 'R15B':(ROOT/'request_bound_fusion_r15b/snapshots/20261105T152223Z',ROOT/'request_bound_fusion_r15b/runs/physical_content_67829/step0400.pt')}

def cosine(a,b):return float(F.cosine_similarity(a.float().flatten(),b.float().flatten(),dim=0))
def pair_cosine(x):
    x=F.normalize(x.float().reshape(-1,x.shape[-1]),dim=-1);n=len(x)
    return float((x.sum(0).square().sum()-n)/(n*(n-1)))

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',choices=REFERENCES,required=True);p.add_argument('--describe',action='store_true');a=p.parse_args()
    source,checkpoint=REFERENCES[a.reference]
    plan=dict(reference=a.reference,source=str(source),checkpoint=str(checkpoint),cases=['VP004','VP038'],steps=[1,15,35,50],layers=[2,14,29],seed=42,gamma=5,training_updates=0,output_videos=0,
        scope='Own normal trajectory replay. Full and no-C velocities at identical latent within each model; between-model latent trajectories differ. Input-only original evidence and fixed images/noise reused. Read-only checkpoint, no API/model downloads.',
        limitation='Two diagnosed failure cases are not a population quality estimate. A common physical write can be useful; native conditional guidance may itself be wrong. Compare mechanism alongside saved videos and scores.')
    if a.describe:print(json.dumps(plan));return
    sys.path.insert(0,str(source));import bootstrap
    from physical_content.model import PhysicalModel
    from physical_content.common import gpu_guard,atomic_json
    # Resolve each historical model before adding only input-reader utilities.
    sys.path.append(str(CURRENT/'snapshots/20261106T011032Z'))
    sys.path.insert(0,str(CURRENT/'snapshots/20261106T011032Z/data_tools'))
    from benchmark_inputs import reference,case_input,load_noise
    from condition_controls import InferenceCache
    folder=CURRENT/'reports/control_helpers'/os.environ['DET_EXPERIMENT_ID']
    gpu_guard(folder);atomic_json(plan,folder/(a.reference+'_plan.json'))
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);assert saved['step']==400
    model=PhysicalModel(saved['config']).eval().requires_grad_(False)
    parameters=dict(model.named_parameters());buffers=dict(model.named_buffers())
    assert set(saved['model'])==set(model.learned_names)
    for name,value in saved['model'].items():parameters[name].copy_(value)
    for name,value in saved['constants'].items():buffers[name].copy_(value)
    del saved
    active={};native={};layers=[];velocities=[]
    def native_hook(site):
        def hook(module,inputs,output):
            if active.get('record'):native[site]=output
        return hook
    def physical_hook(site):
        def hook(module,inputs,output):
            if not active.get('record'):return
            text=native.pop(site)[:,-output.shape[1]:]
            ids=torch.linspace(0,output.shape[1]-1,128,device=output.device).long()
            ph=output.float();tx=text.float();frames=active['frames'];width=output.shape[-1]
            energy=ph.square().mean();sh=ph.reshape(1,frames,-1,width).mean(1).square().mean()
            layers.append(dict(reference=a.reference,case=active['case'],step=active['step'],arm=active['arm'],layer=site,
                physical_rms=float(energy.sqrt()),native_text_rms=float(tx.square().mean().sqrt()),
                physical_to_native_text_rms=float(energy.sqrt()/tx.square().mean().sqrt().clamp_min(1e-20)),
                physical_sampled_token_cosine=pair_cosine(output[:,ids]),
                physical_native_cosine=cosine(output[:,ids],text[:,ids]),
                physical_temporal_variation_energy_fraction=float(1-sh/energy.clamp_min(1e-20))))
        return hook
    handles=[]
    for site in plan['layers']:
        handles.append(model.wan.blocks[site].cross_attn.o.register_forward_hook(native_hook(site)))
        handles.append(model.fusions[str(site)].output.register_forward_hook(physical_hook(site)))
    reader=InferenceCache(CURRENT/'prepared_inputs');cases={v['id']:v for v in reference()['cases']}
    for case_id in plan['cases']:
        atomic_json(dict(status='reference_execution_probe',reference=a.reference,case=case_id),folder/'status.json')
        inputs,item=case_input(cases[case_id]);sample={k:v.cuda() for k,v in inputs.items()};sample['times']=sample['times'][None]
        pos_input=reader.input(item,image=item['image'],duration=item['duration']);neg_input=reader.input(item,image=item['image'],duration=item['duration'],drop=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            c,groups=model.enhancement(pos_input);nc,ngroups=model.enhancement(neg_input)
            x=load_noise(cases[case_id]).cuda().clone();x[:,:,:1]=sample['first'];grid=torch.linspace(1,0,51,device='cuda');grid=5*grid/(1+4*grid)
            for index in range(50):
                active.update(record=index+1 in plan['steps'],case=case_id,step=index+1,frames=x.shape[2]-1)
                def predict():
                    active['arm']='positive';pos,_=model.velocity(x,sample['first'],sample['text'],grid[index],c,sample['times'],groups)
                    active['arm']='negative';neg,_=model.velocity(x,sample['first'],sample['null_text'],grid[index],nc,sample['times'],ngroups)
                    return neg+5*(pos-neg),pos-neg
                full,guidance=predict()
                if index+1 in plan['steps']:
                    active['record']=False
                    fusions=model.fusions
                    try:model.fusions=nn.ModuleDict();base,base_guidance=predict()
                    finally:model.fusions=fusions
                    change=guidance[:,:,1:]-base_guidance[:,:,1:]
                    velocities.append(dict(reference=a.reference,case=case_id,step=index+1,sigma=float(grid[index]),
                        semantic_guidance_change_cosine=cosine(change,base_guidance[:,:,1:]),
                        full_native_guidance_cosine=cosine(guidance[:,:,1:],base_guidance[:,:,1:]),
                        semantic_guidance_change_to_native_rms=float(change.square().mean().sqrt()/base_guidance[:,:,1:].square().mean().sqrt().clamp_min(1e-20)),
                        full_clean_first_distance=float((x[:,:,1:]-grid[index]*full[:,:,1:]-sample['first']).square().mean().sqrt()),
                        no_c_clean_first_distance=float((x[:,:,1:]-grid[index]*base[:,:,1:]-sample['first']).square().mean().sqrt())))
                x=x+(grid[index+1]-grid[index])*full;x[:,:,:1]=sample['first']
        atomic_json(dict(**plan,measurements=dict(layers=layers,velocities=velocities)),folder/(a.reference+'_measurements.json'))
        print(json.dumps(dict(kind='reference_case_complete',reference=a.reference,case=case_id,time=time.time())),flush=True)
        del x,sample,pos_input,neg_input,c,nc
    for handle in handles:handle.remove()

if __name__=='__main__':main()
