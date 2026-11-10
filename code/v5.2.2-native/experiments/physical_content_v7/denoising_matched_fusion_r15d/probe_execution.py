"""Read-only four-case replay: actual physical writes and semantic guidance interference."""
from pathlib import Path
import argparse,json,math,os,sys,time
import torch
from torch.nn import functional as F

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/denoising_matched_fusion_r15d')
SOURCE=ROOT/'snapshots/20261106T011032Z'
CASES=['VP004','VP025','VP027','VP038']
STEPS=[1,15,35,50]

def field_stats(value,temporal_axis):
    value=value.float();energy=value.square().mean()
    return dict(rms=float(energy.sqrt()),temporal_variation_energy_fraction=float(
        1-value.mean(temporal_axis).square().mean()/energy.clamp_min(1e-20)))

def cosine(a,b):
    return float(F.cosine_similarity(a.float().flatten(),b.float().flatten(),dim=0))

def token_cosine(value):
    vectors=F.normalize(value.float().reshape(-1,value.shape[-1]),dim=-1);n=len(vectors)
    return float((vectors.sum(0).square().sum()-n)/(n*(n-1)))

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--describe',action='store_true');args=parser.parse_args()
    checkpoint=ROOT/'runs/physical_content_67920/step0400.pt'
    plan=dict(checkpoint=str(checkpoint),source=str(SOURCE),cases=CASES,inspected_steps=STEPS,
        inspected_layers=[2,14,29],seed=42,gamma=5,output_videos=0,training_updates=0,api_calls=0,
        selection='Diagnostic cases inherited from the pre-existing R15B execution plan; not an aggregate quality sample.',
        scope='Replay normal trajectory. Compare no-C, pre-text and no-time velocities at the identical latent only. Original native text and all parameters remain unchanged.',
        limitations='Physical/text cosine and first-frame proximity diagnose mechanisms; neither is a quality target. Stable physical phenomena may correctly remain close to the first image. Replay measurements do not score alternative generated videos.')
    if args.describe:print(json.dumps(plan));return
    sys.path.insert(0,str(SOURCE));import bootstrap
    from physical_content.common import gpu_guard,atomic_json
    from physical_content.model import PhysicalModel,time_features,group_log_prior
    from condition_controls import InferenceCache
    from benchmark_inputs import reference,case_input,load_noise
    folder=ROOT/'reports/control_helpers'/os.environ['DET_EXPERIMENT_ID']
    gpu_guard(folder);atomic_json(plan,folder/'plan.json')
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);cfg=saved['config']
    assert saved['step']==400 and Path(cfg['source_snapshot'])==SOURCE
    model=PhysicalModel(cfg).eval().requires_grad_(False)
    parameters=dict(model.named_parameters());buffers=dict(model.named_buffers())
    assert set(saved['model'])==set(model.learned_names)
    for key,value in saved['model'].items():parameters[key].copy_(value)
    for key,value in saved['constants'].items():buffers[key].copy_(value)
    del saved
    active={};native={};layers=[];velocities=[]
    def native_hook(site):
        def hook(module,inputs,output):
            if active.get('record'):native[site]=output
        return hook
    def fusion_hook(site):
        def hook(module,inputs,output):
            if not active.get('record'):return
            x,c,times,geometry,attention,groups,noise_features=inputs
            known=geometry[1]*geometry[2];frames=geometry[0]-1;width=x.shape[-1]
            physical=output[:,known:];text=native.pop(site)[:,known:]
            ids=torch.linspace(0,physical.shape[1]-1,128,device=x.device).long()
            ps=physical[:,ids];ts=text[:,ids]
            pp=field_stats(physical.reshape(1,frames,known,width),1)
            tt=field_stats(text.reshape(1,frames,known,width),1)
            heads,dim=attention.num_heads,attention.head_dim
            split=lambda v:v.reshape(1,-1,heads,dim).transpose(1,2)
            time_input=module.time(time_features(times))[:,1:].repeat_interleave(known,1)[:,ids]
            query_shift,value_scale=module.denoising(noise_features).chunk(2,dim=-1)
            q=split(attention.norm_q(attention.q(x[:,known:][:,ids])+time_input+query_shift[:,None]))
            k=split(module.key_norm(module.key(c['routing']))).to(q)
            v=split(module.value(c['execution'])).to(q)
            ref=split(module.reference_key).to(q)
            scores=q.float()@torch.cat((k,ref),2).float().transpose(-1,-2)/math.sqrt(dim)+group_log_prior(groups).to(q)
            weights=scores.softmax(-1);conditional=weights[...,:-1]/weights[...,:-1].sum(-1,keepdim=True).clamp_min(1e-20)
            z=(weights[...,:-1].to(v)@v).transpose(1,2).flatten(2)
            layers.append(dict(case=active['case'],step=active['step'],sigma=active['sigma'],arm=active['arm'],layer=site,
                physical=pp,native_text=tt,physical_to_native_text_rms=pp['rms']/max(tt['rms'],1e-12),
                physical_native_cosine=cosine(ps,ts),physical_sampled_token_cosine=token_cosine(ps),
                read_value_sampled_token_cosine=token_cosine(z),
                reference_mass=float(weights[...,-1].mean()),
                normalized_physical_attention_entropy=float(-(conditional*conditional.clamp_min(1e-20).log()).sum(-1).mean()/math.log(c['routing'].shape[1]))))
        return hook
    handles=[]
    for site in plan['inspected_layers']:
        handles.append(model.wan.blocks[site].cross_attn.o.register_forward_hook(native_hook(site)))
        handles.append(model.fusions[str(site)].register_forward_hook(fusion_hook(site)))
    reader=InferenceCache(cfg['paths']['inference_evidence']);cases={c['id']:c for c in reference()['cases']}
    for case_id in CASES:
        atomic_json(dict(status='execution_probe',case=case_id),folder/'status.json')
        inputs,item=case_input(cases[case_id]);sample={k:v.cuda() for k,v in inputs.items()};sample['times']=sample['times'][None]
        positive=reader.input(item,image=item['image'],duration=item['duration'],mode='real')
        negative=reader.input(item,image=item['image'],duration=item['duration'],drop=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            c,groups=model.enhancement(positive);nc,ngroups=model.enhancement(negative)
            x=load_noise(cases[case_id]).cuda().clone();x[:,:,:1]=sample['first']
            grid=torch.linspace(1,0,cfg['inference']['steps']+1,device='cuda');shift=cfg['inference']['shift'];grid=shift*grid/(1+(shift-1)*grid)
            for index in range(cfg['inference']['steps']):
                active.update(record=index+1 in STEPS,case=case_id,step=index+1,sigma=float(grid[index]))
                def predict(mode):
                    flags=dict(physical_enabled=mode!='no_c',posttext_read=mode!='pretext',denoising_enabled=mode!='no_time')
                    active['arm']='positive';pos,_=model.velocity(x,sample['first'],sample['text'],grid[index],c,sample['times'],groups,**flags)
                    active['arm']='negative';neg,_=model.velocity(x,sample['first'],sample['null_text'],grid[index],nc,sample['times'],ngroups,**flags)
                    return neg+5*(pos-neg),pos-neg
                full,guidance=predict('normal')
                if index+1 in STEPS:
                    active['record']=False
                    base,base_guidance=predict('no_c');pretext,_=predict('pretext');no_time,_=predict('no_time')
                    effect=full[:,:,1:]-base[:,:,1:];direction=guidance[:,:,1:]-base_guidance[:,:,1:]
                    clean_full=x[:,:,1:]-grid[index]*full[:,:,1:];clean_base=x[:,:,1:]-grid[index]*base[:,:,1:]
                    velocities.append(dict(case=case_id,step=index+1,sigma=float(grid[index]),
                        normal=field_stats(full[:,:,1:],2),no_c=field_stats(base[:,:,1:],2),physical_effect=field_stats(effect,2),
                        physical_effect_to_no_c_rms=float(effect.square().mean().sqrt()/base[:,:,1:].square().mean().sqrt().clamp_min(1e-12)),
                        semantic_guidance_change_cosine=cosine(direction,base_guidance[:,:,1:]),
                        full_native_guidance_cosine=cosine(guidance[:,:,1:],base_guidance[:,:,1:]),
                        semantic_guidance_change_to_native_rms=float(direction.square().mean().sqrt()/base_guidance[:,:,1:].square().mean().sqrt().clamp_min(1e-12)),
                        full_clean_first_distance=float((clean_full-sample['first']).square().mean().sqrt()),
                        no_c_clean_first_distance=float((clean_base-sample['first']).square().mean().sqrt()),
                        pretext_effect=field_stats(full[:,:,1:]-pretext[:,:,1:],2),
                        time_effect=field_stats(full[:,:,1:]-no_time[:,:,1:],2)))
                x=x+(grid[index+1]-grid[index])*full;x[:,:,:1]=sample['first']
        atomic_json(dict(**plan,layers=layers,velocities=velocities),folder/'measurements.json')
        del x,sample,positive,negative,c,nc
        print(json.dumps(dict(kind='case_probe_complete',case=case_id,time=time.time())),flush=True)
    for handle in handles:handle.remove()
    atomic_json(dict(status='completed',cases=CASES,output_videos=0),folder/'status.json')

if __name__=='__main__':main()
