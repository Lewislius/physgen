"""Read-only physical-write tracing on the unchanged native R16A rollout."""
from pathlib import Path
import argparse,json,math,os,runpy,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime import activate,HERE


def install_trace(model,logfile,*,sites=None):
    import torch
    import torch.nn.functional as F
    import guidance
    from physical_content import joint_read
    state=dict(step=0,arm=0,site=None,active=False,row=None)
    steps={1,5,10,20,30,40,50};sites={'2','14','29'} if sites is None else sites
    rms=lambda x:x.float().square().mean().sqrt()
    def common_energy(x):
        return x.float().mean(-2).square().mean()/x.float().square().mean().clamp_min(1e-20)
    handles=[]
    for site,fusion in model.fusions.items():
        def begin(module,args,site=site):
            state.update(site=site,active=state['arm']==1 and state['step'] in steps and site in sites)
            if state['active']:
                x,context,c,times,geometry,*_=args
                state['known']=geometry[1]*geometry[2]
                state['row']=dict(step=state['step'],layer=int(site)+1,
                    input_rms=rms(x[:,state['known']:]),C_execution_rms=rms(c['execution']))
        handles.append(fusion.register_forward_pre_hook(begin))
        def native_output(module,args,out,site=site):
            if state['active'] and state['site']==site:
                state['native']=out[:,state['known']:]
                state['row']['native_text_output_rms']=rms(state['native'])
        handles.append(model.wan.blocks[int(site)].cross_attn.o.register_forward_hook(native_output))
        def state_output(module,args,out,site=site):
            if state['active'] and state['site']==site:state['state_value']=out
        handles.append(fusion.state_value.register_forward_hook(state_output))
        def physical_output(module,args,out,site=site):
            if not (state['active'] and state['site']==site):return
            row=state['row'];physical=args[0];idx=state['indices'];factor=state['state_value']
            row.update(state_value_rms=rms(factor),state_input_amplification=rms(physical)/row['message_rms'].clamp_min(1e-20),
                physical_output_rms=rms(out),physical_to_native_rms=rms(out)/row['native_text_output_rms'].clamp_min(1e-20),
                physical_output_common_energy=common_energy(out),physical_output_finite=torch.isfinite(out).all())
            direct=state['direct_message'].transpose(1,2).flatten(2).to(physical)
            alternate=F.linear(direct+factor[:,idx]*direct,module.weight)
            row.update(direct_counterfactual_write_rms=rms(alternate),actual_sampled_write_rms=rms(out[:,idx]),
                request_prior_write_ratio=rms(out[:,idx])/rms(alternate).clamp_min(1e-20),
                actual_direct_write_cosine=F.cosine_similarity(out[:,idx].float().flatten(),alternate.float().flatten(),dim=0))
            converted={k:(float(v) if torch.is_tensor(v) else v) for k,v in row.items()}
            with logfile.open('a') as stream:stream.write(json.dumps(converted)+'\n')
            state.pop('direct_message',None);state.pop('state_value',None);state.pop('native',None)
        handles.append(fusion.output.register_forward_hook(physical_output))
    original_read=joint_read.read_physical
    def read(q,k,v,request_keys,reference,prior,mode='joint',*,request_query,relation_count):
        result=original_read(q,k,v,request_keys,reference,prior,mode,
            request_query=request_query,relation_count=relation_count)
        if not state['active']:return result
        if mode!='joint':raise ValueError('Trace the normal native reader only')
        idx=torch.linspace(0,q.shape[-2]-1,min(64,q.shape[-2]),device=q.device).long()
        state['indices']=idx;scale=math.sqrt(q.shape[-1]);qs=q[:,:,idx].float();rqs=request_query[:,:,idx].float()
        content_scores=qs@k.float().transpose(-1,-2)/scale
        request_weights=(rqs@request_keys.float().transpose(-1,-2)/scale).softmax(-1)
        binding=(request_keys.float()@k.float().transpose(-1,-2)/scale+prior[:-1]).softmax(-1)
        probability=request_weights@binding
        null=qs@reference.float().transpose(-1,-2)/scale+prior[-1:]
        actual=torch.cat((content_scores+probability.clamp_min(torch.finfo(torch.float32).tiny).log()+math.log(relation_count),null),-1).softmax(-1)
        direct=torch.cat((content_scores+prior[:-1],null),-1).softmax(-1)
        values=torch.cat((v,torch.zeros_like(reference)),-2)
        state['direct_message']=direct.to(values)@values
        def entropy(p):return -(p*p.clamp_min(1e-30).log()).sum(-1).mean()/math.log(p.shape[-1])
        state['row'].update(value_rms=rms(v),message_rms=rms(result),message_common_energy=common_energy(result.transpose(1,2).flatten(2)),
            physical_mass=actual[...,:-1].sum(-1).mean(),direct_physical_mass=direct[...,:-1].sum(-1).mean(),
            attention_normalized_entropy=entropy(actual),prior_normalized_entropy=entropy(probability),
            max_slot_mass=actual[...,:-1].max(-1).values.mean())
        return result
    joint_read.read_physical=read
    original_velocity=model.velocity
    def velocity(*args,**kwargs):
        state['arm']+=1
        return original_velocity(*args,**kwargs)
    model.velocity=velocity
    original_guidance=guidance.guided_velocity
    def guided(*args,**kwargs):
        state.update(step=state['step']+1,arm=0,active=False)
        return original_guidance(*args,**kwargs)
    guidance.guided_velocity=guided
    return guided


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pool',choices=['amp-48g','amp-80g','ada-48g'],required=True)
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    import yaml
    cfg=yaml.safe_load(args.config.read_text());source=Path(cfg['source_snapshot'])
    plan=dict(case='VP004',scope='Read-only positive-arm physical reads/writes during unchanged native50-step generation',
        checkpoint=str(args.checkpoint),source=str(source),runtime=str(HERE),pool=args.pool,
        layers=[3,15,30],steps=[1,5,10,20,30,40,50],attention_sample_positions=64,
        no_CFG_ablation_or_attribution=True,no_training=True,primary_outputs_untouched=True,
        direct_counterfactual='At the same video state and C, measure the existing direct reader; never change the actual rollout')
    if args.describe:print(json.dumps(plan));return
    guard=activate(source,args.pool);folder=args.output.parent/'reports/jobs'/os.getenv('DET_EXPERIMENT_ID','local')
    guard(folder);(folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    logfile=folder/'physical_write.jsonl'
    if logfile.exists():raise FileExistsError('Preserve existing trace; do not mix attempts')
    import guidance
    original=guidance.guided_velocity;installed=False;traced=None
    def initialize(model,*args,**kwargs):
        nonlocal installed,traced
        if not installed:
            guidance.guided_velocity=original
            traced=install_trace(model,logfile);installed=True
        return traced(model,*args,**kwargs)
    guidance.guided_velocity=initialize
    sys.argv=[str(source/'infer.py'),'--checkpoint',str(args.checkpoint),'--output',str(args.output),
              '--suite','videophy40','--modes','real','--cases','VP004']
    runpy.run_path(str(source/'infer.py'),run_name='__main__')
    (folder/'status.json').write_text(json.dumps(dict(status='completed',**plan),indent=2)+'\n')


if __name__=='__main__':main()
