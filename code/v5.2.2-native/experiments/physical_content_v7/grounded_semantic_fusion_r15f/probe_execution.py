"""Read-only R15F replay separating content, physical writes and native responses."""
from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/grounded_semantic_fusion_r15f')
SOURCE = ROOT / 'snapshots/20261106T202709Z'
CHECKPOINT = ROOT / 'runs/physical_content_68038/step0400.pt'


def rms(value):
    return float(value.float().square().mean().sqrt())


def cosine(a, b):
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def token_cosine(value):
    unit = F.normalize(value.float().reshape(-1, value.shape[-1]), dim=-1)
    n = len(unit)
    return float((unit.sum(0).square().sum()-unit.square().sum())/(n*(n-1)))


def projection(a, onto):
    a, onto = a.float().flatten(), onto.float().flatten()
    return float((a*onto).sum()/onto.square().sum().clamp_min(1e-20))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    plan = dict(checkpoint=str(CHECKPOINT), source=str(SOURCE), cases=['VP010','VP027','VP034'],
                steps=[1,15,35,50], inspected_layers=[2,14,29], seed=42, gamma=5,
                interventions=['no_c','parallel_fusion','words_only_memory','native_physical_query','anchor_only'],
                training_updates=0, output_videos=0, api_calls=0, model_downloads=0,
                selection='Predeclared visual cases:010/034 match the R15E replay;027 shows reduced shaping with physical fusion versus no_c.',
                scope='Normal full50 trajectory. Each alternative velocity uses the SAME normal-trajectory latent. '
                      'These are intervention diagnostics, not generated-video scores or retrained ablations.',
                limitations='Conditional-minus-null includes native text and Qwen. Projection onto native guidance is not a pure semantic score. '
                            'Anchor-only changes learned correction and C amplitude; observed effects do not prove all common corrections harmful.')
    if args.describe:
        print(json.dumps(plan)); return
    sys.path.insert(0,str(SOURCE))
    import bootstrap
    from physical_content.common import gpu_guard, atomic_json
    from physical_content.model import PhysicalModel, time_features, group_log_prior
    from condition_controls import InferenceCache
    from benchmark_inputs import reference, case_input, load_noise
    folder = ROOT/'reports/execution_probe'/os.environ['DET_EXPERIMENT_ID']
    gpu_guard(folder)
    atomic_json(plan,folder/'plan.json')
    torch.manual_seed(42); torch.cuda.manual_seed_all(42); torch.set_num_threads(4)
    saved = torch.load(CHECKPOINT,map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==400 and Path(saved['config']['source_snapshot'])==SOURCE
    cfg=saved['config']; model=PhysicalModel(cfg).eval().requires_grad_(False)
    parameters,buffers=dict(model.named_parameters()),dict(model.named_buffers())
    assert set(saved['model'])==set(model.learned_names)
    for name,value in saved['model'].items(): parameters[name].copy_(value)
    for name,value in saved['constants'].items(): buffers[name].copy_(value)
    del saved
    active,pending={},{}
    layer_rows,velocity_rows=[],[]

    def fusion_hook(site):
        def hook(module,inputs,output):
            if not active.get('record'): return
            x,c,times,geometry,attention,groups=inputs
            known=geometry[1]*geometry[2]; delta=output[:,known:]
            indices=torch.linspace(0,delta.shape[1]-1,128,device=x.device).long()
            heads,dim=attention.num_heads,attention.head_dim
            split=lambda value:value.reshape(1,-1,heads,dim).transpose(1,2)
            future=x[:,known:][:,indices]
            native_query=split(attention.norm_q(attention.q(future)))
            time_input=module.time(time_features(times))[:,1:].repeat_interleave(known,1)[:,indices]
            query=split(module.query_norm(module.query(future)+time_input))
            keys=split(module.key_norm(module.key(c['routing']))).to(query)
            values=split(module.value(c['execution'])).to(query)
            keys=torch.cat((keys,split(module.reference_key).to(query)),2)
            logits=query.float()@keys.float().transpose(-1,-2)/math.sqrt(dim)
            prior=group_log_prior(groups).to(device=query.device,dtype=torch.float32)
            weights=(logits+prior).softmax(-1)
            conditional=weights[...,:-1]/weights[...,:-1].sum(-1,keepdim=True).clamp_min(1e-30)
            pooled=(weights[...,:-1].to(values)@values).transpose(1,2).flatten(2)
            direct=module.output(pooled)
            selected=delta[:,indices]
            state_component=selected-direct
            row=dict(case=active['case'],step=active['step'],sigma=active['sigma'],arm=active['arm'],layer=site,
                     physical_rms=rms(delta), physical_finite=bool(torch.isfinite(delta).all()),
                     C_value_token_cosine=token_cosine(values.transpose(1,2).flatten(2)),
                     read_value_token_cosine=token_cosine(pooled), direct_output_token_cosine=token_cosine(direct),
                     state_component_token_cosine=token_cosine(state_component), physical_token_cosine=token_cosine(selected),
                     direct_output_rms=rms(direct), state_component_rms=rms(state_component),
                     physical_token_mean_energy_fraction=float(delta.float().mean(1).square().mean()/delta.float().square().mean().clamp_min(1e-20)),
                     physical_query_to_native_cosine=cosine(query,native_query), reference_mass=float(weights[...,-1].mean()),
                     normalized_C_attention_entropy=float(-(conditional*conditional.clamp_min(1e-30).log()).sum(-1).mean()/math.log(c['routing'].shape[1])))
            # Native text attention follows this physical write in the frozen F graph.
            pending[site]=(row,delta,known,indices,native_query)
        return hook

    def native_hook(site):
        def hook(attention,inputs,output):
            if not active.get('record'): return
            row,delta,known,indices,before_query=pending.pop(site)
            text=output[:,known:]
            after_query=attention.norm_q(attention.q(inputs[0][:,known:][:,indices]))
            after_query=after_query.reshape(1,-1,attention.num_heads,attention.head_dim).transpose(1,2)
            row.update(native_text_rms=rms(text), physical_to_native_text_rms=rms(delta)/max(rms(text),1e-20),
                       physical_native_cosine=cosine(delta[:,indices],text[:,indices]),
                       native_query_after_before_cosine=cosine(after_query,before_query),
                       native_query_relative_change=rms(after_query-before_query)/max(rms(before_query),1e-20))
            layer_rows.append(row)
        return hook

    handles=[]
    for site in plan['inspected_layers']:
        handles.append(model.fusions[str(site)].register_forward_hook(fusion_hook(site)))
        handles.append(model.wan.blocks[site].cross_attn.register_forward_hook(native_hook(site)))
    reader=InferenceCache(cfg['paths']['inference_evidence'])
    cases={case['id']:case for case in reference()['cases']}
    for case_id in plan['cases']:
        atomic_json(dict(status='execution_probe',case=case_id),folder/'status.json')
        inputs,item=case_input(cases[case_id]); sample={key:value.cuda() for key,value in inputs.items()}
        sample['times']=sample['times'][None]
        positive=reader.input(item,image=item['image'],duration=item['duration'])
        negative=reader.input(item,image=item['image'],duration=item['duration'],drop=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            contexts={}
            for name,kwargs in [('normal',{}),('words_only_memory',dict(words_only_memory=True)),('anchor_only',dict(anchor_only=True))]:
                c,groups=model.enhancement(positive,**kwargs); nc,ngroups=model.enhancement(negative,**kwargs)
                contexts[name]=(c,groups,nc,ngroups)
            x=load_noise(cases[case_id]).cuda().clone(); x[:,:,:1]=sample['first']
            grid=torch.linspace(1,0,51,device='cuda'); grid=5*grid/(1+4*grid)
            for index in range(50):
                active.update(record=index+1 in plan['steps'],case=case_id,step=index+1,sigma=float(grid[index]))
                def predict(mode):
                    c,groups,nc,ngroups=contexts.get(mode,contexts['normal'])
                    flags=dict(physical_enabled=mode!='no_c',semantic_after_physics=mode!='parallel_fusion',
                               native_physical_query=mode=='native_physical_query')
                    active['arm']='positive'
                    pos,_=model.velocity(x,sample['first'],sample['text'],grid[index],c,sample['times'],groups,**flags)
                    active['arm']='negative'
                    neg,_=model.velocity(x,sample['first'],sample['null_text'],grid[index],nc,sample['times'],ngroups,**flags)
                    return dict(guided=neg+5*(pos-neg),difference=pos-neg,negative=neg)
                full=predict('normal')
                if index+1 in plan['steps']:
                    assert not pending, 'Every recorded fusion must be followed by native text attention'
                    active['record']=False
                    alternatives={mode:predict(mode) for mode in plan['interventions']}
                    base=alternatives['no_c']; future=lambda value:value[:,:,1:]
                    full_difference=future(full['difference']); base_difference=future(base['difference'])
                    effect=future(full['guided']-base['guided'])
                    velocity_rows.append(dict(case=case_id,step=index+1,sigma=float(grid[index]),latent_rms=rms(future(x)),
                        normal_velocity_rms=rms(future(full['guided'])),no_c_velocity_rms=rms(future(base['guided'])),
                        normal_velocity_finite=bool(torch.isfinite(full['guided']).all()),
                        full_native_conditional_difference_cosine=cosine(full_difference,base_difference),
                        native_conditional_direction_retention=projection(full_difference,base_difference),
                        conditional_difference_change_to_native_rms=rms(full_difference-base_difference)/max(rms(base_difference),1e-20),
                        physical_effect_projection_onto_native_guidance=projection(effect,5*base_difference),
                        physical_unconditional_change_rms=rms(future(full['negative']-base['negative'])),
                        physical_conditional_guidance_change_rms=rms(5*(full_difference-base_difference)),
                        normal_clean_estimate_rms=rms(future(x)-grid[index]*future(full['guided'])),
                        intervention_effect_to_no_c_velocity_rms={mode:rms(future(full['guided']-v['guided']))/max(rms(future(base['guided'])),1e-20)
                                                               for mode,v in alternatives.items()}))
                    del alternatives,base
                    atomic_json(dict(**plan,layers=layer_rows,velocities=velocity_rows),folder/'measurements.json')
                    atomic_json(dict(status='execution_probe',case=case_id,step=index+1),folder/'status.json')
                x=x+(grid[index+1]-grid[index])*full['guided']; x[:,:,:1]=sample['first']
        atomic_json(dict(**plan,layers=layer_rows,velocities=velocity_rows),folder/'measurements.json')
        print(json.dumps(dict(kind='execution_case_complete',case=case_id,time=time.time())),flush=True)
        del x,sample,positive,negative,contexts
    for handle in handles:handle.remove()
    atomic_json(dict(status='completed',cases=plan['cases'],training_updates=0,output_videos=0),folder/'status.json')


if __name__=='__main__':
    main()
