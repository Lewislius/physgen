"""Trace Q -> C in FP32 and intervene on evidence and individual update branches."""
from pathlib import Path
import sys,json,copy,gc
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from audit_representations import JOBS,Enhancement,representation,INFERENCE,OUT
from physical_scaling.supervision import slot_groups
torch.set_num_threads(8)


def load_evidence(key):
    d=torch.load(INFERENCE/f'videophy40_{key}.pt',map_location='cpu',weights_only=True)
    return {k:v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k,v in d.items()}


@torch.no_grad()
def trace(model,evidence,mode='real'):
    mem=model.memory_norm(evidence['memory']);mask=evidence['memory_mask'].bool()
    state=model.seed.expand(mem.shape[0],-1,-1);groups=None;out=[]
    def capture(name,x,update=None):
        row=dict(stage=name,**representation(x[0].float(),groups))
        if update is not None:row['update_rms']=float(update.square().mean().sqrt())
        out.append(row)
    if model.grouped:
        rel=model.relation_norm(evidence['relation_vectors']);groups=slot_groups(state.shape[1],rel.shape[1])
        state=state+rel[:,groups]
    capture('seed_plus_relation' if model.grouped else 'seed',state)
    b=model.block;sm=torch.ones(state.shape[:2],dtype=torch.bool)
    rounds=0 if mode=='zero_round' else 1 if mode=='one_round' else 2
    for i in range(rounds):
        if mode!='no_cross':
            d=b.cross(b.query_norm(state),b.memory_norm(mem),mask);state=state+d;capture(f'round{i+1}_cross',state,d)
        if mode!='no_self':
            n=b.exchange_norm(state);d=b.self_attention(n,n,sm);state=state+d;capture(f'round{i+1}_self',state,d)
        if mode!='no_ffn':
            d=b.ffn(b.ffn_norm(state));state=state+d;capture(f'round{i+1}_ffn',state,d)
    state=model.final_norm(state);capture('final_norm',state)
    c=model.project(state);capture('projection',c)
    c=model.output_norm(c);capture('C',c)
    return c[0],out,groups


@torch.no_grad()
def main():
    evidence=load_evidence('VP015_text');null=load_evidence('VP015_null');other=load_evidence('VP032_text')
    results={}
    for name in ['assoc128','assoc1000','write1000','ce1000']:
        ck=torch.load(JOBS[name],map_location='cpu',weights_only=False,mmap=True)
        with torch.device('meta'):model=Enhancement(ck['config'])
        model.load_state_dict({k.removeprefix('enhancement.'):v for k,v in ck['model'].items() if k.startswith('enhancement.')},assign=True)
        model.eval()
        c,stages,g=trace(model,evidence)
        # Verify direct reconstruction against the unchanged implementation.
        direct,_=model(evidence)
        assert torch.allclose(c,direct[0],rtol=1e-5,atol=1e-5)
        with torch.autocast('cpu',dtype=torch.bfloat16):bf,_=model(evidence)
        e={}
        e['real']=evidence;e['image_only']=null;e['unrelated']=other
        e['unrelated_memory']=dict(evidence,memory=other['memory'],memory_mask=other['memory_mask'])
        e['zero_memory']=dict(evidence,memory=torch.zeros_like(evidence['memory']))
        e['zero_relations']=dict(evidence,relation_vectors=torch.zeros_like(evidence['relation_vectors']))
        e['zero_both']=dict(e['zero_memory'],relation_vectors=torch.zeros_like(evidence['relation_vectors']))
        variations=[]
        for mode in ['image_only','unrelated','unrelated_memory','zero_memory','zero_relations','zero_both','no_self','no_cross','no_ffn','one_round','zero_round']:
            x,trace_rows,_=trace(model,e.get(mode,evidence),mode)
            variations.append(dict(mode=mode,relative_difference=float((x-c).norm()/c.norm()),
                mean_cosine=float(F.cosine_similarity(x.mean(0),c.mean(0),dim=0)),
                token_cosine=float(F.cosine_similarity(x,c,dim=-1).mean()),representation=representation(x,g),stages=trace_rows))
        results[name]=dict(stages=stages,ablations=variations,
            fp32_bf16_relative_difference=float((bf[0].float()-c).norm()/c.norm()),
            relation_input=representation(evidence['relation_vectors'][0]),
            memory_sample=representation(evidence['memory'][0,::4]),
            model_parameter_norms={k:float(v.norm()) for k,v in model.state_dict().items()})
        print(name,'stages',[(r['stage'],round(r['rms'],4),round(r['pair_cosine'],7),r.get('update_rms')) for r in stages],flush=True)
        print(name,'ablation',[(r['mode'],round(r['relative_difference'],5),round(r['representation']['pair_cosine'],7)) for r in variations],flush=True)
        del ck,model;gc.collect()
    (OUT/'block_trace_fp32.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
