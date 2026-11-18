"""Audit native text-to-C semantic affinity using actual frozen R10A inputs.

No video is generated and no trained R15E weights exist. This checks whether
the proposed semantic prior has a usable nonconstant input signal, not quality.
"""
from pathlib import Path
import datetime,json,math,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch.nn import functional as F
from safetensors import safe_open
from transformers import AutoTokenizer
from benchmark_inputs import ROOT,reference,case_input
from condition_controls import InferenceCache
from physical_content.model import Enhancement
from physical_scaling.supervision import group_log_prior


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    base=Path('/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B')
    index=json.loads((base/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    def tensor(name):
        with safe_open(base/index[name],framework='pt',device='cpu') as f:return f.get_tensor(name).float()
    embedding={k:tensor(k) for k in index if k.startswith('text_embedding.')}
    def embed(text):
        hidden=F.linear(text.float(),embedding['text_embedding.0.weight'],embedding['text_embedding.0.bias'])
        return F.linear(F.gelu(hidden,approximate='tanh'),embedding['text_embedding.2.weight'],embedding['text_embedding.2.bias'])
    tokenizer=AutoTokenizer.from_pretrained(base/'google/umt5-xxl',local_files_only=True)
    checkpoint=ROOT/'execution_value_r10/runs/physical_content_67003/step0400.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    enhancement=Enhancement(saved['config']).eval().requires_grad_(False)
    enhancement.load_state_dict({k.removeprefix('enhancement.'):v for k,v in
        {**saved['model'],**saved['constants']}.items() if k.startswith('enhancement.')},strict=True)
    enhancement.checkpointing=False
    cache=InferenceCache(ROOT/'denoising_matched_fusion_r15d/prepared_inputs',device='cpu')
    ids={'VP004','VP010','VP034','VP038'};cases=[]
    for case in reference()['cases']:
        if case['id'] not in ids:continue
        inputs,item=case_input(case);evidence=cache.input(item,image=item['image'],duration=item['duration'])
        with torch.autocast('cpu',dtype=torch.bfloat16):
            old,groups=enhancement(evidence,request_refinement=False)
            changed,_=enhancement(evidence,request_refinement=True)
        text=embed(inputs['text'])
        tokens=tokenizer.convert_ids_to_tokens(tokenizer(case['caption'])['input_ids'])
        if len(tokens)!=text.shape[1]:raise ValueError('Native token labels do not match cached T5 length')
        rows=[]
        for layer in (2,14,29):
            prefix=f'blocks.{layer}.cross_attn.'
            weight=tensor(prefix+'k.weight');bias=tensor(prefix+'k.bias');norm=tensor(prefix+'norm_k.weight')
            def keys(value):
                value=F.linear(value.float(),weight,bias)
                value=value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-6)*norm
                return value.view(1,-1,24,128).transpose(1,2)
            kt=keys(text);prior=group_log_prior(groups)[:-1].float();baseline=prior.softmax(0)
            per_variant={}
            for name,content in [('R10A_original',old),('R10A_weights_request_refinement_intervention',changed)]:
                kc=keys(content['routing'])
                probability=(kt@kc.transpose(-1,-2)/math.sqrt(128)+prior).softmax(-1)[0]
                mean=probability.mean(1,keepdim=True)
                total_variation=.5*(probability-mean).abs().sum(-1)
                entropy=-(probability*probability.clamp_min(1e-30).log()).sum(-1)/math.log(probability.shape[-1])
                grouped=torch.stack([probability[...,groups==g].sum(-1) for g in groups.unique()],-1).mean(0)
                physical_value=saved['model'][f'fusions.{layer}.value.weight'].float()
                value=F.linear(content['execution'].float(),physical_value).view(1,-1,24,128).transpose(1,2)[0]
                pooled=probability@value
                denominator=(mean@value).square().mean().sqrt().clamp_min(1e-12)
                per_variant[name]=dict(normalized_C_entropy=float(entropy.mean()),
                    mean_token_variation_TV=float(total_variation.mean()),max_token_variation_TV=float(total_variation.max()),
                    mean_change_from_original_prior_TV=float(.5*(probability-baseline).abs().sum(-1).mean()),
                    token_selected_value_variation_relative_rms=float((pooled-pooled.mean(1,keepdim=True)).square().mean().sqrt()/denominator),
                    token_relation_probabilities=[dict(token=t,probabilities=row.tolist()) for t,row in zip(tokens,grouped)])
            rows.append(dict(layer=layer,variants=per_variant))
            del weight,bias,norm,kt
        cases.append(dict(id=case['id'],caption=case['caption'],relations=evidence['relations'],layers=rows,
            request_refinement_change={k:float((changed[k].float()-old[k].float()).norm()/old[k].float().norm()) for k in old}))
        print(json.dumps(dict(case=case['id'],complete=len(cases))),flush=True)
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint=str(checkpoint),cases=cases,
        scope='Four preselected input cases; frozen R10A400 enhancement and physical values; official frozen Wan text embedding/K/norm. One same-weight request-refinement intervention. No R15E training or video generation.',
        interpretation='Entropy alone is not a quality criterion. Token variation checks whether the prior changes with native request content. Value variation is a zero-physical-logit proxy, not the actual video-dependent output, and cannot establish correctness or improvement.',
        new_model_downloads=0,new_api_calls=0)
    path=Path(__file__).parent/'review/native_key_support4.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    summary=[dict(id=c['id'],request_refinement_change=c['request_refinement_change'],layers=[dict(layer=x['layer'],
        variants={k:{m:v for m,v in y.items() if m!='token_relation_probabilities'} for k,y in x['variants'].items()}) for x in c['layers']]) for c in cases]
    print(json.dumps(dict(output=str(path),cases=summary),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
