"""Bounded held-out comparison of pair-resolved and time-pooled critics.

Time pooling of context-dependent features does not remove the encoder's
temporal information. Compare separately extracted full/pair encodings to
isolate the benefit of pretrained long temporal context.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from physgen_native.runtime import atomic_save,Logger,seed_all
from physgen_native.temporal_critic import TemporalProcessCritic
from physgen_native.sequence import perceptual_process_error


def inputs(record,mode):
    a,b=record['real'],record['generated']
    if mode=='state_only':
        a=a.mean(0,keepdim=True).expand_as(a);b=b.mean(0,keepdim=True).expand_as(b)
    return a,b,record['condition'],record['times'],record['valid']


def evaluate(model,records,mode):
    rows=[]
    for record in records:
        a,b,c,t,v=inputs(record,mode)
        with torch.no_grad():
            loss,metrics=model.calibration_loss(a,b,c,t,v)
            late=record['real'].clone();late[13:]=late[12:13]
            reordered=record['real'].clone();reordered[1:-1]=reordered[1:-1].flip(0)
            if mode=='state_only':
                late=late.mean(0,keepdim=True).expand_as(late)
                reordered=reordered.mean(0,keepdim=True).expand_as(reordered)
            real_score=model(a,c,t,v);late_score=model(late,c,t,v)
            late_margin=torch.stack([(real_score[h]-late_score[h]).mean() for h in real_score]).mean()
            reorder_score=model(reordered,c,t,v)
            order_sensitivity=torch.stack([(real_score[h]-reorder_score[h]).abs().mean() for h in real_score]).mean()
        # This is an explicit validation oracle: does the proposed feature
        # direction agree with reference process correction? It is not a free
        # generation condition or a claim about unique physically valid futures.
        variable=record['generated'].detach().float().requires_grad_(True)
        observed=variable.mean(0,keepdim=True).expand_as(variable) if mode=='state_only' else variable
        adversarial=model.generation_loss(observed,c,t,v)
        gradient=torch.autograd.grad(adversarial,variable)[0]
        oracle=perceptual_process_error(variable,record['real'],v,dict(scale_floor=.03,state_weight=.25))[0]
        target_gradient=torch.autograd.grad(oracle,variable)[0]
        cosine=(gradient*target_gradient).sum()/(gradient.norm()*target_gradient.norm()).clamp_min(1e-12)
        rows.append(dict(id=record['id'],split=record['split'],loss=float(loss),
                         margin=float(metrics['real_score']-metrics['generated_score']),
                         late_feature_stagnation_margin=float(late_margin),
                         same_frame_bag_order_sensitivity=float(order_sensitivity),
                         gradient_oracle_cosine=float(cosine),
                         **{k:float(v) for k,v in metrics.items()}))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--train-cases',type=int,default=16)
    parser.add_argument('--validation-cases',type=int,default=8)
    parser.add_argument('--epochs',type=int,default=16)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,314159,271828])
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();device=torch.device(args.device)
    records=[]
    for path in sorted((args.directory/'features').glob('*.pt')):
        record=torch.load(path,map_location='cpu',weights_only=True)
        records.append({k:v.to(device).float() if torch.is_tensor(v) else v for k,v in record.items()})
    train=[r for r in records if r['split']=='train'];validation=[r for r in records if r['split']=='validation']
    if (len(train),len(validation))!=(args.train_cases,args.validation_cases):
        raise ValueError(f'Wait for the declared cases: found train={len(train)}, held-out={len(validation)}')
    if {r['id'] for r in train}&{r['id'] for r in validation}:raise ValueError('Split leakage')
    feature_width=train[0]['real'].shape[-1];condition_width=train[0]['condition'].shape[-1]
    log=Logger(args.directory/'fit.jsonl');results=[]
    for seed in args.seeds:
        for mode in ['temporal','state_only']:
            seed_all(seed)
            model=TemporalProcessCritic(feature_width=feature_width,condition_width=condition_width).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
            initial=evaluate(model,validation,mode)
            for epoch in range(args.epochs):
                order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
                for index in order:
                    loss,_=model.calibration_loss(*inputs(train[index],mode))
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            learned=evaluate(model,train,mode);heldout=evaluate(model,validation,mode)
            item=dict(mode=mode,seed=seed,updates=args.epochs*len(train),initial=initial,train=learned,validation=heldout)
            results.append(item)
            log.write('fit',args.epochs*len(train),mode=mode,seed=seed,
                      train_pair_accuracy=sum(r['margin']>0 for r in learned)/len(learned),
                      validation_pair_accuracy=sum(r['margin']>0 for r in heldout)/len(heldout),
                      validation_gradient_cosine=sum(r['gradient_oracle_cosine'] for r in heldout)/len(heldout))
            atomic_save(dict(revision='temporal_process_distribution_critic_v1',model=model.state_dict(),
                             feature_width=feature_width,condition_width=condition_width,width=128,seed=seed,mode=mode,
                             evidence_encoder=train[0].get('evidence_encoder','Qwen3-VL visual'),
                             evidence_context=train[0].get('evidence_context','pairs'),
                             train_ids=[r['id'] for r in train],validation_ids=[r['id'] for r in validation]),
                        args.directory/f'critic_{mode}_seed{seed}.pt')
    (args.directory/'fit_results.json').write_text(json.dumps(dict(results=results,
        interpretation='Small held-out discrimination and feature-gradient screening; no full-video or causal validity claim.'),indent=2))
    log.write('complete',0);log.close()


if __name__=='__main__':main()
