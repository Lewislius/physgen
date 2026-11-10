"""Calibrate and inspect video preferences using only existing held-out caches.

This diagnostic never writes cache labels or training checkpoints. It measures
teacher order bias and held-out preference agreement, not physical ground truth.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from physgen_native.config import read_config
from physgen_native.data import VideoDataset, move
from physgen_native.decoder import VideoDecoder
from physgen_native.knowledge import QwenKnowledge
from physgen_native.verifier import ProcessCritic, evidence_indices, rgb_times, omitted_process
from physgen_native.runtime import seed_all, Logger


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=int,default=6)
    parser.add_argument('--output',default=str(ROOT/'analysis/verifier_reasoned'))
    args=parser.parse_args(); cfg=read_config(ROOT/'configs/native_physical_wisa3857.yaml')
    seed_all(42); torch.cuda.set_device(0)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    log=Logger(out/'metrics.jsonl')
    model=QwenKnowledge(cfg,'cuda').cuda().train(); decoder=VideoDecoder(cfg,'cuda').cuda()
    data=VideoDataset(cfg,'validation'); pairs=[]; records=[]
    for index in range(args.cases):
        sample=move(data[index],'cuda'); ids=evidence_indices(sample['times'],cfg['knowledge']['evidence_pairs'])
        times=rgb_times(ids,sample['times'])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            truth=decoder(sample['latent'],resize=cfg['knowledge']['video_size'])
            reasoning=model.dependency_reasoning(truth[:1],sample['caption'],sample['clauses'])
            log.write('dependency_reasoning',index,sample=sample['id'],analysis=reasoning)
            a=model.evidence(truth[ids],times,sample['caption'],sample['clauses'])
            alternatives=dict(identical=truth,stagnant=truth[:1].expand_as(truth),
                              omitted=omitted_process(truth,sample,cfg['verifier'],index),reversed=truth.flip(0))
            for name,alternative in alternatives.items():
                b=model.evidence(alternative[ids],times,sample['caption'],sample['clauses'])
                forward,fp=model.preference(truth[ids],alternative[ids],times,sample['caption'],sample['clauses'])
                forward_reasoning=model.last_comparison_reasoning
                reverse,rp=model.preference(alternative[ids],truth[ids],times,sample['caption'],sample['clauses'])
                target=(forward+1-reverse)/2
                record=dict(sample=sample['id'],caption=sample['caption'],view=name,
                            preference=float(target),forward=float(forward),reverse=float(reverse),
                            forward_reasoning=forward_reasoning,reverse_reasoning=model.last_comparison_reasoning,
                            tie=float((fp[:,2]+rp[:,2])/2),feature_rms=float((a-b).square().mean().sqrt()))
                log.write('preference',index,**record); records.append(record)
                if name!='identical': pairs.append((index,a.detach(),b.detach(),target.detach()))
        del truth,alternatives,sample
    critic=ProcessCritic(cfg).cuda(); optimizer=torch.optim.AdamW(critic.parameters(),lr=1e-3,weight_decay=.01)
    train=[p for p in pairs if p[0]%2==0]; held=[p for p in pairs if p[0]%2]
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss=torch.stack([critic.calibration_loss(a,b,target)[0] for _,a,b,target in train]).mean()
        loss.backward(); optimizer.step()
    with torch.no_grad():
        for split,rows in [('fit',train),('heldout',held)]:
            pred=torch.cat([(critic(a)-critic(b)).sigmoid() for _,a,b,t in rows])
            target=torch.cat([t for _,a,b,t in rows])
            log.write('calibration',200,split=split,preference_mae=(pred-target).abs().mean(),
                      targets=target.tolist(),predictions=pred.tolist())
    (out/'preferences.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    log.write('complete',200,cases=args.cases,peak_gib=torch.cuda.max_memory_allocated()/2**30)
    log.close()


if __name__=='__main__': main()
