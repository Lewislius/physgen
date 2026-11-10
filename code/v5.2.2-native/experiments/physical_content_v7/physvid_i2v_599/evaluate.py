"""Use the same local PhysVid AutoEval implementation and preserve both metrics."""
from pathlib import Path
import argparse
import csv
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT, plan, save, read, aggregate, physvid_score, gpu_guard


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--partial',action='store_true');a=p.parse_args()
    gpu_guard(a.model)
    protocol=plan();cfg=protocol['evaluation'];sys.path.insert(0,cfg['code'])
    import torch
    from transformers.models.llama.tokenization_llama import LlamaTokenizer
    from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
    from mplug_owl_video.processing_mplug_owl import MplugOwlProcessor,MplugOwlImageProcessor
    from template import PROMPT_SA,PROMPT_PHYSICS
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    checkpoint=cfg['checkpoint']
    tokenizer=LlamaTokenizer.from_pretrained(checkpoint,local_files_only=True)
    processor=MplugOwlProcessor(MplugOwlImageProcessor.from_pretrained(checkpoint,local_files_only=True),tokenizer)
    model=MplugOwlForConditionalGeneration.from_pretrained(checkpoint,torch_dtype=torch.bfloat16,device_map={'':'cpu'},local_files_only=True).eval().cuda()
    seed_summaries=[]
    for seed in protocol['generation_seeds']:
        out=ROOT/'outputs'/a.model/f'seed{seed}';rows=[]
        for case in protocol['cases']:
            folder=out/'cases'/case['id']
            if not (folder/'video.json').exists():
                if a.partial:continue
                raise ValueError('Missing declared video '+case['id'])
            row=dict(id=case['id'],caption=case['caption'],physvid_key=case['physvid_key'],seed=seed)
            for task in ('sa','pc'):
                path=folder/(task+'.json')
                if path.exists():record=read(path)
                else:
                    prompt=PROMPT_SA.format(caption=case['caption'].strip()) if task=='sa' else PROMPT_PHYSICS
                    inputs=processor(text=[prompt],videos=[str(folder/'video.mp4')],num_frames=cfg['num_frames'],return_tensors='pt')
                    inputs={k:(v.to(torch.bfloat16) if v.dtype==torch.float32 else v).cuda() for k,v in inputs.items()}
                    with torch.inference_mode():result=model.generate(**inputs,**cfg['generation'])
                    raw=tokenizer.decode(result[0],skip_special_tokens=True)
                    record=dict(task=task,raw=raw,score=physvid_score(raw),case=case['id'],evaluator=cfg)
                    save(path,record)
                row[task]=record['score']
            rows.append(row);print(row,flush=True)
        summary=aggregate(rows);summary.update(seed=seed,status='completed' if len(rows)==len(protocol['cases']) else 'partial')
        save(out/'scores.json',dict(summary=summary,cases=rows,scope=protocol['scope']))
        with (out/'scores.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        seed_summaries.append(summary)
    complete=all(x['status']=='completed' for x in seed_summaries)
    save(ROOT/'outputs'/a.model/'summary.json',dict(model=a.model,status='completed' if complete else 'partial',seeds=seed_summaries,
         seed_mean={k:sum(x[k] for x in seed_summaries)/len(seed_summaries) for k in ('sa_mean','pc_mean','sa_pass_rate','pc_pass_rate','joint_pass_rate')},
         definitions=dict(sa='mean(SA>=4)',pc='mean(PC>=4)',joint='mean(SA>=4 and PC>=4)'),scope=protocol['scope']))


if __name__=='__main__':main()
