"""Bound the existing thinking model's input reasoning and obtain a final text.

The unbounded 2048-token probe is retained separately. No outputs are adopted
into training here; compare factual content before selecting a producer.
"""
from pathlib import Path
import sys,json,copy,argparse,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
ROOT=bootstrap.BASE/'experiments/physical_content_v7'
import torch,yaml
from transformers.utils import logging as transformer_logging
from physgen_native.knowledge import QwenKnowledge
from physical_scaling.data import Data
from probe_mechanism_reasoning import TASK

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(16 if args.device=='cpu' else 4);transformer_logging.disable_progress_bar()
    cfg=copy.deepcopy(yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text()))
    path=Path('/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-2B-Thinking')
    cfg['paths']['qwen_checkpoint']=str(path);cfg['model']['qwen_width']=2048
    knowledge=QwenKnowledge(cfg,device=args.device).eval();data=Data(cfg)
    old=json.loads((ROOT/'reports/reasoning_prompt_probe.json').read_text())['records'];rows=[]
    for record in old:
        sample=data.get(data.rows.index(data.by_id[record['sample']]),device=args.device,targets=False)
        body=TASK+'\nRequested video: '+sample['caption']
        inputs,_=knowledge._inputs([(sample['first_rgb'],sample['first_rgb'].new_zeros(1))],body)
        inputs={k:v for k,v in inputs.items() if k!='position_ids'};prefix=inputs['input_ids'].shape[1]
        settings=dict(do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,
            return_dict_in_generate=False,pad_token_id=knowledge.tokenizer.eos_token_id)
        start=time.monotonic()
        with torch.autocast(args.device,dtype=torch.bfloat16):
            result=knowledge.backbone.generate(**inputs,max_new_tokens=512,**settings)
            raw=knowledge.tokenizer.decode(result[0,prefix:],skip_special_tokens=False)
            forced='</think>' not in raw
            if forced:
                ending=knowledge.tokenizer.encode('\n</think>\n',add_special_tokens=False,return_tensors='pt').to(args.device)
                continuation=torch.cat((result,ending),1)
                following=dict(inputs,input_ids=continuation,attention_mask=torch.ones_like(continuation))
                final=knowledge.backbone.generate(**following,max_new_tokens=160,**settings)
                answer=knowledge.tokenizer.decode(final[0,continuation.shape[1]:],skip_special_tokens=True).strip()
                complete=int(final[0,-1])==knowledge.tokenizer.eos_token_id
            else:
                answer=raw.split('</think>')[-1].replace('<|im_end|>','').strip()
                complete=int(result[0,-1])==knowledge.tokenizer.eos_token_id
        row=dict(sample=sample['id'],caption=sample['caption'],original=record['old'],analysis=answer,
            forced_thinking_boundary=forced,complete=complete,thinking_budget=512,answer_budget=160,
            seconds=time.monotonic()-start)
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        (ROOT/'reports/bounded_reasoning_probe.json').write_text(json.dumps(dict(model=str(path),records=rows,
            prompt=TASK,scope='Input-only training cases; bounded internal reasoning; no video judge; not adopted'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
