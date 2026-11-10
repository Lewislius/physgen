"""Same input-only training cases/prompt, with the official equal-size Thinking model.

No generated video, future frame, benchmark score or evaluator is supplied.
Only final concise answers are recorded; no new conditioning cache is adopted.
"""
from pathlib import Path
import sys,json,time,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch,yaml
from transformers.utils import logging as transformer_logging
from physgen_native.knowledge import QwenKnowledge
from physical_scaling.data import Data
from probe_requested_process import TASK

ROOT=bootstrap.BASE/'experiments/physical_content_v7'


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tokens',type=int,default=2048);args=parser.parse_args()
    torch.set_num_threads(16);torch.manual_seed(42);transformer_logging.disable_progress_bar()
    cfg=yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text())
    cfg['paths']['qwen_checkpoint']='/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-8B-Thinking'
    model=Path(cfg['paths']['qwen_checkpoint'])
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    if any(not (model/name).is_file() for name in set(index.values())):
        raise FileNotFoundError('Wait for the authorized official model download to finish')
    data=Data(cfg);knowledge=QwenKnowledge(cfg,device='cpu').eval()
    original=json.loads((ROOT/'reports/requested_process_reasoning_probe.json').read_text())['records']
    records=[]
    for item in original:
        sample=data.get(data.rows.index(data.by_id[item['sample']]),device='cpu',targets=False)
        inputs,_=knowledge._inputs([(sample['first_rgb'],sample['first_rgb'].new_zeros(1))],
            'Requested video:\n'+sample['caption']+'\n\n'+TASK)
        start=time.monotonic()
        with torch.autocast('cpu',dtype=torch.bfloat16):
            answer,complete=knowledge._reason(inputs,args.tokens,return_complete=True)
        final=answer.rsplit('</think>',1)[-1].strip() if '</think>' in answer else ''
        row=dict(sample=sample['id'],caption=sample['caption'],instruct_answer=item['analysis'],
            thinking_final_answer=final,has_final_answer=bool(final),complete=complete,
            generated_token_count=len(knowledge.tokenizer.encode(answer,add_special_tokens=False)),
            seconds=time.monotonic()-start)
        records.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        result=dict(model=str(model),source='https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking',
            prompt=TASK,budget=args.tokens,records=records,adopted=False,
            scope='Same eight training inputs and exact process prompt, first image/request only; no future or benchmark conditioning')
        (ROOT/'reports/thinking8b_input_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
