"""Compare the already installed thinking model as an input-text producer only.

The running enhancement/read backbone remains the original frozen 8B model.
This experiment neither edits training caches nor evaluates generated videos.
"""
from pathlib import Path
import sys,json,copy,os,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
ROOT=bootstrap.BASE/'experiments/physical_content_v7'
import torch,yaml
from physgen_native.knowledge import QwenKnowledge
from physical_scaling.data import Data
from physical_scaling.common import gpu_guard,atomic_json
from probe_mechanism_reasoning import TASK

@torch.no_grad()
def main():
    torch.set_num_threads(4);gpu_guard(ROOT/'reports/thinking_allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    cfg=yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text());cfg=copy.deepcopy(cfg)
    path=Path('/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-2B-Thinking')
    cfg['paths']['qwen_checkpoint']=str(path);cfg['model']['qwen_width']=json.loads((path/'config.json').read_text())['text_config']['hidden_size']
    knowledge=QwenKnowledge(cfg).eval();data=Data(cfg)
    old=json.loads((ROOT/'reports/reasoning_prompt_probe.json').read_text())['records'];rows=[]
    for record in old:
        sample=data.get(data.rows.index(data.by_id[record['sample']]),targets=False)
        body=TASK+'\nRequested video: '+sample['caption']
        inputs,_=knowledge._inputs([(sample['first_rgb'],sample['first_rgb'].new_zeros(1))],body)
        start=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):raw,complete=knowledge._reason(inputs,2048,return_complete=True)
        answer=raw.split('</think>')[-1].strip() if '</think>' in raw else None
        row=dict(sample=sample['id'],caption=sample['caption'],original=record['old'],raw=raw,
            final_answer=answer,complete=complete,seconds=time.monotonic()-start)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='raw'},ensure_ascii=False),flush=True)
        atomic_json(dict(model=str(path),records=rows,prompt=TASK,scope='Training split; first image/request only; candidate input producer, not adopted'),ROOT/'reports/thinking_reasoning_probe.json')

if __name__=='__main__':main()
