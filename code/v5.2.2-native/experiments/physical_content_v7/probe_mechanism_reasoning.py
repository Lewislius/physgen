"""Input-only prompt/modality comparison on the same eight training examples."""
from pathlib import Path
import sys,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from bootstrap import ROOT
import torch,yaml
from transformers.utils import logging as transformer_logging
from physgen_native.knowledge import QwenKnowledge
from physical_scaling.data import Data

TASK=(
 'Explain the physical development of the process requested below in one or two concise numbered sentences. '
 'In each sentence, connect an applied influence or material interaction to the resulting visible development. '
 'Describe the cause of the requested change and how the relevant material or contact responds. '
 'Include only dependencies that help realize this process. The text is a requested future video, not a claim that '
 'all events have happened in the initial observation. Avoid camera, composition, lighting, safety advice, and decorative '
 'properties. Do not invent a source of force or heat when the request does not establish one. '
 'A steady process may remain steady; do not add an unrelated transition. Maximum 28 words per sentence.'
)

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(16 if args.device=='cpu' else 4);transformer_logging.disable_progress_bar()
    cfg=yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text());data=Data(cfg)
    knowledge=QwenKnowledge(cfg,device=args.device).eval()
    old=json.loads((ROOT/'reports/reasoning_prompt_probe.json').read_text())['records'];rows=[]
    for record in old:
        sample=data.get(data.rows.index(data.by_id[record['sample']]),device=args.device,targets=False)
        body=TASK+'\nRequested video: '+sample['caption']
        row=dict(sample=sample['id'],caption=sample['caption'],original=record['old'],mechanisms={})
        for mode in ['text_only','image_and_text']:
            if mode=='text_only':
                rendered=knowledge.tokenizer.apply_chat_template([{'role':'user','content':body}],tokenize=False,add_generation_prompt=True)
                inputs={k:v.to(args.device) for k,v in knowledge.tokenizer(rendered,add_special_tokens=False,return_tensors='pt').items() if k in ['input_ids','attention_mask']}
            else:inputs,_=knowledge._inputs([(sample['first_rgb'],sample['first_rgb'].new_zeros(1))],body)
            with torch.autocast(args.device,dtype=torch.bfloat16):
                answer,complete=knowledge._reason(inputs,128,return_complete=True)
            row['mechanisms'][mode]=dict(analysis=answer,complete=complete)
            print(json.dumps(dict(sample=sample['id'],mode=mode,analysis=answer),ensure_ascii=False),flush=True)
        rows.append(row)
        (ROOT/'reports/mechanism_reasoning_probe.json').write_text(json.dumps(dict(prompt=TASK,records=rows,
            scope='Training split only; input request/first image; no future video and no generated-video evaluator'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
