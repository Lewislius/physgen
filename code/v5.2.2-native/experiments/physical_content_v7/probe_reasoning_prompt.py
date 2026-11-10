"""Compare input reasoning prompts on training examples; no video/future input."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from bootstrap import ROOT
import torch,yaml
from transformers.utils import logging as transformer_logging
from physgen_native.knowledge import QwenKnowledge
from physical_scaling.data import Data
from physical_content.evidence import find_input

TASK=(
 'The request describes the future process to generate after this initial image. '
 'Give only the physical dependencies needed to realize that process. '
 'Return one to four numbered sentences, each at most 24 words. Each should connect a relevant cause, contact, '
 'or material condition to an observable response and any essential property that must persist. '
 'Prioritize the requested transition over incidental details of the scene. '
 'An event absent in the initial frame may occur later; do not replace the requested process with no change. '
 'Treat requested future actions as conditional, without claiming they are already observed. '
 'Do not invent causes, measurements, timing, or unseen initial facts. '
 'Omit sweat, lighting, generic air resistance and other incidental effects unless essential to the request. '
 'Use fewer sentences when fewer dependencies are relevant; do not pad the list. No introduction or law names.'
)

@torch.no_grad()
def main():
    torch.set_num_threads(16);transformer_logging.disable_progress_bar()
    cfg=yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text())
    data=Data(cfg);knowledge=QwenKnowledge(cfg,device='cpu').eval()
    requested=['source0001638','source0001836']
    selected=[data.rows.index(data.by_id[name]) for name in requested if name in data.by_id]
    selected += [0,len(data)//5,2*len(data)//5,3*len(data)//5,4*len(data)//5,len(data)-1]
    rows=[]
    for index in selected:
        sample=data.get(index,device='cpu',targets=False)
        old=torch.load(find_input(sample['id']+'_text'),map_location='cpu',weights_only=True)
        duration=float(sample['times'][0,-1])
        body=f'Requested future video: {sample["caption"]}\nPresentation interval: {duration:.3f} seconds.\n'+TASK
        inputs,_=knowledge._inputs([(sample['first_rgb'],sample['first_rgb'].new_zeros(1))],body)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            analysis,complete=knowledge._reason(inputs,192,return_complete=True)
        row=dict(sample=sample['id'],caption=sample['caption'],old=old['analysis'],revised=analysis,complete=complete,
                 input_scope='first image and request only; train split; no video or evaluator')
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        (ROOT/'reports/reasoning_prompt_probe.json').write_text(json.dumps(dict(prompt=TASK,records=rows),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
