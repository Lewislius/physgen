"""A bounded input-only trial prompted by observed process omissions.

Same original eight training examples; no benchmark prompt, target video,
generated video or evaluator is supplied to the reasoning producer.
"""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch,yaml
from transformers.utils import logging as transformer_logging
from physgen_native.knowledge import QwenKnowledge
from physical_scaling.data import Data

ROOT=bootstrap.BASE/'experiments/physical_content_v7'
TASK=(
 'Extract the physical evolution that this requested video must depict. The image is the initial state only. '
 'Write one to three short numbered sentences describing the requested material changes and interactions as they develop. '
 'First state the actual requested progression; then include an essential contact or material-consistency constraint if useful. '
 'Base the progression on the request, even when it has not started in the image. '
 'Do not claim the requested future events have already happened. '
 'Do not explain an unstated hidden cause, microscopic mechanism, or force merely to make the answer sound physical. '
 'Omit camera, decorative lighting, background descriptions, safety advice, and statements that irrelevant processes are absent. '
 'Preserve a steady process when that is what is requested. Do not add an unrelated change. '
 'Use fewer sentences when sufficient. Maximum 22 words per sentence. Output the numbered sentences only.'
)


@torch.no_grad()
def main():
    torch.set_num_threads(16);transformer_logging.disable_progress_bar()
    cfg=yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text())
    data=Data(cfg);knowledge=QwenKnowledge(cfg,device='cpu').eval()
    previous=json.loads((ROOT/'reports/reasoning_prompt_probe.json').read_text())['records']
    records=[]
    for item in previous:
        sample=data.get(data.rows.index(data.by_id[item['sample']]),device='cpu',targets=False)
        body='Requested video:\n'+sample['caption']+'\n\n'+TASK
        inputs,_=knowledge._inputs([(sample['first_rgb'],sample['first_rgb'].new_zeros(1))],body)
        start=time.monotonic()
        with torch.autocast('cpu',dtype=torch.bfloat16):
            analysis,complete=knowledge._reason(inputs,160,return_complete=True)
        record={'sample':sample['id'],'caption':sample['caption'],'original':item['old'],
                'analysis':analysis,'complete':complete,'seconds':time.monotonic()-start}
        records.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
        (ROOT/'reports/requested_process_reasoning_probe.json').write_text(json.dumps({
            'prompt':TASK,'records':records,
            'scope':'Original eight training examples, first image/request only. Not adopted into r1 or its evaluation.'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
