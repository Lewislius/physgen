"""Use the existing independent VideoPhy2 AutoEval checkpoint; never train on it."""
import argparse
import json
from pathlib import Path
import random
import re
import sys


def parse_score(raw):
    match=re.fullmatch(r'\s*(?:score\s*[:=]\s*)?(one|two|three|four|five|[1-5])\s*[.!]?\s*',raw,re.I)
    if not match: raise ValueError(f'Unparsed evaluator output: {raw!r}')
    text=match.group(1).lower()
    return int(text) if text.isdigit() else ['one','two','three','four','five'].index(text)+1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory', nargs='+')
    parser.add_argument('--paired-protocol');args=parser.parse_args()
    if args.paired_protocol and len(args.directory)!=1:raise ValueError('One candidate directory per paired protocol')
    plans=[(Path(p),json.loads((Path(p)/'manifest.json').read_text())) for p in args.directory]
    evaluation=plans[0][1]['evaluation']
    if any(plan['evaluation'] != evaluation for _,plan in plans):
        raise ValueError('Batched runs must use the same independent evaluator')
    sys.path.insert(0,evaluation['code'])
    import numpy as np
    import torch
    from transformers.models.llama.tokenization_llama import LlamaTokenizer
    from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
    from mplug_owl_video.processing_mplug_owl import MplugOwlProcessor,MplugOwlImageProcessor
    from template import PROMPT_SA,PROMPT_PHYSICS
    checkpoint=evaluation['checkpoint'];torch.cuda.set_device(0)
    tokenizer=LlamaTokenizer.from_pretrained(checkpoint,local_files_only=True)
    processor=MplugOwlProcessor(MplugOwlImageProcessor.from_pretrained(checkpoint,local_files_only=True),tokenizer)
    model=MplugOwlForConditionalGeneration.from_pretrained(checkpoint,torch_dtype=torch.bfloat16,
             device_map={'':'cpu'},local_files_only=True).eval().cuda()
    for root,plan in plans:
        evaluate(root,plan,evaluation,model,processor,tokenizer,PROMPT_SA,PROMPT_PHYSICS)
    if args.paired_protocol:
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from compare_videophy40 import compare
        compare(args.paired_protocol,args.directory[0])


def evaluate(root,plan,evaluation,model,processor,tokenizer,PROMPT_SA,PROMPT_PHYSICS):
    import numpy as np
    import torch
    rows=[]
    for case in plan['cases']:
        folder=root/'cases'/case['id'];row=dict(id=case['id'],caption=case['caption'])
        if 'generation_seed' in case: row['generation_seed']=case['generation_seed']
        for task in ('sa','pc'):
            path=folder/f'planned.{task}.json'
            if path.exists():record=json.loads(path.read_text())
            else:
                random.seed(evaluation['seed']);np.random.seed(evaluation['seed']);torch.manual_seed(evaluation['seed'])
                prompt=PROMPT_SA.format(caption=case['caption']) if task=='sa' else PROMPT_PHYSICS
                inputs=processor(text=[prompt],videos=[str(folder/'planned.mp4')],num_frames=evaluation['num_frames'],return_tensors='pt')
                inputs={k:(v.to(torch.bfloat16) if v.dtype==torch.float32 else v).cuda() for k,v in inputs.items()}
                with torch.inference_mode():output=model.generate(**inputs,**evaluation['generation'])
                raw=tokenizer.decode(output[0].tolist(),skip_special_tokens=True)
                record=dict(task=task,raw=raw,score=parse_score(raw),case_id=case['id'])
                path.write_text(json.dumps(record,indent=2))
            row[task]=record['score']
        rows.append(row);print(json.dumps(row),flush=True)
    n=len(rows);threshold=evaluation['joint_threshold']
    summary=dict(count=n,sa_mean=sum(r['sa'] for r in rows)/n,pc_mean=sum(r['pc'] for r in rows)/n,
                 joint_pass_rate=sum(r['sa']>=threshold and r['pc']>=threshold for r in rows)/n,
                 scope=plan['scope'],training_verifier_is_independent=True)
    (root/'scores.json').write_text(json.dumps(dict(summary=summary,cases=rows),ensure_ascii=False,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
