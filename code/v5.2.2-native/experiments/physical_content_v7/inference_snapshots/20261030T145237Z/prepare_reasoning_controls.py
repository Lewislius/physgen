"""Re-read the SAME image/request with changed reasoning, never a changed video.

Empty reasoning retains the four structural relation delimiters, but contains no
physical sentences. Their contextual delimiter states are the empty-slot input;
this is distinct from deleting the physical branch or zeroing all of C.
"""
from pathlib import Path
import sys,json,argparse,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
import torch,numpy as np,yaml
from PIL import Image
from transformers.utils import logging as transformer_logging
from physgen_native.knowledge import QwenKnowledge
from physical_content.evidence import find_input,recover_words
from physical_scaling.common import atomic_torch,atomic_json,gpu_guard

CASES=['VP004','VP010','VP015','VP034']
DONORS={'VP004':'VP010','VP010':'VP034','VP015':'VP004','VP034':'VP015'}

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(16 if args.device=='cpu' else 4);transformer_logging.disable_progress_bar()
    if args.device=='cuda':gpu_guard(ROOT/'cache/reasoning_controls/allocations'/__import__('os').getenv('DET_EXPERIMENT_ID','local'))
    cfg=yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text())
    knowledge=QwenKnowledge(cfg,device=args.device).eval()
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from protocol import manifest_for,load_inputs
    manifest=manifest_for('v6_assoc','videophy40');cases={c['id']:c for c in manifest['cases']}
    rows=[];root=ROOT/'cache/reasoning_controls'
    for case_id in CASES:
        case=cases[case_id];inputs,canvas=load_inputs(case)
        rgb=np.array(Image.open(canvas).convert('RGB'),copy=True)
        first=torch.from_numpy(rgb).permute(2,0,1)[None].to(args.device).float()/127.5-1
        duration=float(inputs['times'][-1]);caption=case['caption']
        original=torch.load(find_input(f'videophy40_{case_id}_text'),map_location='cpu',weights_only=True)
        donor=torch.load(find_input(f'videophy40_{DONORS[case_id]}_text'),map_location='cpu',weights_only=True)
        for mode in ['reencoded_real','reasoning_swapped','empty_reasoning']:
            path=root/mode/f'{case_id}.pt'
            if path.is_file():
                print(json.dumps(dict(mode=mode,case=case_id,status='already_prepared')),flush=True);continue
            start=time.monotonic()
            relations=original['relations'] if mode=='reencoded_real' else donor['relations'] if mode=='reasoning_swapped' else ['']*4
            spans=[f'[RELATION {i+1}] {s}' for i,s in enumerate(relations)]
            body=(f'Request: {caption}\nPresentation interval: {duration:.3f} seconds.\n'
                  'Conditional physical analysis:\n'+'\n'.join(spans))
            model_inputs,positions=knowledge._inputs([(first,first.new_zeros(1))],body,extract_spans=spans)
            with torch.autocast(args.device,dtype=torch.bfloat16):
                hidden=knowledge.backbone.model(**model_inputs,use_cache=False,return_dict=True).last_hidden_state
            vectors=torch.stack([hidden[:,p].mean(1) for p in positions],1)
            evidence=dict(memory=hidden.cpu().to(torch.bfloat16),memory_mask=model_inputs['attention_mask'].bool().cpu(),
                relation_vectors=vectors.cpu().to(torch.bfloat16),relations=relations,caption=caption,
                analysis='\n'.join(f'{i+1}. {s}' for i,s in enumerate(relations)) if mode!='empty_reasoning' else '',
                analysis_complete=True,scope='first_image_and_request_only',sample=f'videophy40_{case_id}',
                model=cfg['paths']['qwen_checkpoint'],intervention=mode,
                donor=DONORS[case_id] if mode=='reasoning_swapped' else None,
                intervention_scope='Own first image and request unchanged; only reasoning sentences replaced or emptied')
            if mode=='empty_reasoning':
                evidence.update(words=vectors[0].cpu().to(torch.bfloat16),word_groups=torch.arange(4),
                    word_positions=torch.zeros(4,dtype=torch.long),word_strings=['']*4,
                    empty_slot_definition='Contextual hidden state of each retained structural relation delimiter; no physical sentence')
            else:
                evidence.update(recover_words(evidence,duration,knowledge.tokenizer))
            if mode=='reencoded_real':
                delta=(evidence['memory'].float()-original['memory'].float()).norm()/original['memory'].float().norm()
                evidence['same_input_relative_reencoding_difference']=float(delta)
                if float(delta)>.02:raise ValueError(f'Unexpectedly large same-input re-encoding discrepancy: {float(delta):.6f}')
            atomic_torch(evidence,path)
            row=dict(case=case_id,mode=mode,relations=relations,seconds=time.monotonic()-start,
                     same_input_relative_reencoding_difference=evidence.get('same_input_relative_reencoding_difference'))
            rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
            atomic_json(dict(records=rows,scope='same first image/request; input reasoning only'),ROOT/'reports/reasoning_controls.json')

if __name__=='__main__':main()
