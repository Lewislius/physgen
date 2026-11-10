"""Inspect grounded judgments on existing raw pairs and one RGB derivative."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from physgen_native.config import read_config
from physgen_native.knowledge import QwenKnowledge
from physgen_native.qwen_verifier import QwenProcessCritic


def main(args):
    cfg=read_config(args.config);torch.manual_seed(42);torch.cuda.set_device(0)
    model=QwenKnowledge(cfg).eval().requires_grad_(False)
    critic=QwenProcessCritic(cfg['model']['qwen_width']).cuda().requires_grad_(False)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    with (output/'judgments.jsonl').open('a',buffering=1) as stream:
        for index,path in enumerate(map(Path,args.pairs)):
            metadata=json.loads(path.read_text())
            video=torch.load(path.with_suffix('.pt'),weights_only=True,map_location='cuda').float()
            half=video.shape[-1]//2;first,second=video[:,:,:,:half],video[:,:,:,half:]
            times=torch.tensor(metadata['timestamps'],device='cuda');start=time.monotonic()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                preference,probability=model.preference(first,second,times,metadata['caption'],metadata['clauses'])
            record=dict(sample=metadata['sample'],pair=str(path),input_kind='exact_raw_teacher_frames',
                reference_preference=float(preference),probabilities=probability[0].tolist(),
                individual_order_gap=float(model.last_comparison_order_gap),
                observations=model.last_comparison_observations,seconds=time.monotonic()-start)
            if index==0:
                # Random head checks the RGB path only, never physical reward quality.
                candidate=second.detach().requires_grad_(True);model.train()
                torch.cuda.reset_peak_memory_stats()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    evidence=model.evidence(candidate,times,metadata['caption'],metadata['clauses'])
                    loss=critic.generation_loss(evidence)
                gradient,=torch.autograd.grad(loss,candidate)
                assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
                record['input_gradient_check']=dict(random_head_not_quality=True,
                    rms=float(gradient.square().mean().sqrt()),
                    nonzero_frame_count=int((gradient.abs().flatten(1).sum(1)>0).sum()),
                    peak_gib=torch.cuda.max_memory_allocated()/2**30)
                model.eval();del candidate,evidence,loss,gradient
            text=json.dumps(record,ensure_ascii=False);stream.write(text+'\n');print(text,flush=True)
    print(json.dumps(dict(kind='complete',pairs=len(args.pairs),new_wan_generations=0)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/native_grounded_qwen_process_wisa3857.yaml'))
    parser.add_argument('--pairs',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
