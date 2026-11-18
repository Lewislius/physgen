"""Disposable full-size generation/backpropagation check before fresh400.

Uses an existing training sample and official foundations. No checkpoint from
these few validation updates is saved or passed into the formal training run.
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from physical_content.common import load_config, gpu_guard, atomic_json
from physical_content.model import PhysicalModel
from physical_content.data import TrainingData
from physical_content.teacher import Teacher
from physical_content.objectives import backward_micro


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cfg=load_config(args.config)
    gpu_guard(args.output.parent)
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    started=time.monotonic();model=PhysicalModel(cfg);model.set_phase(0)
    opt=torch.optim.AdamW(model.optim_groups(),betas=(.9,.999),eps=1e-8,foreach=False)
    data=TrainingData(cfg);sample=data.get(0);evidence=Teacher(cfg).input(sample)
    sigma=torch.tensor(.5,device='cuda');noise=torch.randn_like(sample['latent'])
    torch.cuda.reset_peak_memory_stats()
    records=[]
    required_paths = (
        'enhancement.seed', 'enhancement.reasoning_read.value.weight',
        'enhancement.request_read.value.weight', 'enhancement.exchange.value.weight',
        'enhancement.source_query.weight', 'enhancement.caption_query.weight',
        'physical_value.weight', 'state_expert.condition.weight',
        'state_expert.relation.v.weight', 'state_expert.temporal.v.weight',
        'state_expert.ffn.0.weight',
        'mechanisms.2.query.weight', 'mechanisms.2.key.weight',
        'feature_gates.2.output.weight')
    named = dict(model.named_parameters())
    for index in range(4):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            values=backward_micro(model,evidence,sample,sample['text'],sigma,noise,0.,1)
        norms={}
        for role in cfg['train']['peak_lrs']:
            parameters=[p for group in opt.param_groups if group['role']==role
                        for p in group['params'] if p.grad is not None]
            norms[role]=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
        opt.step()
        record=dict(disposable_update=index+1,gradient_norms=norms,
                    functional_gradients={name: (float(named[name].grad.norm())
                        if named[name].grad is not None else None) for name in required_paths},
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    generation=float(values['generation']),
                    source_max_probability=values['source_max_probability'],
                    source_entropy_nats=values['source_entropy_nats'])
        records.append(record);print(json.dumps(record),flush=True)
    if any(records[-1]['gradient_norms'][role]<=0 for role in cfg['train']['peak_lrs']):
        raise RuntimeError('A functional parameter group did not receive generation gradients')
    missing = [name for name, norm in records[-1]['functional_gradients'].items()
               if norm is None or norm <= 0]
    if missing:
        raise RuntimeError(f'Actual generation objective did not reach functional paths: {missing}')
    model.base_bank.assert_frozen()
    atomic_json(dict(passed=True,revision=cfg['revision'],sample=sample['id'],
                     parameter_count=sum(model.trainable_count().values()),records=records,
                     seconds=time.monotonic()-started,checkpoint_saved=False,
                     formal_training_initialization='separate process; official fresh seed42',
                     video_quality_or_score_improvement_proven=False),args.output)


if __name__=='__main__':main()
