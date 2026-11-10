"""Three disposable full-size training passes; never initialize formal training."""
from pathlib import Path
import argparse,json,random,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import numpy as np
import torch
from physical_content.common import load_config,gpu_guard,atomic_json
from physical_content.model import PhysicalModel
from physical_content.objectives import backward_sample
from physical_content.teacher import Teacher
from physical_scaling.data import Data,SampleOrder
from physical_scaling.schedule import rates,auxiliary_coefficient


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cfg=load_config(args.config);folder=args.output
    gpu=gpu_guard(folder)
    torch.set_num_threads(4);random.seed(42);np.random.seed(42)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    atomic_json(dict(status='loading_fresh_official_model',gpu=gpu),folder/'status.json')
    try:
        model=PhysicalModel(cfg).train();model.wan.eval();teacher=Teacher(cfg)
        named=dict(model.named_parameters());groups=model.optim_groups()
        count=sum(named[n].numel() for n in model.learned_names)
        if count!=685739008:raise AssertionError(f'Unexpected trainable ownership: {count}')
        if any(p.requires_grad for block in model.wan.blocks for p in block.cross_attn.parameters()):
            raise AssertionError('Native text projections must stay frozen')
        optimizer=torch.optim.AdamW(groups,betas=(.9,.999),weight_decay=.01,foreach=False)
        data=Data(cfg);index=next(iter(SampleOrder(len(data),cfg['train']['seed'],0)))
        sample=data.get(index);evidence=teacher.input(sample)
        noise=torch.randn(sample['latent'].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(42))
        sigma=torch.tensor(.6,device='cuda');rows=[]
        for iteration,step in enumerate((0,1,128)):
            phase=model.set_phase(step);rates(optimizer,cfg,step);optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                values=backward_sample(model,evidence,sample,sample['text'],sigma,noise,
                                       auxiliary_coefficient(cfg,step),1)
            fusion=model.fusions['2']
            grad=lambda p:None if p.grad is None else float(p.grad.detach().float().norm())
            relevant=dict(output=grad(fusion.output.weight),denoising=grad(fusion.denoising.weight),
                          execution_C=grad(model.enhancement.execution_project.weight),
                          key=grad(fusion.key.weight),value=grad(fusion.value.weight),state_value=grad(fusion.state_value.weight))
            if not relevant['output']>0:raise AssertionError('No generation gradient to physical output')
            if iteration>=1 and not relevant['denoising']>0:raise AssertionError('Diffusion-conditioned physical path did not open')
            if iteration>=1 and not relevant['execution_C']>0:raise AssertionError('Generation gradient did not reach execution C')
            for name,param in model.wan.named_parameters():
                if param.grad is not None and (phase=='interface' or not name.endswith(('lora_a','lora_b'))):
                    raise AssertionError(f'Unexpected native gradient: {phase}/{name}')
            norms={}
            for group in optimizer.param_groups:
                params=[p for p in group['params'] if p.grad is not None]
                norms[group['name']]=float(torch.nn.utils.clip_grad_norm_(params,cfg['train']['grad_clip'][group['name']],error_if_nonfinite=True)) if params else 0.
            if phase=='joint' and not norms['wan_lora']>0:raise AssertionError('Joint phase has no native LoRA gradient')
            optimizer.step();torch.cuda.synchronize()
            row=dict(disposable_iteration=iteration+1,phase=phase,sample=sample['id'],seconds=time.monotonic()-start,
                     peak_gib=torch.cuda.max_memory_allocated()/2**30,gradients=relevant,group_gradients=norms,**values)
            rows.append(row);print(json.dumps(row),flush=True)
            atomic_json(dict(status='running',rows=rows),folder/'status.json')
        report=dict(status='passed',gpu=gpu,parameters=count,
                    parameter_groups={g['name']:sum(p.numel() for p in g['params']) for g in groups},
                    rows=rows,quality_proof=False,temporary_weights_saved=False,
                    formal_training='Separate process reloads official foundations, seed42, cursor0; never uses these temporary weights')
        atomic_json(report,folder/'result.json');atomic_json(report,folder/'status.json')
    except BaseException as error:
        atomic_json(dict(status='failed',error=repr(error)),folder/'status.json');raise


if __name__=='__main__':main()
