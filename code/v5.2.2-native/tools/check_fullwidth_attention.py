"""One bounded real-model check of native C attention and its input gradients.

No benchmark claim: the migrated C has not been aligned to the new reader, and
the uncalibrated critic is used only to verify the complete RGB-to-C derivative.
No extra video dataset, optimizer updates, video encoding or GPU subprocesses.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from physgen_native.config import read_config,FULLWIDTH_REVISION
from physgen_native.data import VideoDataset,load_tensor,move
from physgen_native.generator import load_wan
from physgen_native.system import PhysGenNative
from physgen_native.runtime import initialize_generator,seed_all,parameter_groups,Logger


def rms(value):return float(value.detach().float().square().mean().sqrt())


def gradients(groups):
    result={}
    for group in groups:
        values=[p.grad.detach().float() for p in group['params'] if p.grad is not None]
        if any(not bool(torch.isfinite(v).all()) for v in values):raise FloatingPointError(group['name'])
        result[group['name']]=sum(float(v.square().sum()) for v in values)**.5
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/native_fullwidth_attention_wisa3857.yaml'))
    parser.add_argument('--initialize-generator',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=read_config(args.config)
    if cfg['revision']!=FULLWIDTH_REVISION:raise ValueError('This check requires full-width native C attention')
    seed_all(42);torch.cuda.set_device(0);device=torch.device('cuda',0)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    log=Logger(out/'metrics.jsonl');start=time.monotonic()
    log.write('loading',0,revision=cfg['revision'],device=torch.cuda.get_device_name(0))
    model=PhysGenNative(cfg,load_wan(cfg),device=device).to(device).train()
    initialization=initialize_generator(model,cfg,args.initialize_generator)
    groups=parameter_groups(model,cfg)
    log.write('start',0,initialization=initialization,
              parameters={g['name']:sum(p.numel() for p in g['params']) for g in groups})
    sample=move(VideoDataset(cfg)[0],device)
    null=load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].to(device)
    sigma=sample['latent'].new_tensor(.25);noise=torch.randn_like(sample['latent'])
    noisy=torch.cat((sample['first'],((1-sigma)*sample['latent']+sigma*noise)[:,:,1:]),2)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        c=model.encode_condition(model.conditions(sample))
        known=(sample['latent'].shape[-2]//2)*(sample['latent'].shape[-1]//2)
        length=sample['latent'].shape[2]*known
        times=torch.cat((torch.zeros(known,device=device),sigma.expand(length-known)*1000))[None]
        expected=model.generator.wan([noisy[0]],times,[sample['text'][0]],length)[0][None].float()
        off=model.generator(noisy,sample['first'],sample['text'],sigma,c,sample['times'],0.)
        torch.testing.assert_close(off,expected,rtol=1e-5,atol=1e-5)
        model.generator.interface.capture=True
        on=model.generator(noisy,sample['first'],sample['text'],sigma,c,sample['times'])
        model.generator.interface.capture=False
        observations=model.generator.interface.observations
        log.write('native_equivalence_and_read',0,sample=sample['id'],
                  native_max_absolute_error=float((expected-off).abs().max()),
                  C_shape=list(c.shape),C_rms=rms(c),off_velocity_rms=rms(off),
                  on_minus_off_rms=rms(on-off),late_on_minus_off_rms=rms((on-off)[:,:,16:]),
                  layers=observations)
    del c,expected,off,on,noisy,noise
    model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    capture={}
    def remember(module,inputs,output):output.retain_grad();capture['C']=output
    handle=model.conditioner.register_forward_hook(remember)
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        losses,weights,metrics=model(sample,null,0,sigma_override=.25)
    model.backward_losses(losses,weights);handle.remove()
    main_gradients=gradients(groups)
    for key in ('condition','interface','knowledge','wan_lora'):
        if main_gradients[key]<=0:raise AssertionError(f'No generation gradient for {key}')
    if main_gradients['critic']!=0:raise AssertionError('Generator loss updated the critic')
    log.write('generation_backward',1,gradients=main_gradients,C_gradient_rms=rms(capture['C'].grad),
              losses={key:float(value.detach()) for key,value in losses.items()},
              peak_gib=torch.cuda.max_memory_allocated()/2**30)
    del capture,losses,weights,metrics
    model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    joint=sum(cfg['stages'][p] for p in ('interface','calibration'))
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        losses,weights,metrics=model.feedback_losses(sample,joint)
    model.backward_losses(losses,weights);feedback_gradients=gradients(groups)
    for key in ('condition','knowledge'):
        if feedback_gradients[key]<=0:raise AssertionError(f'No feedback gradient for {key}')
    if any(feedback_gradients[key]!=0 for key in ('critic','interface','wan_lora')):
        raise AssertionError('Latent-only feedback updated the judge or reader')
    log.write('complete_trajectory_feedback',2,gradients=feedback_gradients,
              peak_gib=torch.cuda.max_memory_allocated()/2**30,**metrics)
    result=dict(completed=True,revision=cfg['revision'],seconds=time.monotonic()-start,
                generation_gradients=main_gradients,feedback_gradients=feedback_gradients,
                interpretation='Execution and derivative evidence only; unaligned C, uncalibrated critic, no quality claim')
    (out/'result.json').write_text(json.dumps(result,indent=2))
    log.write('complete',2,**result);log.close()


if __name__=='__main__':main()
