"""Real-model gradients and a controlled C update, using existing cached video.

The C update uses a reference video as a diagnostic oracle. It is not ordinary
inference or evidence of generalization. Same-norm random updates are the control.
"""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from physgen_native.config import read_config
from physgen_native.data import VideoDataset,load_tensor,move
from physgen_native.generator import load_wan
from physgen_native.system import PhysGenNative
from physgen_native.sequence import sequence_error,perceptual_process_error,sequence_metrics
from physgen_native.runtime import parameter_groups,seed_all,Logger
from train.train import export_video


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'archive/20261027_cleanup/analysis/real_gradient_effect_final'))
    parser.add_argument('--random-controls',type=int,default=3)
    parser.add_argument('--skip-resample',action='store_true')
    args=parser.parse_args()
    cfg=read_config(ROOT/'configs/native_physical_wisa3857.yaml'); seed_all(42); torch.cuda.set_device(0)
    out=Path(args.output); log=Logger(out/'metrics.jsonl')
    model=PhysGenNative(cfg,load_wan(cfg),device='cuda').cuda().train()
    data=move(VideoDataset(cfg,'validation')[0],'cuda')
    null=load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        tokens=model.encode_condition(model.conditions(data));sigma=torch.tensor(.5,device='cuda')
        noisy=torch.randn_like(data['latent']);noisy[:,:,:1]=data['first']
        native_input=noisy[0];known=15*26
        time=torch.cat((torch.zeros(known,device='cuda'),torch.full((20*known,),500.,device='cuda')))[None]
        expected=model.generator.wan([native_input],time,[data['text'][0]],21*known)[0][None]
        actual=model.generator(noisy,data['first'],data['text'],sigma,tokens,data['times'],strength=0.)
        delta=(expected-actual).abs().max()
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
        log.write('native_equivalence',0,max_absolute_error=delta)
    groups=parameter_groups(model,cfg); capture={}
    def remember(module,args,output):
        output.retain_grad();capture['c']=output
    handle=model.conditioner.register_forward_hook(remember)
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        # This micro uses all-time RGB + frozen visual-feature supervision only.
        losses,weights,metrics=model(data,null,cfg['stages']['interface'],micro=1,validation=False,sigma_override=.65)
    losses['sequence'].backward(); handle.remove()
    grads={g['name']:sum(float(p.grad.float().square().sum()) for p in g['params'] if p.grad is not None)**.5 for g in groups}
    c=capture['c'].detach().float()
    log.write('sequence_gradient',0,sample=data['id'],gradients=grads,
              condition_token_gradient=capture['c'].grad.float().square().mean(-1).sqrt().tolist(),
              sequence=losses['sequence'],peak_gib=torch.cuda.max_memory_allocated()/2**30)
    assert all(grads[k]>0 for k in ('condition','interface','knowledge','wan_lora'))
    assert grads['critic']==0
    del capture,losses,weights,metrics
    model.zero_grad(set_to_none=True);model.requires_grad_(False)
    sigma=torch.tensor(.65,device='cuda'); seed_all(123)
    z=data['latent']; noisy=(1-sigma)*z+sigma*torch.randn_like(z)
    noisy[:,:,:1]=data['first']
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        truth=model.decoder(z,resize=cfg['knowledge']['video_size'])
        reference_features=model.knowledge.process_features(truth)
    def objective(tokens):
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            v=model.generator(noisy,data['first'],data['text'],sigma,tokens,data['times'])
            clean=torch.cat((data['first'],(noisy-sigma*v)[:,:,1:]),2)
            frames=model.decoder(clean,resize=cfg['knowledge']['video_size'])
            rgb=sequence_error(frames,truth,data['valid'],cfg['sequence'])[0]
            visual=perceptual_process_error(model.knowledge.process_features(frames),reference_features,data['valid'],cfg['sequence'])[0]
            return .5*(rgb+visual)
    c.requires_grad_(); value=objective(c); gradient=torch.autograd.grad(value,c)[0]
    direction=gradient/gradient.square().mean().sqrt().clamp_min(1e-20)
    scale=c.detach().square().mean().sqrt(); gradient_mode_value=float(value.detach())
    del value,gradient
    with torch.no_grad(): repeated=[float(objective(c.detach())) for _ in range(3)]
    baseline=sum(repeated)/len(repeated);best=c.detach();best_loss=baseline
    log.write('repeated_baseline',0,gradient_mode=gradient_mode_value,evaluation_repeats=repeated,
              evaluation_range=max(repeated)-min(repeated),gradient_mode_gap=gradient_mode_value-baseline)
    for fraction in (.02,.1,.5):
        candidate=c.detach()-fraction*scale*direction
        with torch.no_grad():
            a=float(objective(candidate));controls=[]
            for _ in range(args.random_controls):
                random=torch.randn_like(direction);random/=random.square().mean().sqrt()
                controls.append(float(objective(c.detach()-fraction*scale*random)))
        log.write('paired_C_update',0,fraction=fraction,baseline=baseline,gradient_update=a,random_updates=controls,
                  random_mean=sum(controls)/len(controls),random_min=min(controls),random_max=max(controls))
        if a<best_loss:best,best_loss=candidate.detach(),a
    for name,tokens in ([] if args.skip_resample else [('before',c.detach()),('after',best)]):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            video=model.sample(model.conditions(data),data['text'],null,seed=123,token_override=tokens)
            frames=model.decoder(video,resize=cfg['knowledge']['video_size'])
            process=sequence_error(frames,truth,data['valid'],cfg['sequence'])[0]
            horizon=sequence_metrics(frames,truth,data['valid'])
        log.write('full_resample',0,variant=name,oracle_process_error=process,**horizon)
        export_video(model,video,data,out/f'{name}.mp4',0,123)
    log.write('complete',0,paired_before=baseline,paired_after=best_loss,
              sample=data['id'],interpretation='single-case oracle C optimization; not generalization',
              peak_gib=torch.cuda.max_memory_allocated()/2**30)
    log.close()


if __name__=='__main__':main()
