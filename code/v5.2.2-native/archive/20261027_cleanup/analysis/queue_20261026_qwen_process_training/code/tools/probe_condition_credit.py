"""Locate sampler gradient amplification and match C updates to their derivative.

Uses previously saved legal C and noise, with reference supervision explicitly
marked as an oracle. Compare complete resampling with all-step C changes against
complete resampling that preserves the original C before the differentiated
suffix. Every case, direction and update scale is recorded, without best-of picks.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from physgen_native.data import VideoDataset,load_tensor,move
from physgen_native.generator import load_wan
from physgen_native.process_encoder import VideoProcessEncoder
from physgen_native.process_distribution import ordered_process_distribution
from physgen_native.runtime import Logger,load_checkpoint,restore,seed_all,atomic_save
from physgen_native.system import PhysGenNative
from train.train import export_video


def amp():return torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--condition-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',type=int,default=[1,3])
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,314159])
    parser.add_argument('--fractions',nargs='+',type=float,default=[.0001,.001,.01])
    args=parser.parse_args();seed_all(42);torch.cuda.set_device(0)
    state=load_checkpoint(args.checkpoint);cfg=state['config'];step=state['step']
    model=PhysGenNative(cfg,load_wan(cfg),device='cuda').cuda()
    restore(model,None,None,state,cfg);del state
    model.eval().requires_grad_(False)
    # Saved C was encoded with exactly this checkpoint and legal inputs.
    del model.knowledge,model.conditioner,model.critic
    gc.collect();torch.cuda.empty_cache()
    encoder=VideoProcessEncoder('/home/liuzhirui/model/vjepa2',
        '/home/liuzhirui/model/vjepa2/checkpoint/vjepa2_1_vitG_384.pt',checkpointing=True).eval()
    data=VideoDataset(cfg,'validation');null=load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
    policy=dict(cfg['inference'],gradient_steps=50,discrete_adjoint=True)
    log=Logger(args.output/'metrics.jsonl')
    log.write('protocol',step,oracle=True,update_uses_reference_future=True,video_frames=81,
        sampling_steps=50,comparison='all-step update versus original-C prefix plus updated-C suffix',
        fractions=args.fractions,cases=args.cases,seeds=args.seeds)
    for index in args.cases:
        sample=move(data[index],'cuda');condition=model.conditions(sample);valid=sample['valid']
        with torch.no_grad(),amp():
            reference=encoder(model.decoder(sample['latent'],resize=encoder.size))
        for base_seed in args.seeds:
            seed=base_seed+index;name=f'{sample["id"]}_seed{seed}'
            saved=torch.load(args.condition_directory/f'{name}.pt',map_location='cpu',weights_only=True)
            if Path(saved['checkpoint']).resolve()!=Path(args.checkpoint).resolve():raise ValueError('Saved C checkpoint differs')
            noise=torch.randn(sample['latent'].shape,generator=torch.Generator().manual_seed(seed))
            noise_hash=hashlib.sha256(noise.numpy().tobytes()).hexdigest()
            if noise_hash!=saved['noise_sha256']:raise ValueError('Fixed noise differs')
            noise=noise.cuda();c=saved['c'].cuda();negative=saved['negative'].cuda();random=saved['random'].cuda()
            def sample_tokens(tokens,*,boundary=0,differentiable=False,callback=None):
                return model._sample_tokens(condition,sample['text'],null,seed,policy,
                    initial_noise=noise,token_override=tokens,negative_token_override=negative,
                    prefix_token_override=c,condition_update_start=boundary,
                    differentiable=differentiable,adjoint_callback=callback)
            def objective(features):return ordered_process_distribution(features,reference,valid)
            with torch.no_grad(),amp():
                original=sample_tokens(c);frames=model.decoder(original,resize=encoder.size)
            pixels=frames.detach().requires_grad_(True)
            with amp():loss,baseline_metrics=objective(encoder(pixels))
            baseline=float(loss.detach());pixel_gradient=torch.autograd.grad(loss,pixels)[0]
            del pixels,loss,frames
            clean=original.detach().requires_grad_(True)
            with amp():frames=model.decoder(clean,resize=encoder.size)
            latent_gradient=torch.autograd.grad(frames,clean,grad_outputs=pixel_gradient)[0]
            del clean,frames,pixel_gradient
            credits={}
            def record_credit(i,adjoint,credit):
                k=policy['steps']-i
                log.write('step_credit',step,sample=sample['id'],seed=seed,sampler_step=i,suffix_steps=k,
                    state_gradient_rms=adjoint.float().square().mean().sqrt(),
                    condition_gradient_rms=credit.float().square().mean().sqrt(),
                    condition_gradient_max=credit.abs().max())
                if k in (1,4,12,25,50):credits[k]=credit.detach().cpu()
            leaf=c.detach().clone().requires_grad_(True)
            with amp():recomputed=sample_tokens(leaf,differentiable=True,callback=record_credit)
            torch.testing.assert_close(recomputed.detach(),original,atol=1e-5,rtol=1e-5)
            full_gradient=torch.autograd.grad(recomputed,leaf,grad_outputs=latent_gradient)[0]
            torch.testing.assert_close(full_gradient.cpu(),credits[50])
            if not all(bool(torch.isfinite(g).all()) for g in credits.values()):raise FloatingPointError('Nonfinite sampler credit')
            atomic_save(dict(gradients=credits,noise_sha256=noise_hash,condition_source=str(args.condition_directory/f'{name}.pt'),
                             update_uses_reference_future=True),args.output/'credits'/f'{name}.pt')
            del recomputed,leaf,latent_gradient,full_gradient
            log.write('baseline',step,sample=sample['id'],seed=seed,objective_loss=baseline,**baseline_metrics)
            with torch.no_grad():export_video(model,original,sample,args.output/'videos'/f'{name}_original.mp4',step,seed)
            c_rms=c.square().mean().sqrt()
            for k,scope in [(12,'all_steps'),(12,'matched_suffix'),(50,'all_steps')]:
                gradient=credits[k].cuda();g_rms=gradient.square().mean().sqrt()
                direction=gradient/g_rms.clamp_min(1e-20);boundary=50-k if scope=='matched_suffix' else 0
                log.write('direction',step,sample=sample['id'],seed=seed,gradient_steps=k,scope=scope,
                    gradient_rms=g_rms,condition_rms=c_rms,
                    full_gradient_cosine=torch.nn.functional.cosine_similarity(gradient.flatten(),credits[50].cuda().flatten(),dim=0))
                for fraction in args.fractions:
                    for kind,delta in [('descent',-direction),('ascent',direction),('random',-random)]:
                        tokens=c+fraction*c_rms*delta
                        with torch.no_grad(),amp():
                            generated=sample_tokens(tokens,boundary=boundary)
                            value,metrics=objective(encoder(model.decoder(generated,resize=encoder.size)))
                            predicted=baseline+fraction*c_rms*(gradient*delta).sum()
                            if fraction==max(args.fractions) and kind=='descent':
                                file=args.output/'videos'/f'{name}_k{k}_{scope}_{kind}_{fraction}.mp4'
                                export_video(model,generated,sample,file,step,seed)
                                metadata=json.loads(file.with_suffix('.json').read_text())
                                metadata.update(update_uses_reference_future=True,oracle=True,
                                    conditions='Training oracle C update; not ordinary inference or a generalization result',
                                    update_scope=scope,condition_update_start=boundary,gradient_steps=k,
                                    condition_update_fraction=fraction,noise_sha256=noise_hash)
                                file.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
                        log.write('resample',step,sample=sample['id'],seed=seed,gradient_steps=k,
                            scope=scope,update_start=boundary,fraction=fraction,direction=kind,
                            baseline_loss=baseline,objective_loss=value,linear_prediction=predicted,
                            linear_prediction_matches_update_scope=(scope=='matched_suffix' or k==50),**metrics)
                        del generated
            del original
    log.write('complete',step);log.close()


if __name__=='__main__':main()
