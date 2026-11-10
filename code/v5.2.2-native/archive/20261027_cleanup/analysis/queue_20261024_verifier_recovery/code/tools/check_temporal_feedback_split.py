"""Exact staged chain-rule study with resident Wan and video evidence.

This changes memory scheduling, not the chosen suffix derivative. Qwen conditions
are computed from legal inputs once, then Qwen leaves the GPUs for this C-only
experiment. The reference future supervises the update: this is a training oracle,
not an inference method or evidence of generalization.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from physgen_native.data import VideoDataset,load_tensor,move
from physgen_native.generator import load_wan
from physgen_native.process_encoder import VideoProcessEncoder
from physgen_native.process_distribution import ordered_process_distribution
from physgen_native.runtime import load_checkpoint,restore,Logger,seed_all,atomic_save
from physgen_native.sequence import sequence_error,sequence_metrics
from physgen_native.system import PhysGenNative
from train.train import export_video


def autocast():
    return torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',type=int,nargs='+',default=[0,1,2,3,4])
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,314159])
    parser.add_argument('--fractions',type=float,nargs='+',default=[.1,.5])
    parser.add_argument('--gradient-steps',type=int,default=12)
    parser.add_argument('--discrete-adjoint',action='store_true')
    parser.add_argument('--single-device',action='store_true',help='Run all stages on one allocated blk-96g GPU')
    parser.add_argument('--condition-controls',action='store_true',help='Also fully sample with injection off and another sample\'s legal C')
    parser.add_argument('--verifier',type=Path,help='Use a trained process-evidence verifier; reference is then evaluation only')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--noise-directory',type=Path)
    args=parser.parse_args()
    if not args.single_device and torch.cuda.device_count()<2:raise ValueError('Choose --single-device or allocate two GPUs')
    wan_device=torch.device('cuda:0');evidence_device=torch.device('cuda:0' if args.single_device else 'cuda:1')
    devices=sorted({wan_device.index,evidence_device.index})
    torch.cuda.set_device(wan_device);seed_all(42)
    log=Logger(args.output/f'metrics_shard{args.shard}.jsonl')
    started=time.monotonic()
    state=load_checkpoint(args.checkpoint);cfg=state['config']
    if cfg['verifier'].get('kind')=='ordered_process':
        raise ValueError('Use a pre-feedback checkpoint for this capacity experiment')
    model=PhysGenNative(cfg,load_wan(cfg),device=evidence_device)
    model.generator.to(wan_device);model.conditioner.to(evidence_device)
    completed,_=restore(model,None,None,state,cfg);del state
    uses_reference=args.verifier is None
    verifier_state=torch.load(args.verifier,map_location='cpu',weights_only=True) if args.verifier else None
    model.eval().requires_grad_(False)
    policy=dict(cfg['inference'],gradient_steps=args.gradient_steps,step_checkpointing=True,
                discrete_adjoint=args.discrete_adjoint)
    if not 1<=args.gradient_steps<=policy['steps']:raise ValueError('Invalid suffix length')
    data=VideoDataset(cfg,'validation')
    indices=[index for order,index in enumerate(args.cases) if order%args.shards==args.shard]
    if args.condition_controls and len(args.cases)<2:raise ValueError('C swapping needs two different samples')
    partners={index:args.cases[(order+1)%len(args.cases)] for order,index in enumerate(args.cases)}
    encode_indices=sorted(set(indices+[partners[i] for i in indices])) if args.condition_controls else indices
    conditions={};fixed_conditions={}
    for index in encode_indices:
        sample=move(data[index],evidence_device);condition=model.conditions(sample)
        with torch.no_grad(),autocast():
            first=model.decoder(sample['first'],[0])
            positive=model.encode_condition(condition,first).float().cpu()
            rationale=model.last_condition_analysis
            negative=model.encode_condition(dict(condition,caption='',clauses=[]),first).float().cpu()
            if verifier_state is not None:
                fixed_conditions[index]=model.knowledge.encode(first,condition['caption'],condition['clauses'],
                    adapt=False,readout='queries')['contextual'].detach().cpu()
        conditions[index]=(positive,negative,rationale)
        log.write('condition_precomputed',completed,sample=sample['id'],condition_uses_reference_future=False)
    del sample,condition,first,positive,negative
    # No Qwen weight is updated or needed after both legal CFG branches exist.
    model.knowledge.cpu();model.conditioner.cpu();gc.collect()
    torch.cuda.empty_cache()
    encoder=VideoProcessEncoder('/home/liuzhirui/model/vjepa2',
        '/home/liuzhirui/model/vjepa2/checkpoint/vjepa2_1_vitG_384.pt',
        device=evidence_device,checkpointing=True).eval()
    if verifier_state is not None:
        from physgen_native.evidence_verifier import ProcessEvidenceVerifier
        if {data.rows[i]['id'] for i in indices}&set(verifier_state['train_ids']):raise ValueError('Verifier training overlap')
        verifier=ProcessEvidenceVerifier(verifier_state['feature_width'],verifier_state['condition_width'],
            verifier_state['width'],lags=verifier_state['lags']).to(evidence_device).eval().requires_grad_(False)
        verifier.load_state_dict(verifier_state['model']);del verifier_state
    null=load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].to(wan_device)
    objective_name='ordered_reference' if uses_reference else 'learned_process_verifier'
    log.write('protocol',completed,objective=objective_name,update_uses_reference_future=uses_reference,
        verifier=str(args.verifier.resolve()) if args.verifier else None,
        gradient_steps=args.gradient_steps,step_checkpointing=True,steps=policy['steps'],
        discrete_adjoint=args.discrete_adjoint,
        devices=[torch.cuda.get_device_name(i) for i in devices],
        checkpoint=str(Path(args.checkpoint).resolve()),
        derivative='Exact staged VJP for the specified denoising suffix; prefix detached',
        knowledge_on_cpu_after_legal_condition_encoding=True)

    for index in indices:
        cpu_sample=data[index];sample=move(cpu_sample,wan_device);condition=model.conditions(sample)
        valid=sample['valid'].to(evidence_device)
        fixed=fixed_conditions[index].to(evidence_device) if not uses_reference else None
        c,negative,rationale=conditions[index];c=c.to(wan_device);negative=negative.to(wan_device)
        with torch.no_grad(),autocast():
            truth=model.decoder(cpu_sample['latent'].to(evidence_device),resize=encoder.size)
            reference=encoder(truth)
        def objective(features):
            return (ordered_process_distribution(features,reference,valid)[0] if uses_reference else
                    verifier.generation_loss(features,fixed,sample['times'].to(evidence_device),valid))

        for base_seed in args.seeds:
            seed=base_seed+index
            noise=(load_tensor(args.noise_directory/f'seed{seed}.pt') if args.noise_directory else
                   torch.randn(sample['latent'].shape,generator=torch.Generator().manual_seed(seed)))
            noise_hash=hashlib.sha256(noise.numpy().tobytes()).hexdigest();noise=noise.to(wan_device)
            with torch.no_grad(),autocast():
                original=model._sample_tokens(condition,sample['text'],null,seed,policy,
                    token_override=c,negative_token_override=negative,initial_noise=noise)
                frames=model.decoder(original.to(evidence_device),resize=encoder.size)
            log.write('stage',completed,sample=sample['id'],seed=seed,stage='sample_complete',seconds=time.monotonic()-started)
            # Stage 1: dL/dRGB, without retaining the VAE or Wan graph.
            pixels=frames.detach().requires_grad_(True)
            with autocast():loss=objective(encoder(pixels))
            pixel_gradient=torch.autograd.grad(loss,pixels)[0]
            loss_value=float(loss.detach());del pixels,loss,frames
            log.write('stage',completed,sample=sample['id'],seed=seed,stage='observer_vjp',seconds=time.monotonic()-started)
            # Stage 2: dL/dz through the complete native causal VAE.
            clean=original.detach().to(evidence_device).requires_grad_(True)
            with autocast():decoded=model.decoder(clean,resize=encoder.size)
            latent_gradient=torch.autograd.grad(decoded,clean,grad_outputs=pixel_gradient)[0]
            del decoded,clean,pixel_gradient
            log.write('stage',completed,sample=sample['id'],seed=seed,stage='vae_vjp',seconds=time.monotonic()-started)
            # Stage 3: dL/dC through the selected unbroken Wan suffix.
            leaf=c.detach().clone().requires_grad_(True)
            with autocast():
                latent=model._sample_tokens(condition,sample['text'],null,seed,policy,differentiable=True,
                    token_override=leaf,negative_token_override=negative,initial_noise=noise)
            difference=(latent.detach()-original).abs().max()
            torch.testing.assert_close(latent.detach(),original,atol=1e-5,rtol=1e-5)
            gradient=torch.autograd.grad(latent,leaf,grad_outputs=latent_gradient.to(wan_device))[0]
            rms=gradient.square().mean().sqrt();direction=gradient/rms.clamp_min(1e-20)
            generator=torch.Generator().manual_seed(seed+1000003)
            random=torch.randn(c.shape,generator=generator).to(wan_device)
            random=random/random.square().mean().sqrt()
            log.write('gradient',completed,sample=sample['id'],seed=seed,objective_loss=loss_value,
                condition_gradient_rms=rms,recomputed_latent_max_difference=difference,noise_sha256=noise_hash,
                seconds=time.monotonic()-started,update_uses_reference_future=uses_reference,
                peak_gib={str(i):torch.cuda.max_memory_allocated(i)/2**30 for i in devices})
            atomic_save(dict(c=c.cpu(),negative=negative.cpu(),direction=direction.detach().cpu(),
                random=random.cpu(),noise_sha256=noise_hash,gradient_steps=args.gradient_steps,
                checkpoint=args.checkpoint,update_uses_reference_future=uses_reference,
                verifier=str(args.verifier.resolve()) if args.verifier else None),
                args.output/'directions'/f'{sample["id"]}_seed{seed}.pt')
            del latent,leaf,gradient,latent_gradient
            candidates=[('original',c,0.)]
            for fraction in args.fractions:
                scale=fraction*c.square().mean().sqrt()
                candidates += [(f'gradient_{fraction}',c-scale*direction,fraction),
                               (f'random_{fraction}',c-scale*random,fraction)]
            if args.condition_controls:
                candidates += [('injection_off',c,0.),('other_condition',conditions[partners[index]][0].to(wan_device),0.)]
            for variant,tokens,fraction in candidates:
                with torch.no_grad(),autocast():
                    latent=original if variant=='original' else model._sample_tokens(
                        condition,sample['text'],null,seed,policy,token_override=tokens,
                        negative_token_override=negative,initial_noise=noise,
                        strength=0. if variant=='injection_off' else cfg['inference']['condition_strength'])
                    frames=model.decoder(latent.to(evidence_device),resize=encoder.size)
                    observed=encoder(frames)
                    reference_error,features_metrics=ordered_process_distribution(observed,reference,valid)
                    score=objective(observed)
                    rgb_error=sequence_error(frames,truth,valid,cfg['sequence'])[0]
                    metrics=sequence_metrics(frames,truth,valid)
                    file=args.output/'videos'/f'{sample["id"]}_seed{seed}_{variant}.mp4'
                    export_video(model,latent.to(evidence_device),sample,file,completed,seed)
                metadata=json.loads(file.with_suffix('.json').read_text())
                metadata.update(variant=variant,relative_condition_update=fraction,
                    physical_dependency_analysis=rationale,objective=objective_name,
                    verifier=str(args.verifier.resolve()) if args.verifier else None,
                    update_uses_reference_future=uses_reference and variant.startswith('gradient'),noise_sha256=noise_hash,
                    gradient_steps=args.gradient_steps,
                    condition_source=(data.rows[partners[index]]['id'] if variant=='other_condition' else sample['id']),
                    injection_strength=0. if variant=='injection_off' else cfg['inference']['condition_strength'],
                    conditions=('Training oracle: C updated using the reference future; not ordinary inference'
                                if uses_reference and variant.startswith('gradient') else
                                'first frame, caption, existing requirements; no future target'))
                file.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
                log.write('full_resample',completed,sample=sample['id'],seed=seed,variant=variant,
                    objective_loss=score,reference_process_error=rgb_error,
                    ordered_reference_error=reference_error,
                    seconds=time.monotonic()-started,**features_metrics,**metrics)
                del latent,frames
            del original
    log.write('complete',completed,shard=args.shard,seconds=time.monotonic()-started,
        peak_gib={str(i):torch.cuda.max_memory_allocated(i)/2**30 for i in devices})
    log.close()


if __name__=='__main__':main()
