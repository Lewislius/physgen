"""Does a process-supervision C gradient survive complete resampling?

The critic mode uses no reference future for its update; ordered_reference is an
explicit training-target oracle, not deployable inference or generalization.
Same-norm random directions, complete resampling and failures are all recorded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from physgen_native.data import VideoDataset,load_tensor,move
from physgen_native.generator import load_wan
from physgen_native.runtime import load_checkpoint,restore,Logger,seed_all
from physgen_native.system import PhysGenNative
from physgen_native.temporal_critic import TemporalProcessCritic
from physgen_native.sequence import sequence_error,sequence_metrics
from physgen_native.process_distribution import ordered_process_distribution
from train.train import export_video


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--critic')
    parser.add_argument('--objective',choices=['critic','ordered_reference'],default='critic')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',type=int,nargs='+',default=[0,1,3,4])
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,314159])
    parser.add_argument('--fractions',type=float,nargs='+',default=[.1,.5])
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--gradient-steps',type=int,default=None)
    parser.add_argument('--step-checkpointing',action='store_true')
    parser.add_argument('--noise-directory',type=Path)
    args=parser.parse_args();torch.cuda.set_device(0);seed_all(42)
    state=load_checkpoint(args.checkpoint);cfg=state['config']
    model=PhysGenNative(cfg,load_wan(cfg),device='cuda').cuda()
    completed,_=restore(model,None,None,state,cfg);del state
    model.eval().requires_grad_(False)
    uses_reference=args.objective=='ordered_reference'
    if uses_reference:
        critic_state=dict(evidence_encoder='V-JEPA 2.1 ViT-g',evidence_context='full',train_ids=[])
    else:
        if args.critic is None:raise ValueError('A trained critic is required for the critic objective')
        critic_state=torch.load(args.critic,map_location='cpu',weights_only=True)
        critic=TemporalProcessCritic(critic_state['feature_width'],critic_state['condition_width'],
                                    critic_state['width']).cuda().eval().requires_grad_(False)
        critic.load_state_dict(critic_state['model'],strict=True)
        if critic_state['mode']!='temporal':raise ValueError('Use the temporal critic, not its time-pooled control')
    evidence_encoder=critic_state.get('evidence_encoder','Qwen3-VL visual')
    if evidence_encoder=='V-JEPA 2.1 ViT-g':
        from physgen_native.process_encoder import VideoProcessEncoder
        encoder=VideoProcessEncoder('/home/liuzhirui/model/vjepa2',
            '/home/liuzhirui/model/vjepa2/checkpoint/vjepa2_1_vitG_384.pt',checkpointing=True).eval()
        evidence=lambda frames:encoder(frames,context=critic_state['evidence_context'])
    elif evidence_encoder=='Qwen3-VL visual':
        evidence=model.knowledge.process_features
    else:raise ValueError(f'Unknown trained evidence encoder: {evidence_encoder}')
    policy=dict(cfg['rollout'])
    if args.gradient_steps is not None:policy['gradient_steps']=args.gradient_steps
    if args.step_checkpointing:policy['step_checkpointing']=True
    if not 1<=policy['gradient_steps']<=policy['steps']:raise ValueError('Invalid differentiable suffix length')
    data=VideoDataset(cfg,'validation')
    if {data.rows[i]['id'] for i in args.cases}&set(critic_state['train_ids']):raise ValueError('Critic training overlap')
    null=load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
    log=Logger(args.output/f'metrics_shard{args.shard}.jsonl')
    log.write('protocol',completed,evidence_encoder=evidence_encoder,
              evidence_context=critic_state.get('evidence_context','pairs'),
              gradient_steps=policy['gradient_steps'],device=torch.cuda.get_device_name(0),
              step_checkpointing=policy.get('step_checkpointing',False),
              objective=args.objective,update_uses_reference_future=uses_reference)
    for order,index in enumerate(args.cases):
        if order%args.shards!=args.shard:continue
        sample=move(data[index],'cuda');condition=model.conditions(sample)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            first=model.decoder(sample['first'],[0])
            c=model.encode_condition(condition,first).detach().float()
            fixed=model.knowledge.encode(first,sample['caption'],sample['clauses'],adapt=False,readout='queries')['contextual'].detach()
            truth=model.decoder(sample['latent'],resize=cfg['knowledge']['video_size'])
            reference_features=evidence(truth) if uses_reference else None
        def objective(features):
            return (ordered_process_distribution(features,reference_features,sample['valid'])[0] if uses_reference else
                    critic.generation_loss(features,fixed,sample['times'],sample['valid']))
        for base_seed in args.seeds:
            seed=base_seed+index;leaf=c.clone().requires_grad_(True)
            noise=(load_tensor(args.noise_directory/f'seed{seed}.pt') if args.noise_directory else
                   torch.randn(sample['latent'].shape,generator=torch.Generator().manual_seed(seed)))
            noise_hash=hashlib.sha256(noise.numpy().tobytes()).hexdigest();noise=noise.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                latent=model._sample_tokens(condition,sample['text'],null,seed,policy,
                                            differentiable=True,token_override=leaf,initial_noise=noise)
                frames=model.decoder(latent,resize=cfg['knowledge']['video_size'])
                features=evidence(frames)
                loss=objective(features)
            gradient=torch.autograd.grad(loss,leaf)[0]
            rms=gradient.square().mean().sqrt()
            direction=gradient/rms.clamp_min(1e-20)
            generator=torch.Generator(device='cuda').manual_seed(seed+1000003)
            random=torch.randn(c.shape,device='cuda',generator=generator)
            random=random/random.square().mean().sqrt()
            log.write('gradient',completed,sample=sample['id'],seed=seed,objective_loss=loss,
                      condition_gradient_rms=rms,gradient_steps=policy['gradient_steps'],noise_sha256=noise_hash,
                      update_uses_reference_future=uses_reference,critic=str(Path(args.critic).resolve()) if args.critic else None)
            del latent,frames,features,loss,leaf,gradient
            candidates=[('original',c,0.)]
            for fraction in args.fractions:
                scale=fraction*c.square().mean().sqrt()
                candidates += [(f'gradient_{fraction}',c-scale*direction,fraction),
                               (f'random_{fraction}',c-scale*random,fraction)]
            for variant,tokens,fraction in candidates:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                    latent=model.sample(condition,sample['text'],null,seed=seed,token_override=tokens,initial_noise=noise)
                    frames=model.decoder(latent,resize=cfg['knowledge']['video_size'])
                    score=objective(evidence(frames))
                    oracle=sequence_error(frames,truth,sample['valid'],cfg['sequence'])[0]
                    metrics=sequence_metrics(frames,truth,sample['valid'])
                    file=args.output/'videos'/f'{sample["id"]}_seed{seed}_{variant}.mp4'
                    export_video(model,latent,sample,file,completed,seed)
                metadata=json.loads(file.with_suffix('.json').read_text())
                metadata.update(variant=variant,relative_condition_update=fraction,
                                critic=str(Path(args.critic).resolve()) if args.critic else None,
                                objective=args.objective,update_uses_reference_future=uses_reference,
                                noise_sha256=noise_hash,gradient_steps=policy['gradient_steps'])
                file.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
                log.write('full_resample',completed,sample=sample['id'],seed=seed,variant=variant,
                          objective_loss=score,reference_process_error_for_evaluation_only=oracle,**metrics)
                del latent,frames
        del sample,truth,c,fixed
    log.write('complete',completed,shard=args.shard,peak_gib=torch.cuda.max_memory_allocated()/2**30)
    log.close()


if __name__=='__main__':main()
