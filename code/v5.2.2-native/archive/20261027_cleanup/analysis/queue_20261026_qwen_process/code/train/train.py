"""Train the native process revision on one resident blk-96g GPU."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
import yaml
from physgen_native.config import read_config, smoke_config
from physgen_native.data import VideoDataset, SampleOrder, audit, load_tensor, move, manifest
from physgen_native.generator import load_wan
from physgen_native.system import PhysGenNative
from physgen_native.runtime import parameter_groups, schedule, seed_all, save_checkpoint, load_checkpoint, restore, Logger
from physgen_native.execution_trace import trace_micro, traced_call


@torch.no_grad()
def export_video(model, latent, sample, path, step, seed):
    import imageio.v2 as imageio
    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        frames = model.decoder(latent)
    frames = ((frames.permute(0, 2, 3, 1).clamp(-1, 1)+1)*127.5).round().byte().cpu().numpy()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fps = 4/float(sample['times'][1]-sample['times'][0])
    temporary = path.with_suffix('.partial.mp4')
    try:
        with imageio.get_writer(str(temporary), fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None,
                               quality=None,ffmpeg_params=['-crf','18','-profile:v','baseline']) as stream:
            for frame in frames: stream.append_data(frame)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    path.with_suffix('.json').write_text(json.dumps(dict(id=sample['id'], caption=sample['caption'],
        revision=model.cfg['revision'], optimizer_step=step, seed=seed, fps=fps,
        conditions='first frame, caption, existing requirements; no future target',
        physical_dependency_analysis=getattr(model,'last_sample_analysis',''),
        latent_reasoning_steps=model.cfg['knowledge'].get('latent_reasoning_steps',0),
        latent_feedback=getattr(model,'last_sample_feedback',None),
        sampler=model.cfg['inference']), ensure_ascii=False, indent=2))


@torch.no_grad()
def validate(model, dataset, null_text, completed, run, device):
    cfg = model.cfg; policy = cfg['validation']
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), random.getstate())
    training = model.training; model.eval(); values = defaultdict(list)
    records = Logger(run/'validation_cases.jsonl')
    try:
        if model.qwen_process and model.phase == 'calibration':
            # The generator is unchanged during this phase. Validate the learned
            # judge, rather than repeatedly exporting that same generator.
            for index in torch.linspace(0, len(dataset)-1, min(len(dataset), policy['samples'])).long().tolist():
                sample = move(dataset[index], device)
                for view in (0, 1):
                    seed_all(cfg['train']['seed']+index)
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        _, _, metrics = model.qwen_calibration(sample, 0, view)
                    records.write('qwen_judge_validation', completed, sample=sample['id'], **metrics)
                    for key, value in metrics.items(): values[key].append(float(value))
            return {k:sum(v)/len(v) for k,v in values.items()}
        for index in torch.linspace(0, len(dataset)-1, min(len(dataset), policy['samples'])).long().tolist():
            sample = move(dataset[index], device)
            for sigma in policy['sigmas']:
                seed_all(cfg['train']['seed']+index)
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    _, _, metrics = model(sample, null_text, max(0, completed-1), validation=True, sigma_override=sigma)
                records.write('paired_validation', completed, sample=sample['id'], **metrics)
                for key, value in metrics.items(): values[f'sigma{sigma}/{key}'].append(float(value))
        if completed % policy['sample_every'] == 0:
            for index in range(min(len(dataset), policy['sample_cases'])):
                sample = move(dataset[index], device)
                for base_seed in policy.get('sample_seeds', [cfg['train']['seed']]):
                    seed = base_seed+index
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        latent = model.sample(model.conditions(sample), sample['text'], null_text, seed=seed)
                    filename = f'step{completed:04d}_{sample["id"]}_seed{seed}.mp4'
                    export_video(model, latent, sample, Path(cfg['paths']['outputs'])/run.name/filename, completed, seed)
        result={k:sum(v)/len(v) for k,v in values.items()}
        for sigma in policy['sigmas']:
            for h in range(1,cfg['data']['latent_shape'][1]):
                prefix=f'sigma{sigma}/rgb/h{h}_'
                energy=sum(values[prefix+'true_change_mse'])
                predicted=sum(values[prefix+'predicted_change_mse'])
                error=sum(values[prefix+'change_error'])
                result[prefix+'pooled_change_skill']=1-error/max(1e-8,energy)
                result[prefix+'pooled_change_rms_ratio']=(predicted/max(1e-8,energy))**.5
        return result
    finally:
        records.close()
        torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1]); random.setstate(rng[2])
        model.train(training); model.set_phase(completed)


def train(args):
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError('--stop-after must be a positive optimizer-step count')
    cfg = read_config(args.config)
    if args.smoke: cfg = smoke_config(cfg)
    if args.smoke_accumulation is not None:
        if not args.smoke or args.smoke_accumulation < 1:
            raise ValueError('--smoke-accumulation requires --smoke and a positive value')
        cfg['train']['accumulation'] = args.smoke_accumulation
        cfg['run_name'] += f'-accum{args.smoke_accumulation}'
    report = audit(cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report['ready']: raise RuntimeError('Existing resources are incomplete; no automatic preprocessing or downloads')
    if args.check_only: return
    if not torch.cuda.is_available(): raise RuntimeError('Use the supplied blk-96g YAML or an allocated GPU')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1: raise ValueError('This recipe is single-GPU')
    torch.cuda.set_device(0); device = torch.device('cuda', 0)
    suffix = os.environ.get('DET_EXPERIMENT_ID') or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run = Path(args.resume) if args.resume else Path(cfg['paths']['runs'])/f'{cfg["run_name"]}_{suffix}'
    resume = bool(args.resume) or (os.environ.get('AUTO_RESUME') == '1' and (run/'latest.pt').is_file())
    if cfg['verifier'].get('initialization_required') and not resume and not args.initialize_critic:
        raise ValueError('This recipe needs the calibrated train-split verifier via --initialize-critic before generation feedback')
    state = load_checkpoint(run/'latest.pt') if resume else None
    if resume and (args.initialize_generator or args.initialize_critic):
        raise ValueError('Resume restores its own learned weights; initialization starts a separate run')
    if state is not None:
        if state['config'] != cfg: raise ValueError('Saved configuration differs; pass its config.yaml to resume')
    else:
        if run.exists():
            if os.environ.get('AUTO_RESUME') != '1' or not (run/'config.yaml').is_file():
                raise FileExistsError(run)
            if yaml.safe_load((run/'config.yaml').read_text()) != cfg:
                raise ValueError('An incomplete run exists with a different configuration')
        else:
            run.mkdir(parents=True)
        (run/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    seed_all(cfg['train']['seed'])
    dataset, heldout = VideoDataset(cfg), VideoDataset(cfg, 'validation')
    print('[load] native Wan / grounded Qwen / video critic / VAE', flush=True)
    model = PhysGenNative(cfg, load_wan(cfg), device=device).to(device)
    if args.initialize_generator:
        source=load_checkpoint(args.initialize_generator)
        for key in ('wan_checkpoint','qwen_checkpoint'):
            if source['config']['paths'][key] != cfg['paths'][key]:raise ValueError('Initialization base model differs')
        learned={n:p for n,p in model.named_parameters() if not n.startswith('critic.') and
                 (n.startswith(('conditioner.','generator.interface.')) or n.endswith(('lora_a','lora_b')))}
        saved={n:p for n,p in source['model'].items() if not n.startswith('critic.')}
        if set(learned)!=set(saved):raise ValueError('Generator initialization parameter sets differ')
        with torch.no_grad():
            for name,parameter in learned.items():parameter.copy_(saved[name].to(parameter))
        del source,saved,learned
    if args.initialize_critic:
        source=torch.load(args.initialize_critic,map_location='cpu',weights_only=True)
        if model.calibrated_verifier:
            if source.get('method')!=cfg['verifier'].get('calibration','regression'):
                raise ValueError('Initialized verifier method differs from online calibration')
        elif not model.temporal_verifier or source.get('mode')!='temporal':raise ValueError('Use a temporal critic initialization')
        training_ids={row['id'] for row in manifest(cfg) if row['split']=='train'}
        if not set(source['train_ids'])<=training_ids:raise ValueError('Critic initialization trained outside the training split')
        model.critic.load_state_dict(source['model'],strict=True)
        del source
    groups = parameter_groups(model, cfg)
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['train']['weight_decay'], foreach=False)
    scheduler = schedule(optimizer, cfg)
    completed, cursor = restore(model, optimizer, scheduler, state, cfg) if state is not None else (0, 0)
    del state
    model.set_phase(completed)
    loader = DataLoader(dataset, batch_size=None, sampler=SampleOrder(len(dataset), cfg['train']['seed'], cursor),
        num_workers=cfg['data']['workers'], pin_memory=True,
        multiprocessing_context='spawn' if cfg['data']['workers'] else None,
        generator=torch.Generator().manual_seed(cfg['train']['seed']))
    iterator = iter(loader)
    null = load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].to(device)
    logger = Logger(run/'metrics.jsonl')
    total = sum(cfg['stages'].values())
    stop = min(total, completed+args.stop_after) if args.stop_after else total
    logger.write('start', completed, revision=cfg['revision'], budgets=cfg['stages'],
        device_name=torch.cuda.get_device_name(0),device_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
        devices=1, resident_training=True, cpu_offload=False, cursor=cursor, smoke=bool(cfg.get('smoke')),
        parameter_counts={g['name']: sum(p.numel() for p in g['params']) for g in groups},
        initialized_generator=args.initialize_generator,initialized_critic=args.initialize_critic)
    main_params = [p for g in optimizer.param_groups if g['name'] != 'critic' for p in g['params']]
    head_params = [p for g in optimizer.param_groups if g['name'] == 'critic' for p in g['params']]
    accumulation = cfg['train']['accumulation']
    model.train()
    try:
        if completed == 0 and cfg['validation'].get('initial', False):
            logger.write('validation', 0, **validate(model, heldout, null, 0, run, device))
        for step in range(completed, stop):
            phase = model.set_phase(step)
            optimizer.zero_grad(set_to_none=True); sums = defaultdict(float); counts = defaultdict(int)
            start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            process_update=(model.ordered_process or model.calibrated_verifier) and model.should_rollout(step)
            for micro in range(1 if process_update else accumulation):
                sample = move(next(iterator), device)
                if model.operator_training and not model.qwen_process and (phase == 'calibration' or
                        cfg['condition_use'].get('mode') != 'positive_only'):
                    sample['alternative'] = dataset.alternative_condition(sample, cursor)
                if process_update:
                    # A genuine process update has its own optimizer step. Its
                    # signal is not buried underneath eight paired FM gradients.
                    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                        losses,weights,metrics=model.rollout_losses(sample,null,step,seed=cfg['train']['seed']+cursor)
                    if any(not bool(torch.isfinite(value)) for value in losses.values()):
                        raise FloatingPointError(f'Nonfinite sampled objective at {step}, sample {sample["id"]}')
                    model.backward_losses(losses,weights)
                    for key,value in metrics.items():sums[key]+=float(value);counts[key]+=1
                    cursor+=1
                    logger.write('rollout',step+1,phase=phase,sample=sample['id'],
                                 peak_gib=torch.cuda.max_memory_allocated()/2**30,**metrics)
                    del losses,weights,metrics,sample
                    continue
                with trace_micro(args.smoke, step=step+1, phase=phase, micro=micro+1, sample=sample['id']):
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        losses, weights, metrics = traced_call('forward', model, sample, null, step, micro=micro, sigma_override=.5 if args.smoke else None)
                    if any(not bool(torch.isfinite(value)) for value in losses.values()):
                        raise FloatingPointError(f'Nonfinite objective at {step}, sample {sample["id"]}')
                    traced_call('backward', model.backward_losses, losses, weights, accumulation)
                for key, value in metrics.items():
                    sums[key] += float(value); counts[key] += 1
                cursor += 1
                if args.smoke:
                    logger.write('micro', step+1, phase=phase, sample=sample['id'], micro=micro+1,
                                 peak_gib=torch.cuda.max_memory_allocated()/2**30, **metrics)
                if model.operator_training and micro == accumulation-1 and model.should_feedback(step):
                    # Release the paired graph before the complete short-window
                    # verifier graph. Feedback is a separate joint objective
                    # (third in v8, fourth in the legacy operator revisions).
                    del losses, metrics
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        feedback, feedback_weights, feedback_metrics = model.feedback_losses(sample, step)
                    if any(not bool(torch.isfinite(value)) for value in feedback.values()):
                        raise FloatingPointError(f'Nonfinite completion feedback at {step}, sample {sample["id"]}')
                    model.backward_losses(feedback, feedback_weights)
                    logger.write('completion_feedback', step+1, phase=phase, sample=sample['id'], **feedback_metrics)
                    for key, value in feedback_metrics.items():
                        sums['feedback/'+key] += float(value); counts['feedback/'+key] += 1
                    del feedback, feedback_weights, feedback_metrics
                elif micro == accumulation-1 and model.should_rollout(step):
                    # Finish/release the paired graph before building the sampled graph.
                    del losses, metrics
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        rollout_losses, rollout_weights, rollout_metrics = model.rollout_losses(
                            sample, null, step, seed=cfg['train']['seed']+cursor)
                    if any(not bool(torch.isfinite(value)) for value in rollout_losses.values()):
                        raise FloatingPointError(f'Nonfinite sampled objective at {step}, sample {sample["id"]}')
                    model.backward_losses(rollout_losses, rollout_weights)
                    logger.write('rollout', step+1, phase=phase, sample=sample['id'],
                                 peak_gib=torch.cuda.max_memory_allocated()/2**30, **rollout_metrics)
                    del rollout_losses, rollout_metrics
                else:
                    del losses, metrics
                del sample
            # Critic calibration has detached features and its own clipping scale.
            main_norm = torch.nn.utils.clip_grad_norm_(main_params, cfg['train']['grad_clip'], error_if_nonfinite=True)
            head_norm = torch.nn.utils.clip_grad_norm_(head_params, cfg['train']['grad_clip'], error_if_nonfinite=True)
            group_grad = {g['name']: sum(float(p.grad.detach().float().square().sum()) for p in g['params'] if p.grad is not None)**.5
                          for g in optimizer.param_groups}
            rates = {g['name']: g['lr'] for g in optimizer.param_groups}
            optimizer.step(); scheduler.step(); completed = step+1
            torch.cuda.synchronize()
            logger.write('train', completed, phase=phase, cursor=cursor, seconds=time.monotonic()-start,
                         update_kind=('verifier_calibration' if phase == 'calibration' else
                                      'conditional_operator' if model.operator_training else
                                      'full_process' if process_update else 'paired_fm'),
                         main_grad_norm=main_norm, critic_grad_norm=head_norm, gradients=group_grad, learning_rates=rates,
                         peak_gib=torch.cuda.max_memory_allocated()/2**30,
                         reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                         **{key: value/counts[key] for key, value in sums.items()})
            optimizer.zero_grad(set_to_none=True)
            boundary = completed in [sum(list(cfg['stages'].values())[:i+1]) for i in range(len(cfg['stages']))]
            if completed % cfg['train']['save_every'] == 0 or boundary or completed == stop:
                save_checkpoint(model, optimizer, scheduler, cfg, completed, cursor, run/'latest.pt')
                logger.write('checkpoint', completed, path=str(run/'latest.pt'))
            if not args.smoke and (completed % cfg['validation']['every'] == 0 or completed == stop):
                logger.write('validation', completed, **validate(model, heldout, null, completed, run, device))
        logger.write('complete' if completed == total else 'paused', completed, checkpoint=str(run/'latest.pt'))
    finally:
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/native_qwen_process_wisa3857.yaml'))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-accumulation',type=int)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--initialize-generator',help='Copy only native-revision generator/knowledge weights into a new run')
    parser.add_argument('--initialize-critic',help='Initialize the temporal critic from the bounded train-split study')
    train(parser.parse_args())
