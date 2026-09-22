import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physgen_v4.environments import require_environment
require_environment('runtime')

import torch
import torch.distributed as dist

from physgen_v4.runtime import (ROOT, read_config, add_external_paths, init_distributed, seed_all,
    Logger, reduce_metrics, synchronize_gradients, group_metrics, memory_metrics,
    make_scheduler, save_checkpoint, restore_rng, rng_state, configure_batch, resume_rng)
from physgen_v4.corrector import Corrector, CONDITION_INITIALIZATIONS
from physgen_v4.checkpoint_paths import name_training_run
from physgen_v4.backbone import load_wan, ProcessWan
from physgen_v4.coordinates import Coordinates
from physgen_v4.data import CachedWISA, TrainingOrder, move_sample, training_loader
from physgen_v4.diagnostics import RankDiagnostics
from physgen_v4.encoders import load_vae, DifferentiableDecoder
from physgen_v4.remote_teacher import RemoteVideoTeacher
from physgen_v4.losses import noisy_view, training_loss, prior_supervision_enabled


def configured_run(args):
    if args.resume and getattr(args, "init_from", None):
        raise ValueError("RESUME and INIT_FROM are mutually exclusive")
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if args.resume:
        config = read_config(Path(args.resume) / "config.yaml")
        # Preserve the saved optimizer schedule; adapt accumulation to this allocation.
        if getattr(args, 'run_name', None):
            config['run_name_base'] = args.run_name
        if args.wandb is not None:
            config['logging']['wandb']['enabled'] = args.wandb
        return runtime_overrides(configure_batch(config, world_size), args)
    config = read_config(args.config)
    phase = config["phases"][args.phase]
    config["stage"] = phase["stage"]
    config["phase"] = args.phase
    config["corrector"]["iterations"] = phase["iterations"]
    config["loss"]["out_weight"] = phase["out_weight"]
    config["train"]["steps"] = phase["steps"]
    config["train"]["learning_rate"] = phase["learning_rate"]
    config["hardware_name"] = args.hardware
    config["run_name"] = args.run_name or f"wisa_native_p_{args.phase}_{args.hardware}"
    config["run_name_base"] = config["run_name"]
    config["init_from"] = args.init_from
    if args.steps is not None:
        config["train"]["steps"] = args.steps
    if args.wandb is not None:
        config['logging']['wandb']['enabled'] = args.wandb
    return runtime_overrides(configure_batch(config, world_size), args)


def runtime_overrides(config, args):
    """Loader/diagnostic overrides also apply to resumes; optimizer schedule stays saved."""
    cfg = config['train']
    cfg.setdefault('pin_memory', True)
    cfg.setdefault('loader_timeout_seconds', 120)
    cfg.setdefault('state_warmup_every', 0)
    if type(cfg['state_warmup_every']) is not int or cfg['state_warmup_every'] < 0:
        raise ValueError('state_warmup_every must be a nonnegative integer; 0 disables the optional warm pass')
    if config['corrector'].get('initialization') == 'image_text_gt':
        if not 0 <= cfg.get('gt_condition_dropout', -1) <= 1:
            raise ValueError('image_text_gt training requires gt_condition_dropout between 0 and 1')
    for key in ('workers', 'pin_memory', 'loader_timeout_seconds'):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    options = config.setdefault('diagnostics', {})
    for key, default in (('trace_steps', 1), ('stack_timeout_seconds', 120), ('trace_wan_blocks', False)):
        options.setdefault(key, default)
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    # A short check never changes the saved schedule or writes a checkpoint.
    options['check_steps'] = getattr(args, 'check_steps', 0)
    if any(value < 0 for value in (cfg['workers'], cfg['loader_timeout_seconds'],
                                   options['trace_steps'], options['stack_timeout_seconds'], options['check_steps'])):
        raise ValueError('Loader and diagnostic counts/timeouts must be nonnegative')
    return config


def sample_sigma(config, step, micro, device, output_active):
    cfg = config["train"]
    generator = torch.Generator().manual_seed(cfg["seed"] + 100003 * step + micro)
    if output_active:
        lower, upper = config["loss"]["out_sigma_min"], config["loss"]["out_sigma_max"]
    else:
        lower = cfg["sigma_min"]
        upper = config["corrector"]["handoff_sigma"] if config["stage"] == "B" else cfg["sigma_max"]
    sigma = lower + (upper - lower) * torch.rand((), generator=generator)
    previous = sigma + (1 - sigma) * torch.rand((), generator=generator)
    return sigma.to(device), previous.to(device)


def use_gt_condition(config, step):
    """One reproducible modality choice per optimizer step, identical on every rank.

    This keeps unused GT-only gradients and collective ordering identical across ranks,
    including accumulation and checkpoint resumes. It does not carry P between videos.
    """
    if config['corrector'].get('initialization') != 'image_text_gt':
        return False
    cfg = config['train']
    generator = torch.Generator().manual_seed(cfg['seed'] + 700001 * (step + 1))
    return bool(torch.rand((), generator=generator) >= cfg['gt_condition_dropout'])


def geometry(sample, config):
    height, width = sample["latent"].shape[-2] * 16, sample["latent"].shape[-1] * 16
    return Coordinates.build(sample["times"], height, width, config["data"]["teacher_frames"],
                             config["corrector"]["seconds_scale"], config["data"]["teacher_size"])


@torch.no_grad()
def validate(model, dataset, config, step, device, decoder=None, teacher=None):
    saved_rng = rng_state()
    results = {}
    cfg = config["validation"]
    # Every rank executes exactly samples_per_rank * len(sigmas) Wan forwards.
    for sigma_value in cfg["sigmas"]:
        out_active = (config["loss"]["out_weight"] > 0 and
                      config["loss"]["out_sigma_min"] <= sigma_value <= config["loss"]["out_sigma_max"])
        sums = defaultdict(list)
        for local_index in range(cfg["samples_per_rank"]):
            index = dist.get_rank() + local_index * dist.get_world_size()
            sample = move_sample(dataset[index], device)
            generator = torch.Generator(device=device).manual_seed(cfg["seed"] + sample["record"]["index"])
            noise = torch.randn(sample["latent"].shape, generator=generator, device=device, dtype=torch.float32)
            sigma = torch.tensor(sigma_value, device=device)
            noisy = noisy_view(sample["latent"], sample["first"], noise, sigma)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction, _, info = model(noisy, sample["first"], sample["text"].unsqueeze(0), sigma, geometry(sample, config))
                _, metrics = training_loss(prediction, noise, sample["latent"], sample["first"], noisy, sigma,
                                            info, sample["target"], sample["target_weight"], step, config, out_active, decoder, teacher)
            for key, value in metrics.items():
                sums[key].append(torch.as_tensor(value, device=device))
        reduced = reduce_metrics({key: torch.stack(values).mean() for key, values in sums.items()}, device)
        results.update({f"sigma_{sigma_value:.2f}/{key}": value for key, value in reduced.items()})
    results["validation_loss"] = sum(results[f"sigma_{s:.2f}/loss/fm"] for s in cfg["sigmas"]) / len(cfg["sigmas"])
    restore_rng(saved_rng)
    return results


def train(args):
    with ExitStack() as cleanup:
        _train(args, cleanup)


def _train(args, cleanup):
    config = configured_run(args)
    trace = cleanup.enter_context(RankDiagnostics(config))
    add_external_paths(config)
    with trace.stage('init_distributed'):
        rank, world, device = init_distributed()
    def destroy_group():
        with trace.stage('destroy_process_group'):
            dist.destroy_process_group()
    cleanup.callback(destroy_group)
    session = [datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8] if rank == 0 else None]
    with trace.stage('session_broadcast'):
        dist.broadcast_object_list(session, src=0)
    # Use the rank-zero timestamp/unique suffix on every worker. Resumes keep the
    # optimizer schedule but receive a new destination, even from older checkpoints.
    checkpoint_directory = name_training_run(config, session[0], args.resume)
    config['logging']['session_id'] = session[0]
    # Create both ranks' files before loading the model or starting DataLoader workers.
    with trace.stage('logger_init'):
        logger = cleanup.enter_context(Logger(config, rank))
    trace.emit('session_ready', session_id=session[0], log_directory=str(logger.root),
               checkpoint_directory=str(checkpoint_directory))
    with trace.stage('device_metadata'):
        properties = torch.cuda.get_device_properties(device)
        trace.emit('device_metadata', device=str(device), gpu=properties.name,
                   total_memory_gib=properties.total_memory / 2**30, torch_version=torch.__version__,
                   cuda_version=torch.version.cuda, nccl_version=list(torch.cuda.nccl.version()))
    cfg = config["train"]
    with trace.stage('corrector_init'):
        seed_all(cfg["seed"])
        corrector = Corrector(config["corrector"]).to(device)
    start, cursor = 0, 0
    source = args.resume or args.init_from
    if source:
        with trace.stage('corrector_restore'):
            restored = corrector.load_checkpoint_weights(
                torch.load(Path(source) / "corrector.pt", map_location="cpu", weights_only=True),
                read_config(Path(source) / "config.yaml")["corrector"],
                allow_new_initializer=bool(args.init_from) and not args.resume and config["stage"] in ("A", "AB"),
                allow_new_fusion=bool(args.init_from) and not args.resume and config["stage"] in ("A", "AB"),
                allow_new_parameter_sharing=bool(args.init_from) and not args.resume and config["stage"] in ("A", "AB"))
            trace.emit('corrector_weights_loaded', **restored)
    corrector.set_stage(config["stage"])
    groups = corrector.parameter_groups()
    gt_parameter_ids = {id(p) for p in corrector.gt_condition_parameters()}
    optimizer = torch.optim.AdamW([dict(params=parameters, name=name) for name, parameters in groups.items()],
                                  lr=cfg["learning_rate"], betas=(0.9, 0.999), weight_decay=cfg["weight_decay"],
                                  foreach=False)
    scheduler = make_scheduler(optimizer, config)
    if args.resume:
        with trace.stage('optimizer_restore'):
            state = torch.load(Path(args.resume) / "training.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start, cursor = state["next_step"], state["data_cursor"]
    if config['diagnostics']['check_steps'] and start >= cfg['steps']:
        raise ValueError('No training steps remain for --check-steps; use an unfinished checkpoint or a fresh run')
    with trace.stage('wan_load_and_shard'):
        wan = load_wan(config["paths"]["wan_checkpoint"], device, True, True)
    model = ProcessWan(wan, corrector, config["stage"] in ("B", "AB"))
    model.diagnostics = trace
    teacher = decoder = None
    if config["loss"]["out_weight"] > 0:
        with trace.stage('online_teacher_init'):
            teacher = cleanup.enter_context(RemoteVideoTeacher(config["paths"], device, sharded=True, recompute=True))
        with trace.stage('vae_init'):
            vae = load_vae(config["paths"], device)
            decoder = DifferentiableDecoder(vae, recompute=True)
    with trace.stage('dataset_init'):
        dataset = CachedWISA(config["paths"]["cache_root"], "train", limit=cfg.get("sample_limit", 0))
        validation = CachedWISA(config["paths"]["cache_root"], "validation")
    keys = [(record["view"]["frames"], record["view"]["height"], record["view"]["width"]) for record in dataset.records]
    order = TrainingOrder(keys, cfg["seed"], rank, world, cursor)
    with trace.stage('dataloader_create'):
        loader = training_loader(dataset, order, cfg, rank)
    with trace.stage('dataloader_iter', workers=cfg['workers'],
                     start_method='spawn' if cfg['workers'] else 'single_process',
                     pin_memory=loader.pin_memory, timeout_seconds=loader.timeout):
        iterator = iter(loader)
    with trace.stage('rank_seed'):
        seed_all(cfg["seed"] + rank)
        if args.resume:
            resume_rng(args.resume, rank, world, state['world_size'], start, cfg['seed'])
    trace.emit('rank_ready', session_id=session[0], start_step=start)
    if rank == 0:
        logger.start(dict(stage=config["stage"], phase=config["phase"], run=config["run_name"],
            trainable_parameters={name: sum(p.numel() for p in parameters) for name, parameters in groups.items()},
            total_corrector_parameters=sum(p.numel() for p in corrector.parameters()),
            train_samples=len(dataset), validation_samples=len(validation),
            global_batch=world * cfg["accumulation_steps"], accumulation_steps=cfg['accumulation_steps'],
            target_global_batch=cfg['global_batch_size'], torch_version=torch.__version__,
            runtime=require_environment('runtime'), teacher_runtime=teacher.runtime if teacher else None,
            resume_from=args.resume, start_step=start, world_size=world,
            loader=dict(workers=cfg['workers'], start_method='spawn' if cfg['workers'] else 'single_process',
                        pin_memory=loader.pin_memory, timeout_seconds=loader.timeout),
            diagnostic_directory=str(trace.root), check_steps=config['diagnostics']['check_steps']))
        trace.emit('prior_training_policy', initialization=config['corrector'].get('initialization', 'latent'),
                   gt_condition_dropout=cfg.get('gt_condition_dropout'),
                   state_warmup_every=cfg['state_warmup_every'], validation_uses_gt_condition=False)
    latest = {}
    latest_validation = {}
    check_steps = config['diagnostics']['check_steps']
    end = min(cfg['steps'], start + check_steps) if check_steps else cfg['steps']
    for step in range(start, end):
        with trace.iteration(step + 1, step - start):
            with trace.stage('step_start_sync'):
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            collected = defaultdict(list)
            out_active = config["loss"]["out_weight"] > 0 and step % config["loss"]["out_every"] == 0
            warm_active = cfg["state_warmup_every"] > 0 and step % cfg["state_warmup_every"] == 0
            gt_active = use_gt_condition(config, step)
            b_active = False
            for micro in range(cfg["accumulation_steps"]):
                trace.context = dict(step=step + 1, micro=micro)
                data_started = time.perf_counter()
                with trace.stage('data_next'):
                    cpu_sample = next(iterator)
                trace.context.update(index=cpu_sample['record']['index'])
                with trace.stage('data_to_device', latent_shape=list(cpu_sample['latent'].shape)):
                    sample = move_sample(cpu_sample, device)
                    del cpu_sample
                with trace.stage('data_sync'):
                    torch.cuda.synchronize(device)
                data_seconds = time.perf_counter() - data_started
                with trace.stage('sample_prepare'):
                    sigma, previous_sigma = sample_sigma(config, step, micro, device, out_active)
                    b_active |= model.enable_b and sigma.item() < config["corrector"]["handoff_sigma"]
                    noise = torch.randn(sample["latent"].shape, device=device, dtype=torch.float32)
                    noisy = noisy_view(sample["latent"], sample["first"], noise, sigma)
                    dropped = torch.rand((), device=device).item() < cfg["text_dropout"]
                    text = (dataset.null_text.to(device) if dropped else sample["text"]).unsqueeze(0)
                    coords = geometry(sample, config)
                forward_started = time.perf_counter()
                # Warm no_grad forwards must not populate a weight-cast cache reused
                # by the grad-enabled forward. In PyTorch 2.8 that changes tensors
                # saved by non-reentrant checkpointing and can also drop gradients.
                # Checkpoint captures cache_enabled, so recomputation uses it too.
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    process = None
                    prior = None
                    gt_inputs = dict(gt_video=sample['latent']) if gt_active else {}
                    if warm_active:
                        # Optional same-video two-noise-view experiment. Construct P0 once.
                        # The carried state is detached; its initializer keeps direct JEPA supervision.
                        if config['corrector'].get('initialization') in CONDITION_INITIALIZATIONS:
                            with trace.stage('condition_prior'):
                                prior = corrector.initialize(sample['first'], text, coords, **gt_inputs)
                            process = prior.detach()
                        with torch.no_grad():
                            previous_noisy = noisy_view(sample["latent"], sample["first"], noise, previous_sigma)
                            with trace.stage('warm_forward'):
                                warm_prediction, process, warm_info = model(previous_noisy, sample["first"], text,
                                                                           previous_sigma, coords, process)
                            process = process.detach()
                            del warm_prediction, warm_info, previous_noisy
                    with trace.stage('forward'):
                        prediction, next_process, info = model(noisy, sample["first"], text, sigma, coords, process,
                                                              **({} if warm_active else gt_inputs))
                    if warm_active and prior_supervision_enabled(config):
                        info['prior'] = prior
                    info['prior_gt_conditioned'] = gt_active
                    with trace.stage('loss'):
                        loss, metrics = training_loss(prediction, noise, sample["latent"], sample["first"], noisy,
                            sigma, info, sample["target"], sample["target_weight"], step, config, out_active, decoder, teacher)
                with trace.stage('forward_sync'):
                    torch.cuda.synchronize(device)
                forward_seconds = time.perf_counter() - forward_started
                backward_started = time.perf_counter()
                with trace.stage('backward'):
                    (loss / cfg["accumulation_steps"]).backward()
                with trace.stage('backward_sync'):
                    torch.cuda.synchronize(device)
                metrics.update({"time/data_seconds": data_seconds, "time/forward_seconds": forward_seconds,
                    "time/backward_seconds": time.perf_counter() - backward_started,
                    "state/warm_active": float(warm_active), "text/dropped": float(dropped),
                    "prior/gt_conditioned": float(gt_active),
                    "compute/wan_forwards": 1 + int(warm_active),
                    "data/frames": sample["video_metadata"]["frames"],
                    "data/height": sample["video_metadata"]["height"],
                    "data/width": sample["video_metadata"]["width"],
                    "data/source_fps": sample["video_metadata"]["source_fps"],
                    "data/window_seconds": sample["video_metadata"]["duration"],
                    "data/teacher_frames": sample["video_metadata"]["teacher_frames"],
                    "data/window_caption": float(sample["record"]["caption_scope"] == "window"),
                    "state/previous_sigma": previous_sigma.item() if warm_active else 0.0})
                if trace.enabled:
                    with trace.stage('micro_memory'):
                        trace.emit('cuda_memory', allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                                   reserved_gib=torch.cuda.memory_reserved(device) / 2**30,
                                   peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                                   peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
                with trace.stage('micro_log'):
                    logger.micro(step + 1, micro, sample["record"]["index"], metrics, variables=dict(
                        latent_shape=list(sample['latent'].shape), text_shape=list(text.shape),
                        target_shape=list(sample['target'].shape), video_metadata=sample['video_metadata'],
                        times_seconds=sample['times'].detach().cpu().tolist()))
                for key, value in metrics.items():
                    # Frame counts and wan_forwards are integers; mean() requires a
                    # floating dtype even though these are diagnostic quantities.
                    collected[key].append(torch.as_tensor(value, device=device, dtype=torch.float32).detach())
                cursor += world
                del loss, prediction, next_process, info, process, prior, gt_inputs, noisy, sample
            trace.context = dict(step=step + 1)
            update_started = time.perf_counter()
            # Warm views bypass initialization unless its independent prior loss is enabled.
            # Both choices are shared across ranks; unused parameters receive no optimizer update.
            active_groups = {name: parameters for name, parameters in groups.items()
                             if not (warm_active and not prior_supervision_enabled(config) and name == "initialization")
                             and not (name == "B" and not b_active)}
            if not gt_active:
                active_groups = {name: [p for p in parameters if id(p) not in gt_parameter_ids]
                                 for name, parameters in active_groups.items()}
            with trace.stage('gradient_all_reduce'):
                synchronize_gradients(active_groups)
            metrics = {key: torch.stack(values).mean() for key, values in collected.items()}
            metrics.update(group_metrics(active_groups))
            metrics["grad/initialization_active"] = float("initialization" in active_groups)
            metrics['grad/initialization_gt_active'] = float(gt_active and 'initialization' in active_groups)
            metrics["grad/B_active"] = float("B" in active_groups)
            norm = torch.nn.utils.clip_grad_norm_([p for ps in active_groups.values() for p in ps], cfg["grad_clip"])
            metrics["grad/global_norm_before_clip"] = norm
            metrics["grad/clip_coefficient"] = (cfg["grad_clip"] / (norm + 1e-6)).clamp(max=1)
            metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            with trace.stage('optimizer_step'):
                optimizer.step()
                scheduler.step()
            with trace.stage('optimizer_sync'):
                torch.cuda.synchronize(device)
            metrics["time/update_seconds"] = time.perf_counter() - update_started
            elapsed = time.perf_counter() - started
            metrics["time/step_seconds"] = elapsed
            metrics["throughput/samples_per_second"] = world * cfg["accumulation_steps"] / elapsed
            metrics["data/cursor"] = cursor
            with trace.stage('metric_reduce'):
                latest = reduce_metrics(metrics, device)
                latest.update(memory_metrics(device))
            with trace.stage('step_log'):
                logger.step(step + 1, latest)
            if not check_steps and (step + 1) % config["validation"]["every"] == 0:
                with trace.stage('validation', step=step + 1, always=True):
                    latest_validation = validate(model, validation, config, step, device, decoder, teacher)
                    logger.step(step + 1, latest_validation, validation=True)
            if not check_steps and (step + 1) % cfg["save_every"] == 0:
                with trace.stage('checkpoint', step=step + 1, always=True):
                    save_checkpoint(config, corrector, optimizer, scheduler, step + 1, cursor,
                                    dict(train=latest, validation=latest_validation))
    if check_steps:
        trace.emit('training_check_complete', completed_steps=max(0, end - start), next_step=end,
                   validation_skipped=True, checkpoint_saved=False)
        return
    if cfg["steps"] % config["validation"]["every"] != 0:
        with trace.stage('final_validation'):
            latest_validation = validate(model, validation, config, cfg["steps"] - 1, device, decoder, teacher)
            logger.step(cfg["steps"], latest_validation, validation=True)
    with trace.stage('final_checkpoint'):
        save_checkpoint(config, corrector, optimizer, scheduler, cfg["steps"], cursor,
                        dict(train=latest, validation=latest_validation), final=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/wisa_native_p.yaml"))
    parser.add_argument("--phase", choices=("A1", "A2", "A3", "B1", "AB"), default="A1")
    parser.add_argument("--hardware", choices=("4x48g", "4x80g", "4x96g"), default="4x48g",
                        help="GPU pool label for run/checkpoint paths; process count is set by the allocation")
    parser.add_argument("--run-name")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init-from")
    source.add_argument("--resume")
    parser.add_argument("--steps", type=int)
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--workers', type=int, help='0 for single-process cache loading; positive values use spawn')
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--loader-timeout-seconds', type=float)
    parser.add_argument('--trace-steps', type=int, help='Trace the first N steps of this invocation')
    parser.add_argument('--stack-timeout-seconds', type=float, help='Dump Python stacks during a stalled traced stage; 0 disables')
    parser.add_argument('--trace-wan-blocks', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--check-steps', type=int, default=0,
                        help='Run at most N optimizer steps, preserving the LR schedule; skip validation/checkpoint writes')
    train(parser.parse_args())


if __name__ == "__main__":
    main()
