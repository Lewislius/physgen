"""V4.3 training: native-width joint P/H, FM + STRUCT + verified Flow-Oracle."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physgen_v43.environments import require_environment
require_environment("runtime")

import torch
import torch.distributed as dist

from physgen_v43.runtime import (ROOT, read_config, configure_batch, seed_all, init_distributed,
    add_external_paths, make_scheduler, save_checkpoint, resume_rng, rng_state, restore_rng,
    synchronize_gradients, group_metrics, reduce_metrics, memory_metrics, Logger)
from physgen_v43.config import validate_config
from physgen_v43.corrector import Corrector, ARCHITECTURE
from physgen_v43.backbone import ProcessWan, load_wan
from physgen_v43.checkpoint_paths import name_training_run
from physgen_v43.coordinates import Coordinates
from physgen_v43.data import CachedWISA, TrainingOrder, training_loader, move_sample
from physgen_v43.diagnostics import RankDiagnostics
from physgen_v43.losses import noisy_view, flow_target, training_loss
from physgen_v43.oracle import select_sites, build_oracle_targets


def configured_run(args):
    requested = validate_config(read_config(args.config))
    if args.resume:
        config = validate_config(read_config(Path(args.resume) / "config.yaml"))
        for key in ("corrector", "loss", "oracle", "data"):
            if config[key] != requested[key]:
                raise ValueError(f"RESUME must retain {key}; use a new run for an ablation")
        if args.steps is not None:
            raise ValueError("RESUME retains its saved optimizer schedule; --steps is only for a new run")
    else:
        config = requested
        if args.steps is not None:
            config["train"]["steps"] = args.steps
    if args.run_name:
        config["run_name_base"] = args.run_name
        if not args.resume:
            config["run_name"] = args.run_name
    for key in ("workers", "sample_limit"):
        value = getattr(args, key, None)
        if value is not None:
            config["train"][key] = value
    config["diagnostics"]["check_steps"] = args.check_steps
    if args.check_steps < 0:
        raise ValueError("check_steps must be nonnegative")
    return configure_batch(validate_config(config), int(os.environ.get("WORLD_SIZE", "1")))


def sample_sigma(config, step, micro, device):
    """Retain V4.2 stage-A sampling, independent of global RNG and Oracle work."""
    cfg = config["train"]
    generator = torch.Generator().manual_seed(cfg["seed"] + 100003 * step + micro)
    sigma = cfg["sigma_min"] + (cfg["sigma_max"] - cfg["sigma_min"]) * torch.rand((), generator=generator)
    return sigma.to(device)


def geometry(sample, config):
    height, width = [s * 16 for s in sample["latent"].shape[-2:]]
    return Coordinates.build(sample["times"], height, width, config["data"]["teacher_frames"],
        config["corrector"]["seconds_scale"], config["data"]["teacher_size"], config["data"]["teacher_sampling"])


def check_config(config):
    with torch.device("meta"):
        corrector = Corrector(config["corrector"])
    corrector.set_stage("A")
    dataset = CachedWISA(config["paths"]["cache_root"], "train", config=config,
                         limit=config["train"]["sample_limit"])
    sample = dataset[0]
    coords = geometry(sample, config)
    if sample["target"].shape[1:] != (coords.position.shape[1], config["corrector"]["jepa_width"]):
        raise ValueError("Cached JEPA geometry is incompatible")
    wan_config = json.loads((Path(config["paths"]["wan_checkpoint"]) / "config.json").read_text())
    if wan_config["dim"] != config["corrector"]["hidden_width"]:
        raise ValueError("Working width must equal the selected Wan checkpoint width")
    parameters = {name: sum(p.numel() for p in ps) for name, ps in corrector.parameter_groups().items()}
    report = dict(architecture=ARCHITECTURE, sites=list(corrector.a_blocks), parameters=parameters,
        total_parameters=sum(parameters.values()), p_working_width=config["corrector"]["hidden_width"],
        h_width=wan_config["dim"], jepa_width=config["corrector"]["jepa_width"], losses=config["loss"],
        oracle=config["oracle"], train_samples=len(dataset), first_sample_frames=sample["video_metadata"]["frames"],
        cache=config["paths"]["cache_root"], gt_future_input=False,
        global_batch=config["train"]["effective_global_batch_size"],
        # FP32 parameters + gradients + two Adam moments; excludes all activations/Wan/workspaces.
        fp32_parameters_grad_adam_gib=sum(parameters.values()) * 16 / 2**30)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def training_micro(model, sample, text, noise, sigma, config, step, micro):
    """Teacher prepass/inner graphs are freed BEFORE constructing the training graph."""
    coords = geometry(sample, config)
    noisy = noisy_view(sample["latent"], sample["first"], noise, sigma)
    targets, oracle_metrics = {}, {"oracle/acceptance": 0.0, "oracle/selected_sites": 0}
    if config["loss"]["oracle_weight"] > 0:
        labels = select_sites([label for _, _, label in model.correction_sites()], step, micro,
                              config["train"]["accumulation_steps"], config["oracle"])
        with torch.no_grad():
            reference_prediction, reference_process, reference_info = model(
                noisy, sample["first"], text, sigma, coords, capture=labels)
            snapshots = reference_info["snapshots"]
            del reference_prediction, reference_process, reference_info
        targets, oracle_metrics = build_oracle_targets(model, snapshots, labels, text, sigma, coords,
            flow_target(sample["latent"], noise), config["oracle"], sample["first"] is not None)
        del snapshots
    prediction, _, info = model(noisy, sample["first"], text, sigma, coords)
    loss, metrics, terms = training_loss(prediction, sample["latent"], noise, sample["first"], sigma,
        info, sample["target"], sample["target_weight"], targets, step, config)
    metrics.update(oracle_metrics)
    metrics["sigma"] = sigma.detach()
    metrics["compute/full_wan_forwards"] = 1 + int(config["loss"]["oracle_weight"] > 0)
    return loss, metrics, terms


@torch.no_grad()
def validate(model, dataset, config, step, device):
    saved = rng_state()
    results = {}
    try:
        for sigma_value in config["validation"]["sigmas"]:
            sums = defaultdict(list)
            for local_index in range(config["validation"]["samples_per_rank"]):
                sample = move_sample(dataset[dist.get_rank() + local_index * dist.get_world_size()], device)
                generator = torch.Generator(device=device).manual_seed(config["validation"]["seed"] + sample["record"]["index"])
                noise = torch.randn(sample["latent"].shape, generator=generator, device=device)
                sigma = torch.tensor(sigma_value, device=device)
                noisy = noisy_view(sample["latent"], sample["first"], noise, sigma)
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    prediction, _, info = model(noisy, sample["first"], sample["text"].unsqueeze(0), sigma, geometry(sample, config))
                    _, metrics, _ = training_loss(prediction, sample["latent"], noise, sample["first"], sigma,
                        info, sample["target"], sample["target_weight"], {}, step, config)
                # Validation never has access to the oracle or future GT conditions.
                metrics = {k: v for k, v in metrics.items() if k not in ("loss/oracle", "loss/weighted_oracle", "loss/total")}
                for key, value in metrics.items():
                    sums[key].append(torch.as_tensor(value, device=device, dtype=torch.float32))
            current = reduce_metrics({k: torch.stack(v).mean() for k, v in sums.items()}, device)
            results.update({f"sigma_{sigma_value:.2f}/{k}": v for k, v in current.items()})
        results["validation_loss"] = sum(results[f"sigma_{s:.2f}/loss/fm"] for s in config["validation"]["sigmas"]) / len(config["validation"]["sigmas"])
        return results
    finally:
        restore_rng(saved)


def train(args):
    config = configured_run(args)
    with ExitStack() as cleanup:
        trace = cleanup.enter_context(RankDiagnostics(config))
        add_external_paths(config)
        rank, world, device = init_distributed()
        cleanup.callback(dist.destroy_process_group)
        session = [datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8] if rank == 0 else None]
        dist.broadcast_object_list(session, src=0)
        destination = name_training_run(config, session[0], args.resume)
        config["logging"]["session_id"] = session[0]
        logger = cleanup.enter_context(Logger(config, rank))
        seed_all(config["train"]["seed"])
        with trace.stage("corrector_init"):
            corrector = Corrector(config["corrector"]).to(device)
            if args.resume or args.init_from:
                source = Path(args.resume or args.init_from)
                corrector.load_checkpoint_weights(torch.load(source / "corrector.pt", map_location="cpu", weights_only=True),
                                                   read_config(source / "config.yaml")["corrector"])
            corrector.set_stage("A")
        groups = corrector.parameter_groups()
        cfg = config["train"]
        optimizer = torch.optim.AdamW([dict(params=ps, name=name) for name, ps in groups.items()],
            lr=cfg["learning_rate"], betas=(0.9, 0.999), weight_decay=cfg["weight_decay"], foreach=False)
        scheduler = make_scheduler(optimizer, config)
        start, cursor = 0, 0
        if args.resume:
            state = torch.load(Path(args.resume) / "training.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start, cursor = state["next_step"], state["data_cursor"]
        if start >= cfg["steps"]:
            raise ValueError("No training steps remain")
        with trace.stage("wan_load_and_shard"):
            wan = load_wan(config["paths"]["wan_checkpoint"], device, sharded=True, recompute=True)
        model = ProcessWan(wan, corrector)
        model.diagnostics = trace
        dataset = CachedWISA(config["paths"]["cache_root"], "train", config=config, limit=cfg["sample_limit"])
        validation = CachedWISA(config["paths"]["cache_root"], "validation", config=config)
        keys = [(r["view"]["frames"], r["view"]["height"], r["view"]["width"]) for r in dataset.records]
        order = TrainingOrder(keys, cfg["seed"], rank, world, cursor)
        iterator = iter(training_loader(dataset, order, cfg, rank))
        seed_all(cfg["seed"] + rank)
        if args.resume:
            resume_rng(args.resume, rank, world, state["world_size"], start, cfg["seed"])
            del state  # Release the CPU copy of the multi-billion-parameter Adam state.
        if rank == 0:
            logger.start(dict(architecture=ARCHITECTURE, stage="A", run=config["run_name"],
                trainable_parameters={name: sum(p.numel() for p in ps) for name, ps in groups.items()},
                total_corrector_parameters=sum(p.numel() for p in corrector.parameters()),
                train_samples=len(dataset), validation_samples=len(validation), world_size=world,
                global_batch=cfg["effective_global_batch_size"], gt_future_input=False,
                losses=["fm", "struct", "oracle"], oracle=config["oracle"], checkpoint_directory=str(destination),
                runtime=require_environment("runtime"), torch_version=torch.__version__))
        latest, latest_validation = {}, {}
        end = min(cfg["steps"], start + args.check_steps) if args.check_steps else cfg["steps"]
        for step in range(start, end):
            with trace.iteration(step + 1, step - start):
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                optimizer.zero_grad(set_to_none=True)
                collected = defaultdict(list)
                for micro in range(cfg["accumulation_steps"]):
                    trace.context = dict(step=step + 1, micro=micro)
                    sample = move_sample(next(iterator), device)
                    sigma = sample_sigma(config, step, micro, device)
                    noise = torch.randn_like(sample["latent"], dtype=torch.float32)
                    dropped = bool(torch.rand((), device=device) < cfg["text_dropout"])
                    text = (dataset.null_text.to(device) if dropped else sample["text"]).unsqueeze(0)
                    # No autocast weight cache may cross no_grad/grad oracle and training passes.
                    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False), trace.stage("oracle_and_forward"):
                        loss, metrics, terms = training_micro(model, sample, text, noise, sigma, config, step, micro)
                    with trace.stage("backward"):
                        (loss / cfg["accumulation_steps"]).backward()
                    metrics.update({"data/frames": sample["video_metadata"]["frames"],
                                    "data/height": sample["video_metadata"]["height"],
                                    "data/width": sample["video_metadata"]["width"], "text/dropped": float(dropped)})
                    logger.micro(step + 1, micro, sample["record"]["index"], metrics, variables=dict(
                        latent_shape=list(sample["latent"].shape), jepa_target_shape=list(sample["target"].shape),
                        h_width=config["corrector"]["hidden_width"], p_working_width=config["corrector"]["hidden_width"],
                        decoded_p_width=config["corrector"]["jepa_width"], gt_future_input=False))
                    for key, value in metrics.items():
                        collected[key].append(torch.as_tensor(value, device=device, dtype=torch.float32).detach())
                    cursor += world
                    del loss, terms, sample
                synchronize_gradients(groups)
                metrics = {k: torch.stack(v).mean() for k, v in collected.items()}
                metrics.update(group_metrics(groups))
                norm = torch.nn.utils.clip_grad_norm_(corrector.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
                metrics["grad/global_norm_before_clip"] = norm
                metrics["grad/clip_coefficient"] = (cfg["grad_clip"] / (norm + 1e-6)).clamp(max=1.)
                metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                torch.cuda.synchronize(device)
                metrics["time/step_seconds"] = time.perf_counter() - started
                metrics["data/cursor"] = cursor
                latest = reduce_metrics(metrics, device)
                latest.update(memory_metrics(device))
                logger.step(step + 1, latest)
                if not args.check_steps and (step + 1) % config["validation"]["every"] == 0:
                    latest_validation = validate(model, validation, config, step, device)
                    logger.step(step + 1, latest_validation, validation=True)
                if not args.check_steps and (step + 1) % cfg["save_every"] == 0:
                    save_checkpoint(config, corrector, optimizer, scheduler, step + 1, cursor,
                                    dict(train=latest, validation=latest_validation))
        if args.check_steps:
            trace.emit("training_check_complete", completed_steps=end - start, checkpoint_saved=False)
            return
        if cfg["steps"] % config["validation"]["every"]:
            latest_validation = validate(model, validation, config, cfg["steps"] - 1, device)
            logger.step(cfg["steps"], latest_validation, validation=True)
        save_checkpoint(config, corrector, optimizer, scheduler, cfg["steps"], cursor,
                        dict(train=latest, validation=latest_validation), final=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/flow_oracle.yaml"))
    source = p.add_mutually_exclusive_group()
    source.add_argument("--resume")
    source.add_argument("--init-from", help="Compatible V4.3 weights only; creates a fresh optimizer/run")
    p.add_argument("--run-name")
    p.add_argument("--steps", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--sample-limit", type=int)
    p.add_argument("--check-steps", type=int, default=0)
    p.add_argument("--check-config", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    check_config(configured_run(args)) if args.check_config else train(args)
