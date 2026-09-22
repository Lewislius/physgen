"""Single-GPU Wan LoRA: FM only, microbatch 1 x accumulation 8, 1200 updates."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from static_lora import TRAINING_RECIPE
from static_lora.backbone import fix_first, load_model
from static_lora.checkpoint import load_training, save_checkpoint
from static_lora.config import DEFAULT_CONFIG, configuration
from static_lora.data import check_data, ExposurePlan
from static_lora.lora import assert_lora_only, lora_named_parameters
from static_lora.optim import EMA, optimizer_for, restore_rng, rng_state, set_schedule
from static_lora.runtime import (code_fingerprint, digest, job_lock, lock_assets, log, output_path,
                                 read_json, require_environment, require_single_gpu, save_json, seed_all)
from physgen_v42.losses import fm_loss


def train_micro(model, sample, exposure, cfg):
    xgt, first = sample["latent"], sample["first"]
    low, high = cfg["noise"]["low" if exposure["low_noise"] else "ordinary"]
    sigma = torch.empty((), device=xgt.device).uniform_(low, high)
    noise = torch.randn_like(xgt)
    noisy = fix_first((1 - sigma) * xgt + sigma * noise, first)
    prediction = model(noisy, first, sample["text"], sigma)
    loss = fm_loss(prediction, noise - xgt, first is not None)
    weighted = cfg["loss"]["fm"] * loss / cfg["train"]["accumulation"]
    weighted.backward()
    return dict(fm=float(loss.detach()), weighted_fm=float(weighted.detach()), sigma=float(sigma),
                low_noise=exposure["low_noise"], fm_active=True, repair=False)


@torch.no_grad()
def health_check(model, dataset, step, path):
    saved_rng = rng_state()
    by_id = {r["id"]: i for i, r in enumerate(dataset.records)}
    totals = []
    device = next(model.parameters()).device
    try:
        seed_all(93001)
        for number, key in enumerate(dataset.manifest["health_ids"]):
            mode = "t2v" if number % 4 == 3 else "i2v"
            sample = dataset.load(by_id[key], mode, device)
            sigma = torch.tensor((.98, .85, .60, .35, .10)[number % 5], device=device)
            noise = torch.randn_like(sample["latent"])
            noisy = fix_first((1 - sigma) * sample["latent"] + sigma * noise, sample["first"])
            prediction = model(noisy, sample["first"], sample["text"], sigma)
            loss = fm_loss(prediction, noise - sample["latent"], mode == "i2v")
            totals.append(float(loss))
            log(path, console=False, event="health", step=step, id=key, mode=mode, sigma=float(sigma), fm=float(loss))
    finally:
        restore_rng(saved_rng)
    return sum(totals) / len(totals)


def train(cfg, args, device):
    dataset, validation = check_data(cfg)
    assets = lock_assets(cfg)
    if assets != dataset.manifest["assets"]:
        raise ValueError("Base asset inventory differs from the shared v4.2 data manifest")
    identity = dict(manifest=digest(dataset.manifest), assets=digest(assets),
                    code=digest(code_fingerprint()), reference=cfg["reference_sha256"])
    seed_all(cfg["train"]["seed"])
    model = load_model(cfg, device, training=True)
    parameters = assert_lora_only(model)
    optimizer = optimizer_for(model, cfg)
    plan = ExposurePlan(dataset.records, cfg["train"]["seed"])
    start, ema = 0, EMA(model, cfg["train"]["ema_decay"])
    checkpoint_root = Path(cfg["paths"]["checkpoints"]).resolve()
    if args.resume:
        resume = Path(args.resume).expanduser().resolve()
        if checkpoint_root not in resume.parents:
            raise ValueError("Resume must use this experiment's separate checkpoint directory")
        start, ema, saved_rng = load_training(resume, model, optimizer, plan, cfg, identity)
        run_dir = output_path(resume.parent)
        restore_rng(saved_rng)  # Constructors/loading finished before restoring RNG.
    else:
        name = args.run_name or "lora_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("run-name must be a simple directory name")
        run_dir = output_path(checkpoint_root / name)
    log_dir = output_path(Path(cfg["paths"]["logs"]) / run_dir.name)
    if start == 0:
        run_dir.mkdir(parents=True, exist_ok=False)
    with job_lock(run_dir / ".training.lock"):
        save_json(dict(config=cfg, identity=identity, training_recipe=TRAINING_RECIPE,
                       trainable_parameters=parameters, lora_targets=model.targets,
                       trainable_names=[name for name, _ in lora_named_parameters(model)],
                       gpu=torch.cuda.get_device_name(device), batch_size=8, microbatch=1, accumulation=8,
                       loss_reporting="mean of eight conditional FM losses; no STRUCT/TEMP/repair"), log_dir / "run.json")
        log(log_dir / "events.jsonl", console=False, event="start", run=str(run_dir), step=start, parameters=parameters)
        trainable = [p for _, p in lora_named_parameters(model)]
        for step in range(start + 1, cfg["train"]["steps"] + 1):
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            set_schedule(optimizer, cfg, step)
            optimizer.zero_grad(set_to_none=True)
            exposures = plan.batch(step)
            print(f"step {step}/1200 | micro 0/8", flush=True)
            metrics = []
            for micro, exposure in enumerate(exposures):
                sample = dataset.load(exposure["index"], exposure["mode"], device)
                values = train_micro(model, sample, exposure, cfg)
                log(log_dir / "micro.jsonl", console=False, event="micro", step=step, micro=micro,
                    id=sample["record"]["id"], mode=exposure["mode"], exposure=exposure["exposure"],
                    duration=sample["record"]["duration"], **values)
                metrics.append(values)
                del sample
                if micro < 7:
                    print(f"step {step}/1200 | micro {micro + 1}/8", flush=True)
            norm = torch.nn.utils.clip_grad_norm_(trainable, cfg["train"]["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            ema.update(model)
            torch.cuda.synchronize(device)
            loss = sum(m["weighted_fm"] for m in metrics)
            update = dict(event="update", step=step, phase="fm", batch_size=8, accumulation=8,
                          loss_fm=loss, loss_total=loss, global_grad_norm=float(norm),
                          lr=optimizer.param_groups[0]["lr"], ema_updates=ema.updates,
                          low_noise_active=sum(m["low_noise"] for m in metrics), repair_active=0,
                          sigma=[m["sigma"] for m in metrics], seconds=time.perf_counter() - started,
                          peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
            log(log_dir / "updates.jsonl", console=False, **update)
            print(f"step {step}/1200 | micro 8/8 | FM={loss:.6g} | grad_norm={float(norm):.6g} | lr={update['lr']:.6g}", flush=True)
            if step % 200 == 0:
                validation_loss = health_check(model, validation, step, log_dir / "health.jsonl")
                log(log_dir / "updates.jsonl", event="validation", step=step, validation_loss=validation_loss)
            if step % cfg["train"]["save_every"] == 0:
                path = save_checkpoint(run_dir, step, model, optimizer, ema, plan, cfg, identity)
                log(log_dir / "events.jsonl", event="checkpoint", step=step, path=str(path))
        save_json(dict(checkpoint=str(run_dir / "step1200"), step=1200), checkpoint_root / "latest_final.json")
        save_json(plan.state_dict(), log_dir / "exposure_coverage.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume")
    parser.add_argument("--run-name")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-config", action="store_true")
    action.add_argument("--check-gpu", action="store_true")
    action.add_argument("--prepare-only", action="store_true", help="Validate/reuse existing VAE/T5 caches; no teacher/preprocessing run")
    args = parser.parse_args()
    require_environment("runtime")
    cfg = configuration(args.config)
    if args.check_config or args.prepare_only:
        dataset, validation = check_data(cfg)
        assets = lock_assets(cfg)
        if assets != dataset.manifest["assets"]:
            raise ValueError("Base asset inventory differs from the cached data")
        model_cfg = read_json(Path(cfg["paths"]["wan_checkpoint"]) / "config.json")
        count = model_cfg["num_layers"] * len(cfg["lora"]["targets"]) * 2 * model_cfg["dim"] * cfg["lora"]["rank"]
        log(event="lora_config_valid", train=len(dataset), validation=len(validation),
            trainable_parameters=count, config=cfg)
        return
    device = require_single_gpu()
    if args.check_gpu:
        log(event="gpu_ready", gpu=torch.cuda.get_device_name(device), python=sys.executable,
            torch_version=torch.__version__, torch_cuda=torch.version.cuda)
        return
    train(cfg, args, device)


if __name__ == "__main__":
    main()
