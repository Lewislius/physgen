"""One fixed single-GPU training run; no v4 imports or optional legacy losses."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from physgen_v42.backbone import ProcessWan, fix_first, load_wan
from physgen_v42.data import Dataset, ExposurePlan, UniformExposurePlan
from physgen_v42.diagnostics import state_relations
from physgen_v42.encoders import Decoder, load_vae
from physgen_v42.geometry import Geometry
from physgen_v42.losses import (fm_loss, local_pair_specs, struct_loss, temporal_loss,
                               generation_objective, auxiliary_objective, struct_noise_weight, v4_struct_coefficient,
                               correction_benefit, structure_components)
from physgen_v42.model import Adapter
from physgen_v42.optim import (EMA, GradientMixer, TRAINING_RECIPE, GATED_RECIPE, load_training, optimizer_for, ramp, restore_rng,
                              rng_state, save_checkpoint, set_schedule)
from physgen_v42.runtime import (ROOT, autocast, code_fingerprint, digest, lock_assets, log, output_path,
                                read_config, read_json, require_environment, require_single_gpu,
                                save_json, seed_all, sha256, job_lock)
from physgen_v42.training_log import TrainingProgress, loss_breakdown


def norm_group(module):
    values = [p.grad.float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def train_micro_gated(model, sample, exposure, step, mixer, accumulation, weights):
    """Conditional FM plus STRUCT on core and TextInit, with v4 coefficient limits."""
    a = model.adapter
    device = sample["latent"].device
    geo = Geometry.build(sample["record"]["geometry"], device)
    first, xgt, text = sample["first"], sample["latent"], sample["text"]
    duration = sample["record"]["duration"]
    sigma = torch.empty((), device=device).uniform_(.02, .999)
    noise = torch.randn_like(xgt)
    x = fix_first((1 - sigma) * xgt + sigma * noise, first)
    target = noise - xgt
    with autocast(device):
        p0 = a.initialize(text, geo, exposure["mode"], sample["anchor"])
        result = model(x, first, text, sigma, duration, geo, p0, ramp=ramp(step))
        fm = fm_loss(result["cond"], target, first is not None)
        base_fm = fm_loss(result["base"], target, first is not None)
        weighted_fm = weights["fm"] * fm / accumulation
    weighted_fm.backward()
    # Rebuild deployable P0 after FM backward, retaining its initializer gradient.
    # Only observed H is detached: STRUCT cannot change writers via H15.
    with autocast(device):
        if weights["struct"] > 0:
            aux_p0 = a.initialize(text, geo, exposure["mode"], sample["anchor"])
            p15 = a.auxiliary(aux_p0, result["observations"], text, sigma, duration, geo, detach_initial=False)
        else:
            p15 = result["p15"]
        struct = struct_loss(p15, sample["target"], a.s_z, geo.p_weight, sample["instances"])
        coefficient = v4_struct_coefficient(sigma, step, weights)
        weighted_struct = coefficient * struct / accumulation
    if weights["struct"] > 0:
        mixer.accumulate_aux(weighted_struct)
    return dict(fm=float(fm.detach()), fm_base=float(base_fm), fm_delta=float(fm.detach() - base_fm),
                struct=float(struct.detach()), temp=0., weighted_fm=float(weighted_fm.detach()),
                weighted_struct=float(weighted_struct.detach()), weighted_temp=0.,
                sigma=float(sigma), sigma_a=float(sigma), fm_active=True,
                struct_scheme="v4_limits", struct_coefficient=float(coefficient),
                correction_benefit=correction_benefit(result["cond"], result["base"], target, first is not None),
                structure_components=structure_components(p15.detach(), sample["target"], a.s_z, geo.p_weight),
                struct_warmup=min(1., step / weights["struct_warmup_steps"]),
                struct_noise_weight=float(((1 - sigma) / weights["struct_noise_width"]).clamp(0, 1)),
                temporal_active=False, repair=False, out_of_range=None, interventions=result["metrics"])


def train_micro(model, decoder, sample, exposure, step, mixer, negative, accumulation=8, loss_weights=None):
    weights = loss_weights or dict(fm=1., struct=.1, temp=.05)
    a = model.adapter
    device = sample["latent"].device
    geo = Geometry.build(sample["record"]["geometry"], device)
    mode, active = exposure["mode"], exposure["temporal"]
    first, xgt, text = sample["first"], sample["latent"], sample["text"]
    duration = sample["record"]["duration"]
    sigma = torch.empty((), device=device).uniform_(.05 if active else .02, .15 if active else .999)
    sigma_a = float(sigma)
    noise = torch.randn_like(xgt)
    x = fix_first((1 - sigma) * xgt + sigma * noise, first)
    target = noise - xgt
    with autocast(device):
        if exposure["repair"]:
            with torch.no_grad(), autocast(device):
                p0_a = a.initialize(text, geo, mode, sample["anchor"])
                initial = model(x, first, text, sigma, duration, geo, p0_a, negative, ramp(step))
                sigma_b = .8 * sigma
                x = fix_first(x + (sigma_b - sigma) * initial["guided"], first).detach()
                sigma = sigma_b
                target = ((x - xgt) / sigma).detach()
                del p0_a, initial
        # Reset again after repair construction; no graph or state from that prediction survives.
        p0 = a.initialize(text, geo, mode, sample["anchor"])
        result = model(x, first, text, sigma, duration, geo, p0, negative if active else None, ramp(step))
        prediction = result["guided"] if exposure["repair"] else result["cond"]
        loss_fm = fm_loss(prediction, target, first is not None)
        loss_temp = loss_fm.new_zeros(())
        outside = 0.
        if active:
            xhat = fix_first(x - sigma * result["guided"], first)
            specs = local_pair_specs(sample["record"], exposure["exposure"])
            with torch.no_grad():
                reference = decoder.views(fix_first(xgt, first), specs)
            decoded = decoder.views(xhat, specs)
            loss_temp = temporal_loss(decoded, reference, sample["record"]["geometry"], specs)
            outside = float(decoded["out_of_range"])
        generation = generation_objective(loss_fm, loss_temp, accumulation, weights["fm"], weights["temp"])
    generation.backward()
    # Detached observations cut BOTH the TextInit path and the H15 -> writer5 path.
    with autocast(device):
        p15_aux = a.auxiliary(result["p0"], result["observations"], text, sigma, duration, geo)
        loss_struct = struct_loss(p15_aux, sample["target"], a.s_z, geo.p_weight, sample["instances"])
        weighted_struct = auxiliary_objective(loss_struct, sigma, accumulation, weights["struct"])
    mixer.accumulate_aux(weighted_struct)
    return dict(fm=float(loss_fm.detach()), struct=float(loss_struct.detach()), temp=float(loss_temp.detach()),
                weighted_fm=weights["fm"] * float(loss_fm.detach()) / accumulation, weighted_struct=float(weighted_struct.detach()),
                weighted_temp=weights["temp"] * float(loss_temp.detach()), sigma=float(sigma), sigma_a=sigma_a,
                fm_active=True, struct_noise_weight=float(struct_noise_weight(sigma)),
                temporal_active=active, repair=exposure["repair"], out_of_range=outside,
                interventions=result["metrics"])


@torch.no_grad()
def health_check(model, dataset, step, log_path, *, console=True, weights="raw", return_summary=False):
    saved_rng = rng_state()
    totals, baselines, modes = [], [], []
    records = {r["id"]: i for i, r in enumerate(dataset.records)}
    device = next(model.adapter.parameters()).device
    try:
        seed_all(93001)
        for number, key in enumerate(dataset.manifest["health_ids"]):
            mode = "t2v" if number % 2 else "i2v"
            sample = dataset.load(records[key], mode, device)
            geo = Geometry.build(sample["record"]["geometry"], device)
            sigma = torch.tensor((.98, .85, .60, .35, .10)[number % 5], device=device)
            noise = torch.randn_like(sample["latent"])
            x = fix_first((1 - sigma) * sample["latent"] + sigma * noise, sample["first"])
            with autocast(device):
                p0 = model.adapter.initialize(sample["text"], geo, mode, sample["anchor"])
                result = model(x, sample["first"], sample["text"], sigma, sample["record"]["duration"], geo, p0, ramp=ramp(step))
                fm = fm_loss(result["cond"], noise - sample["latent"], mode == "i2v")
                fm_base = fm_loss(result["base"], noise - sample["latent"], mode == "i2v")
                struct = struct_loss(result["p15"], sample["target"], model.adapter.s_z, geo.p_weight, sample["instances"])
            totals.append(float(fm))
            baselines.append(float(fm_base))
            modes.append(mode)
            log(log_path, console=console, event="health", step=step, id=key, mode=mode, sigma=float(sigma),
                weights=weights,
                fm=float(fm), fm_base=float(fm_base), fm_delta=float(fm - fm_base),
                struct=float(struct), interventions=result["metrics"],
                correction_benefit=correction_benefit(result["cond"], result["base"], noise-sample["latent"], mode=="i2v"),
                structure_components=structure_components(result["p15"], sample["target"], model.adapter.s_z, geo.p_weight),
                state_to_gt=state_relations(result, sample["target"], geo, model.adapter.s_z, sample["instances"]))
    finally:
        restore_rng(saved_rng)
    summary = dict(validation_loss=sum(totals)/len(totals), base_loss=sum(baselines)/len(baselines),
                   count=len(totals), weights=weights, improves=sum(a<b for a,b in zip(totals,baselines)))
    summary["fm_delta"] = summary["validation_loss"]-summary["base_loss"]
    summary["by_mode"] = {mode: dict(count=modes.count(mode),
                          corrected=sum(x for x,m in zip(totals,modes) if m==mode)/modes.count(mode),
                          base=sum(x for x,m in zip(baselines,modes) if m==mode)/modes.count(mode))
                          for mode in set(modes)}
    log(log_path, console=console, event="health_summary", step=step, **summary)
    return summary if return_summary else summary["validation_loss"]


def train(cfg, args, device):
    corrected = cfg["train"]["recipe"] == GATED_RECIPE
    dataset, validation = Dataset(cfg, "train"), Dataset(cfg, "validation", require_complete=False)
    root = Path(cfg["paths"]["cache"])
    assets = lock_assets(cfg)
    ready = read_json(root / "ready.json")
    if ready["statistics_sha256"] != sha256(root / "stats.pt") or assets != dataset.manifest["assets"]:
        raise ValueError("Cache calibration/assets do not match finalized data")
    stats = torch.load(root / "stats.pt", map_location="cpu", weights_only=True)
    if stats["manifest_hash"] != digest(dataset.manifest) or stats["train_count"] != len(dataset):
        raise ValueError("Statistics are not calibrated on the current training sources")
    seed_all(cfg["train"]["seed"])
    adapter = Adapter(cfg, stats).float().to(device)
    count = sum(p.numel() for p in adapter.parameters())
    if count >= 100_000_000:
        raise ValueError(f"Trainable parameter budget exceeded: {count}")
    optimizer = optimizer_for(adapter, cfg)
    plan_type = UniformExposurePlan if corrected else ExposurePlan
    plan = plan_type(dataset.records, cfg["train"]["seed"])
    identity = dict(manifest=digest(dataset.manifest), statistics=ready["statistics_sha256"],
                    assets=digest(assets), code=digest(code_fingerprint()))
    start, ema, resume_rng = 0, EMA(adapter, cfg["train"]["ema_decay"]), None
    if args.resume:
        start, ema, resume_rng = load_training(args.resume, adapter, optimizer, plan, cfg, identity)
        run_dir = output_path(Path(args.resume).parent)
        run_name = run_dir.name
    else:
        run_name = args.run_name or "stability_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        if Path(run_name).name != run_name:
            raise ValueError("run-name must be a simple directory name")
        run_dir = output_path(Path(cfg["paths"]["checkpoints"]) / run_name)
        if run_dir.exists():
            raise FileExistsError("Run already exists; use --resume explicitly")
        run_dir.mkdir(parents=True)
    log_dir = Path(cfg["paths"]["logs"]) / run_name
    save_json(dict(config=cfg, identity=identity, training_recipe=cfg["train"]["recipe"],
                   parameters=count, groups={k: sum(p.numel() for p in v.parameters())
                    for k, v in adapter.groups().items()}, gpu=torch.cuda.get_device_name(device),
                   loss_reporting=(dict(
                       fm="mean conditional FM on all eight ordinary samples; paired base FM is diagnostic only",
                       struct=f"mean({cfg['loss']['struct']} * min(step/200,1) * clamp((1-sigma)/0.2,0,1) * L_struct)",
                       struct_gradient="core branches and T2V TextInit; detached H blocks teacher gradients to writers; no ratio rescaling",
                       temp="disabled; no VAE/negative branch/repair in training",
                       total="Nominal FM + STRUCT; auxiliary_gradient_policy is recorded separately, then global grad_clip=1",
                       auxiliary_gradient_policy=cfg["train"]["auxiliary_gradient_policy"]
                   ) if corrected else dict(
                       fm="mean(L_fm); active on all eight samples from step 1",
                       struct=f"{cfg['loss']['struct']} * aux_alpha * mean(q_i * L_struct_i); q_i=clamp((1-sigma_i)/0.5,0,1)^2",
                       struct_gradient="core branches only; ||g_struct||/(||g_generation_all||+||g_struct||) <= 0.20",
                       temp="0.05 * sum(active L_temp); one active micro per joint/repair update",
                       total="FM + effective STRUCT + TEMP, with aux_alpha treated as a fixed multiplier; "
                             "STRUCT gradients route only to core; before global gradient clipping and AdamW"))),
              log_dir / "run.json")
    log(log_dir / "events.jsonl", console=False, event="start", run=str(run_dir), step=start,
        parameters=count, identity=identity)
    model = ProcessWan(load_wan(cfg["paths"]["wan_checkpoint"], device), adapter)
    decoder = None if corrected else Decoder(load_vae(cfg["paths"], device), save_on_cpu=cfg["train"]["decoder_save_on_cpu"])
    negative = None if corrected else torch.load(root / "negative.pt", map_location=device, weights_only=True)[None]
    if resume_rng is not None:
        restore_rng(resume_rng)
    with job_lock(run_dir / ".training.lock"), TrainingProgress(cfg["train"]["steps"]) as progress:
        if corrected and start == 0:
            baseline = health_check(model, validation, 0, log_dir / "health.jsonl", console=False,
                                    weights="initial", return_summary=True)
            log(log_dir / "updates.jsonl", event="validation", step=0, **baseline)
        for step in range(start + 1, 1201):
            started = time.perf_counter()
            set_schedule(adapter, optimizer, cfg, step)
            optimizer.zero_grad(set_to_none=True)
            mixer = GradientMixer(adapter.auxiliary_parameters() if corrected else adapter.core_parameters(),
                                  adapter.parameters(), diagnostic_parameters=adapter.core_parameters())
            exposures = plan.batch(step)
            progress.start_step(step, len(exposures))
            metrics = []
            for micro, exposure in enumerate(exposures):
                sample = dataset.load(exposure["index"], exposure["mode"], device)
                if corrected:
                    values = train_micro_gated(model, sample, exposure, step, mixer,
                                              cfg["train"]["accumulation"], cfg["loss"])
                else:
                    values = train_micro(model, decoder, sample, exposure, step, mixer, negative,
                                         accumulation=cfg["train"]["accumulation"], loss_weights=cfg["loss"])
                log(log_dir / "micro.jsonl", console=False, event="micro", step=step, micro=micro, id=sample["record"]["id"],
                    category=sample["record"]["category"], mode=exposure["mode"], duration=sample["record"]["duration"],
                    exposure=exposure["exposure"], **values)
                metrics.append(values)
                del sample
                progress.advance(micro + 1)
            alignment = None
            if corrected and cfg["train"]["auxiliary_gradient_policy"] == "project_conflicts":
                alignment = mixer.align_to_generation({k:v for k,v in adapter.groups().items()
                                                       if k in ("core5", "core15", "text_init")})
            combined = mixer.combine(max_share=None if corrected else cfg["train"]["aux_ratio"])
            if corrected:
                combined.update(auxiliary_gradient_policy=cfg["train"]["auxiliary_gradient_policy"],
                                auxiliary_alignment=alignment)
            before_update = {name: [p.detach().clone() for p in group.parameters()] for name, group in adapter.groups().items()}
            norms = {name: norm_group(module) for name, module in adapter.groups().items()}
            total_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg["train"]["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            ema.update(adapter)
            updates = {name: float(torch.stack([(p.detach() - old).square().sum() for p, old in
                         zip(group.parameters(), before_update[name])]).sum().sqrt()) for name, group in adapter.groups().items()}
            del before_update, mixer
            torch.cuda.synchronize(device)
            values = dict(event="update", step=step, phase="fm_struct" if corrected else ("repair" if step >= 801 else "joint"),
                          seconds=time.perf_counter() - started,
                          **loss_breakdown(metrics, cfg["loss"], combined["aux_alpha"]),
                          global_grad_norm=float(total_norm), grad_groups=norms, parameter_update_norms=updates,
                          lr={g["name"]: g["lr"] for g in optimizer.param_groups}, ramp=ramp(step),
                          temporal_active=sum(m["temporal_active"] for m in metrics),
                          repair_active=sum(m["repair"] for m in metrics), **combined,
                          peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
            if corrected:
                values['loss_accounting'] = 'nominal_before_gradient_projection'
                values["fm_by_mode"] = {
                    mode: dict(count=len(selected),
                               corrected=sum(m["fm"] for m in selected) / len(selected),
                               base=sum(m["fm_base"] for m in selected) / len(selected),
                               delta=sum(m["fm_delta"] for m in selected) / len(selected))
                    for mode in ("i2v", "t2v")
                    if (selected := [m for e, m in zip(exposures, metrics) if e["mode"] == mode])}
            log(log_dir / "updates.jsonl", console=False, **values)
            progress.finish_step(values)
            if step % cfg["train"].get("validate_every", 200) == 0:
                if corrected:
                    for weights in ("raw", "ema"):
                        if weights == "ema":
                            with ema.average_parameters(adapter):
                                summary = health_check(model, validation, step, log_dir / "health.jsonl",
                                                       console=False, weights=weights, return_summary=True)
                        else:
                            summary = health_check(model, validation, step, log_dir / "health.jsonl",
                                                   console=False, weights=weights, return_summary=True)
                        log(log_dir / "updates.jsonl", event="validation", step=step, **summary)
                else:
                    validation_loss = health_check(model, validation, step, log_dir / "health.jsonl", console=False)
                    log(log_dir / "updates.jsonl", console=False, event="validation", step=step, validation_loss=validation_loss)
            if step % cfg["train"]["save_every"] == 0:
                path = save_checkpoint(run_dir, step, adapter, optimizer, ema, plan, cfg, identity)
                log(log_dir / "events.jsonl", console=False, event="checkpoint", path=str(path), step=step)
        save_json(dict(checkpoint=str(run_dir / "step1200"), step=1200), Path(cfg["paths"]["checkpoints"]) / "latest_final.json")
        save_json(plan.state_dict(), log_dir / "exposure_coverage.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/stability.yaml"))
    parser.add_argument("--resume")
    parser.add_argument("--run-name")
    parser.add_argument("--check-gpu", action="store_true", help="Check the training CUDA environment and exit")
    args = parser.parse_args()
    require_environment("runtime")
    cfg = read_config(args.config)
    if args.resume:
        saved_recipe = read_json(Path(args.resume) / "complete.json").get("training_recipe")
        if saved_recipe != cfg["train"]["recipe"]:
            raise ValueError("Cannot resume a different recipe: start corrected gated training without RESUME; "
                             "the old step0100 remains available for legacy inference")
    device = require_single_gpu()
    if args.check_gpu:
        log(event="gpu_ready", python=sys.executable, torch_version=torch.__version__,
            torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device))
        return
    train(cfg, args, device)


if __name__ == "__main__":
    main()
