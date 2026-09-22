"""Compatible intermediate/final EMA inference; UniPC/50/shift5/CFG5, seed42."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    print("[inference] Loading Python/PyTorch ...", flush=True)
import numpy as np
from PIL import Image
import torch

from physgen_v42 import VERSION
from physgen_v42.backbone import ProcessWan, fix_first, load_wan
from physgen_v42.data import Dataset
from physgen_v42.diagnostics import state_relations
from physgen_v42.encoders import Decoder, encode_vae, load_text, load_vae
from physgen_v42.geometry import Geometry, canvas, fit_geometry, letterbox, teacher_indices
from physgen_v42.model import Adapter
from physgen_v42.runtime import (ROOT, autocast, digest, log, native_negative, output_path,
                                read_json, require_environment, require_single_gpu, save_json, save_tensor,
                                seed_all, job_lock)
from inference.conditions import (condition_key, encoder_stamps, file_stamp, image_cache, text_cache)

if __name__ == "__main__":
    print("[inference] Python/PyTorch ready.", flush=True)


RECIPE = dict(solver="unipc", steps=50, shift=5, guidance=5, adapter_guidance=1, seed=42)


def read_config(path):
    # Inference needs model settings, not the training/preparation contract.
    import yaml
    path = Path(path)
    source = path.read_text().replace("${V42_ROOT}", str(ROOT))
    cfg = json.loads(source) if path.suffix == ".json" else yaml.safe_load(source)
    sys.path.insert(0, cfg["paths"]["wan_code"])
    return cfg


def checkpoint_path(cfg, value):
    if value:
        path = Path(value).expanduser().resolve()
    else:
        path = Path(read_json(Path(cfg["paths"]["checkpoints"]) / "latest_final.json")["checkpoint"]).resolve()
    return path


def collect_cases(cfg, args):
    cases = []
    if args.suite == "single":
        if not args.prompt or not args.prompt.strip() or (args.mode == "i2v" and not args.image):
            raise ValueError("Single generation needs --prompt and, for I2V, --image")
        if args.portrait and args.mode != "t2v":
            raise ValueError("--portrait applies to single T2V; I2V follows the input image geometry")
        return [dict(key="single_" + args.mode, id="single", group="new_prompt", mode=args.mode,
                     image=args.image if args.mode == "i2v" else None, prompt=args.prompt, duration=args.duration,
                     portrait=args.portrait)]
    if args.portrait or args.duration != 5.:
        raise ValueError("The fixed final/demo suite uses its predefined canvas and duration; use --suite single for new conditions")
    if args.suite == "final":
        manifest = Dataset(cfg, "train").manifest
        by_id = {r["id"]: r for r in manifest["records"]}
        for split in ("train", "validation"):
            for fixed in manifest["diagnostics"][split]:
                record = by_id[fixed["id"]]
                cases.append(dict(key=f'{split}_{record["id"]}_{fixed["mode"]}', id=record["id"], group=split,
                                  mode=fixed["mode"], prompt=record["caption"], duration=record["duration"], gt_record=record))
    cases.append(dict(key="reference_i2v", id="reference", group="reference", mode="i2v",
                      image=cfg["paths"]["reference_image"], prompt=cfg["paths"]["reference_prompt"], duration=5.))
    demo = Path(cfg["paths"]["demo_root"])
    for number in range(1, 21):
        key = f"P{number:02d}"
        folder = demo / key
        text_file = folder / (key + "-origin.txt")
        prompt = text_file.read_text(encoding="utf-8-sig").strip()
        image = next((folder / (key + "-" + suffix) for suffix in
                      ("i0-1280x704.png", "i0-1280x704.jpg", "i0.png", "i0.jpg") if (folder / (key + "-" + suffix)).exists()), None)
        if image is None or not prompt:
            raise FileNotFoundError(f"Missing image/prompt for {key}")
        for mode in ("i2v", "t2v"):
            cases.append(dict(key=f"{key}_{mode}", id=key, group="demo", mode=mode, prompt=prompt,
                              image=str(image) if mode == "i2v" else None, duration=5.))
    return cases


def build_plan(cfg, args):
    checkpoint = checkpoint_path(cfg, args.checkpoint)
    cases = collect_cases(cfg, args)
    checkpoint_status = read_json(checkpoint / "complete.json") if (checkpoint / "complete.json").exists() else {}
    for case in cases:
        if not np.isfinite(case["duration"]) or case["duration"] <= 0:
            raise ValueError("Duration must be finite and positive")
        if case.get("image"):
            case["image"] = str(Path(case["image"]).expanduser().resolve())
            case["image_file"] = file_stamp(case["image"])
            with Image.open(case["image"]) as image:
                h, w = canvas(image.height, image.width)
                geo = fit_geometry(image.height, image.width, h, w, upscale=False)
        elif "gt_record" in case:
            geo = dict(case["gt_record"]["geometry"])
        else:
            w, h = cfg["inference"]["canvas_wh"]
            if case.get("portrait"):
                h, w = w, h
            geo = fit_geometry(h, w, h, w)
        geo.setdefault("frames", cfg["data"]["frames"])
        case["geometry"] = geo
    return dict(version=VERSION, schema_version=3, suite=args.suite, checkpoint=str(checkpoint),
                    checkpoint_step=checkpoint_status.get("step"), ema_updates=checkpoint_status.get("ema_updates"),
                    ema_file=file_stamp(checkpoint / "ema.pt"), encoders=encoder_stamps(cfg["paths"]),
                    config=cfg, cases=cases, negative=native_negative(), negative_sha256=digest(native_negative()),
                    recipe=dict(RECIPE))


def prepare(cfg, args, root, device=None):
    metadata = build_plan(cfg, args)
    cases = metadata["cases"]
    plan_file = root / "cases.json"
    if plan_file.exists() and read_json(plan_file) != metadata:
        raise ValueError("Output folder already belongs to different cases/checkpoint; choose a new output folder")
    save_json(metadata, plan_file)
    ready_file = root / "conditions_ready.json"
    pending = []
    for case in cases:
        folder = root / case["key"]
        marker = folder / "condition.meta.json"
        if ((folder / "condition.pt").exists() and marker.exists()
                and read_json(marker).get("key") == condition_key(metadata, case)
                and (case["mode"] != "i2v" or (folder / (
                    "anchor.pt" if "gt_record" in case else "first_image.pt")).exists())):
            continue
        pending.append(case)
    log(event="conditions", total=len(cases), cached=len(cases) - len(pending), pending=len(pending))
    if not pending:
        save_json(dict(plan_hash=digest(metadata)), ready_file)
        return
    prompts = list(dict.fromkeys([c["prompt"] for c in pending] + [metadata["negative"]]))
    missing_texts = [p for p in prompts if not text_cache(metadata, p).exists()]
    log(event="text_cache", total=len(prompts), missing=len(missing_texts))
    if missing_texts:
        log(event="loading", model="T5", detail="Only uncached prompts need encoding")
        device = device if device is not None else require_single_gpu()
        encoder = load_text(cfg["paths"], device)
        with torch.no_grad(), autocast(device):
            for index, prompt in enumerate(missing_texts, 1):
                save_tensor(encoder([prompt], device)[0].cpu().bfloat16(), text_cache(metadata, prompt))
                log(event="text_encoded", completed=index, total=len(missing_texts))
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    texts = {p: torch.load(text_cache(metadata, p), map_location="cpu", weights_only=True) for p in prompts}
    needs_vae = any(c["mode"] == "i2v" and "gt_record" not in c and not image_cache(metadata, c, "image").exists()
                    for c in pending)
    vae = None
    if needs_vae:
        log(event="loading", model="VAE encoder", detail="Only uncached first images need encoding")
        device = device if device is not None else require_single_gpu()
        vae = load_vae(cfg["paths"], device)
    for index, case in enumerate(pending, 1):
        folder = root / case["key"]
        geo = dict(case["geometry"])
        condition = dict(text=texts[case["prompt"]], negative=texts[metadata["negative"]], first=None)
        if "gt_record" in case:
            if case["mode"] == "i2v":
                cache = Path(cfg["paths"]["cache"])
                condition["first"] = torch.load(cache / "vae" / (case["id"] + ".pt"), map_location="cpu", weights_only=True)["first"]
                anchor = torch.load(cache / "anchors" / (case["id"] + ".pt"), map_location="cpu", weights_only=True)
                save_tensor(anchor, folder / "anchor.pt")
        elif case["mode"] == "i2v":
            cached = image_cache(metadata, case, "image")
            if cached.exists():
                image_value = torch.load(cached, map_location="cpu", weights_only=True)
                pixels, condition["first"] = image_value["pixels"], image_value["first"]
            else:
                with Image.open(case["image"]) as image:
                    pixels = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)[:, None].float() / 127.5 - 1
                pixels = letterbox(pixels, geo)[None]
                condition["first"] = encode_vae(vae, pixels.to(device)).cpu()
                save_tensor(dict(pixels=pixels, first=condition["first"]), cached)
            save_tensor(pixels, folder / "first_image.pt")
        condition["geometry"] = geo
        save_tensor(condition, folder / "condition.pt")
        save_json(dict(key=condition_key(metadata, case)), folder / "condition.meta.json")
        log(event="condition_ready", case=case["key"], completed=index, total=len(pending))
    save_json(dict(plan_hash=digest(metadata)), ready_file)
    log(event="conditions_prepared", cases=len(cases), teacher_anchors_pending=sum(
        c["mode"] == "i2v" and "gt_record" not in c for c in cases))


def load_adapter(plan, device):
    checkpoint = Path(plan["checkpoint"])
    log(event="loading", model="step checkpoint EMA", path=str(checkpoint / "ema.pt"))
    values = torch.load(checkpoint / "ema.pt", map_location="cpu", weights_only=True, mmap=True)
    stats = {k: values[k] for k in ("mu_first", "s_z", "s_v")}
    # Construct shapes only; do not randomly initialize 65M parameters just to overwrite them.
    with torch.device("meta"):
        adapter = Adapter(plan["config"], stats)
    adapter.load_state_dict(values, strict=True, assign=True)
    return adapter.eval().requires_grad_(False).to(device)


def case_signature(case, plan):
    return digest(dict(case=case, ema=plan["ema_file"], recipe=plan["recipe"], checkpoint=plan["checkpoint"]))


def completed_case(case, plan, root):
    folder = root / case["key"]
    marker = folder / "metadata.json"
    if marker.exists():
        saved = read_json(marker)
        if saved.get("status") == "completed" and saved.get("signature") == case_signature(case, plan):
            return all((folder / name).is_file() and (folder / name).stat().st_size == stamp["bytes"]
                       for name, stamp in saved["files"].items())
    return False


@torch.no_grad()
def sample_case(model, decoder, case, plan, root, device):
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    import imageio.v2 as imageio
    folder = root / case["key"]
    signature = case_signature(case, plan)
    complete = folder / "metadata.json"
    if completed_case(case, plan, root):
        log(event="case_reused", key=case["key"])
        return
    save_json(dict(status="running", signature=signature, case=case), complete)
    condition = torch.load(folder / "condition.pt", map_location="cpu", weights_only=True)
    if condition["geometry"] != case["geometry"]:
        raise ValueError("Prepared geometry differs from the case plan")
    geo = Geometry.build(condition["geometry"], device)
    text, negative = condition["text"][None].to(device), condition["negative"][None].to(device)
    first = condition["first"].to(device) if condition["first"] is not None else None
    anchor = torch.load(folder / "anchor.pt", map_location=device, weights_only=True) if case["mode"] == "i2v" else None
    if (case["mode"] == "i2v") != (first is not None):
        raise ValueError("Mode/first-slice condition mismatch")
    h, w = geo.geo["h"], geo.geo["w"]
    frames = geo.geo.get("frames", 121)
    target = instances = None
    if "gt_record" in case:
        cache = Path(plan["config"]["paths"]["cache"])
        target = torch.load(cache / "teacher" / (case["id"] + ".pt"), map_location=device, weights_only=True).float()
        if case["gt_record"]["objects_reliable"]:
            instances = torch.load(cache / "instances" / (case["id"] + ".pt"), map_location=device, weights_only=True)
    recipe = plan["recipe"]
    generator = torch.Generator(device=device).manual_seed(recipe["seed"])
    x = fix_first(torch.randn((1, 48, 1 + (frames - 1) // 4, h // 16, w // 16), generator=generator, device=device), first)
    sampler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sampler.set_timesteps(recipe["steps"], device=device, shift=recipe["shift"])
    with autocast(device):
        # Cache ONLY deployable P0; every solver call starts from this same tensor.
        p0 = model.adapter.initialize(text, geo, case["mode"], anchor)
    trajectory = folder / "trajectory.jsonl"
    trajectory.write_text("")
    started = time.perf_counter()
    last_p, last_sigma = None, None
    for index, timestep in enumerate(sampler.timesteps):
        sigma = sampler.sigmas[index].to(device=device, dtype=torch.float32)
        with autocast(device):
            result = model(x, first, text, sigma, case["duration"], geo, p0, negative)
        x = sampler.step(result["guided"].float(), timestep, x.float(), return_dict=False, generator=generator)[0]
        x = fix_first(x, first)
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError(f"Nonfinite sampling latent: {case['key']}, step {index}")
        relations = state_relations(result, target, geo, model.adapter.s_z, instances) if target is not None else None
        log(trajectory, event="solver_call", step=index, sigma=float(sigma), state_to_gt=relations,
            metrics=result["metrics"], known_first_fixed=first is not None, console=False)
        log(event="sampling", case=case["key"], step=index + 1, steps=recipe["steps"],
            seconds=round(time.perf_counter() - started, 1))
        last_p, last_sigma = result["p15"], float(sigma)
        del result
    torch.cuda.synchronize(device)
    sampling_seconds = time.perf_counter() - started
    save_tensor(x.cpu(), folder / "final_latent.pt")
    save_tensor(dict(p15=last_p.cpu(), sigma=last_sigma, geometry=geo.geo), folder / "final_state.pt")
    # GT single-step diagnostics use a separate task/RNG AFTER deployment-only full generation.
    if target is not None:
        gt = torch.load(cache / "vae" / (case["id"] + ".pt"), map_location=device, weights_only=True)["latent"]
        generator = torch.Generator(device=device).manual_seed(42)
        noise = torch.randn(gt.shape, generator=generator, device=device)
        single_path = folder / "gt_noised.jsonl"
        single_path.write_text("")
        for value in (.98, .85, .60, .35, .10):
            sigma = torch.tensor(value, device=device)
            noisy = fix_first((1 - sigma) * gt + sigma * noise, first)
            with autocast(device):
                result = model(noisy, first, text, sigma, case["duration"], geo, p0, negative)
            log(single_path, event="gt_noised", sigma=value,
                state_to_gt=state_relations(result, target, geo, model.adapter.s_z, instances), metrics=result["metrics"])
    # Stream export; retain only the 32 teacher frames after the SAME 384 letterbox geometry.
    teacher_order = teacher_indices(frames).tolist()
    indices = set(teacher_order)
    evidence_indices = set(np.linspace(0, frames - 1, min(frames, 12)).round().astype(int).tolist())
    if "gt_record" in case:
        evidence_indices.update(case["gt_record"]["interactions"])
    teacher_frames = {}
    diagnostics_seconds = time.perf_counter() - started - sampling_seconds
    tg = fit_geometry(h, w, 384, 384)
    with imageio.get_writer(str(folder / "video.mp4"), fps=(frames - 1) / case["duration"], codec="libx264", quality=8,
                            macro_block_size=16) as writer:
        for index, frame in enumerate(decoder.frames(x)):
            if not bool(torch.isfinite(frame).all()):
                raise FloatingPointError(f"Nonfinite FP32 VAE output: {case['key']}, frame {index}")
            exported = frame.clamp(0, 1)
            pixels = (exported * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            if index in indices:
                # Use the same uint8 RGB given to the exporter, with no premature BF16 rounding.
                teacher_pixels = torch.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1
                teacher_frames[index] = letterbox(teacher_pixels[:, None], tg)
            writer.append_data(pixels)
            if index in evidence_indices:
                Image.fromarray(pixels).save(folder / f"event_{index:03d}.jpg", quality=90)
    if set(teacher_frames) != indices or index != frames - 1:
        raise RuntimeError("Export did not cover the source window and ordered teacher frames")
    save_tensor(torch.cat([teacher_frames[i] for i in teacher_order], dim=1)[None], folder / "generated_teacher_view.pt")
    names = ["video.mp4", "final_latent.pt", "final_state.pt", "trajectory.jsonl", "generated_teacher_view.pt"]
    if target is not None:
        names.append("gt_noised.jsonl")
    save_json(dict(status="completed", signature=signature, case=case, steps=recipe["steps"], sigma_of_final_p15=last_sigma,
                   checkpoint=plan["checkpoint"], checkpoint_step=plan["checkpoint_step"],
                   ema_updates=plan["ema_updates"], ema_file=plan["ema_file"], recipe=recipe,
                   native_negative=plan["negative"], negative_sha256=plan["negative_sha256"],
                   fps=(frames - 1) / case["duration"], duration=case["duration"], frames=frames, geometry=geo.geo,
                   source_pts=case.get("gt_record", {}).get("pts"),
                   source_fps=case.get("gt_record", {}).get("source_fps"),
                   cfg_scheme="native_cfg_plus_bounded_positive_delta", p_policy="reset",
                   generation_logical_predictions=3 * recipe["steps"], generation_block_executions=85 * recipe["steps"],
                   gt_noised_logical_predictions=15 if target is not None else 0,
                   sampling_seconds=sampling_seconds, gt_diagnostics_seconds=diagnostics_seconds,
                   total_seconds=time.perf_counter() - started,
                   teacher_rgb_source="pre_codec_uint8_export_frames",
                   files={name: file_stamp(folder / name) for name in names},
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30), complete)
    log(event="case_complete", key=case["key"])


def sample(root, device=None):
    plan = read_json(root / "cases.json")
    ready = read_json(root / "conditions_ready.json")
    if ready["plan_hash"] != digest(plan):
        raise ValueError("Prepared conditions do not match the fixed case plan")
    statuses = [dict(key=c["key"], mode=c["mode"], video=str(root / c["key"] / "video.mp4"),
                     status="completed" if completed_case(c, plan, root) else "pending") for c in plan["cases"]]
    summary = dict(status="running", total=len(statuses), cases=statuses, checkpoint=plan["checkpoint"],
                   checkpoint_step=plan["checkpoint_step"], ema_updates=plan["ema_updates"], weights="ema.pt",
                   suite=plan["suite"], recipe=plan["recipe"])
    def progress():
        summary["completed"] = sum(c["status"] == "completed" for c in statuses)
        save_json(summary, root / "summary.json")
    progress()
    log(event="batch_start", total=summary["total"], completed=summary["completed"],
        checkpoint=plan["checkpoint"], step=plan["checkpoint_step"], weights="ema.pt")
    active = None
    try:
        if summary["completed"] != summary["total"]:
            log(event="loading", model="CUDA device")
            device = device if device is not None else require_single_gpu()
            adapter = load_adapter(plan, device)
            log(event="loading", model="Wan2.2 TI2V 5B")
            wan = load_wan(plan["config"]["paths"]["wan_checkpoint"], device)
            model = ProcessWan(wan, adapter, recompute=False).eval()
            log(event="loading", model="FP32 VAE decoder")
            decoder = Decoder(load_vae(plan["config"]["paths"], device), recompute=False, save_on_cpu=False)
            log(event="models_ready", pending=summary["total"] - summary["completed"])
            for case, active in zip(plan["cases"], statuses):
                if active["status"] == "completed":
                    continue
                active["status"] = "running"
                progress()
                seed_all(plan["recipe"]["seed"])
                torch.cuda.reset_peak_memory_stats(device)
                sample_case(model, decoder, case, plan, root, device)
                active["status"] = "completed"
                progress()
                log(event="batch_progress", completed=summary["completed"], total=summary["total"], key=case["key"])
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        if active is not None and active["status"] == "running":
            active.update(status="failed", error=summary["error"])
        raise
    finally:
        progress()
    log(event="batch_complete", completed_cases=summary["completed"], output=str(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Defaults to the explicit checkpoint's saved config.json")
    parser.add_argument("--phase", choices=("prepare", "sample"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", choices=("final", "demo", "single"), default="demo")
    parser.add_argument("--mode", choices=("i2v", "t2v"), default="i2v")
    parser.add_argument("--image")
    parser.add_argument("--prompt")
    parser.add_argument("--duration", type=float, default=5.)
    parser.add_argument("--portrait", action="store_true")
    args = parser.parse_args()
    log(event="inference_start", phase=args.phase, output=args.output)
    require_environment("runtime")
    root = output_path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with job_lock(root / ".inference.lock"):
        if args.phase == "sample":
            sample(root)
        else:
            cfg_path = args.config or (Path(args.checkpoint).expanduser() / "config.json" if args.checkpoint
                                       else ROOT / "configs/stability.yaml")
            cfg = read_config(cfg_path)
            prepare(cfg, args, root)


if __name__ == "__main__":
    main()
