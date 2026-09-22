"""LoRA EMA inference with native Wan CFG and the shared reference/P01-P20 suite."""
import argparse
from collections import Counter
import gc
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    print(f"[inference] Python started: {sys.executable}; importing PyTorch ...", flush=True)
import numpy as np
from PIL import Image
import torch

from static_lora import ROOT, VERSION
from static_lora.backbone import fix_first, load_model
from static_lora.checkpoint import checkpoint_path
from static_lora.config import DEFAULT_CONFIG, configuration
from static_lora.data import Dataset
from static_lora.lora import load_lora_state_dict
from static_lora.runtime import (autocast, digest, file_identity, job_lock, log, native_negative,
                                 output_path, read_json, require_environment, require_single_gpu,
                                 save_json, save_tensor, seed_all, sha256, verify_files)
from static_lora.encoders import Decoder, encode_vae, load_text, load_vae
from static_lora.geometry import canvas, fit_geometry, letterbox

if __name__ == "__main__":
    print("[inference] Python/PyTorch ready.", flush=True)


def collect_cases(cfg, args):
    if args.suite == "single":
        if not args.prompt or not args.prompt.strip() or (args.mode == "i2v" and not args.image):
            raise ValueError("Single generation needs a prompt and, for I2V, an image")
        if args.portrait and args.mode != "t2v":
            raise ValueError("--portrait applies only to single T2V")
        return [dict(key="single_" + args.mode, id="single", group="new_prompt", mode=args.mode,
                     image=args.image if args.mode == "i2v" else None, prompt=args.prompt,
                     duration=args.duration, portrait=args.portrait)]
    if args.portrait or args.duration != 5.:
        raise ValueError("Use --suite single for a custom canvas/duration")
    cases = []
    if args.suite == "final":
        manifest = Dataset(cfg, "train").manifest
        by_id = {r["id"]: r for r in manifest["records"]}
        for split in ("train", "validation"):
            for fixed in manifest["diagnostics"][split]:
                record = by_id[fixed["id"]]
                cases.append(dict(key=f'{split}_{record["id"]}_{fixed["mode"]}', id=record["id"],
                                  group=split, mode=fixed["mode"], prompt=record["caption"],
                                  duration=record["duration"], gt_record=record))
    cases.append(dict(key="reference_i2v", id="reference", group="reference", mode="i2v",
                      image=cfg["paths"]["reference_image"], prompt=cfg["paths"]["reference_prompt"], duration=5.))
    for number in range(1, 21):
        key = f"P{number:02d}"
        folder = Path(cfg["paths"]["demo_root"]) / key
        text_file = folder / (key + "-origin.txt")
        prompt = text_file.read_text(encoding="utf-8-sig").strip()
        image = next((folder / (key + "-" + suffix) for suffix in
                      ("i0-1280x704.png", "i0-1280x704.jpg", "i0.png", "i0.jpg")
                      if (folder / (key + "-" + suffix)).exists()), None)
        if image is None or not prompt:
            raise FileNotFoundError(f"Missing reference image/prompt for {key}")
        for mode in ("i2v", "t2v"):
            cases.append(dict(key=f"{key}_{mode}", id=key, group="demo", mode=mode, prompt=prompt,
                              image=str(image) if mode == "i2v" else None,
                              prompt_sha256=sha256(text_file), duration=5.))
    return cases


def build_plan(cfg, args):
    checkpoint = checkpoint_path(cfg, args.checkpoint)
    status = read_json(checkpoint / "complete.json")
    cases = collect_cases(cfg, args)
    negative = native_negative()
    if args.suite == "final":
        manifest = read_json(Path(cfg["paths"]["cache"]) / "manifest.json")
        if digest(manifest) != status["identity"]["manifest"]:
            raise ValueError("Final evaluation must use the same source split as training")
        if negative != manifest["negative_prompt"]:
            raise ValueError("Cached negative prompt differs from native Wan")
    for case in cases:
        if not np.isfinite(case["duration"]) or case["duration"] <= 0:
            raise ValueError("Duration must be finite and positive")
        if case.get("image"):
            case["image"] = str(Path(case["image"]).expanduser().resolve())
            case["image_sha256"] = sha256(case["image"])
            with Image.open(case["image"]) as image:
                h, w = canvas(image.height, image.width, cfg["data"]["max_long_side"], cfg["data"]["max_area"])
                geo = fit_geometry(image.height, image.width, h, w, upscale=False)
                image.verify()
        elif "gt_record" in case:
            geo = dict(case["gt_record"]["geometry"])
        else:
            w, h = cfg["inference"]["canvas_wh"]
            if case.get("portrait"):
                h, w = w, h
            geo = fit_geometry(h, w, h, w)
        geo.setdefault("frames", cfg["data"]["frames"])
        case["geometry"] = geo
    inference = cfg["inference"]
    return dict(version=VERSION, suite=args.suite, checkpoint=str(checkpoint), checkpoint_step=status["step"],
                ema_updates=status.get("ema_updates"),
                ema_sha256=sha256(checkpoint / "ema.pt"), config=cfg, cases=cases,
                negative=negative, negative_sha256=digest(negative), identity=status["identity"],
                recipe=dict(solver="unipc", steps=inference["steps"], shift=inference["shift"],
                            guidance=inference["guidance"], seed=inference["seed"],
                            cfg_scheme="native_cfg_with_lora_on_both_branches"))


def validate_ema(cfg, checkpoint):
    """Check real adapter tensors against original safetensors header shapes on CPU."""
    from safetensors import safe_open
    base = Path(cfg["paths"]["wan_checkpoint"])
    native = read_json(base / "config.json")
    if (native["num_layers"], native["dim"], native["in_dim"], native["out_dim"]) != (30, 3072, 48, 48):
        raise ValueError("Expected original Wan2.2-TI2V-5B")
    index = read_json(base / "diffusion_pytorch_model.safetensors.index.json")["weight_map"]
    expected = {}
    shards = {}
    for block in range(native["num_layers"]):
        for target in cfg["lora"]["targets"]:
            name = f"blocks.{block}.{target}"
            shards.setdefault(index[name + ".weight"], []).append(name)
    for shard, names in shards.items():
        with safe_open(str(base / shard), framework="pt", device="cpu") as reader:
            for name in names:
                out_features, in_features = reader.get_slice(name + ".weight").get_shape()
                expected[f"wan.{name}.lora_A"] = (cfg["lora"]["rank"], in_features)
                expected[f"wan.{name}.lora_B"] = (out_features, cfg["lora"]["rank"])
    values = torch.load(Path(checkpoint) / "ema.pt", map_location="cpu", weights_only=True)
    if set(values) != set(expected):
        raise ValueError("EMA LoRA target names differ from the original Wan model")
    for name, shape in expected.items():
        value = values[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.float32:
            raise ValueError(f"Invalid EMA LoRA shape/dtype: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite EMA LoRA weight: {name}")
    return dict(tensors=len(values), parameters=sum(v.numel() for v in values.values()),
                dtype="float32", finite=True, base_tensor_shapes_verified=True)


def check(cfg, args, root):
    plan = build_plan(cfg, args)
    weights = validate_ema(cfg, plan["checkpoint"])
    if args.suite != "final":
        load_local_negative(plan["negative"])
    save_json(plan, root / "preflight_cases.json")
    result = dict(status="passed", checkpoint=plan["checkpoint"], checkpoint_step=plan["checkpoint_step"],
                  ema_updates=plan["ema_updates"], ema_sha256=plan["ema_sha256"], weights=weights,
                  suite=plan["suite"], total_cases=len(plan["cases"]),
                  modes=dict(Counter(c["mode"] for c in plan["cases"])), recipe=plan["recipe"],
                  gpu_sampling_run=False)
    save_json(result, root / "preflight.json")
    log(event="preflight_passed", **result)


def load_local_negative(prompt):
    """Reuse the exact training-time native negative encoding from this project."""
    path = ROOT / "assets/negative.pt"
    metadata = read_json(ROOT / "assets/negative.json")
    if metadata["prompt"] != prompt or sha256(path) != metadata["sha256"]:
        raise ValueError("Local negative embedding/prompt differs from its pinned identity")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (value.ndim != 2 or not 1 <= value.shape[0] <= 512 or value.shape[1] != 4096 or
            value.dtype not in (torch.float32, torch.bfloat16) or not bool(torch.isfinite(value).all())):
        raise ValueError("Invalid native Wan negative text embedding")
    return value


def prepare(cfg, args, root, device):
    plan = build_plan(cfg, args)
    cases = plan["cases"]
    cache = Path(cfg["paths"]["cache"])
    plan_file = root / "cases.json"
    if plan_file.exists() and read_json(plan_file) != plan:
        raise ValueError("Output folder belongs to another checkpoint/case plan; set a different OUTPUT")
    save_json(plan, plan_file)
    stamp = root / "conditions_ready.json"
    if stamp.exists():
        ready = read_json(stamp)
        if ready["plan_hash"] != digest(plan):
            raise ValueError("Prepared conditions differ from the case plan")
        verify_files(root, ready["files"], full=True)
        log(event="conditions_reused", cases=len(cases))
        return
    cached_names = {"negative.pt"} if args.suite == "final" else set()
    for case in cases:
        if "gt_record" in case:
            cached_names.add(f"text/{case['id']}.pt")
            if case["mode"] == "i2v":
                cached_names.add(f"vae/{case['id']}.pt")
    if cached_names:
        ready = read_json(cache / "ready.json")
        verify_files(cache, {name: ready["files"][name] for name in cached_names})
    negative_embedding = (torch.load(cache / "negative.pt", map_location="cpu", weights_only=True)
                          if args.suite == "final" else load_local_negative(plan["negative"]))
    external_prompts = list(dict.fromkeys(case["prompt"] for case in cases if "gt_record" not in case))
    texts = {}
    if external_prompts:
        log(event="loading_text_encoder", prompts=len(external_prompts))
        encoder = load_text(cfg["paths"], device)
        with torch.no_grad(), autocast(device):
            for prompt in external_prompts:
                texts[prompt] = encoder([prompt], device)[0].cpu().bfloat16()
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    vae = load_vae(cfg["paths"], device) if any(case.get("image") for case in cases) else None
    files = {}
    for case in cases:
        folder = root / case["key"]
        text = (torch.load(cache / "text" / (case["id"] + ".pt"), map_location="cpu", weights_only=True)
                if "gt_record" in case else texts[case["prompt"]])
        condition = dict(text=text, negative=negative_embedding, first=None)
        geo = dict(case["geometry"])
        if "gt_record" in case:
            if case["mode"] == "i2v":
                condition["first"] = torch.load(cache / "vae" / (case["id"] + ".pt"),
                                                 map_location="cpu", weights_only=True)["first"]
        elif case["mode"] == "i2v":
            with Image.open(case["image"]) as image:
                pixels = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)[:, None].float() / 127.5 - 1
            pixels = letterbox(pixels, geo)[None]
            condition["first"] = encode_vae(vae, pixels.to(device)).cpu()
        condition["geometry"] = geo
        save_tensor(condition, folder / "condition.pt")
        files[str((folder / "condition.pt").relative_to(root))] = file_identity(folder / "condition.pt")
        log(event="condition_ready", key=case["key"], completed=len(files), total=len(cases))
    save_json(dict(plan_hash=digest(plan), files=files), stamp)
    log(event="conditions_prepared", cases=len(cases))


def case_signature(case, plan):
    return digest(dict(case=case, ema=plan["ema_sha256"], recipe=plan["recipe"]))


def completed_case(case, plan, root):
    folder = root / case["key"]
    marker = folder / "metadata.json"
    if not marker.exists():
        return False
    prior = read_json(marker)
    if prior.get("status") != "completed" or prior.get("signature") != case_signature(case, plan):
        return False
    verify_files(folder, prior["files"], full=True)
    return True


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
    if "geometry" in case and condition["geometry"] != case["geometry"]:
        raise ValueError("Prepared geometry differs from the case plan")
    text, negative = condition["text"][None].to(device), condition["negative"][None].to(device)
    first = condition["first"].to(device) if condition["first"] is not None else None
    if (case["mode"] == "i2v") != (first is not None):
        raise ValueError("Mode/first-slice condition mismatch")
    geo, recipe = condition["geometry"], plan["recipe"]
    h, w, frames = geo["h"], geo["w"], geo["frames"]
    generator = torch.Generator(device=device).manual_seed(recipe["seed"])
    x = fix_first(torch.randn((1, 48, 1 + (frames - 1) // 4, h // 16, w // 16),
                              generator=generator, device=device), first)
    sampler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sampler.set_timesteps(recipe["steps"], device=device, shift=recipe["shift"])
    trajectory = folder / "trajectory.jsonl"
    trajectory.write_text("")
    started = time.perf_counter()
    for index, timestep in enumerate(sampler.timesteps):
        sigma = sampler.sigmas[index].to(device=device, dtype=torch.float32)
        prediction = model.guided(x, first, text, negative, sigma, recipe["guidance"])
        x = sampler.step(prediction.float(), timestep, x.float(), return_dict=False, generator=generator)[0]
        x = fix_first(x, first)
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError(f"Nonfinite latent in {case['key']} at solver step {index}")
        log(trajectory, console=False, event="solver_call", step=index, sigma=float(sigma), known_first_fixed=first is not None)
        if (index + 1) % 5 == 0 or index == 0:
            log(event="sampling", key=case["key"], step=index + 1, steps=recipe["steps"])
    torch.cuda.synchronize(device)
    sampling_seconds = time.perf_counter() - started
    save_tensor(x.cpu(), folder / "final_latent.pt")
    evidence = set(np.linspace(0, frames - 1, min(frames, 12)).round().astype(int).tolist())
    if "gt_record" in case:
        evidence.update(case["gt_record"]["interactions"])
    count = 0
    with imageio.get_writer(str(folder / "video.mp4"), fps=(frames - 1) / case["duration"], codec="libx264",
                            quality=8, macro_block_size=16) as writer:
        for index, frame in enumerate(decoder.frames(x)):
            if not bool(torch.isfinite(frame).all()):
                raise FloatingPointError(f"Nonfinite VAE output in {case['key']}, frame {index}")
            pixels = (frame.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            writer.append_data(pixels)
            if index in evidence:
                Image.fromarray(pixels).save(folder / f"event_{index:03d}.jpg", quality=90)
            count += 1
    if count != frames:
        raise RuntimeError(f"Expected {frames} decoded frames, got {count}")
    names = ["video.mp4", "final_latent.pt", "trajectory.jsonl"]
    save_json(dict(status="completed", signature=signature, case=case, checkpoint_step=plan["checkpoint_step"],
                   recipe=recipe, native_negative=plan["negative"], negative_sha256=plan["negative_sha256"],
                   fps=(frames - 1) / case["duration"], duration=case["duration"], frames=frames, geometry=geo,
                   source_pts=case.get("gt_record", {}).get("pts"),
                   source_fps=case.get("gt_record", {}).get("source_fps"),
                   generation_logical_predictions=2 * recipe["steps"],
                   sampling_seconds=sampling_seconds, total_seconds=time.perf_counter() - started,
                   peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                   files={name: file_identity(folder / name) for name in names}), complete)
    log(event="case_complete", key=case["key"], sampling_seconds=sampling_seconds)


def sample(root, device):
    plan = read_json(root / "cases.json")
    cfg = configuration(Path(plan["checkpoint"]) / "config.json")
    checkpoint_path(cfg, plan["checkpoint"])
    if digest(cfg) != digest(plan["config"]) or sha256(Path(plan["checkpoint"]) / "ema.pt") != plan["ema_sha256"]:
        raise ValueError("Case plan configuration/EMA differs from the checkpoint")
    ready = read_json(root / "conditions_ready.json")
    if ready["plan_hash"] != digest(plan):
        raise ValueError("Prepared conditions differ from the case plan")
    verify_files(root, ready["files"], full=True)
    statuses = [dict(key=c["key"], mode=c["mode"], status="completed" if completed_case(c, plan, root) else "pending")
                for c in plan["cases"]]
    summary = dict(status="running", checkpoint=plan["checkpoint"], checkpoint_step=plan["checkpoint_step"],
                   total=len(statuses), cases=statuses)
    def save_summary():
        summary["completed"] = sum(c["status"] == "completed" for c in statuses)
        save_json(summary, root / "summary.json")
    save_summary()
    pending = [(case, state) for case, state in zip(plan["cases"], statuses) if state["status"] != "completed"]
    log(event="batch_start", total=len(statuses), pending=len(pending), checkpoint=plan["checkpoint"])
    active = None
    try:
        if pending:
            log(event="loading_wan", path=cfg["paths"]["wan_checkpoint"])
            model = load_model(cfg, device, training=False)
            load_lora_state_dict(model, torch.load(Path(plan["checkpoint"]) / "ema.pt", map_location="cpu", weights_only=True))
            log(event="ema_loaded", checkpoint_step=plan["checkpoint_step"], path=plan["checkpoint"])
            decoder = Decoder(load_vae(cfg["paths"], device), recompute=False, save_on_cpu=False)
            for case, active in pending:
                active["status"] = "running"
                save_summary()
                seed_all(plan["recipe"]["seed"])
                torch.cuda.reset_peak_memory_stats(device)
                sample_case(model, decoder, case, plan, root, device)
                active["status"] = "completed"
                save_summary()
                log(event="batch_progress", completed=summary["completed"], total=len(statuses), key=case["key"])
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        if active is not None and active["status"] == "running":
            active.update(status="failed", error=summary["error"])
        raise
    finally:
        save_summary()


def report(root):
    plan = read_json(root / "cases.json")
    results = []
    for case in plan["cases"]:
        folder = root / case["key"]
        result = read_json(folder / "metadata.json")
        if result.get("status") != "completed" or result.get("signature") != case_signature(case, plan):
            raise ValueError(f"Incomplete or stale case: {case['key']}")
        verify_files(folder, result["files"])
        results.append(dict(key=case["key"], mode=case["mode"], group=case["group"],
                            video=str(folder / "video.mp4"), sampling_seconds=result["sampling_seconds"]))
    result = dict(checkpoint=plan["checkpoint"], checkpoint_step=plan["checkpoint_step"],
                  completed_cases=len(results), recipe=plan["recipe"], cases=results)
    save_json(result, root / "report.json")
    log(event="report", completed_cases=len(results), path=str(root / "report.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Defaults to the selected checkpoint's saved config.json")
    parser.add_argument("--phase", choices=("check", "prepare", "sample", "report"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", choices=("final", "demo", "single"), default="demo")
    parser.add_argument("--mode", choices=("i2v", "t2v"), default="i2v")
    parser.add_argument("--image")
    parser.add_argument("--prompt")
    parser.add_argument("--duration", type=float, default=5.)
    parser.add_argument("--portrait", action="store_true")
    args = parser.parse_args()
    require_environment("runtime")
    root = output_path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with job_lock(root / ".inference.lock"):
        if args.phase == "report":
            report(root)
        elif args.phase in ("check", "prepare"):
            cfg = configuration(args.config or (Path(args.checkpoint).expanduser() / "config.json"
                                                if args.checkpoint else DEFAULT_CONFIG))
            if args.phase == "check":
                check(cfg, args, root)
            else:
                prepare(cfg, args, root, require_single_gpu())
        else:
            sample(root, require_single_gpu())


if __name__ == "__main__":
    main()
