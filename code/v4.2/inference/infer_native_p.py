"""Single-GPU image/text-to-video sampling with the architecture saved by v4 training."""
import argparse
from collections import Counter
import copy
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physgen_v4.environments import require_environment
require_environment("runtime")

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

from physgen_v4.runtime import ROOT, read_config, seed_all
from physgen_v4.corrector import Corrector
from physgen_v4.backbone import load_wan, ProcessWan
from physgen_v4.coordinates import Coordinates
from physgen_v4.data import letterbox
from physgen_v4.encoders import load_vae, load_text_encoder, DifferentiableDecoder
from physgen_v4.views import choose_canvas


DEFAULT_CHECKPOINT = "latest"


def resolve_checkpoint(value):
    if str(value) in ('latest', 'latest_with_prior', 'latest_no_prior'):
        variant = 'without_prior' if str(value) == 'latest_no_prior' else 'with_prior'
        pointer = ROOT / f'checkpoints/v42_v4_fullwidth3_write_{variant}_latest.json'
        if not pointer.is_file():
            raise FileNotFoundError('No trained full-width three-corrector checkpoint exists yet. '
                                    'Train the new recipe first, or explicitly supply a compatible --checkpoint.')
        saved = json.loads(pointer.read_text())
        if saved['architecture'] != 'v4_fullwidth3_write_v1' or saved.get('loss_variant') != variant:
            raise ValueError('Latest checkpoint has the wrong architecture')
        value = saved['checkpoint']
    return str(Path(value).expanduser().resolve())


def state_policy(config, args):
    warmup = config.get("train", {}).get("state_warmup_every")
    requested = getattr(args, "reset_state", None)
    reset = warmup == 0 if requested is None else requested
    return dict(mode="reset" if reset else "persistent", reset_state=reset,
                selection="checkpoint" if requested is None else "explicit",
                training_state_warmup_every=warmup)


def load_checkpoint(checkpoint):
    """Strict restore, without INIT_FROM migration or current-training config merging."""
    checkpoint = Path(resolve_checkpoint(checkpoint))
    if not (checkpoint / 'corrector.pt').is_file():
        raise FileNotFoundError(f"Native V4 checkpoint required: {checkpoint}/corrector.pt. "
                                "Old 512-wide V4.2 adapter.pt/ema.pt cannot be loaded into this architecture.")
    config = read_config(checkpoint / "config.yaml")
    if config["stage"] not in ("A", "B", "AB"):
        raise ValueError(f"Unknown checkpoint stage: {config['stage']}")
    # Avoid initializing another ~1B random parameters and duplicating CPU weights.
    with torch.device("meta"):
        corrector = Corrector(config["corrector"])
    weights = torch.load(checkpoint / "corrector.pt", map_location="cpu", mmap=True, weights_only=True)
    corrector.load_state_dict(weights, strict=True, assign=True)
    corrector.eval().requires_grad_(False)
    report = dict(
        checkpoint=str(checkpoint), phase=config.get("phase"), stage=config["stage"],
        run_name=config.get("run_name"),
        initialization=corrector.initialization, fusion=corrector.fusion,
        parameter_sharing=corrector.parameter_sharing,
        a_blocks=list(corrector.a_blocks) if corrector.parameter_sharing == "per_block" else None,
        block_b=config["corrector"]["block_b"], iterations=config["corrector"]["iterations"],
        enable_b=config["stage"] in ("B", "AB"), strict_load=True,
        state_dict_tensors=len(weights), parameters=sum(p.numel() for p in corrector.parameters()),
        weight_bytes=sum(v.numel() * v.element_size() for v in weights.values()),
        weight_dtypes=dict(Counter(str(v.dtype) for v in weights.values())),
        optimizer_loaded=False, teacher_loaded=False,
        inference_gt_conditioned=False,
        training_gt_condition_dropout=config.get("train", {}).get("gt_condition_dropout"),
    )
    return config, corrector, report


def validate_assets(config):
    """Check only generation assets; neither WISA caches nor JEPA are needed."""
    root = Path(config["paths"]["wan_checkpoint"])
    wan_config = json.loads((root / "config.json").read_text())
    expected = dict(model_type="ti2v", dim=3072, in_dim=48, out_dim=48, num_layers=30, text_len=512)
    for key, value in expected.items():
        if wan_config.get(key) != value:
            raise ValueError(f"Wan TI2V-5B requires {key}={value}; got {wan_config.get(key)}")
    index = root / "diffusion_pytorch_model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values())
    files = [root / name for name in sorted(shards)] + [
        root / "Wan2.2_VAE.pth", root / "models_t5_umt5-xxl-enc-bf16.pth",
        root / "google/umt5-xxl/spiece.model",
        Path(config["paths"]["wan_code"]) / "wan/modules/model.py",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local inference assets: {missing}")
    return dict(wan_checkpoint=str(root), wan_config=wan_config, wan_shards=len(shards))


def validate_writers(model, writer_off):
    valid = {"A", "B"} | {label for _, _, label in model.correction_sites()}
    unknown = set(writer_off) - valid
    if unknown:
        raise ValueError(f"Unknown writer locations: {sorted(unknown)}; available: {sorted(valid)}")


def video_spec(config, frames, fps, duration):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and positive")
    requested = config["data"]["max_frames"] if frames is None else frames
    actual = 1 + 4 * ((min(requested, config["data"]["max_frames"]) - 1) // 4)
    if actual < 5:
        raise ValueError("Video generation requires at least 5 frames (4n+1)")
    duration = (actual - 1) / fps if duration is None else duration
    return dict(requested_frames=requested, frames=actual, duration=duration,
                output_fps=(actual - 1) / duration)


def validate_sampling(args):
    if getattr(args, "mode", "i2v") not in ("i2v", "t2v"):
        raise ValueError("mode must be i2v or t2v")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if not math.isfinite(args.shift) or args.shift <= 0:
        raise ValueError("shift must be finite and positive")
    if not math.isfinite(args.guidance) or args.guidance < 0:
        raise ValueError("guidance must be finite and nonnegative")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if args.device < 0:
        raise ValueError("device must be a nonnegative visible CUDA index")
    if args.solver not in ("euler", "unipc", "dpm++"):
        raise ValueError(f"Unknown solver: {args.solver}")
    if args.variant not in ("full", "wan", "A"):
        raise ValueError(f"Unknown variant: {args.variant}")


def text_only_canvas(config, args):
    height, width = args.height, args.width
    if any(type(size) is not int or size < 32 or size % 32 for size in (height, width)):
        raise ValueError("T2V height/width must be positive multiples of 32")
    if max(height, width) > config["data"]["max_long_side"] or height * width > config["data"]["max_area"]:
        raise ValueError("T2V canvas exceeds the checkpoint's training geometry limits")
    return height, width


def schedule(solver, steps, shift, device):
    if steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("Sampling steps and shift must be positive")
    if solver == "euler":
        raw = torch.linspace(1, 0, steps + 1, device=device)
        sigmas = shift * raw / (1 + (shift - 1) * raw)
        return None, sigmas[:-1] * 1000, sigmas
    if solver == "unipc":
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        sampler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        sampler.set_timesteps(steps, device=device, shift=shift)
    elif solver == "dpm++":
        from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
        sampler = FlowDPMSolverMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        retrieve_timesteps(sampler, device=device, sigmas=get_sampling_sigmas(steps, shift))
    else:
        raise ValueError(f"Unknown solver: {solver}")
    return sampler, sampler.timesteps, sampler.sigmas.to(device)


@torch.no_grad()
def sample_latents(model, first, conditional, unconditional, coords, args, on_step=None):
    """Two independent P chains, two Wan calls and one scheduler update per step."""
    validate_sampling(args)
    validate_writers(model, args.writer_off)
    if (getattr(args, "mode", "i2v") == "t2v") != (first is None):
        raise ValueError("I2V requires a first latent; T2V must pass first=None")
    device = first.device if first is not None else conditional.device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    spatial = first.shape[-2:] if first is not None else tuple(axis.numel() for axis in coords.latent[1:])
    shape = (1, 48, coords.latent[0].numel(), *spatial)
    latent = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    if first is not None:
        latent = torch.cat((first.float(), latent[:, :, 1:]), dim=2)
    sampler, timesteps, sigmas = schedule(args.solver, args.steps, args.shift, device)
    states = dict(cond=None, uncond=None)
    disabled = {"full": (), "wan": ("A", "B"), "A": ("B",)}[args.variant]
    counters = dict(wan_forwards=0, scheduler_updates=0, corrector_calls=0,
                    image_conditioned=first is not None, first_frame_clamped=first is not None)
    for step, timestep in enumerate(timesteps):
        started = time.perf_counter()
        sigma = sigmas[step].float()
        outputs, metrics = {}, {}
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda", cache_enabled=False):
            for branch, text in (("cond", conditional), ("uncond", unconditional)):
                process = None if args.reset_state else states[branch]
                outputs[branch], states[branch], info = model(
                    latent, first, text, sigma, coords, process,
                    disable=disabled, writer_off=tuple(args.writer_off))
                # Copy scalar diagnostics in one transfer, then release intermediate P states.
                values = info["metrics"]
                metrics[branch] = (dict(zip(values, torch.stack(list(values.values())).float().cpu().tolist()))
                                   if values else {})
                counters["wan_forwards"] += 1
                counters["corrector_calls"] += len(info["states"])
                del info
            vector = outputs["uncond"].float() + args.guidance * (
                outputs["cond"].float() - outputs["uncond"].float())
        if sampler is None:
            latent = latent + (sigmas[step + 1] - sigmas[step]) * vector
        else:
            latent = sampler.step(vector, timestep, latent, return_dict=False, generator=generator)[0]
        if first is not None:
            latent = torch.cat((first.float(), latent[:, :, 1:]), dim=2)
        counters["scheduler_updates"] += 1
        if (not bool(torch.isfinite(latent).all())
                or any(not math.isfinite(value) for branch in metrics.values() for value in branch.values())):
            raise FloatingPointError(f"Non-finite sampling values at step {step + 1}, sigma={float(sigma)}")
        if on_step is not None:
            on_step(dict(step=step, sigma=float(sigma), metrics=metrics,
                         wall_seconds=time.perf_counter() - started))
    counters["process_dtypes"] = {key: str(value.dtype) for key, value in states.items() if value is not None}
    return latent, counters


def check_checkpoint(args):
    """Read all actual weights on CPU; do not load Wan/T5/VAE or initialize CUDA."""
    validate_sampling(args)
    config, corrector, report = load_checkpoint(args.checkpoint)
    report["assets"] = validate_assets(config)
    from types import SimpleNamespace
    model = ProcessWan(SimpleNamespace(blocks=[None] * 30), corrector, report["enable_b"])
    validate_writers(model, args.writer_off)
    report["correction_sites"] = model.correction_sites()
    report["video"] = video_spec(config, args.frames, args.fps, args.duration)
    report["state_policy"] = state_policy(config, args)
    from inference.cases import collect_cases, describe_cases
    if getattr(args, "suite", "single") != "single" or args.image is not None or getattr(args, "mode", "i2v") == "t2v":
        report["cases"] = describe_cases(collect_cases(args), config, args)
        report["case_count"] = len(report["cases"])
        report["mode_counts"] = dict(Counter(case["mode"] for case in report["cases"]))
        report["sampling_steps"] = args.steps
        if report["mode_counts"].get("t2v"):
            report["t2v_prior"] = "Text-only context; image tokens omitted. This checkpoint was trained with first images."
    bad = [name for name, value in corrector.state_dict().items() if not bool(torch.isfinite(value).all())]
    if bad:
        raise FloatingPointError(f"Non-finite checkpoint tensors: {bad}")
    report["all_weights_finite"] = True
    sites = ({f"A@{n}": unit.site for n, unit in corrector.A.items()}
             if corrector.parameter_sharing == "per_block" else {"A": corrector.A})
    report["writer_norms"] = {name: float(site.writer.weight.float().norm()) for name, site in sites.items()}
    if corrector.include_b:
        report["writer_norms"]["B"] = float(corrector.B.writer.weight.float().norm())
        report["b_state_scale_parameter"] = float(corrector.B.b)
    report["runtime"] = require_environment("runtime")
    report["torch_version"] = str(torch.__version__)
    report["status"] = "checkpoint_compatible"
    if getattr(args, "report", None):
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


class InferenceRuntime:
    """Load each checkpoint once per job; reuse it across independent videos."""

    def __init__(self, args):
        validate_sampling(args)
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Use a single Python process for inference, not a multi-process torchrun launch")
        self.config, self.corrector, self.checkpoint_report = load_checkpoint(args.checkpoint)
        self.assets = validate_assets(self.config)
        self.device = torch.device("cuda", args.device)
        torch.cuda.set_device(self.device)
        sys.path.insert(0, self.config["paths"]["wan_code"])
        seed_all(args.seed)
        self.texts = {}
        self.vae = None
        self.model = None
        self.vae_dtype = torch.float32 if self.config['data'].get('vae_dtype') == 'float32' else torch.bfloat16

    @torch.no_grad()
    def prepare_texts(self, requests):
        prompts = dict.fromkeys(text for args in requests for text in (args.prompt, args.negative_prompt))
        print(f"Loading UMT5 once; encoding {len(prompts)} distinct prompts", flush=True)
        encoder = load_text_encoder(self.config["paths"], self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            for prompt in prompts:
                self.texts[prompt] = encoder([prompt], self.device)[0].unsqueeze(0).cpu()
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

    def image_vae(self):
        if self.vae is None:
            self.vae = load_vae(self.config["paths"], self.device, dtype=self.vae_dtype)
        else:
            self.vae.model.to(self.device)
        return self.vae

    def sampling_model(self):
        if self.model is None:
            print("Loading frozen Wan and trained correctors once", flush=True)
            self.corrector.to(device=self.device, dtype=torch.float32)
            wan = load_wan(self.config["paths"]["wan_checkpoint"], self.device, sharded=False, recompute=False)
            self.model = ProcessWan(wan, self.corrector, self.checkpoint_report["enable_b"]).eval()
        else:
            self.model.to(self.device)
        return self.model

    def close(self):
        self.model = self.corrector = self.vae = None
        self.texts.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@torch.no_grad()
def generate(args, runtime=None):
    started = time.perf_counter()
    validate_sampling(args)
    if runtime is None:
        with InferenceRuntime(args) as runtime:
            runtime.prepare_texts([args])
            return _generate(args, runtime, started)
    return _generate(args, runtime, started)


def _generate(args, runtime, started):
    config, assets = runtime.config, runtime.assets
    checkpoint_report = copy.deepcopy(runtime.checkpoint_report)
    if (str(Path(args.checkpoint).expanduser().resolve()) != checkpoint_report["checkpoint"]
            or args.device != runtime.device.index):
        raise ValueError("A shared inference runtime requires the same checkpoint and CUDA device")
    policy = state_policy(config, args)
    spec = video_spec(config, args.frames, args.fps, args.duration)
    destination = Path(args.output).expanduser().resolve()
    if destination.suffix.lower() != ".mp4":
        raise ValueError("output must have the .mp4 extension")
    mode = getattr(args, "mode", "i2v")
    reference = transform = None
    if mode == "i2v":
        with Image.open(args.image) as source:
            pixels = torch.from_numpy(np.array(source.convert("RGB"))).unsqueeze(0)
        height, width = choose_canvas(pixels.shape[1], pixels.shape[2],
                                      config["data"]["max_long_side"], config["data"]["max_area"])
        reference, transform = letterbox(pixels, height, width)
    else:
        height, width = text_only_canvas(config, args)
    seed_all(args.seed)
    device = runtime.device
    torch.cuda.reset_peak_memory_stats(device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(event="start", checkpoint=checkpoint_report, video=spec,
                          state_policy=policy, case=getattr(args, "case", None),
                          height=height, width=width, gpu=torch.cuda.get_device_name(device))), flush=True)
    conditional = runtime.texts[args.prompt].to(device)
    unconditional = runtime.texts[args.negative_prompt].to(device)

    first = None
    if reference is not None:
        print("Encoding the reference image with VAE", flush=True)
        vae = runtime.image_vae()
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False,
                            enabled=runtime.vae_dtype != torch.float32):
            first = vae.model.encode(reference.unsqueeze(0).to(device), vae.scale).float()
        vae.model.cpu()
        torch.cuda.empty_cache()

    model = runtime.sampling_model()
    times = torch.linspace(0, spec["duration"], spec["frames"], device=device)
    coords = Coordinates.build(times, height, width, config["data"]["teacher_frames"],
                               config["corrector"]["seconds_scale"], config["data"]["teacher_size"],
                               config['data'].get('teacher_sampling', 'uniform'))
    checkpoint_report["correction_sites"] = model.correction_sites()
    with destination.with_suffix(".steps.jsonl").open("w", buffering=1) as stream:
        def on_step(record):
            stream.write(json.dumps(record) + "\n")
            print(json.dumps(dict(step=record["step"] + 1, total=args.steps, sigma=record["sigma"],
                                  wall_seconds=record["wall_seconds"])), flush=True)
        sampling_args = copy.copy(args)
        sampling_args.reset_state = policy["reset_state"]
        latent, sampling = sample_latents(model, first, conditional, unconditional, coords, sampling_args, on_step)
    # Keep reusable weights on CPU while decoding; every video gets fresh CFG/P chains.
    model.cpu()
    del model, conditional, unconditional
    gc.collect()
    torch.cuda.empty_cache()
    print("Decoding the final video", flush=True)
    vae = runtime.image_vae()
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False,
                        enabled=runtime.vae_dtype != torch.float32):
        video = DifferentiableDecoder(vae, recompute=False)(latent)
    if tuple(video.shape) != (1, 3, spec["frames"], height, width) or not bool(torch.isfinite(video).all()):
        raise RuntimeError(f"Invalid decoded video: shape={tuple(video.shape)}")
    array = ((video[0].permute(1, 2, 3, 0).float().cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    imageio.mimwrite(destination, array, fps=spec["output_fps"], codec="libx264", quality=8, macro_block_size=1)
    metadata = dict(
        status="completed", arguments=dict(vars(args), image=args.image if mode == "i2v" else None),
        checkpoint=checkpoint_report, training_config=config,
        generation_mode=mode, p0_conditioning="image_text" if mode == "i2v" else "text_only",
        t2v_training_note=("This checkpoint was trained with first images; text-only inference is an evaluation mode."
                           if mode == "t2v" else None),
        state_policy=policy, case=getattr(args, "case", None),
        assets=assets, frame_times=times.cpu().tolist(), height=height, width=width, **spec,
        image_transform=transform, wall_seconds=time.perf_counter() - started, **sampling,
        runtime=require_environment("runtime"), torch_version=str(torch.__version__),
        gpu=dict(name=torch.cuda.get_device_name(device), device_index=args.device, inference_gpus=1,
                 total_bytes=torch.cuda.get_device_properties(device).total_memory,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(device)),
    )
    destination.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    vae.model.cpu()
    print(str(destination), flush=True)
    return metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--image")
    parser.add_argument("--mode", choices=("i2v", "t2v"), default="i2v", help="Single/reference generation mode")
    parser.add_argument("--demo-modes", choices=("both", "i2v", "t2v"), default="i2v")
    parser.add_argument("--width", type=int, default=512, help="T2V output width; no reference image is read")
    parser.add_argument("--height", type=int, default=288, help="T2V output height")
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output")
    parser.add_argument("--device", type=int, default=0, help="Index within CUDA_VISIBLE_DEVICES; uses only this GPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=5)
    parser.add_argument("--guidance", type=float, default=5)
    parser.add_argument("--solver", choices=("euler", "unipc", "dpm++"), default="euler")
    parser.add_argument("--variant", choices=("full", "wan", "A"), default="full",
                        help="full enables the saved stage (A1/A2/A3: A; B1/AB: A+B)")
    parser.add_argument("--writer-off", nargs="*", default=[], help="A/B or individual calls, e.g. A@5 A@10")
    parser.add_argument("--reset-state", action="store_true", default=None,
                        help="Reinitialize P each step; default auto resets when training state_warmup_every=0")
    parser.add_argument("--persistent-state", action="store_false", dest="reset_state",
                        help="Explicitly carry independent CFG P states across denoising steps")
    parser.add_argument("--check-only", action="store_true", help="CPU check of actual weights and assets; no generation")
    parser.add_argument("--report", help="Optional JSON path for --check-only")
    parser.add_argument("--suite", choices=("single", "demo", "both"), default="single")
    parser.add_argument("--demo-root", default=str(ROOT.parent / "v1/demo"))
    parser.add_argument("--sample-ids", default="all", help="all means exactly P01-P20; or comma-separated IDs")
    parser.add_argument("--cases", help="JSON condition list replacing the reference case in suite=both")
    parser.add_argument("--output-dir", help="Batch run directory; each case gets its own MP4/JSON/step log")
    parser.add_argument("--resume", action="store_true", help="Resume an identical batch plan, skipping completed cases")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.checkpoint = resolve_checkpoint(args.checkpoint)
    if args.check_only:
        check_checkpoint(args)
    elif args.suite != "single":
        from inference.cases import run_batch
        run_batch(args)
    else:
        if args.prompt is None or args.output is None or (args.mode == "i2v" and args.image is None):
            parser.error("generation requires --prompt and --output; i2v also requires --image")
        generate(args)


if __name__ == "__main__":
    main()
