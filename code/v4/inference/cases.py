"""Fixed reference/demo conditions, batch provenance and resumable single-GPU jobs."""
import copy
import hashlib
import json
from pathlib import Path
import re
import time

from PIL import Image


DEMO_IDS = tuple(f"P{n:02d}" for n in range(1, 21))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def collect_cases(args):
    suite = getattr(args, "suite", "single")
    cases = []
    if suite in ("single", "both"):
        source = getattr(args, "cases", None)
        if source:
            if suite == "single":
                raise ValueError("--cases requires --suite both")
            source = Path(source).expanduser().resolve()
            entries = json.loads(source.read_text())
            if not isinstance(entries, list) or not entries:
                raise ValueError("--cases must contain a nonempty JSON list of image/prompt conditions")
            for entry in entries:
                entry = dict(entry)
                if entry.get("image"):
                    image = Path(entry["image"]).expanduser()
                    entry["image"] = str((source.parent / image).resolve())
                entry.update(group="reference", source_manifest=str(source))
                cases.append(entry)
        else:
            cases.append(dict(case_id="reference", group="reference", image=args.image, prompt=args.prompt,
                              mode=getattr(args, "mode", "i2v")))
    if suite in ("demo", "both"):
        root = Path(args.demo_root).expanduser().resolve()
        ids = list(DEMO_IDS) if args.sample_ids == "all" else args.sample_ids.split(",")
        if not ids or len(set(ids)) != len(ids) or any(cid not in DEMO_IDS for cid in ids):
            raise ValueError("sample-ids must be all or distinct comma-separated IDs from P01 to P20")
        for cid in ids:
            folder = root / cid
            origin = folder / f"{cid}-origin.txt"
            raw = origin.read_bytes()
            selection = getattr(args, "demo_modes", "both")
            modes = ("i2v", "t2v") if selection == "both" else (selection,)
            for mode in modes:
                image = None
                if mode == "i2v":
                    # Same preferred first frames as v1 I2V; no archive or image lookup for T2V.
                    images = [folder / f"{cid}-{suffix}" for suffix in
                              ("i0-1280x704.png", "i0-1280x704.jpg", "i0.png", "i0.jpg")]
                    image = next((path for path in images if path.is_file()), None)
                    if image is None:
                        raise FileNotFoundError(f"No first-frame image for {cid} in {folder}")
                cases.append(dict(case_id=cid, group="demo20", mode=mode,
                                  image=str(image) if image is not None else None,
                                  prompt=raw.decode("utf-8-sig").strip(), origin_txt=str(origin),
                                  origin_sha256=digest(raw)))
    seen = set()
    for case in cases:
        cid = case.get("case_id")
        if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", cid):
            raise ValueError(f"Invalid case_id: {cid!r}")
        mode = case.setdefault("mode", getattr(args, "mode", "i2v"))
        if mode not in ("i2v", "t2v"):
            raise ValueError(f"Invalid generation mode for {cid}: {mode}")
        key = (case["group"], cid, mode)
        if key in seen:
            raise ValueError(f"Duplicate case: {key}")
        seen.add(key)
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
            raise ValueError(f"Empty prompt for {cid}")
        if mode == "t2v":
            case.update(image=None, source_size=None, image_sha256=None)
            continue
        if not case.get("image"):
            raise ValueError(f"Missing image for {cid}")
        image = Path(case["image"]).expanduser().resolve()
        with Image.open(image) as source:
            case["source_size"] = list(source.size)
            source.verify()
        case["image"] = str(image)
        case["image_sha256"] = digest(image.read_bytes())
    return cases


def case_args(args, case):
    request = copy.copy(args)
    request.image, request.prompt = case["image"], case["prompt"]
    for key in ("frames", "fps", "duration", "negative_prompt", "mode", "width", "height"):
        if key in case:
            setattr(request, key, case[key])
    request.case = case
    request.suite = "single"
    request.cases = None
    return request


def describe_cases(cases, config, args):
    from inference.infer_native_p import validate_sampling, video_spec, text_only_canvas
    from physgen_v4.views import choose_canvas
    result = []
    for case in cases:
        request = case_args(args, case)
        validate_sampling(request)
        spec = video_spec(config, request.frames, request.fps, request.duration)
        if request.mode == "t2v":
            height, width = text_only_canvas(config, request)
            if config.get("corrector", {}).get("initialization", "image_text") not in ("image_text", "image_text_gt"):
                raise ValueError("T2V requires a checkpoint with an image_text or image_text_gt condition prior")
        else:
            width, height = case["source_size"]
            height, width = choose_canvas(height, width, config["data"]["max_long_side"], config["data"]["max_area"])
        result.append(dict(case, video=spec, height=height, width=width))
    return result


def batch_plan(args, cases, config):
    from inference.infer_native_p import state_policy
    if not args.output_dir:
        raise ValueError("Batch generation requires --output-dir")
    root = Path(args.output_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    weights = (checkpoint / "corrector.pt").stat()
    common = dict(checkpoint=str(checkpoint), config_sha256=digest((checkpoint / "config.yaml").read_bytes()),
                  weights_bytes=weights.st_size, weights_mtime_ns=weights.st_mtime_ns,
                  seed=args.seed, steps=args.steps, shift=args.shift, guidance=args.guidance,
                  solver=args.solver, variant=args.variant, writer_off=args.writer_off,
                  state_policy=state_policy(config, args))
    requests, records = [], []
    for case in describe_cases(cases, config, args):
        request = case_args(args, case)
        request.output = str(root / case["group"] / case["case_id"] / case["mode"] / f"seed{args.seed}_{args.variant}.mp4")
        signature = digest(json.dumps(dict(common=common, case=case, negative_prompt=request.negative_prompt),
                                      sort_keys=True, ensure_ascii=False).encode())
        request.case = dict(case, signature=signature)
        requests.append(request)
        records.append(dict(request.case, output=request.output))
    return dict(schema_version=2, settings=common, cases=records), requests


def completed_case(request):
    destination = Path(request.output)
    if not all(path.is_file() and path.stat().st_size > 0 for path in
               (destination, destination.with_suffix(".json"), destination.with_suffix(".steps.jsonl"))):
        return False
    try:
        saved = json.loads(destination.with_suffix(".json").read_text())
        return (saved["status"] == "completed" and saved["case"]["signature"] == request.case["signature"]
                and saved["scheduler_updates"] == request.steps and saved["wan_forwards"] == request.steps * 2)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_batch(args):
    from inference.infer_native_p import InferenceRuntime, generate
    from physgen_v4.runtime import read_config
    started = time.perf_counter()
    config = read_config(Path(args.checkpoint).expanduser() / "config.yaml")
    cases = collect_cases(args)
    plan, requests = batch_plan(args, cases, config)
    root = Path(args.output_dir).expanduser().resolve()
    plan_path = root / "cases.json"
    if plan_path.exists():
        if not args.resume:
            raise FileExistsError(f"Batch already exists: {root}; use --resume for the identical plan")
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Cannot resume: checkpoint, inputs or sampling settings differ from cases.json")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Nonempty output directory has no matching batch plan: {root}")
    root.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, plan)
    statuses = [dict(case_id=r.case["case_id"], group=r.case["group"], mode=r.mode, output=r.output,
                     status="completed" if args.resume and completed_case(r) else "pending") for r in requests]
    summary = dict(status="running", total=len(requests), cases=statuses, plan=str(plan_path))

    def save_summary():
        summary["completed"] = sum(case["status"] == "completed" for case in statuses)
        summary["wall_seconds"] = time.perf_counter() - started
        write_json(root / "summary.json", summary)

    pending = [(request, status) for request, status in zip(requests, statuses) if status["status"] != "completed"]
    save_summary()
    print(json.dumps(dict(event="batch_start", total=len(requests), pending=len(pending), output=str(root))), flush=True)
    active = None
    try:
        if pending:
            with InferenceRuntime(args) as runtime:
                runtime.prepare_texts([request for request, _ in pending])
                for request, active in pending:
                    active["status"] = "running"
                    save_summary()
                    # Invalidate any stale completion marker before replacing partial outputs.
                    destination = Path(request.output)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    write_json(destination.with_suffix(".json"), dict(status="running", case=request.case))
                    result = generate(request, runtime=runtime)
                    active.update(status="completed", wall_seconds=result["wall_seconds"])
                    save_summary()
                    print(json.dumps(dict(event="case_completed", case_id=active["case_id"],
                                          mode=active["mode"],
                                          completed=summary["completed"], total=len(requests))), flush=True)
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        if active is not None and active["status"] == "running":
            active.update(status="failed", error=summary["error"])
        raise
    finally:
        save_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
