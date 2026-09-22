#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
CODE_ROOT = Path("/home/liuzhirui/Project/physGen/code")
V3_ROOT = Path("/home/liuzhirui/Project/physGen/code/v3")
for root in (CODE_ROOT, V3_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from v2.ace_router.demo_inputs import DemoInputError, DemoTraceInput, parse_sample_ids, resolve_demo_inputs
from v2.ace_router.trace_schema import TracePlan
from v2.ace_router.upstream_guard import assert_upstream_unchanged, capture_upstream, verify_expected_upstream

from trace_writer_v3.compile import CompiledTraceV3, compile_trace_plan_v3
from trace_writer_v3.config import TraceWriterConfigV3
from trace_writer_v3.control_schema import merge_overlay
from trace_writer_v3.controllers import AuditReader
from trace_writer_v3.dynamic_support import build_dynamic_support
from trace_writer_v3.verifier import verify_compiled_contract, verify_video_tensor


@dataclass(frozen=True)
class PreparedSample:
    source: DemoTraceInput
    raw_plan: Mapping[str, Any]
    compiled: CompiledTraceV3
    plan_sha256: str
    image_sha256: str | None
    image_size: tuple[int, int] | None
    overlay_path: Path | None
    support_manifest: Mapping[str, Any]
    context_token_lengths: Mapping[str, int]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="TRACE-Writer v3 fixed-five M3/M4 inference")
    value.add_argument("--mode", choices=("i2v", "t2v"), required=True)
    value.add_argument("--wan_repo", default="/home/liuzhirui/model/Wan2.2")
    value.add_argument("--checkpoint_dir", default="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B")
    value.add_argument("--demo_root", default="/home/liuzhirui/Project/physGen/code/v1/demo")
    value.add_argument("--plan_root", default="/home/liuzhirui/Project/physGen/code/v1/demo")
    value.add_argument("--control_overlay_root", default="/home/liuzhirui/Project/physGen/code/v3/control_overlays")
    value.add_argument("--output_root", default="/home/liuzhirui/Project/physGen/code/v3/outputs/trace_writer")
    value.add_argument("--run_id", default=None)
    value.add_argument("--sample_ids", default="all")
    value.add_argument("--seeds", default="42")
    value.add_argument("--minimal_pair_mode", choices=("compat", "strict"), default="strict")
    value.add_argument("--crossfade_tokens", type=int, choices=(1, 2), default=2)
    value.add_argument("--event_clock_mode", choices=("fixed", "plan_prior", "reader_hysteresis"), default="plan_prior")
    value.add_argument("--boundary_morph", choices=("linear", "smoothstep", "cosine_trust"), default="cosine_trust")
    value.add_argument("--dynamic_support_mode", choices=("off", "planned", "planned_saliency"), default=None)
    value.add_argument("--cap_mode", choices=("legacy", "aggregate_estimated", "aggregate_strict"), default="aggregate_strict")
    value.add_argument("--reader_mode", choices=("off", "audit"), default="audit")
    value.add_argument("--verify_mode", choices=("off", "audit"), default="audit")
    value.add_argument("--width", type=int, default=1280)
    value.add_argument("--height", type=int, default=704)
    value.add_argument("--max_area", type=int, default=901120)
    value.add_argument("--frame_num", type=int, default=97)
    value.add_argument("--sampling_steps", type=int, default=50)
    value.add_argument("--sample_solver", choices=("unipc", "dpm++"), default="unipc")
    value.add_argument("--guide_scale", type=float, default=3.5)
    value.add_argument("--shift", type=float, default=5.0)
    value.add_argument("--fps", type=int, default=24)
    value.add_argument("--lambda0", type=float, default=0.05)
    value.add_argument("--token_cap_ratio", type=float, default=0.05)
    value.add_argument("--layer_cap_ratio", type=float, default=0.01)
    value.add_argument("--group_cap_ratio", type=float, default=0.01)
    value.add_argument("--cfg_cap_ratio", type=float, default=0.01)
    value.add_argument("--cfg_outside_cap_ratio", type=float, default=0.0025)
    value.add_argument("--temporal_derivative_ratio", type=float, default=0.05)
    value.add_argument("--device_id", type=int, default=0)
    value.add_argument("--offload_model", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--t5_cpu", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--convert_model_dtype", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--show_progress", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--allow_upstream_drift", action="store_true")
    value.add_argument("--parse_only", action="store_true")
    value.add_argument("--check_only", action="store_true")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise DemoInputError("seeds must be comma-separated integers") from exc
    if not result or any(x < 0 for x in result) or len(set(result)) != len(result):
        raise DemoInputError("seeds must be unique non-negative integers")
    return result


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _overlay_path(root: Path, sample_id: str, mode: str) -> Path | None:
    method = "m3" if mode == "i2v" else "m4"
    candidates = (
        root / sample_id / f"{sample_id}-{method}-control.json",
        root / f"{sample_id}-{method}-control.json",
    )
    matches = [item.resolve() for item in candidates if item.is_file()]
    if len(matches) > 1:
        raise DemoInputError(f"{sample_id}: ambiguous control overlays: {matches}")
    return matches[0] if matches else None


def _check_variant(raw: Mapping[str, Any], sample_id: str, mode: str) -> None:
    basis = raw.get("generation_basis")
    expected = "visual_grounding" if mode == "i2v" else "identifier_only"
    if not isinstance(basis, Mapping) or basis.get("mode") != expected:
        raise DemoInputError(f"{sample_id}: expected generation_basis.mode={expected!r}")


def _prepare(
    sources: tuple[DemoTraceInput, ...],
    *,
    mode: str,
    overlay_root: Path,
    config: TraceWriterConfigV3,
    minimal_pair_mode: str,
    tokenizer: Any,
) -> tuple[PreparedSample, ...]:
    from PIL import Image

    prepared: list[PreparedSample] = []
    errors: list[str] = []
    for source in sources:
        try:
            raw = json.loads(source.plan_path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise DemoInputError("plan root must be an object")
            _check_variant(raw, source.sample_id, mode)
            overlay_path = _overlay_path(overlay_root, source.sample_id, mode)
            overlay = json.loads(overlay_path.read_text(encoding="utf-8")) if overlay_path else None
            merged = merge_overlay(raw, overlay)
            plan = TracePlan.from_dict(merged)
            compiled = compile_trace_plan_v3(
                plan,
                merged,
                mode=mode,
                crossfade_tokens=config.crossfade_tokens,
                event_clock_mode=config.event_clock_mode,
                minimal_pair_mode=minimal_pair_mode,
            )
            context_token_lengths = {
                name: len(
                    tokenizer(
                        text,
                        add_special_tokens=True,
                        padding=False,
                        truncation=False,
                    )["input_ids"]
                )
                for name, text in compiled.context_texts().items()
            }
            over_budget = {
                name: length for name, length in context_token_lengths.items() if length > 512
            }
            if over_budget:
                raise DemoInputError(f"T5 context exceeds 512 tokens: {over_budget}")
            image_size = None
            image_hash = None
            if source.image_path is not None:
                with Image.open(source.image_path) as image:
                    image.verify()
                with Image.open(source.image_path) as image:
                    image_size = (int(image.width), int(image.height))
                if image_size != (1280, 704):
                    raise DemoInputError(f"{source.sample_id}: first frame is {image_size}, expected 1280x704")
                image_hash = _sha256(source.image_path)
            _, support_manifest = build_dynamic_support(
                compiled,
                height=22,
                width=40,
                support_mode=config.dynamic_support_mode,
                epsilon=config.mask_epsilon,
                device="cpu",
            )
            prepared.append(
                PreparedSample(
                    source=source,
                    raw_plan=merged,
                    compiled=compiled,
                    plan_sha256=_sha256(source.plan_path),
                    image_sha256=image_hash,
                    image_size=image_size,
                    overlay_path=overlay_path,
                    support_manifest=support_manifest,
                    context_token_lengths=context_token_lengths,
                )
            )
        except Exception as exc:
            errors.append(f"{source.sample_id}: {type(exc).__name__}: {exc}")
    if errors:
        raise DemoInputError("v3 preflight failed:\n- " + "\n- ".join(errors))
    return tuple(prepared)


def _config(args: argparse.Namespace) -> TraceWriterConfigV3:
    support = args.dynamic_support_mode or ("planned" if args.mode == "i2v" else "planned_saliency")
    return TraceWriterConfigV3(
        crossfade_tokens=args.crossfade_tokens,
        lambda0=args.lambda0,
        token_cap_ratio=args.token_cap_ratio,
        layer_cap_ratio=args.layer_cap_ratio,
        group_cap_ratio=args.group_cap_ratio,
        cfg_cap_ratio=args.cfg_cap_ratio,
        cfg_outside_cap_ratio=args.cfg_outside_cap_ratio,
        temporal_derivative_ratio=args.temporal_derivative_ratio,
        cap_mode=args.cap_mode,
        boundary_morph=args.boundary_morph,
        event_clock_mode=args.event_clock_mode,
        dynamic_support_mode=support,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if (args.width, args.height, args.max_area, args.frame_num) != (1280, 704, 901120, 97):
        raise DemoInputError("v3 HD preset requires 1280x704, max_area=901120, frame_num=97")
    if args.sampling_steps <= 0 or args.fps <= 0 or args.device_id < 0:
        raise DemoInputError("steps/fps must be positive and device_id non-negative")
    if not math.isfinite(args.guide_scale) or args.guide_scale < 0:
        raise DemoInputError("guide_scale must be finite and non-negative")
    if args.mode == "t2v" and args.dynamic_support_mode == "planned":
        raise DemoInputError("M4 without a frame must use planned_saliency or off")
    if args.event_clock_mode == "reader_hysteresis":
        raise DemoInputError("reader_hysteresis is implemented as an experimental controller but not enabled before calibration")


def _static_artifacts(path: Path, sample: PreparedSample) -> None:
    manifest = sample.compiled.manifest()
    _write_json(path / "trace.plan.input.json", sample.raw_plan)
    _write_json(path / "trace.control.normalized.json", sample.compiled.entity_ledger.manifest())
    context_manifest = sample.compiled.contexts.manifest()
    context_manifest["actual_t5_token_lengths"] = dict(sample.context_token_lengths)
    context_manifest["max_t5_token_length"] = max(sample.context_token_lengths.values())
    _write_json(path / "trace.contexts.compiled.json", context_manifest)
    _write_json(
        path / "trace.constraints.compiled.json",
        {
            "closed_world": asdict(sample.compiled.entity_ledger.closed_world),
            "lifecycles": [asdict(x) for x in sample.compiled.entity_ledger.lifecycles],
            "conservation": [asdict(x) for x in sample.compiled.entity_ledger.conservation],
        },
    )
    _write_json(path / "trace.route.compiled.json", manifest)
    _write_json(path / "trace.support.compiled.json", sample.support_manifest)
    _write_json(path / "trace.verify.compiled.json", verify_compiled_contract(sample.compiled))


def main() -> int:
    args = parser().parse_args()
    if args.parse_only:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    try:
        _validate_args(args)
        config = _config(args)
        sample_ids = parse_sample_ids(args.sample_ids)
        seeds = _seeds(args.seeds)
        demo_root = Path(args.demo_root).expanduser().resolve()
        plan_root = Path(args.plan_root).expanduser().resolve()
        overlay_root = Path(args.control_overlay_root).expanduser().resolve()
        sources = resolve_demo_inputs(
            demo_root=demo_root,
            plan_root=plan_root,
            sample_ids=sample_ids,
            mode=args.mode,
        )
        from transformers import AutoTokenizer

        tokenizer_path = Path(args.checkpoint_dir).expanduser().resolve() / "google" / "umt5-xxl"
        if not tokenizer_path.is_dir():
            raise DemoInputError(f"missing local UMT5 tokenizer: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        prepared = _prepare(
            sources,
            mode=args.mode,
            overlay_root=overlay_root,
            config=config,
            minimal_pair_mode=args.minimal_pair_mode,
            tokenizer=tokenizer,
        )
        upstream_before = capture_upstream(args.wan_repo, args.checkpoint_dir)
        upstream_mismatches = verify_expected_upstream(
            upstream_before, allow_drift=args.allow_upstream_drift
        )
    except (DemoInputError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[TRACE V3 PREFLIGHT][ERROR] {exc}", file=sys.stderr)
        return 2
    preflight = {
        "status": "ready",
        "method": "TRACE-V3-M3" if args.mode == "i2v" else "TRACE-V3-M4",
        "sample_ids": list(sample_ids),
        "sample_count": len(prepared),
        "mode": args.mode,
        "model_uses_first_frame": args.mode == "i2v",
        "config": config.manifest(),
        "upstream": upstream_before,
        "upstream_mismatches": upstream_mismatches,
        "samples": [
            {
                "sample_id": item.source.sample_id,
                "plan": str(item.source.plan_path),
                "image": str(item.source.image_path) if item.source.image_path else None,
                "overlay": str(item.overlay_path) if item.overlay_path else None,
                "issues": list(item.compiled.issues),
                "pair_quality": [x.pair_quality for x in item.compiled.contexts.stage_pairs],
                "context_token_lengths": dict(item.context_token_lengths),
                "max_context_token_length": max(item.context_token_lengths.values()),
                "support": item.support_manifest,
            }
            for item in prepared
        ],
    }
    if args.check_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    wan_repo = str(Path(args.wan_repo).expanduser().resolve())
    if wan_repo not in sys.path:
        sys.path.insert(0, wan_repo)
    import torch
    from PIL import Image
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import save_video

    from trace_writer_v3.wan_pipeline import TraceWanTI2VV3

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for generation")
    pipeline = TraceWanTI2VV3(
        config=WAN_CONFIGS["ti2v-5B"],
        checkpoint_dir=str(Path(args.checkpoint_dir).expanduser().resolve()),
        device_id=args.device_id,
        rank=0,
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
        convert_model_dtype=args.convert_model_dtype,
    )
    method = "m3" if args.mode == "i2v" else "m4"
    run_id = args.run_id or datetime.now(timezone.utc).strftime(f"trace-v3-{method}-%Y%m%d-%H%M%S")
    run_root = Path(args.output_root).expanduser().resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    _write_json(run_root / "preflight.json", preflight)
    results: list[dict[str, Any]] = []
    for sample in prepared:
        for seed in seeds:
            artifact = run_root / sample.source.sample_id / args.mode / f"seed_{seed:06d}"
            artifact.mkdir(parents=True, exist_ok=False)
            _static_artifacts(artifact, sample)
            image = None
            reader = AuditReader() if args.reader_mode == "audit" else None
            temporary = artifact / "video.incomplete.mp4"
            video_path = artifact / "video.mp4"
            try:
                if sample.source.image_path is not None:
                    with Image.open(sample.source.image_path) as opened:
                        image = opened.convert("RGB")
                output = pipeline.generate_trace_v3(
                    compiled=sample.compiled,
                    image=image,
                    size=(args.width, args.height),
                    max_area=args.max_area,
                    frame_num=args.frame_num,
                    sampling_steps=args.sampling_steps,
                    sample_solver=args.sample_solver,
                    guide_scale=args.guide_scale,
                    shift=args.shift,
                    seed=seed,
                    writer_config=config,
                    reader=reader,
                    offload_model=args.offload_model,
                    show_progress=args.show_progress,
                )
                if int(output.video.shape[1]) != args.frame_num:
                    raise RuntimeError(f"decoded {output.video.shape[1]} frames, expected {args.frame_num}")
                save_video(
                    tensor=output.video[None],
                    save_file=str(temporary),
                    fps=args.fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise RuntimeError("video writer produced no output")
                os.replace(temporary, video_path)
                _write_json(artifact / "trace.support.runtime.json", output.support_manifest)
                _write_jsonl(artifact / "trace.cap.audit.jsonl", list(output.cfg_audit))
                if reader is not None:
                    _write_jsonl(artifact / "reader.audit.jsonl", reader.records + reader.cfg_records)
                if args.verify_mode == "audit":
                    _write_json(artifact / "trace.verify.json", verify_video_tensor(output.video, sample.compiled))
                manifest = {
                    "status": "success",
                    "run_id": run_id,
                    "method": preflight["method"],
                    "sample_id": sample.source.sample_id,
                    "mode": args.mode,
                    "seed": seed,
                    "plan": str(sample.source.plan_path),
                    "plan_sha256": sample.plan_sha256,
                    "image": str(sample.source.image_path) if sample.source.image_path else None,
                    "image_sha256": sample.image_sha256,
                    "image_is_model_condition": args.mode == "i2v",
                    "video": str(video_path),
                    "video_sha256": _sha256(video_path),
                    "size": [output.width, output.height],
                    "frame_num": args.frame_num,
                    "context_token_lengths": output.context_token_lengths,
                    "config": config.manifest(),
                    "compiled_issues": list(sample.compiled.issues),
                }
                _write_json(artifact / "manifest.json", manifest)
                results.append(manifest)
                del output
            except Exception as exc:
                for path in (temporary, video_path):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                failure = {
                    "status": "failed",
                    "sample_id": sample.source.sample_id,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                _write_json(artifact / "failure.json", failure)
                results.append(failure)
                if not args.continue_on_error:
                    raise
            finally:
                if image is not None:
                    image.close()
                torch.cuda.empty_cache()
    upstream_after = capture_upstream(args.wan_repo, args.checkpoint_dir)
    assert_upstream_unchanged(upstream_before, upstream_after)
    summary = {
        "run_id": run_id,
        "method": preflight["method"],
        "success_count": sum(x["status"] == "success" for x in results),
        "failure_count": sum(x["status"] == "failed" for x in results),
        "results": results,
        "upstream_unchanged": True,
    }
    _write_json(run_root / "results.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
