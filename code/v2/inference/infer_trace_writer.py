#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
INFERENCE_ROOT = Path(__file__).resolve().parent
V2_ROOT = INFERENCE_ROOT.parent
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from ace_router.controllers import AuditReader
from ace_router.demo_inputs import (
    DemoInputError,
    DemoTraceInput,
    parse_sample_ids,
    resolve_demo_inputs,
)
from ace_router.trace_compile import CompiledTrace, compile_trace_plan
from ace_router.trace_config import TraceWriterConfig
from ace_router.trace_masks import build_stage_spatial_masks
from ace_router.trace_schema import TracePlan, load_trace_plan
from ace_router.upstream_guard import (
    assert_upstream_unchanged,
    capture_upstream,
    verify_expected_upstream,
)


@dataclass(frozen=True)
class PreparedSample:
    source: DemoTraceInput
    plan: TracePlan
    compiled: CompiledTrace
    plan_sha256: str
    image_sha256: str | None
    image_size: tuple[int, int] | None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "TRACE-Writer fixed-five inference. M3/i2v reads *-V2-planimg.json "
            "and a *1280x704 first frame; M4/t2v reads *-V2-plan.json and never "
            "loads an image."
        )
    )
    parser.add_argument("--mode", choices=("i2v", "t2v"), required=True)
    parser.add_argument("--wan_repo", default="/home/liuzhirui/model/Wan2.2")
    parser.add_argument(
        "--checkpoint_dir",
        default="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B",
    )
    parser.add_argument("--demo_root", default=str(V2_ROOT.parent / "v1" / "demo"))
    parser.add_argument(
        "--plan_root",
        default=None,
        help="Plan tree; defaults to demo_root so PXX/PXX-V2-plan[img].json is used.",
    )
    parser.add_argument("--output_root", default=str(V2_ROOT / "outputs" / "trace_writer"))
    parser.add_argument("--run_id", default=None)
    parser.add_argument(
        "--sample_ids",
        default="all",
        help="Comma-separated Pxx ids or 'all' for exactly P01..P20.",
    )
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--crossfade_tokens", type=int, choices=(1, 2), default=1)
    parser.add_argument("--spatial_mode", choices=("time", "spacetime"), default="time")
    parser.add_argument("--reader_mode", choices=("off", "audit"), default="off")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--max_area", type=int, default=1280 * 704)
    parser.add_argument("--frame_num", type=int, default=97)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--sample_solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--guide_scale", type=float, default=3.5)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--lambda0", type=float, default=0.10)
    parser.add_argument("--token_cap_ratio", type=float, default=0.10)
    parser.add_argument("--global_cap_ratio", type=float, default=0.02)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--offload_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--t5_cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--convert_model_dtype", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--show_progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--allow_upstream_drift", action="store_true")
    parser.add_argument("--parse_only", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    return parser


def _seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise DemoInputError("seeds must be comma-separated integers") from exc
    if not result or any(seed < 0 for seed in result):
        raise DemoInputError("seeds must be non-negative integers")
    if len(set(result)) != len(result):
        raise DemoInputError("seeds must not contain duplicates")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validation_record(compiled: CompiledTrace) -> dict[str, Any]:
    return {
        "error_count": compiled.validation.error_count,
        "warning_count": compiled.validation.warning_count,
        "fallback_selected": False,
        "issues": [
            {
                "severity": issue.severity,
                "code": issue.code,
                "path": issue.path,
                "message": issue.message,
            }
            for issue in compiled.validation.issues
        ],
    }


def _check_plan_variant(source: DemoTraceInput, mode: str) -> None:
    raw = json.loads(source.plan_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise DemoInputError(f"{source.sample_id}: plan root must be an object")
    basis = raw.get("generation_basis")
    if not isinstance(basis, dict):
        raise DemoInputError(
            f"{source.sample_id}: selected V2 plan has no generation_basis object"
        )
    expected_basis = "visual_grounding" if mode == "i2v" else "identifier_only"
    actual_basis = basis.get("mode")
    if actual_basis != expected_basis:
        raise DemoInputError(
            f"{source.sample_id}: {source.plan_path.name} generation_basis.mode="
            f"{actual_basis!r}, expected {expected_basis!r} for {mode}"
        )


def _prepare_samples(
    sources: tuple[DemoTraceInput, ...],
    *,
    mode: str,
    spatial_mode: str,
    crossfade_tokens: int,
) -> tuple[PreparedSample, ...]:
    from PIL import Image

    prepared: list[PreparedSample] = []
    errors: list[str] = []
    for source in sources:
        try:
            _check_plan_variant(source, mode)
            plan = load_trace_plan(source.plan_path)
            if plan.sample_id != source.sample_id:
                raise DemoInputError(
                    f"plan sample_id {plan.sample_id!r} != requested {source.sample_id!r}"
                )
            compiled = compile_trace_plan(
                plan,
                crossfade_tokens=crossfade_tokens,
                emit_validation=False,
            )
            image_size = None
            image_hash = None
            if source.image_path is not None:
                with Image.open(source.image_path) as image:
                    image.verify()
                with Image.open(source.image_path) as image:
                    image_size = (int(image.width), int(image.height))
                if image_size != (1280, 704):
                    raise DemoInputError(
                        f"first frame must be 1280x704, got {image_size} at {source.image_path}"
                    )
                image_hash = _sha256(source.image_path)
            if spatial_mode == "spacetime":
                if mode != "i2v":
                    raise DemoInputError(
                        "M4/t2v cannot use image-grounded spacetime routing; use spatial_mode=time"
                    )
                # HD97 has a fixed 25x22x40 Wan token grid. This catches missing
                # boxes before any model weights are loaded.
                build_stage_spatial_masks(plan, height=22, width=40, device="cpu")
            prepared.append(
                PreparedSample(
                    source=source,
                    plan=plan,
                    compiled=compiled,
                    plan_sha256=_sha256(source.plan_path),
                    image_sha256=image_hash,
                    image_size=image_size,
                )
            )
        except Exception as exc:
            errors.append(f"{source.sample_id}: {type(exc).__name__}: {exc}")
    if errors:
        raise DemoInputError(
            f"TRACE plan preflight failed for {len(errors)} requested sample(s):\n- "
            + "\n- ".join(errors)
        )
    return tuple(prepared)


def _preflight_record(
    sample: PreparedSample,
    *,
    mode: str,
    spatial_mode: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample.source.sample_id,
        "mode": mode,
        "plan_variant": "planimg" if mode == "i2v" else "plan",
        "plan": str(sample.source.plan_path),
        "plan_sha256": sample.plan_sha256,
        "image": str(sample.source.image_path) if sample.source.image_path else None,
        "image_sha256": sample.image_sha256,
        "image_size": list(sample.image_size) if sample.image_size else None,
        "image_is_model_condition": mode == "i2v",
        "spatial_mode": spatial_mode,
        "validation": _validation_record(sample.compiled),
    }


def _write_static_artifacts(
    artifact_dir: Path,
    sample: PreparedSample,
    *,
    mode: str,
    spatial_mode: str,
) -> None:
    raw_plan = json.loads(sample.source.plan_path.read_text(encoding="utf-8"))
    _write_json(artifact_dir / "trace.plan.input.json", raw_plan)
    _write_json(artifact_dir / "trace.plan.validation.json", _validation_record(sample.compiled))
    route = sample.compiled.manifest()
    route.update(
        {
            "input_plan": str(sample.source.plan_path),
            "input_plan_sha256": sample.plan_sha256,
            "requested_mode": mode,
            "spatial_mode": spatial_mode,
            "flatten_order": "f-y-x",
            "fallback_selected": False,
        }
    )
    if spatial_mode == "spacetime":
        spatial_masks = build_stage_spatial_masks(
            sample.plan,
            height=22,
            width=40,
            device="cpu",
        )
    else:
        spatial_masks = sample.compiled.temporal_weights.new_ones(
            len(sample.compiled.stages), 22, 40
        )
    route["spatial_masks"] = [
        {
            "stage_id": stage.stage_id,
            "positive_token_count": int((mask > 0).sum().item()),
            "soft_area": round(float(mask.sum().item()), 6),
            "grid_area": int(mask.numel()),
        }
        for stage, mask in zip(sample.compiled.stages, spatial_masks)
    ]
    route["active_stage_summary"] = [
        {
            "stage_id": stage.stage_id,
            "temporal_token_count": int((weights > 0).sum().item()),
            "temporal_weight_sum": round(float(weights.sum().item()), 6),
        }
        for stage, weights in zip(
            sample.compiled.stages,
            sample.compiled.temporal_weights,
        )
    ]
    _write_json(artifact_dir / "trace.route.compiled.json", route)


def _validate_args(args: argparse.Namespace) -> None:
    if args.frame_num != 97:
        raise DemoInputError("TRACE fixed5-causal-r3 requires frame_num=97")
    if args.width <= 0 or args.height <= 0 or args.max_area <= 0:
        raise DemoInputError("width, height, and max_area must be positive")
    if (args.width, args.height, args.max_area) != (1280, 704, 1280 * 704):
        raise DemoInputError(
            "TRACE fixed5-causal-r3 is pinned to 1280x704 / max_area=901120"
        )
    if args.mode == "t2v" and args.width * args.height != args.max_area:
        raise DemoInputError("M4/t2v requires width * height == max_area")
    if args.mode == "t2v" and args.spatial_mode != "time":
        raise DemoInputError("M4/t2v must use spatial_mode=time because no image is used")
    if args.sampling_steps <= 0 or args.fps <= 0:
        raise DemoInputError("sampling_steps and fps must be positive")
    if not math.isfinite(args.guide_scale) or args.guide_scale < 0.0:
        raise DemoInputError("guide_scale must be finite and non-negative")
    if not math.isfinite(args.shift) or args.shift <= 0.0:
        raise DemoInputError("shift must be finite and positive")
    if args.device_id < 0:
        raise DemoInputError("device_id must be non-negative")


def main() -> int:
    args = _parser().parse_args()
    if args.parse_only:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    try:
        _validate_args(args)
        writer_config = TraceWriterConfig(
            crossfade_tokens=args.crossfade_tokens,
            lambda0=args.lambda0,
            token_cap_ratio=args.token_cap_ratio,
            global_cap_ratio=args.global_cap_ratio,
        )
        sample_ids = parse_sample_ids(args.sample_ids)
        seeds = _seeds(args.seeds)
        demo_root = Path(args.demo_root).expanduser().resolve()
        plan_root = (
            Path(args.plan_root).expanduser().resolve()
            if args.plan_root
            else demo_root
        )
        sources = resolve_demo_inputs(
            demo_root=demo_root,
            plan_root=plan_root,
            sample_ids=sample_ids,
            mode=args.mode,
        )
        prepared = _prepare_samples(
            sources,
            mode=args.mode,
            spatial_mode=args.spatial_mode,
            crossfade_tokens=args.crossfade_tokens,
        )
        upstream_before = capture_upstream(args.wan_repo, args.checkpoint_dir)
        upstream_mismatches = verify_expected_upstream(
            upstream_before,
            allow_drift=args.allow_upstream_drift,
        )
    except (
        DemoInputError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[TRACE PREFLIGHT][ERROR] {exc}", file=sys.stderr)
        print(
            "[TRACE PREFLIGHT] stopped before Wan was loaded; "
            "no fallback was selected",
            file=sys.stderr,
        )
        return 2

    preflight = {
        "status": "ready",
        "method": "TRACE-M3" if args.mode == "i2v" else "TRACE-M4",
        "requested_sample_count": len(sample_ids),
        "sample_ids": list(sample_ids),
        "mode": args.mode,
        "plan_variant": "planimg" if args.mode == "i2v" else "plan",
        "model_uses_first_frame": args.mode == "i2v",
        "upstream": upstream_before,
        "upstream_mismatches": upstream_mismatches,
        "samples": [
            _preflight_record(sample, mode=args.mode, spatial_mode=args.spatial_mode)
            for sample in prepared
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

    from ace_router.wan_pipeline import TraceWanTI2V

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    pipeline = TraceWanTI2V(
        config=WAN_CONFIGS["ti2v-5B"],
        checkpoint_dir=str(Path(args.checkpoint_dir).expanduser().resolve()),
        device_id=args.device_id,
        rank=0,
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
        convert_model_dtype=args.convert_model_dtype,
    )
    method = "m3" if args.mode == "i2v" else "m4"
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        f"trace-{method}-%Y%m%d-%H%M%S"
    )
    run_root = Path(args.output_root).expanduser().resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    _write_json(run_root / "preflight.json", preflight)

    results: list[dict[str, Any]] = []
    for sample in prepared:
        sample_id = sample.source.sample_id
        for seed in seeds:
            artifact_dir = run_root / sample_id / args.mode / f"seed_{seed:06d}"
            artifact_dir.mkdir(parents=True, exist_ok=False)
            _write_static_artifacts(
                artifact_dir,
                sample,
                mode=args.mode,
                spatial_mode=args.spatial_mode,
            )
            image = None
            reader = AuditReader() if args.reader_mode == "audit" else None
            temporary_video = artifact_dir / "video.incomplete.mp4"
            video_path = artifact_dir / "video.mp4"
            try:
                if sample.source.image_path is not None:
                    with Image.open(sample.source.image_path) as opened:
                        image = opened.convert("RGB")
                output = pipeline.generate_trace(
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
                    spatial_mode=args.spatial_mode,
                    writer_config=writer_config,
                    reader=reader,
                    offload_model=args.offload_model,
                    show_progress=args.show_progress,
                )
                if int(output.video.shape[1]) != args.frame_num:
                    raise RuntimeError(
                        f"expected {args.frame_num} decoded frames, "
                        f"got {output.video.shape[1]}"
                    )
                if (output.width, output.height) != (args.width, args.height):
                    raise RuntimeError(
                        "resolved output size differs from requested HD size: "
                        f"{(output.width, output.height)} vs {(args.width, args.height)}"
                    )
                save_video(
                    tensor=output.video[None],
                    save_file=str(temporary_video),
                    fps=args.fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                if not temporary_video.is_file() or temporary_video.stat().st_size == 0:
                    raise RuntimeError("video writer produced no output")
                os.replace(temporary_video, video_path)
                diagnostics = [
                    {
                        "type": "generation",
                        "sample_id": sample_id,
                        "seed": seed,
                        "mode": args.mode,
                        "spatial_mode": args.spatial_mode,
                        "reader_mode": args.reader_mode,
                    }
                ]
                if reader is not None:
                    diagnostics.extend(
                        {"type": "writer_residual", **record} for record in reader.records
                    )
                    _write_jsonl(artifact_dir / "reader.audit.jsonl", reader.records)
                _write_jsonl(artifact_dir / "trace.diagnostics.jsonl", diagnostics)
                manifest = {
                    "status": "success",
                    "run_id": run_id,
                    "method": "TRACE-M3" if args.mode == "i2v" else "TRACE-M4",
                    "sample_id": sample_id,
                    "mode": args.mode,
                    "plan_variant": "planimg" if args.mode == "i2v" else "plan",
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
                    "crossfade_tokens": args.crossfade_tokens,
                    "spatial_mode": args.spatial_mode,
                    "reader_mode": args.reader_mode,
                    "validation": _validation_record(sample.compiled),
                    "context_token_lengths": output.context_token_lengths,
                    "writer": {
                        "trace_active": args.lambda0 > 0.0,
                        "lambda0": args.lambda0,
                        "token_cap_ratio": args.token_cap_ratio,
                        "global_cap_ratio": args.global_cap_ratio,
                        "writer_blocks": [14, 23],
                    },
                }
                _write_json(artifact_dir / "manifest.json", manifest)
                results.append(manifest)
                del output
            except Exception as exc:
                for incomplete in (temporary_video, video_path):
                    try:
                        incomplete.unlink()
                    except FileNotFoundError:
                        pass
                failure = {
                    "status": "failed",
                    "run_id": run_id,
                    "method": "TRACE-M3" if args.mode == "i2v" else "TRACE-M4",
                    "sample_id": sample_id,
                    "mode": args.mode,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                _write_json(artifact_dir / "failure.json", failure)
                failure_diagnostics = [
                    {
                        "type": "failure",
                        "sample_id": sample_id,
                        "seed": seed,
                        "mode": args.mode,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                ]
                if reader is not None:
                    _write_jsonl(artifact_dir / "reader.audit.jsonl", reader.records)
                    failure_diagnostics.extend(
                        {"type": "writer_residual", **record} for record in reader.records
                    )
                _write_jsonl(
                    artifact_dir / "trace.diagnostics.jsonl",
                    failure_diagnostics,
                )
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
        "method": "TRACE-M3" if args.mode == "i2v" else "TRACE-M4",
        "mode": args.mode,
        "requested_sample_count": len(prepared),
        "result_count": len(results),
        "success_count": sum(item["status"] == "success" for item in results),
        "failure_count": sum(item["status"] == "failed" for item in results),
        "results": results,
        "upstream_unchanged": True,
        "upstream_mismatches": upstream_mismatches,
    }
    _write_json(run_root / "results.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
