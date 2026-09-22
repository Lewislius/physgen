#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

INFERENCE_ROOT = Path(__file__).resolve().parent
V1_ROOT = INFERENCE_ROOT.parent
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))

from ace_router.artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    context_record,
    jsonable_path_hashes,
    runtime_versions,
    utc_run_id,
)
from ace_router.conditioning import (  # noqa: E402
    CONDITIONING_MODES,
    CompiledConditioning,
    compile_conditioning,
)
from ace_router.paths import ensure_runtime_dirs, require_write_path  # noqa: E402
from ace_router.schema import AceSample, discover_sample_ids, load_sample  # noqa: E402
from ace_router.schedules import (  # noqa: E402
    LAYER_PRESETS,
    STEP_PRESETS,
    layer_gates_for_preset,
    parse_layer_gates,
    step_gates,
)
from ace_router.sizing import (  # noqa: E402
    LANDSCAPE_REFERENCE_SIZE,
    PORTRAIT_REFERENCE_SIZE,
    best_output_size,
    orientation_and_reference_size,
    prepare_reference_image,
)
from ace_router.upstream_guard import (  # noqa: E402
    assert_upstream_unchanged,
    capture_upstream,
    sha256_file,
    verify_expected_upstream,
)


METHOD_LABELS = {
    "M0": "M0_T2V_WAN",
    "M1": "M1_I2V_WAN",
    "M2": "M2_I2V_CPLUS",
    "M3": "M3_I2V_ACE",
    "M4": "M4_T2V_ACE",
}
I2V_METHODS = frozenset({"M1", "M2", "M3"})
T2V_METHODS = frozenset({"M0", "M4"})
INJECTION_METHODS = frozenset({"M2", "M3", "M4"})
CONTRASTIVE_METHODS = frozenset({"M3", "M4"})
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wan2.2-TI2V-5B M0-M4 baseline and ACE-Router inference"
    )
    parser.add_argument("--method", choices=sorted(METHOD_LABELS), required=True)
    parser.add_argument(
        "--wan_repo", default="/home/liuzhirui/model/Wan2.2"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B",
    )
    parser.add_argument("--demo_root", default=str(V1_ROOT / "demo"))
    parser.add_argument(
        "--output_root", default=str(V1_ROOT / "outputs" / "ace_router")
    )
    parser.add_argument("--run_id", default=None)
    parser.add_argument(
        "--sample_ids",
        default="all",
        help="Comma-separated Pxx ids or 'all'",
    )
    parser.add_argument("--seeds", default="42", help="Comma-separated integers")
    parser.add_argument("--frame_num", type=int, default=97)
    parser.add_argument("--max_area", type=int, default=1280 * 704)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--sample_solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--lambda0", type=float, default=0.5)
    parser.add_argument(
        "--residual_cap_ratio",
        type=float,
        default=None,
        help="Optional per-block ACE/semantic L2 cap; omitted for legacy behavior",
    )
    parser.add_argument("--residual_cap_eps", type=float, default=1e-12)
    parser.add_argument(
        "--conditioning_mode",
        choices=sorted(CONDITIONING_MODES),
        default="original",
    )
    parser.add_argument(
        "--cfg_negative_mode",
        choices=("empty", "wan_default"),
        default="empty",
        help="CFG unconditional text; wan_default reproduces the initial V1 pilot",
    )
    parser.add_argument(
        "--layer_preset", choices=sorted(LAYER_PRESETS), default="mvp_mid16"
    )
    parser.add_argument(
        "--layer_gates",
        default=None,
        help="Optional comma-separated 30 values; overrides --layer_preset",
    )
    parser.add_argument(
        "--diagnostics", choices=("off", "summary", "full"), default="summary"
    )
    parser.add_argument(
        "--step_preset", choices=sorted(STEP_PRESETS), default="legacy"
    )
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--m4_orientation",
        choices=("match_image", "landscape", "portrait"),
        default="landscape",
        help="Legacy M4 T2V output-shape policy (also the M0 fallback)",
    )
    parser.add_argument(
        "--t2v_orientation",
        choices=("match_image", "landscape", "portrait"),
        default=None,
        help="M0/M4 output-shape policy; overrides --m4_orientation",
    )
    parser.add_argument("--minimum_confidence", type=float, default=0.7)
    parser.add_argument(
        "--offload_model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--t5_cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--convert_model_dtype", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--show_progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--allow_upstream_drift", action="store_true")
    parser.add_argument("--require_selection_manifest", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--parse_only", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    return parser


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid seed list: {value!r}") from exc
    if not seeds:
        raise ValueError("at least one seed is required")
    if any(seed < 0 for seed in seeds):
        raise ValueError("batch inference seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds are not allowed")
    return seeds


def _resolve_sample_ids(demo_root: Path, value: str) -> list[str]:
    available = discover_sample_ids(demo_root)
    if value.strip().lower() == "all":
        return available
    requested = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not requested:
        raise ValueError("at least one sample id is required")
    unknown = sorted(set(requested).difference(available))
    if unknown:
        raise ValueError(f"unknown sample ids: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate sample ids are not allowed")
    return requested


def _validate_basic_args(args: argparse.Namespace) -> None:
    if args.frame_num <= 0 or (args.frame_num - 1) % 4:
        raise ValueError("frame_num must satisfy 4n+1")
    if args.max_area <= 0 or args.sampling_steps <= 0 or args.fps <= 0:
        raise ValueError("max_area, sampling_steps, and fps must be positive")
    if not math.isfinite(args.lambda0) or args.lambda0 < 0.0:
        raise ValueError("lambda0 must be finite and non-negative")
    if args.residual_cap_ratio is not None and (
        not math.isfinite(args.residual_cap_ratio)
        or args.residual_cap_ratio <= 0.0
    ):
        raise ValueError("residual_cap_ratio must be finite and positive")
    if not math.isfinite(args.residual_cap_eps) or args.residual_cap_eps <= 0.0:
        raise ValueError("residual_cap_eps must be finite and positive")
    if not math.isfinite(args.guide_scale) or args.guide_scale < 0.0:
        raise ValueError("guide_scale must be finite and non-negative")
    if not math.isfinite(args.shift) or args.shift <= 0.0:
        raise ValueError("shift must be finite and positive")
    if not 0.0 <= args.minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")
    if args.run_id is not None and not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError("run_id may contain only letters, digits, dot, underscore, dash")


def _load_samples(args: argparse.Namespace, sample_ids: list[str]) -> list[AceSample]:
    inspect_image = args.method in I2V_METHODS or (
        args.method in T2V_METHODS and _t2v_orientation_policy(args) == "match_image"
    )
    require_image = args.method in I2V_METHODS
    return [
        load_sample(
            args.demo_root,
            sample_id,
            require_image=require_image,
            inspect_image=inspect_image,
            require_selection_manifest=(
                args.require_selection_manifest and args.method in I2V_METHODS
            ),
            minimum_confidence=args.minimum_confidence,
        )
        for sample_id in sample_ids
    ]


def _method_warnings(sample: AceSample, args: argparse.Namespace) -> list[str]:
    warnings = list(sample.warnings)
    if args.method in T2V_METHODS:
        warnings = [
            warning
            for warning in warnings
            if warning not in {"initial_image_missing", "selection_manifest_missing"}
        ]
        if _t2v_orientation_policy(args) == "match_image" and sample.image_path is None:
            warnings.append("sizing_image_missing_fallback_landscape")
    return warnings


def _t2v_orientation_policy(args: argparse.Namespace) -> str:
    return args.t2v_orientation or args.m4_orientation


def _residual_mode(method: str) -> str:
    if method == "M2":
        return "positive_only"
    if method in CONTRASTIVE_METHODS:
        return "positive_minus_counterfactual"
    return "none"


def _conditioning_for(
    sample: AceSample, args: argparse.Namespace
) -> CompiledConditioning:
    return compile_conditioning(
        sample,
        mode=args.conditioning_mode,
        use_initial_image=args.method in I2V_METHODS,
    )


def _sample_orientation(
    sample: AceSample, args: argparse.Namespace
) -> tuple[str, tuple[int, int]]:
    if args.method in I2V_METHODS or _t2v_orientation_policy(args) == "match_image":
        if sample.image_path is None:
            if args.method in T2V_METHODS:
                return "landscape", LANDSCAPE_REFERENCE_SIZE
            raise ValueError(f"sample {sample.sample_id} has no image for orientation")
        return orientation_and_reference_size(sample.image_path)
    if _t2v_orientation_policy(args) == "portrait":
        return "portrait", PORTRAIT_REFERENCE_SIZE
    return "landscape", LANDSCAPE_REFERENCE_SIZE


def _resolved_size(sample: AceSample, args: argparse.Namespace) -> dict[str, Any]:
    orientation, reference_size = _sample_orientation(sample, args)
    width, height = best_output_size(
        reference_size[0], reference_size[1], 32, 32, args.max_area
    )
    if width * height != args.max_area:
        raise ValueError(
            f"max_area must resolve exactly to width * height, got "
            f"{args.max_area} vs {width} * {height}"
        )
    return {
        "orientation": orientation,
        "reference_size": list(reference_size),
        "width": width,
        "height": height,
        "max_area": args.max_area,
    }


def _preflight_context_token_lengths(
    samples: list[AceSample], checkpoint_dir: str | Path, args: argparse.Namespace
) -> dict[str, dict[str, int]]:
    from ftfy import fix_text
    from transformers import AutoTokenizer

    tokenizer_root = (
        Path(checkpoint_dir).expanduser().resolve() / "google" / "umt5-xxl"
    )
    if not tokenizer_root.is_dir():
        raise ValueError(f"missing local UMT5 tokenizer: {tokenizer_root}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root, local_files_only=True
    )
    names = ["semantic"]
    if args.method in INJECTION_METHODS:
        names.append("positive")
    if args.method in CONTRASTIVE_METHODS:
        names.append("counterfactual")
    flattened: list[str] = []
    for sample in samples:
        conditioning = _conditioning_for(sample, args)
        texts = [conditioning.semantic_text]
        if args.method in INJECTION_METHODS:
            texts.append(conditioning.positive_text)
        if args.method in CONTRASTIVE_METHODS:
            texts.append(conditioning.counterfactual_text)
        flattened.extend(
            re.sub(r"\s+", " ", html.unescape(html.unescape(fix_text(text)))).strip()
            for text in texts
        )
    encoded = tokenizer(
        flattened,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    lengths = [len(token_ids) for token_ids in encoded["input_ids"]]
    result: dict[str, dict[str, int]] = {}
    for sample_index, sample in enumerate(samples):
        start = sample_index * len(names)
        sample_lengths = dict(zip(names, lengths[start : start + len(names)]))
        too_long = {
            name: length for name, length in sample_lengths.items() if length > 512
        }
        if too_long:
            raise ValueError(
                f"sample {sample.sample_id} contexts exceed Wan text_len=512: "
                f"{too_long}"
            )
        result[sample.sample_id] = sample_lengths
    return result


def _check_inputs(
    args: argparse.Namespace,
    samples: list[AceSample],
    upstream: dict[str, Any],
    mismatches: list[str],
    token_lengths: dict[str, dict[str, int]],
) -> dict[str, Any]:
    checks = []
    for sample in samples:
        size = _resolved_size(sample, args)
        conditioning = _conditioning_for(sample, args)
        sizing_image = (
            sample.image_path
            if args.method in T2V_METHODS
            and _t2v_orientation_policy(args) == "match_image"
            else None
        )
        checks.append(
            {
                "sample_id": sample.sample_id,
                "method": METHOD_LABELS[args.method],
                "model_condition_image_path": (
                    str(sample.image_path)
                    if args.method in I2V_METHODS and sample.image_path
                    else None
                ),
                "sizing_only_image_path": str(sizing_image) if sizing_image else None,
                "image_is_model_condition": args.method in I2V_METHODS,
                "residual_mode": _residual_mode(args.method),
                "conditioning_mode": conditioning.mode,
                "semantic_variant": conditioning.semantic_variant,
                "causal_origin_prompt_prefixed": (
                    conditioning.causal_origin_prompt_prefixed
                ),
                "cfg_negative_mode": args.cfg_negative_mode,
                "size": size,
                "context_token_lengths": token_lengths[sample.sample_id],
                "warnings": _method_warnings(sample, args),
            }
        )
    return {
        "status": "ok",
        "method": METHOD_LABELS[args.method],
        "sample_count": len(samples),
        "checks": checks,
        "upstream": upstream,
        "upstream_mismatches": mismatches,
    }


def _reference_record(
    sample: AceSample,
    args: argparse.Namespace,
    size: dict[str, Any],
    prepared: Any | None,
    prepared_path: Path | None,
) -> dict[str, Any]:
    uses_source = args.method in I2V_METHODS or (
        args.method in T2V_METHODS
        and _t2v_orientation_policy(args) == "match_image"
    )
    source_path = sample.image_path if uses_source else None
    source_hash = sha256_file(source_path) if source_path else None
    record: dict[str, Any] = {
        "source_path": str(source_path) if source_path else None,
        "source_sha256": source_hash,
        "selection_manifest": (
            str(sample.selection_manifest_path)
            if args.method in I2V_METHODS and sample.selection_manifest_path
            else None
        ),
        "orientation": size["orientation"],
        "resolved_output_size": [size["width"], size["height"]],
        "used_as_model_condition": args.method in I2V_METHODS,
        "used_for_output_sizing_only": (
            args.method in T2V_METHODS
            and _t2v_orientation_policy(args) == "match_image"
        ),
    }
    if prepared is not None:
        if prepared_path is None or not prepared_path.is_file():
            raise ValueError("prepared I2V reference artifact is missing")
        record.update(
            {
                "original_size": list(prepared.original_size),
                "crop_box": list(prepared.crop_box),
                "prepared_size": list(prepared.prepared_size),
                "prepared_artifact_path": str(prepared_path),
                "prepared_artifact_sha256": sha256_file(prepared_path),
                "crop_policy": "center_crop_to_20x11_or_11x20_then_lanczos_resize",
            }
        )
    return record


def _step_timing_record(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    stable = values[1:] if len(values) > 1 else values
    return {
        "count": len(values),
        "per_step_seconds": values,
        "warmup_first_step_seconds": values[0],
        "stable_median_seconds": statistics.median(stable),
        "all_steps_median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def _method_context_record(
    sample: AceSample,
    args: argparse.Namespace,
    token_lengths: dict[str, int] | None = None,
    cfg_negative_prompt: str | None = None,
) -> dict[str, Any]:
    conditioning = _conditioning_for(sample, args)
    if cfg_negative_prompt is None:
        cfg_negative_prompt = ""
    record = context_record(
        conditioning.semantic_text,
        conditioning.positive_text,
        conditioning.counterfactual_text,
        cfg_negative_prompt,
        token_lengths,
    )
    record["conditioning_mode"] = conditioning.mode
    record["semantic_variant"] = conditioning.semantic_variant
    record["compiler_version"] = conditioning.compiler_version
    record["causal_origin_prompt_prefixed"] = (
        conditioning.causal_origin_prompt_prefixed
    )
    record["used_by_model"] = {
        "semantic": True,
        "positive": args.method in INJECTION_METHODS,
        "counterfactual": args.method in CONTRASTIVE_METHODS,
        "cfg_negative": args.cfg_negative_mode == "wan_default",
        "cfg_unconditional_empty": args.cfg_negative_mode == "empty",
    }
    return record


def _resolved_config(
    args: argparse.Namespace,
    sample: AceSample,
    seed: int,
    size: dict[str, Any],
    gates: list[float],
    gates_per_step: list[float],
    run_id: str,
    upstream: dict[str, Any],
) -> dict[str, Any]:
    injection_enabled = args.method in INJECTION_METHODS
    residual_mode = _residual_mode(args.method)
    conditioning = _conditioning_for(sample, args)
    return {
        "run_id": run_id,
        "sample_id": sample.sample_id,
        "method": METHOD_LABELS[args.method],
        "paths": {
            "wan_repo": str(Path(args.wan_repo).resolve()),
            "checkpoint": str(Path(args.checkpoint_dir).resolve()),
            "demo_root": str(Path(args.demo_root).resolve()),
            "sample_dir": str(sample.sample_dir),
            "output_root": str(Path(args.output_root).resolve()),
        },
        "generation": {
            "frame_num": args.frame_num,
            "width": size["width"],
            "height": size["height"],
            "max_area": args.max_area,
            "sampling_steps": args.sampling_steps,
            "sample_solver": args.sample_solver,
            "shift": args.shift,
            "guide_scale": args.guide_scale,
            "fps": args.fps,
            "seed": seed,
            "offload_model": args.offload_model,
            "t5_cpu": args.t5_cpu,
            "convert_model_dtype": args.convert_model_dtype,
        },
        "ace": {
            "enabled": injection_enabled,
            "residual_mode": residual_mode,
            "formula": (
                "semantic + scale * (positive - semantic)"
                if residual_mode == "positive_only"
                else (
                    (
                        "semantic + relative_cap(scale * "
                        "(positive - counterfactual), semantic)"
                        if args.residual_cap_ratio is not None
                        else "semantic + scale * (positive - counterfactual)"
                    )
                    if residual_mode == "positive_minus_counterfactual"
                    else "semantic (unmodified Wan)"
                )
            ),
            "lambda0": args.lambda0 if injection_enabled else 0.0,
            "residual_cap_ratio": (
                args.residual_cap_ratio if injection_enabled else None
            ),
            "residual_cap_eps": (
                args.residual_cap_eps if injection_enabled else None
            ),
            "layer_preset": args.layer_preset if injection_enabled else None,
            "step_preset": args.step_preset if injection_enabled else None,
            "layer_gates": gates,
            "step_gates": gates_per_step,
            "cfg_scope": "conditional_only" if injection_enabled else None,
            "fp32_delta": injection_enabled,
            "diagnostics": args.diagnostics if injection_enabled else "off",
        },
        "conditioning": {
            "initial_image_to_model": args.method in I2V_METHODS,
            "t2v_orientation_policy": (
                _t2v_orientation_policy(args) if args.method in T2V_METHODS else None
            ),
            "conditioning_mode": conditioning.mode,
            "semantic_variant": conditioning.semantic_variant,
            "compiler_version": conditioning.compiler_version,
            "causal_origin_prompt_prefixed": (
                conditioning.causal_origin_prompt_prefixed
            ),
            "positive_context_injected": args.method in INJECTION_METHODS,
            "counterfactual_context_injected": args.method in CONTRASTIVE_METHODS,
            "counterfactual_as_cfg_negative": False,
            "negative_prompt_enabled": args.cfg_negative_mode == "wan_default",
            "cfg_negative_mode": args.cfg_negative_mode,
        },
        "sample_warnings": _method_warnings(sample, args),
        "upstream_identity": upstream,
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_basic_args(args)
    if args.parse_only:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    demo_root = Path(args.demo_root).expanduser().resolve()
    output_root = require_write_path(args.output_root)
    ensure_runtime_dirs(output_root)
    sample_ids = _resolve_sample_ids(demo_root, args.sample_ids)
    seeds = _parse_seeds(args.seeds)
    samples = _load_samples(args, sample_ids)
    preflight_token_lengths = _preflight_context_token_lengths(
        samples, args.checkpoint_dir, args
    )
    upstream_before = capture_upstream(args.wan_repo, args.checkpoint_dir)
    upstream_mismatches = verify_expected_upstream(
        upstream_before, allow_drift=args.allow_upstream_drift
    )
    check_report = _check_inputs(
        args,
        samples,
        upstream_before,
        upstream_mismatches,
        preflight_token_lengths,
    )
    if args.check_only:
        print(json.dumps(check_report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    run_id = args.run_id or utc_run_id(args.method)
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("resolved run_id is unsafe")
    run_root = require_write_path(output_root / run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_root / "preflight.json", check_report)

    wan_repo = str(Path(args.wan_repo).expanduser().resolve())
    if wan_repo not in sys.path:
        sys.path.insert(0, wan_repo)
    import diffusers  # noqa: E402
    import torch  # noqa: E402
    from wan.configs import WAN_CONFIGS  # noqa: E402
    from wan.utils.utils import save_video  # noqa: E402

    from ace_router.diagnostics import ResidualDiagnostics  # noqa: E402
    from ace_router.pipeline import AceWanTI2V  # noqa: E402

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan ACE inference")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Loading one frozen Wan2.2-TI2V-5B pipeline")
    pipeline = AceWanTI2V(
        config=WAN_CONFIGS["ti2v-5B"],
        checkpoint_dir=str(Path(args.checkpoint_dir).expanduser().resolve()),
        device_id=args.device_id,
        rank=0,
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
        convert_model_dtype=args.convert_model_dtype,
    )
    if args.method not in INJECTION_METHODS:
        gates = [0.0] * pipeline.model.num_layers
        gates_per_step = [0.0] * args.sampling_steps
    elif args.layer_gates:
        gates = parse_layer_gates(args.layer_gates, num_layers=pipeline.model.num_layers)
        gates_per_step = step_gates(args.sampling_steps, preset=args.step_preset)
    else:
        gates = layer_gates_for_preset(
            args.layer_preset, num_layers=pipeline.model.num_layers
        )
        gates_per_step = step_gates(args.sampling_steps, preset=args.step_preset)
    runtime = runtime_versions(torch, diffusers)
    results: list[dict[str, Any]] = []

    for sample in samples:
        size = _resolved_size(sample, args)
        conditioning = _conditioning_for(sample, args)
        for seed in seeds:
            method_label = METHOD_LABELS[args.method]
            artifact_dir = require_write_path(
                run_root / sample.sample_id / method_label / f"seed_{seed:06d}"
            )
            if artifact_dir.exists() and any(artifact_dir.iterdir()):
                raise FileExistsError(
                    f"refusing to overwrite non-empty artifact directory: {artifact_dir}"
                )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            video_path = require_write_path(artifact_dir / "video.mp4")
            prepared = None
            prepared_path = None
            image_for_model = None
            if args.method in I2V_METHODS:
                if sample.image_path is None:
                    raise ValueError(
                        f"{args.method} sample has no image: {sample.sample_id}"
                    )
                prepared = prepare_reference_image(sample.image_path)
                image_for_model = prepared.image
                prepared_path = require_write_path(artifact_dir / "reference_input.png")
                prepared.image.save(prepared_path, format="PNG")

            reference = _reference_record(
                sample, args, size, prepared, prepared_path
            )
            input_hashes = jsonable_path_hashes(
                {
                    "origin_txt": sample.origin_path,
                    "causal_json": (
                        sample.causal_path
                        if args.method in INJECTION_METHODS
                        or args.conditioning_mode == "compiled"
                        else None
                    ),
                    "i0_metadata_json": (
                        sample.i0_metadata_path
                        if args.method in I2V_METHODS
                        or args.conditioning_mode == "compiled"
                        else None
                    ),
                    "i0_image": (
                        sample.image_path if args.method in I2V_METHODS else None
                    ),
                    "model_reference_image": prepared_path,
                    "sizing_image": (
                        sample.image_path
                        if args.method in T2V_METHODS
                        and _t2v_orientation_policy(args) == "match_image"
                        else None
                    ),
                    "selection_manifest": (
                        sample.selection_manifest_path
                        if args.method in I2V_METHODS
                        else None
                    ),
                }
            )
            cfg_negative_prompt = (
                pipeline.sample_neg_prompt
                if args.cfg_negative_mode == "wan_default"
                else ""
            )
            contexts = _method_context_record(
                sample,
                args,
                cfg_negative_prompt=cfg_negative_prompt,
            )
            config = _resolved_config(
                args,
                sample,
                seed,
                size,
                gates,
                gates_per_step,
                run_id,
                upstream_before,
            )
            atomic_write_json(artifact_dir / "config.resolved.json", config)
            atomic_write_json(artifact_dir / "contexts.json", contexts)
            atomic_write_json(
                artifact_dir / "conditioning.compiled.json",
                conditioning.to_dict(),
            )
            atomic_write_json(artifact_dir / "reference_image.json", reference)

            diagnostics = ResidualDiagnostics(
                args.diagnostics if args.method in INJECTION_METHODS else "off"
            )
            started = time.monotonic()
            temporary_video_path = require_write_path(
                artifact_dir / ".video.incomplete.mp4"
            )
            torch.cuda.reset_peak_memory_stats(args.device_id)
            logging.info(
                "Generating sample=%s method=%s seed=%s size=%sx%s",
                sample.sample_id,
                method_label,
                seed,
                size["width"],
                size["height"],
            )
            try:
                common_generation_args = {
                    "input_prompt": conditioning.semantic_text,
                    "image": image_for_model,
                    "size": (size["width"], size["height"]),
                    "max_area": args.max_area,
                    "frame_num": args.frame_num,
                    "shift": args.shift,
                    "sample_solver": args.sample_solver,
                    "sampling_steps": args.sampling_steps,
                    "guide_scale": args.guide_scale,
                    "seed": seed,
                    "offload_model": args.offload_model,
                }
                if args.method in INJECTION_METHODS:
                    output = pipeline.generate_ace(
                        **common_generation_args,
                        positive_text=conditioning.positive_text,
                        counterfactual_text=(
                            conditioning.counterfactual_text
                            if args.method in CONTRASTIVE_METHODS
                            else None
                        ),
                        residual_mode=_residual_mode(args.method),
                        layer_gates=gates,
                        step_gates=gates_per_step,
                        lambda0=args.lambda0,
                        residual_cap_ratio=args.residual_cap_ratio,
                        residual_cap_eps=args.residual_cap_eps,
                        cfg_negative_mode=args.cfg_negative_mode,
                        diagnostic_sink=diagnostics,
                        show_progress=args.show_progress,
                    )
                else:
                    output = pipeline.generate_baseline(**common_generation_args)
                if (output.width, output.height) != (size["width"], size["height"]):
                    raise RuntimeError(
                        "pipeline output size differs from preflight: "
                        f"{(output.width, output.height)} vs "
                        f"{(size['width'], size['height'])}"
                    )
                if int(output.video.shape[1]) != args.frame_num:
                    raise RuntimeError(
                        f"expected {args.frame_num} frames, got {output.video.shape[1]}"
                    )
                save_video(
                    tensor=output.video[None],
                    save_file=str(temporary_video_path),
                    fps=args.fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                if (
                    not temporary_video_path.is_file()
                    or temporary_video_path.stat().st_size == 0
                ):
                    raise RuntimeError(
                        f"video writer did not produce {temporary_video_path}"
                    )
                os.replace(temporary_video_path, video_path)
                elapsed = time.monotonic() - started
                contexts = _method_context_record(
                    sample,
                    args,
                    output.context_token_lengths,
                    cfg_negative_prompt=cfg_negative_prompt,
                )
                atomic_write_json(artifact_dir / "contexts.json", contexts)
                atomic_write_jsonl(
                    artifact_dir / "diagnostics.jsonl", diagnostics.records
                )
                manifest = {
                    "status": "success",
                    "run_id": run_id,
                    "sample_id": sample.sample_id,
                    "method": method_label,
                    "seed": output.seed,
                    "output_video": str(video_path),
                    "output_video_sha256": sha256_file(video_path),
                    "input_hashes": input_hashes,
                    "contexts": contexts,
                    "reference": reference,
                    "ace": config["ace"],
                    "generation": {
                        **config["generation"],
                        "first_frame_reanchor_count": output.first_frame_reanchor_count,
                    },
                    "diagnostics": diagnostics.summary(),
                    "upstream": upstream_before,
                    "runtime": {
                        **runtime,
                        "elapsed_seconds": elapsed,
                        "peak_cuda_bytes": torch.cuda.max_memory_allocated(
                            args.device_id
                        ),
                        "cuda_memory_reserved_bytes": torch.cuda.memory_reserved(
                            args.device_id
                        ),
                        "denoising_step_timing": _step_timing_record(
                            output.step_elapsed_seconds
                        ),
                    },
                    "warnings": _method_warnings(sample, args) + upstream_mismatches,
                }
                atomic_write_json(artifact_dir / "manifest.json", manifest)
                results.append(
                    {
                        "status": "success",
                        "sample_id": sample.sample_id,
                        "method": method_label,
                        "seed": seed,
                        "artifact_dir": str(artifact_dir),
                        "video": str(video_path),
                    }
                )
                del output
            except Exception as exc:
                for incomplete_path in (temporary_video_path, video_path):
                    try:
                        incomplete_path.unlink()
                    except FileNotFoundError:
                        pass
                elapsed = time.monotonic() - started
                failure = {
                    "status": "failed",
                    "run_id": run_id,
                    "sample_id": sample.sample_id,
                    "method": method_label,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "elapsed_seconds": elapsed,
                    "config": config,
                    "contexts": contexts,
                    "diagnostics": diagnostics.summary(),
                }
                atomic_write_jsonl(
                    artifact_dir / "diagnostics.jsonl", diagnostics.records
                )
                atomic_write_json(artifact_dir / "failure.json", failure)
                results.append(
                    {
                        "status": "failed",
                        "sample_id": sample.sample_id,
                        "method": method_label,
                        "seed": seed,
                        "artifact_dir": str(artifact_dir),
                        "error": str(exc),
                    }
                )
                logging.exception(
                    "generation failed for %s seed=%s", sample.sample_id, seed
                )
                if not args.continue_on_error:
                    raise
            finally:
                if prepared is not None:
                    prepared.image.close()
                torch.cuda.empty_cache()

    upstream_after = capture_upstream(args.wan_repo, args.checkpoint_dir)
    assert_upstream_unchanged(upstream_before, upstream_after)
    run_summary = {
        "run_id": run_id,
        "method": METHOD_LABELS[args.method],
        "result_count": len(results),
        "success_count": sum(result["status"] == "success" for result in results),
        "failure_count": sum(result["status"] == "failed" for result in results),
        "results": results,
        "upstream_unchanged": True,
    }
    atomic_write_json(run_root / "run_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if run_summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
