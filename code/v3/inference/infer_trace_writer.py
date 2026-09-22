#!/usr/bin/env python3
"""Direct staged Wan inference with optional structured JSON text guidance."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

EXPECTED_IDS = tuple(f"P{number:02d}" for number in range(1, 21))
FIXED_SIZE = (1280, 704)
FIXED_MAX_AREA = 1280 * 704
STAGE_IDS = ("setup", "onset", "evolution", "completion", "terminal")
LATENT_TOKENS = 25
CONDITIONING_VARIANTS = ("baseline", "strong_stage", "strong_stage_json")
CFG_MODES = ("legacy", "separate")
JSON_GLOBAL_CONTEXT = "json.global"
JSON_STAGE_PREFIX = "json.stage."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PhysGen direct five-stage prompts for Wan"
    )
    parser.add_argument("--mode", choices=("i2v", "t2v"), required=True)
    parser.add_argument("--wan_repo", default="/home/liuzhirui/model/Wan2.2")
    parser.add_argument(
        "--checkpoint_dir",
        default="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B",
    )
    parser.add_argument(
        "--demo_root", default="/home/liuzhirui/Project/physGen/code/v1/demo"
    )
    parser.add_argument(
        "--plan_root", default="/home/liuzhirui/Project/physGen/code/v1/demo"
    )
    parser.add_argument(
        "--output_root",
        default="/home/liuzhirui/Project/physGen/code/v3/outputs/trace_writer",
    )
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--sample_ids", default="all")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--global_prompt_override", default=None)
    parser.add_argument("--stage_prompt_override", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--max_area", type=int, default=901120)
    parser.add_argument("--frame_num", type=int, default=97)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--sample_solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--cfg_mode", choices=CFG_MODES, default="legacy")
    parser.add_argument("--phys_guidance_scale", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--stage_strength", type=float, default=0.05)
    parser.add_argument(
        "--conditioning_variant",
        choices=CONDITIONING_VARIANTS,
        default="baseline",
        help=(
            "baseline keeps the original 0.05 residual; strong_stage RMS-matches "
            "stage c+ to global semantics; strong_stage_json also injects compact "
            "structured JSON facts through independent cross-attention contexts"
        ),
    )
    parser.add_argument("--stage_ratio_to_global", type=float, default=1.0)
    parser.add_argument("--json_global_ratio_to_global", type=float, default=0.25)
    parser.add_argument("--json_stage_ratio_to_global", type=float, default=0.50)
    parser.add_argument("--condition_extra_cap_ratio", type=float, default=1.75)
    parser.add_argument("--condition_norm_max_scale", type=float, default=10.0)
    parser.add_argument("--stage_block_start", type=int, default=14)
    parser.add_argument("--stage_block_stop", type=int, default=24)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--offload_model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--t5_cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--convert_model_dtype", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--continue_on_error", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--show_progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--conditioning_audit", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--parse_only", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    return parser


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def selected_sample_ids(value: str) -> tuple[str, ...]:
    if value == "all":
        return EXPECTED_IDS
    selected = tuple(item.upper() for item in value.split(","))
    if not selected or any(item not in EXPECTED_IDS for item in selected):
        raise ValueError("sample_ids must be all or comma-separated IDs from P01 to P20")
    if len(selected) != len(set(selected)):
        raise ValueError("sample_ids contains a duplicate")
    return selected


def selected_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(","))
    if not seeds or any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique non-negative integers")
    return seeds


def check_arguments(args: argparse.Namespace) -> None:
    if (args.width, args.height) != FIXED_SIZE or args.max_area != FIXED_MAX_AREA:
        raise ValueError("P0 uses the fixed Wan size 1280x704 and max_area 901120")
    if args.frame_num != 97:
        raise ValueError("the direct five-stage route requires 97 frames / 25 latent tokens")
    if args.sampling_steps <= 0 or args.fps <= 0 or args.device_id < 0:
        raise ValueError("sampling_steps/fps must be positive and device_id non-negative")
    if not math.isfinite(args.guide_scale) or args.guide_scale < 0:
        raise ValueError("guide_scale must be finite and non-negative")
    if not math.isfinite(args.phys_guidance_scale) or args.phys_guidance_scale < 0:
        raise ValueError("phys_guidance_scale must be finite and non-negative")
    non_negative = {
        "stage_strength": args.stage_strength,
        "stage_ratio_to_global": args.stage_ratio_to_global,
        "json_global_ratio_to_global": args.json_global_ratio_to_global,
        "json_stage_ratio_to_global": args.json_stage_ratio_to_global,
        "condition_extra_cap_ratio": args.condition_extra_cap_ratio,
        "condition_norm_max_scale": args.condition_norm_max_scale,
    }
    for name, value in non_negative.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if args.conditioning_variant != "baseline" and args.condition_extra_cap_ratio == 0:
        raise ValueError("enhanced conditioning requires condition_extra_cap_ratio > 0")
    if args.conditioning_variant != "baseline" and args.condition_norm_max_scale == 0:
        raise ValueError("enhanced conditioning requires condition_norm_max_scale > 0")
    if not 0 <= args.stage_block_start < args.stage_block_stop <= 30:
        raise ValueError("stage block interval must lie inside the 30 Wan blocks")


def validate_direct_stage_data(plan: dict, sample_id: str) -> None:
    stages = plan["stages"]
    if tuple(stage["id"] for stage in stages) != STAGE_IDS:
        raise ValueError(f"{sample_id}: stages must be the fixed five-stage order")
    if any(not isinstance(stage["positive"], str) or not stage["positive"] for stage in stages):
        raise ValueError(f"{sample_id}: every stage positive must be a non-empty string")
    route = plan["temporal_route"]
    if route["latent_tokens"] != LATENT_TOKENS or route["crossfade_tokens"] != 2:
        raise ValueError(f"{sample_id}: route must use 25 latent tokens and two-token crossfades")
    if tuple(route["stage_order"]) != STAGE_IDS:
        raise ValueError(f"{sample_id}: temporal stage order does not match stages")
    weights = route["weights"]
    if len(weights) != len(STAGE_IDS) or any(len(row) != LATENT_TOKENS for row in weights):
        raise ValueError(f"{sample_id}: temporal weights must have shape 5x25")


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a non-empty string array")
    return tuple(item.strip() for item in value)


def _natural_join(values: tuple[str, ...]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _count_label(value: int) -> str:
    words = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four"}
    return words.get(value, str(value))


def compile_structured_guidance(plan: dict, sample_id: str) -> tuple[str, tuple[str, ...]]:
    """Turn a small typed JSON schema into concise, positive T5 contexts.

    This is deliberately not the archived entity-ledger compiler: every fact is
    explicitly authored, no forbidden categories are inferred, and stage facts
    remain time-local instead of being appended to the global semantic prompt.
    """

    guidance = plan.get("structured_guidance")
    if not isinstance(guidance, dict):
        raise ValueError(
            f"{sample_id}: strong_stage_json requires structured_guidance"
        )
    if guidance.get("schema_version") != "structured-text-guidance-v1":
        raise ValueError(f"{sample_id}: unsupported structured_guidance schema")

    declared_entities = {
        item["id"] for item in plan.get("entities", []) if isinstance(item, dict)
    }
    entity_specs = guidance.get("entities")
    if not isinstance(entity_specs, list) or not entity_specs:
        raise ValueError(f"{sample_id}: structured_guidance.entities must be non-empty")

    labels: dict[str, str] = {}
    global_sentences: list[str] = []
    for index, entity in enumerate(entity_specs):
        field = f"{sample_id}.structured_guidance.entities[{index}]"
        if not isinstance(entity, dict):
            raise ValueError(f"{field} must be an object")
        entity_id = entity.get("id")
        label = entity.get("label")
        count = entity.get("count")
        if entity_id not in declared_entities or not isinstance(label, str) or not label.strip():
            raise ValueError(f"{field} must reference a declared entity with a label")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"{field}.count must be a non-negative integer")
        if entity_id in labels:
            raise ValueError(f"{sample_id}: duplicate structured entity {entity_id}")
        invariants = _string_list(entity.get("invariants"), f"{field}.invariants")
        labels[entity_id] = label.strip()
        persistent = entity.get("persistent", True)
        if not isinstance(persistent, bool):
            raise ValueError(f"{field}.persistent must be a boolean")
        if persistent:
            global_sentences.append(
                f"There is exactly {_count_label(count)} {label.strip()}, kept as the same "
                f"tracked object with unchanged {_natural_join(invariants)}."
            )
        else:
            global_sentences.append(
                f"Across the video there is at most {_count_label(count)} {label.strip()}, "
                "visible only in the stages that explicitly require it, with unchanged "
                f"{_natural_join(invariants)}."
            )
    if set(labels) != declared_entities:
        missing = sorted(declared_entities - set(labels))
        extra = sorted(set(labels) - declared_entities)
        raise ValueError(
            f"{sample_id}: structured entities must exactly cover plan entities; "
            f"missing={missing}, extra={extra}"
        )

    initial = guidance.get("initial_condition")
    if not isinstance(initial, dict) or not isinstance(initial.get("type"), str):
        raise ValueError(f"{sample_id}: structured initial_condition is invalid")
    active_initial_stages = tuple(initial.get("active_stages", ()))
    if not active_initial_stages or any(stage not in STAGE_IDS for stage in active_initial_stages):
        raise ValueError(f"{sample_id}: initial_condition.active_stages is invalid")
    initial_facts = _string_list(
        initial.get("facts"), f"{sample_id}.structured_guidance.initial_condition.facts"
    )

    stage_specs = guidance.get("stages")
    if not isinstance(stage_specs, list) or tuple(
        stage.get("id") if isinstance(stage, dict) else None for stage in stage_specs
    ) != STAGE_IDS:
        raise ValueError(f"{sample_id}: structured stages must follow the fixed order")

    stage_texts: list[str] = []
    allowed_presence = {"required", "transitioning", "absent"}
    for stage in stage_specs:
        stage_id = stage["id"]
        sentences = list(initial_facts) if stage_id in active_initial_stages else []
        states = stage.get("entity_states")
        if not isinstance(states, list) or not states:
            raise ValueError(f"{sample_id}.{stage_id}: entity_states must be non-empty")
        for index, state in enumerate(states):
            field = f"{sample_id}.{stage_id}.entity_states[{index}]"
            if not isinstance(state, dict) or state.get("entity_id") not in labels:
                raise ValueError(f"{field} must reference a structured entity")
            entity_id = state["entity_id"]
            presence = state.get("presence")
            count = state.get("count")
            if presence not in allowed_presence:
                raise ValueError(f"{field}.presence is invalid")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"{field}.count must be a non-negative integer")
            state_phrases = _string_list(state.get("states"), f"{field}.states")
            if presence == "required":
                presence_text = "is visible"
            elif presence == "transitioning":
                presence_text = "stays continuously visible through the transition"
            else:
                presence_text = "is absent from the frame"
            sentences.append(
                f"The {labels[entity_id]} {presence_text} as exactly {_count_label(count)} "
                f"instance, {_natural_join(state_phrases)}."
            )
            mutable = state.get("mutable", [])
            if not isinstance(mutable, list) or any(
                not isinstance(item, str) or not item.strip() for item in mutable
            ):
                raise ValueError(f"{field}.mutable must be a string array")
            if mutable:
                sentences.append(
                    f"For the {labels[entity_id]}, this stage changes only "
                    f"{_natural_join(tuple(item.strip() for item in mutable))}."
                )
        relations = stage.get("relations", [])
        if not isinstance(relations, list) or any(
            not isinstance(item, str) or not item.strip() for item in relations
        ):
            raise ValueError(f"{sample_id}.{stage_id}.relations must be a string array")
        sentences.extend(item.strip() for item in relations)
        stage_texts.append(" ".join(sentences))

    return " ".join(global_sentences), tuple(stage_texts)


def load_inputs(args: argparse.Namespace, sample_ids: tuple[str, ...]) -> tuple[dict, ...]:
    from PIL import Image

    demo_root = Path(args.demo_root).expanduser().resolve()
    plan_root = Path(args.plan_root).expanduser().resolve()
    expected_basis = "visual_grounding" if args.mode == "i2v" else "identifier_only"
    plan_suffix = "planimg" if args.mode == "i2v" else "plan"
    prepared = []

    for sample_id in sample_ids:
        sample_dir = demo_root / sample_id
        plan_path = plan_root / sample_id / f"{sample_id}-V2-{plan_suffix}.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

        if plan["schema_version"] != "trace-direct-v1":
            raise ValueError(f"{sample_id}: plan is not trace-direct-v1")
        if plan["design_revision"] != "p0-stage-direct-json-r2":
            raise ValueError(f"{sample_id}: plan revision is not p0-stage-direct-json-r2")
        if plan["sample_id"] != sample_id:
            raise ValueError(f"{sample_id}: plan sample_id does not match its directory")
        if plan["generation_basis"]["mode"] != expected_basis:
            raise ValueError(f"{sample_id}: plan mode is not {expected_basis}")
        if not isinstance(plan["global_semantic"], str) or not plan["global_semantic"]:
            raise ValueError(f"{sample_id}: global_semantic must be a non-empty string")
        if plan["cfg_negative"] != " ":
            raise ValueError(
                f"{sample_id}: cfg_negative must be one literal space so Wan does not load its default negative prompt"
            )
        validate_direct_stage_data(plan, sample_id)

        structured_global_prompt = None
        structured_stage_prompts: tuple[str, ...] = ()
        if args.conditioning_variant == "strong_stage_json":
            structured_global_prompt, structured_stage_prompts = compile_structured_guidance(
                plan, sample_id
            )

        image_path = None
        if args.mode == "i2v":
            image_path = sample_dir / plan["source_files"]["reference_input"]
            with Image.open(image_path) as image:
                if image.size != FIXED_SIZE:
                    raise ValueError(
                        f"{sample_id}: reference image is {image.size}, expected {FIXED_SIZE}"
                    )
                image.verify()
        global_prompt = args.global_prompt_override or plan["global_semantic"]
        stage_prompts = (
            tuple(args.stage_prompt_override for _ in STAGE_IDS)
            if args.stage_prompt_override
            else tuple(stage["positive"] for stage in plan["stages"])
        )
        prepared.append(
            {
                "sample_id": sample_id,
                "plan_path": plan_path,
                "prompt": global_prompt,
                "negative_prompt": plan["cfg_negative"],
                "stage_prompts": stage_prompts,
                "structured_global_prompt": structured_global_prompt,
                "structured_stage_prompts": structured_stage_prompts,
                "temporal_route": plan["temporal_route"],
                "image_path": image_path,
            }
        )
    return tuple(prepared)


def preflight_manifest(args: argparse.Namespace, prepared: tuple[dict, ...]) -> dict:
    enhanced = args.conditioning_variant != "baseline"
    structured = args.conditioning_variant == "strong_stage_json"
    return {
        "status": "ready",
        "method": f"TRACE-V3-{args.conditioning_variant.upper()}-{args.mode.upper()}",
        "mode": args.mode,
        "conditioning_variant": args.conditioning_variant,
        "conditioning": {
            "global_positive": (
                "command-line override"
                if args.global_prompt_override
                else "plan.global_semantic verbatim"
            ),
            "stage_c_plus": (
                "one command-line override repeated across all five stages"
                if args.stage_prompt_override
                else "each stages[].positive verbatim"
            ),
            "stage_c_minus": "empty; shared plan.cfg_negative whitespace context",
            "temporal_route": "plan.temporal_route.weights verbatim",
            "stage_fusion": (
                "per-stage RMS matched to global cross-attention on active temporal tokens"
                if enhanced
                else "legacy fixed residual coefficient"
            ),
            "stage_ratio_to_global_in_selected_blocks": (
                args.stage_ratio_to_global if enhanced else None
            ),
            "structured_json_text_guidance": structured,
            "json_global_ratio_to_global": (
                args.json_global_ratio_to_global if structured else None
            ),
            "json_stage_ratio_to_global": (
                args.json_stage_ratio_to_global if structured else None
            ),
            "extra_condition_cap_ratio": (
                args.condition_extra_cap_ratio if enhanced else None
            ),
            "denoise_step_gate": "constant 1.0" if enhanced else "legacy 0.5/1.0/0.3/0.1",
            "stage_checkpoints_used_by_model": True,
            "json_renderer": (
                "explicit typed facts to concise positive stage-local T5 contexts"
                if structured
                else False
            ),
            "legacy_writer_corrector_reader_subsystem": False,
            "direct_parameter_free_stage_residual": True,
        },
        "fixed_output_size": [args.width, args.height],
        "frame_num": args.frame_num,
        "sampling_steps": args.sampling_steps,
        "sample_solver": args.sample_solver,
        "guide_scale": args.guide_scale,
        "cfg_mode": args.cfg_mode,
        "phys_guidance_scale": (
            args.phys_guidance_scale if args.cfg_mode == "separate" else None
        ),
        "shift": args.shift,
        "fps": args.fps,
        "stage_strength": args.stage_strength,
        "stage_ratio_to_global": args.stage_ratio_to_global if enhanced else None,
        "json_global_ratio_to_global": args.json_global_ratio_to_global if structured else None,
        "json_stage_ratio_to_global": args.json_stage_ratio_to_global if structured else None,
        "condition_extra_cap_ratio": args.condition_extra_cap_ratio if enhanced else None,
        "stage_block_interval": [args.stage_block_start, args.stage_block_stop],
        "samples": [
            {
                "sample_id": item["sample_id"],
                "plan": str(item["plan_path"]),
                "temporal_route": {
                    "profile": item["temporal_route"]["profile"],
                    "latent_tokens": item["temporal_route"]["latent_tokens"],
                    "crossfade_tokens": item["temporal_route"]["crossfade_tokens"],
                },
                "image": str(item["image_path"]) if item["image_path"] else None,
                "structured_guidance": item["structured_global_prompt"] is not None,
            }
            for item in prepared
        ],
    }


@contextmanager
def no_sync():
    yield


def finite_tensor(value, message: str) -> None:
    import torch

    condition = torch.isfinite(value).all()
    if value.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    elif not bool(condition.item()):
        raise FloatingPointError(message)


def encode_direct_contexts(pipeline, sample: dict, offload_model: bool):
    import torch

    pairs = [
        ("global_semantic", sample["prompt"]),
        ("cfg_negative", sample["negative_prompt"]),
        *zip(STAGE_IDS, sample["stage_prompts"]),
    ]
    if sample["structured_global_prompt"] is not None:
        pairs.append((JSON_GLOBAL_CONTEXT, sample["structured_global_prompt"]))
        pairs.extend(
            (JSON_STAGE_PREFIX + stage_id, text)
            for stage_id, text in zip(STAGE_IDS, sample["structured_stage_prompts"])
        )
    names = tuple(name for name, _ in pairs)
    texts = tuple(value for _, value in pairs)
    if pipeline.t5_cpu:
        encoded = pipeline.text_encoder(list(texts), torch.device("cpu"))
        encoded = [item.to(pipeline.device) for item in encoded]
    else:
        pipeline.text_encoder.model.to(pipeline.device)
        encoded = pipeline.text_encoder(list(texts), pipeline.device)
        if offload_model:
            pipeline.text_encoder.model.cpu()
    return dict(zip(names, encoded)), {
        name: int(context.size(0)) for name, context in zip(names, encoded)
    }


def project_direct_contexts(model, encoded: dict):
    import torch

    names = tuple(encoded)
    padded = []
    for name in names:
        context = encoded[name]
        if context.ndim != 2 or context.size(0) > model.text_len:
            raise ValueError(f"{name}: invalid T5 context shape {tuple(context.shape)}")
        padded.append(
            torch.cat(
                [
                    context,
                    context.new_zeros(model.text_len - context.size(0), context.size(1)),
                ]
            )
        )
    projected = model.text_embedding(torch.stack(padded))
    return {
        name: projected[index].unsqueeze(0) for index, name in enumerate(names)
    }


def build_direct_stage_gates(
    route: dict,
    *,
    height: int,
    width: int,
    seq_len: int,
    device,
):
    import torch

    weights = torch.tensor(route["weights"], dtype=torch.float32, device=device)
    valid = weights[:, :, None, None].expand(-1, -1, height, width).reshape(
        len(STAGE_IDS), -1, 1
    )
    if valid.size(1) > seq_len:
        raise ValueError("stage route exceeds Wan sequence length")
    gates = torch.cat(
        [valid, valid.new_zeros(len(STAGE_IDS), seq_len - valid.size(1), 1)], dim=1
    ).unsqueeze(1)
    indices = tuple(
        torch.nonzero(gates[index, 0, :, 0] > 0, as_tuple=False).flatten()
        for index in range(len(STAGE_IDS))
    )
    return gates, indices


def stage_step_gate(step_index: int, total_steps: int) -> float:
    ratio = step_index / total_steps
    if ratio < 0.30:
        return 0.5
    if ratio < 0.70:
        return 1.0
    if ratio < 0.90:
        return 0.3
    return 0.1


def _tensor_rms(value):
    import torch

    return torch.sqrt(torch.mean(value.float().square()))


def rms_match(candidate, reference, *, epsilon: float = 1e-6, max_scale: float = 10.0):
    """Match candidate RMS to reference RMS without trainable parameters."""

    import torch

    candidate_rms = _tensor_rms(candidate)
    reference_rms = _tensor_rms(reference)
    scale = reference_rms / (candidate_rms + float(epsilon))
    scale = torch.clamp(scale, min=0.0, max=float(max_scale))
    return candidate.float() * scale, scale


def rms_cap(candidate, reference, *, ratio: float, epsilon: float = 1e-6):
    """Cap candidate RMS relative to reference while preserving its direction."""

    import torch

    candidate_rms = _tensor_rms(candidate)
    reference_rms = _tensor_rms(reference)
    maximum = float(ratio) * reference_rms
    scale = torch.minimum(
        torch.ones_like(candidate_rms), maximum / (candidate_rms + float(epsilon))
    )
    return candidate.float() * scale, scale


def _rms_ratio(value, reference, valid_length: int) -> float:
    numerator = _tensor_rms(value[:, :valid_length])
    denominator = _tensor_rms(reference[:, :valid_length])
    return float((numerator / (denominator + 1e-6)).item())


def summarize_conditioning_audit(rows: list[dict]) -> dict:
    if not rows:
        return {"row_count": 0}
    keys = (
        "stage_to_global",
        "json_global_to_global",
        "json_stage_to_global",
        "extra_to_global",
        "extra_cap_scale",
    )
    summary = {"row_count": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows]
        summary[key] = {
            "minimum": min(values),
            "mean": sum(values) / len(values),
            "maximum": max(values),
        }
    summary["extra_cap_trigger_rate"] = sum(
        float(row["extra_cap_scale"]) < 0.9999 for row in rows
    ) / len(rows)
    return summary


def staged_model_forward(
    model,
    x,
    t,
    *,
    seq_len: int,
    global_context,
    empty_context,
    stage_contexts,
    json_global_context,
    json_stage_contexts,
    stage_gates,
    stage_indices,
    conditioning_variant: str,
    stage_strength: float,
    stage_ratio_to_global: float,
    json_global_ratio_to_global: float,
    json_stage_ratio_to_global: float,
    condition_extra_cap_ratio: float,
    condition_norm_max_scale: float,
    stage_block_start: int,
    stage_block_stop: int,
    sampling_step: int,
    conditioning_audit: list[dict] | None,
):
    import torch
    from wan.modules.model import sinusoidal_embedding_1d

    device = model.patch_embedding.weight.device
    if model.freqs.device != device:
        model.freqs = model.freqs.to(device)
    patches = [model.patch_embedding(latent.unsqueeze(0)) for latent in x]
    grid_sizes = torch.stack(
        [torch.tensor(item.shape[2:], dtype=torch.long) for item in patches]
    )
    expected_grid = (LATENT_TOKENS, stage_gates.grid_height, stage_gates.grid_width)
    for grid in grid_sizes.tolist():
        if tuple(grid) != expected_grid:
            raise ValueError(f"Wan grid {tuple(grid)} differs from stage grid {expected_grid}")
    tokens = [item.flatten(2).transpose(1, 2) for item in patches]
    seq_lens = torch.tensor([item.size(1) for item in tokens], dtype=torch.long)
    if int(seq_lens.max().item()) > seq_len:
        raise ValueError("patch sequence exceeds padded sequence length")
    hidden = torch.cat(
        [
            torch.cat(
                [item, item.new_zeros(1, seq_len - item.size(1), item.size(2))],
                dim=1,
            )
            for item in tokens
        ]
    )

    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    flattened_t = t.flatten()

    def fp32_context():
        return (
            torch.amp.autocast("cuda", dtype=torch.float32)
            if device.type == "cuda"
            else nullcontext()
        )

    with fp32_context():
        time_embedding = model.time_embedding(
            sinusoidal_embedding_1d(model.freq_dim, flattened_t)
            .unflatten(0, (t.size(0), seq_len))
            .float()
        )
        modulation = model.time_projection(time_embedding).unflatten(2, (6, model.dim))
    if time_embedding.dtype != torch.float32 or modulation.dtype != torch.float32:
        raise TypeError("Wan time embeddings must remain float32")

    shared = {
        "e": modulation,
        "seq_lens": seq_lens,
        "grid_sizes": grid_sizes,
        "freqs": model.freqs,
        "context_lens": None,
    }
    for layer_id, block in enumerate(model.blocks):
        if not stage_block_start <= layer_id < stage_block_stop:
            hidden = block(hidden, context=global_context, **shared)
            continue

        with fp32_context():
            block_modulation = (block.modulation.unsqueeze(0) + modulation).chunk(6, dim=2)
        self_attention = block.self_attn(
            block.norm1(hidden).float()
            * (1 + block_modulation[1].squeeze(2))
            + block_modulation[0].squeeze(2),
            seq_lens,
            grid_sizes,
            model.freqs,
        )
        with fp32_context():
            hidden = hidden + self_attention * block_modulation[2].squeeze(2)

        query = block.norm3(hidden)
        semantic = block.cross_attn(query, global_context, None)
        empty_semantic = block.cross_attn(query, empty_context, None).float()
        stage_residual = torch.zeros_like(semantic, dtype=torch.float32)
        for stage_index, stage_context in enumerate(stage_contexts):
            indices = stage_indices[stage_index]
            if indices.numel() == 0:
                continue
            query_slice = query.index_select(1, indices)
            positive = block.cross_attn(query_slice, stage_context, None).float()
            negative = empty_semantic.index_select(1, indices)
            gate = stage_gates.tensor[stage_index].index_select(1, indices)
            delta = positive - negative
            if conditioning_variant != "baseline":
                delta, _ = rms_match(
                    delta,
                    semantic.index_select(1, indices),
                    max_scale=condition_norm_max_scale,
                )
            stage_residual.index_add_(1, indices, gate.float() * delta)

        if conditioning_variant == "baseline":
            stage_residual.mul_(float(stage_strength))
            finite_tensor(stage_residual, "non-finite direct stage residual")
            hidden = hidden + semantic + stage_residual.to(semantic.dtype)
        else:
            valid_length = int(seq_lens[0].item())
            json_global_residual = torch.zeros_like(semantic, dtype=torch.float32)
            json_stage_residual = torch.zeros_like(semantic, dtype=torch.float32)
            if conditioning_variant == "strong_stage_json":
                if json_global_context is None or len(json_stage_contexts) != len(STAGE_IDS):
                    raise ValueError("strong_stage_json requires all structured contexts")
                json_global_residual = (
                    block.cross_attn(query, json_global_context, None).float()
                    - empty_semantic
                )
                matched, _ = rms_match(
                    json_global_residual[:, :valid_length],
                    semantic[:, :valid_length],
                    max_scale=condition_norm_max_scale,
                )
                json_global_residual[:, :valid_length] = matched
                json_global_residual[:, valid_length:] = 0

                for stage_index, json_context in enumerate(json_stage_contexts):
                    indices = stage_indices[stage_index]
                    if indices.numel() == 0:
                        continue
                    query_slice = query.index_select(1, indices)
                    delta = (
                        block.cross_attn(query_slice, json_context, None).float()
                        - empty_semantic.index_select(1, indices)
                    )
                    delta, _ = rms_match(
                        delta,
                        semantic.index_select(1, indices),
                        max_scale=condition_norm_max_scale,
                    )
                    gate = stage_gates.tensor[stage_index].index_select(1, indices)
                    json_stage_residual.index_add_(1, indices, gate.float() * delta)

            extra = (
                float(stage_ratio_to_global) * stage_residual
                + float(json_global_ratio_to_global) * json_global_residual
                + float(json_stage_ratio_to_global) * json_stage_residual
            )
            capped, cap_scale = rms_cap(
                extra[:, :valid_length],
                semantic[:, :valid_length],
                ratio=condition_extra_cap_ratio,
            )
            extra[:, :valid_length] = capped
            finite_tensor(extra, "non-finite enhanced conditioning residual")
            if conditioning_audit is not None:
                conditioning_audit.append(
                    {
                        "sampling_step": sampling_step,
                        "block": layer_id,
                        "stage_to_global": _rms_ratio(
                            float(stage_ratio_to_global) * stage_residual,
                            semantic,
                            valid_length,
                        ),
                        "json_global_to_global": _rms_ratio(
                            float(json_global_ratio_to_global) * json_global_residual,
                            semantic,
                            valid_length,
                        ),
                        "json_stage_to_global": _rms_ratio(
                            float(json_stage_ratio_to_global) * json_stage_residual,
                            semantic,
                            valid_length,
                        ),
                        "extra_to_global": _rms_ratio(extra, semantic, valid_length),
                        "extra_cap_scale": float(cap_scale.item()),
                    }
                )
            hidden = hidden + semantic + extra.to(semantic.dtype)

        feed_forward = block.ffn(
            block.norm2(hidden).float()
            * (1 + block_modulation[4].squeeze(2))
            + block_modulation[3].squeeze(2)
        )
        with fp32_context():
            hidden = hidden + feed_forward * block_modulation[5].squeeze(2)

    output = model.head(hidden, time_embedding)
    output = model.unpatchify(output, grid_sizes)
    return [item.float() for item in output]


class DirectStageGates:
    def __init__(self, tensor, grid_height: int, grid_width: int):
        self.tensor = tensor
        self.grid_height = grid_height
        self.grid_width = grid_width


def create_scheduler(pipeline, solver: str, steps: int, shift: float):
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    if solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=pipeline.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(steps, device=pipeline.device, shift=shift)
        return scheduler, scheduler.timesteps
    scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=pipeline.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    timesteps, _ = retrieve_timesteps(
        scheduler,
        device=pipeline.device,
        sigmas=get_sampling_sigmas(steps, shift),
    )
    return scheduler, timesteps


def predict_with_direct_stages(
    pipeline,
    latent,
    token_timestep,
    *,
    raw_global_context,
    raw_empty_context,
    global_context,
    empty_context,
    stage_contexts,
    json_global_context,
    json_stage_contexts,
    stage_gates,
    stage_indices,
    seq_len: int,
    guide_scale: float,
    cfg_mode: str,
    phys_guidance_scale: float,
    conditioning_variant: str,
    stage_strength: float,
    stage_ratio_to_global: float,
    json_global_ratio_to_global: float,
    json_stage_ratio_to_global: float,
    condition_extra_cap_ratio: float,
    condition_norm_max_scale: float,
    stage_block_start: int,
    stage_block_stop: int,
    sampling_step: int,
    conditioning_audit: list[dict] | None,
):
    effective_strength = (
        stage_strength / max(guide_scale, 1.0)
        if conditioning_variant == "baseline"
        else 0.0
    )
    conditional = staged_model_forward(
        pipeline.model,
        [latent],
        token_timestep,
        seq_len=seq_len,
        global_context=global_context,
        empty_context=empty_context,
        stage_contexts=stage_contexts,
        json_global_context=json_global_context,
        json_stage_contexts=json_stage_contexts,
        stage_gates=stage_gates,
        stage_indices=stage_indices,
        conditioning_variant=conditioning_variant,
        stage_strength=effective_strength,
        stage_ratio_to_global=stage_ratio_to_global,
        json_global_ratio_to_global=json_global_ratio_to_global,
        json_stage_ratio_to_global=json_stage_ratio_to_global,
        condition_extra_cap_ratio=condition_extra_cap_ratio,
        condition_norm_max_scale=condition_norm_max_scale,
        stage_block_start=stage_block_start,
        stage_block_stop=stage_block_stop,
        sampling_step=sampling_step,
        conditioning_audit=conditioning_audit,
    )[0]
    unconditional = pipeline.model(
        [latent],
        t=token_timestep,
        context=[raw_empty_context],
        seq_len=seq_len,
    )[0]
    finite_tensor(conditional, "non-finite staged conditional prediction")
    finite_tensor(unconditional, "non-finite unconditional prediction")
    if cfg_mode == "legacy":
        return unconditional + guide_scale * (conditional - unconditional)

    global_conditional = pipeline.model(
        [latent],
        t=token_timestep,
        context=[raw_global_context],
        seq_len=seq_len,
    )[0]
    finite_tensor(global_conditional, "non-finite global conditional prediction")
    return (
        unconditional
        + guide_scale * (global_conditional - unconditional)
        + phys_guidance_scale * (conditional - global_conditional)
    )


def generate_with_direct_stages(pipeline, sample: dict, image, args, seed: int):
    import torch
    import torchvision.transforms.functional as transform_functional
    from tqdm import tqdm
    from wan.utils.utils import masks_like

    encoded, token_lengths = encode_direct_contexts(
        pipeline, sample, offload_model=args.offload_model
    )
    if args.offload_model or pipeline.init_on_cpu:
        pipeline.model.to(pipeline.device)
        torch.cuda.empty_cache()
    with torch.amp.autocast("cuda", dtype=pipeline.param_dtype):
        projected = project_direct_contexts(pipeline.model, encoded)

    width, height = FIXED_SIZE
    grid_height = height // (pipeline.vae_stride[1] * pipeline.patch_size[1])
    grid_width = width // (pipeline.vae_stride[2] * pipeline.patch_size[2])
    seq_len = int(
        math.ceil(LATENT_TOKENS * grid_height * grid_width / pipeline.sp_size)
    ) * pipeline.sp_size
    gate_tensor, stage_indices = build_direct_stage_gates(
        sample["temporal_route"],
        height=grid_height,
        width=grid_width,
        seq_len=seq_len,
        device=pipeline.device,
    )
    stage_gates = DirectStageGates(gate_tensor, grid_height, grid_width)
    audit_rows = [] if args.conditioning_audit else None
    json_global_context = projected.get(JSON_GLOBAL_CONTEXT)
    json_stage_contexts = tuple(
        projected[JSON_STAGE_PREFIX + stage_id]
        for stage_id in STAGE_IDS
        if JSON_STAGE_PREFIX + stage_id in projected
    )

    shape = (
        pipeline.vae.model.z_dim,
        LATENT_TOKENS,
        height // pipeline.vae_stride[1],
        width // pipeline.vae_stride[2],
    )
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    latent = torch.randn(*shape, dtype=torch.float32, device=pipeline.device, generator=generator)
    image_condition = None
    if image is None:
        _, mask2 = masks_like([latent], zero=False)
    else:
        image = image.convert("RGB")
        if image.size != FIXED_SIZE:
            raise ValueError(f"I2V image must remain exactly {FIXED_SIZE}")
        image_tensor = (
            transform_functional.to_tensor(image)
            .sub_(0.5)
            .div_(0.5)
            .to(pipeline.device)
            .unsqueeze(1)
        )
        image_condition = pipeline.vae.encode([image_tensor])[0]
        _, mask2 = masks_like([latent], zero=True)
        latent = (1.0 - mask2[0]) * image_condition + mask2[0] * latent

    no_sync_context = getattr(pipeline.model, "no_sync", no_sync)
    with (
        torch.amp.autocast("cuda", dtype=pipeline.param_dtype),
        torch.no_grad(),
        no_sync_context(),
    ):
        scheduler, timesteps = create_scheduler(
            pipeline, args.sample_solver, args.sampling_steps, args.shift
        )
        iterator = tqdm(
            timesteps,
            disable=not args.show_progress,
            desc="DIRECT-STAGE I2V" if image is not None else "DIRECT-STAGE T2V",
        )
        for step_index, timestep_value in enumerate(iterator):
            timestep = torch.stack([timestep_value]).to(pipeline.device)
            token_timestep = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
            token_timestep = torch.cat(
                [
                    token_timestep,
                    token_timestep.new_ones(seq_len - token_timestep.numel()) * timestep,
                ]
            ).unsqueeze(0)
            prediction = predict_with_direct_stages(
                pipeline,
                latent,
                token_timestep,
                raw_global_context=encoded["global_semantic"],
                raw_empty_context=encoded["cfg_negative"],
                global_context=projected["global_semantic"],
                empty_context=projected["cfg_negative"],
                stage_contexts=tuple(projected[stage_id] for stage_id in STAGE_IDS),
                json_global_context=json_global_context,
                json_stage_contexts=json_stage_contexts,
                stage_gates=stage_gates,
                stage_indices=stage_indices,
                seq_len=seq_len,
                guide_scale=args.guide_scale,
                cfg_mode=args.cfg_mode,
                phys_guidance_scale=args.phys_guidance_scale,
                conditioning_variant=args.conditioning_variant,
                stage_strength=(
                    args.stage_strength * stage_step_gate(step_index, args.sampling_steps)
                    if args.conditioning_variant == "baseline"
                    else 0.0
                ),
                stage_ratio_to_global=args.stage_ratio_to_global,
                json_global_ratio_to_global=args.json_global_ratio_to_global,
                json_stage_ratio_to_global=args.json_stage_ratio_to_global,
                condition_extra_cap_ratio=args.condition_extra_cap_ratio,
                condition_norm_max_scale=args.condition_norm_max_scale,
                stage_block_start=args.stage_block_start,
                stage_block_stop=args.stage_block_stop,
                sampling_step=step_index,
                conditioning_audit=audit_rows,
            )
            latent = scheduler.step(
                prediction.unsqueeze(0),
                timestep_value,
                latent.unsqueeze(0),
                return_dict=False,
                generator=generator,
            )[0].squeeze(0)
            if image_condition is not None:
                latent = (1.0 - mask2[0]) * image_condition + mask2[0] * latent

        if args.offload_model:
            pipeline.model.cpu()
            torch.cuda.empty_cache()
        video = pipeline.vae.decode([latent])[0]
    finite_tensor(video, "non-finite decoded video")
    if args.offload_model:
        gc.collect()
    audit = None
    if audit_rows is not None:
        audit = {
            "summary": summarize_conditioning_audit(audit_rows),
            "rows": audit_rows,
        }
    return video, token_lengths, audit


def video_checks(video, frame_num: int) -> dict:
    import torch

    if video is None or video.ndim != 4:
        raise RuntimeError("Wan did not return a CxFxHxW video tensor")
    if int(video.shape[1]) != frame_num:
        raise RuntimeError(f"Wan returned {video.shape[1]} frames, expected {frame_num}")
    if not bool(torch.isfinite(video).all().item()):
        raise RuntimeError("Wan returned non-finite video values")
    return {
        "status": "numeric_pass",
        "shape": [int(value) for value in video.shape],
        "minimum": float(video.min().item()),
        "maximum": float(video.max().item()),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.parse_only:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return 0

    try:
        check_arguments(args)
        sample_ids = selected_sample_ids(args.sample_ids)
        seeds = selected_seeds(args.seeds)
        prepared = load_inputs(args, sample_ids)
        preflight = preflight_manifest(args, prepared)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"[P0 STAGED PREFLIGHT][ERROR] {error}", file=sys.stderr)
        return 2

    if args.check_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    wan_repo = str(Path(args.wan_repo).expanduser().resolve())
    if wan_repo not in sys.path:
        sys.path.insert(0, wan_repo)

    import torch
    from PIL import Image
    from wan.configs import WAN_CONFIGS
    from wan.textimage2video import WanTI2V
    from wan.utils.utils import save_video

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan generation")

    pipeline = WanTI2V(
        config=WAN_CONFIGS["ti2v-5B"],
        checkpoint_dir=str(Path(args.checkpoint_dir).expanduser().resolve()),
        device_id=args.device_id,
        rank=0,
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
        convert_model_dtype=args.convert_model_dtype,
    )

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        f"trace-v3-p0-staged-{args.mode}-%Y%m%d-%H%M%S"
    )
    run_root = Path(args.output_root).expanduser().resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "preflight.json", preflight)

    results = []
    for sample in prepared:
        for seed in seeds:
            artifact = run_root / sample["sample_id"] / args.mode / f"seed_{seed:06d}"
            artifact.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(sample["plan_path"], artifact / "plan.input.json")
            (artifact / "prompt.used.txt").write_text(sample["prompt"], encoding="utf-8")
            (artifact / "negative_prompt.used.txt").write_text(
                sample["negative_prompt"], encoding="utf-8"
            )
            write_json(
                artifact / "stage_prompts.used.json",
                [
                    {
                        "stage_id": stage_id,
                        "c_plus": sample["stage_prompts"][index],
                        "c_minus": sample["negative_prompt"],
                    }
                    for index, stage_id in enumerate(STAGE_IDS)
                ],
            )
            if sample["structured_global_prompt"] is not None:
                write_json(
                    artifact / "structured_prompts.used.json",
                    {
                        "global": sample["structured_global_prompt"],
                        "stages": [
                            {"stage_id": stage_id, "text": text}
                            for stage_id, text in zip(
                                STAGE_IDS, sample["structured_stage_prompts"]
                            )
                        ],
                    },
                )
            write_json(artifact / "temporal_route.used.json", sample["temporal_route"])

            image = None
            incomplete = artifact / "video.incomplete.mp4"
            video_path = artifact / "video.mp4"
            try:
                if sample["image_path"] is not None:
                    with Image.open(sample["image_path"]) as opened:
                        image = opened.convert("RGB")

                video, context_token_lengths, conditioning_audit = generate_with_direct_stages(
                    pipeline,
                    sample,
                    image,
                    args,
                    seed,
                )
                numeric = video_checks(video, args.frame_num)
                save_video(
                    tensor=video[None],
                    save_file=str(incomplete),
                    fps=args.fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                if not incomplete.is_file() or incomplete.stat().st_size == 0:
                    raise RuntimeError("video writer produced no output")
                os.replace(incomplete, video_path)
                write_json(artifact / "video.numeric.json", numeric)
                if conditioning_audit is not None:
                    write_json(artifact / "conditioning.audit.json", conditioning_audit)

                manifest = {
                    "status": "success",
                    "run_id": run_id,
                    "method": preflight["method"],
                    "sample_id": sample["sample_id"],
                    "mode": args.mode,
                    "seed": seed,
                    "plan": str(sample["plan_path"]),
                    "context_token_lengths": context_token_lengths,
                    "temporal_route_file": str(artifact / "temporal_route.used.json"),
                    "stage_c_minus": "one whitespace context for every stage",
                    "stage_strength": args.stage_strength,
                    "conditioning_variant": args.conditioning_variant,
                    "stage_ratio_to_global": (
                        args.stage_ratio_to_global
                        if args.conditioning_variant != "baseline"
                        else None
                    ),
                    "json_global_ratio_to_global": (
                        args.json_global_ratio_to_global
                        if args.conditioning_variant == "strong_stage_json"
                        else None
                    ),
                    "json_stage_ratio_to_global": (
                        args.json_stage_ratio_to_global
                        if args.conditioning_variant == "strong_stage_json"
                        else None
                    ),
                    "condition_extra_cap_ratio": (
                        args.condition_extra_cap_ratio
                        if args.conditioning_variant != "baseline"
                        else None
                    ),
                    "conditioning_audit_summary": (
                        conditioning_audit["summary"]
                        if conditioning_audit is not None
                        else None
                    ),
                    "stage_block_interval": [
                        args.stage_block_start,
                        args.stage_block_stop,
                    ],
                    "image": str(sample["image_path"]) if sample["image_path"] else None,
                    "image_is_model_condition": args.mode == "i2v",
                    "video": str(video_path),
                    "size": [int(video.shape[-1]), int(video.shape[-2])],
                    "frame_num": args.frame_num,
                    "sampling_steps": args.sampling_steps,
                    "sample_solver": args.sample_solver,
                    "guide_scale": args.guide_scale,
                    "cfg_mode": args.cfg_mode,
                    "phys_guidance_scale": (
                        args.phys_guidance_scale
                        if args.cfg_mode == "separate"
                        else None
                    ),
                    "shift": args.shift,
                    "fps": args.fps,
                }
                write_json(artifact / "manifest.json", manifest)
                results.append(manifest)
                del video
            except Exception as error:
                for path in (incomplete, video_path):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                failure = {
                    "status": "failed",
                    "sample_id": sample["sample_id"],
                    "seed": seed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                write_json(artifact / "failure.json", failure)
                results.append(failure)
                if not args.continue_on_error:
                    raise
            finally:
                if image is not None:
                    image.close()
                torch.cuda.empty_cache()

    summary = {
        "run_id": run_id,
        "method": preflight["method"],
        "success_count": sum(item["status"] == "success" for item in results),
        "failure_count": sum(item["status"] == "failed" for item in results),
        "results": results,
    }
    write_json(run_root / "results.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
