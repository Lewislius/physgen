from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from .compile import CompiledTraceV3


def verify_compiled_contract(compiled: CompiledTraceV3) -> dict[str, Any]:
    pair_failures = [
        pair.stage_id
        for pair in compiled.contexts.stage_pairs
        if pair.pair_quality != "minimal_verified"
    ]
    unresolved_counts = [
        slot.entity_id
        for slot in compiled.entity_ledger.slots
        if slot.cardinality.kind in {"group", "unknown"}
    ]
    protected_missing = [
        item
        for item in compiled.plan.protected_predicates
        if item not in compiled.contexts.protected_coverage
    ]
    return {
        "status": "pass" if not pair_failures and not protected_missing else "warning",
        "pair_failures": pair_failures,
        "unresolved_exact_counts": unresolved_counts,
        "protected_predicates_missing_from_compiler": protected_missing,
        "closed_world": asdict(compiled.entity_ledger.closed_world),
        "lifecycle_rule_count": len(compiled.entity_ledger.lifecycles),
        "conservation_rule_count": len(compiled.entity_ledger.conservation),
        "limitations": [
            "Pixel-space entity count and identity require a configured detector/tracker.",
            "Material conservation is reported as an observable proxy, not physical mass.",
        ],
    }


def verify_video_tensor(video: torch.Tensor, compiled: CompiledTraceV3) -> dict[str, Any]:
    """Model-independent temporal checks over decoded [C,T,H,W] output."""

    if video.ndim != 4:
        raise ValueError("video must have shape [C,T,H,W]")
    value = video.detach().float().cpu()
    if not bool(torch.isfinite(value).all().item()):
        return {"status": "fail", "reason": "non_finite_video"}
    adjacent = (value[:, 1:] - value[:, :-1]).abs().mean((0, 2, 3))
    second = (value[:, 2:] - 2 * value[:, 1:-1] + value[:, :-2]).abs().mean((0, 2, 3))
    boundary_latents = sorted(
        {
            frame
            for item in compiled.temporal_manifest.get("boundaries", [])
            for frame in item.get("frames", [])
        }
    )
    boundary_output_frames = sorted(
        {min(value.size(1) - 1, int(frame) * 4) for frame in boundary_latents}
    )
    boundary_mad = [
        float(adjacent[max(0, frame - 1)].item())
        for frame in boundary_output_frames
        if adjacent.numel()
    ]
    return {
        "status": "pass",
        "frame_count": int(value.size(1)),
        "adjacent_mad_mean": float(adjacent.mean().item()) if adjacent.numel() else 0.0,
        "adjacent_mad_max": float(adjacent.max().item()) if adjacent.numel() else 0.0,
        "second_difference_mean": float(second.mean().item()) if second.numel() else 0.0,
        "second_difference_max": float(second.max().item()) if second.numel() else 0.0,
        "boundary_output_frames": boundary_output_frames,
        "boundary_mad": boundary_mad,
        "semantic_checks": "requires configured detector/tracker; not fabricated",
    }
