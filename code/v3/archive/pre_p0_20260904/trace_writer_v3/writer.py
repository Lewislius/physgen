from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn

from .cap_ledger import masked_energy
from .runtime import RoutingRuntimeV3, StageRuntime


@dataclass(frozen=True)
class TraceResidualResult:
    candidate: torch.Tensor
    applied: torch.Tensor
    token_cap_fraction: float
    layer_cap_coefficient: float
    group_cap_coefficient: float


def _operator(
    block: nn.Module,
    query: torch.Tensor,
    stage: StageRuntime,
    context_lens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = block.cross_attn(query, stage.contexts.positive, context_lens).float()
    negative = torch.zeros_like(positive)
    for weight, context in zip(
        stage.contexts.violation_weights,
        stage.contexts.violations,
    ):
        negative.add_(float(weight) * block.cross_attn(query, context, context_lens).float())
    return positive, negative, positive - negative


def _saliency_gate(
    runtime: RoutingRuntimeV3,
    stage: StageRuntime,
    positive: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if runtime.saliency_state is None:
        return positive.new_ones(1, indices.numel(), 1)
    return runtime.saliency_state.local_mask(
        stage.stage_id, positive, indices, runtime.seq_len
    ).to(positive)


def _cosine_morph(
    left: torch.Tensor,
    right: torch.Tensor,
    left_gate: torch.Tensor,
    right_gate: torch.Tensor,
    *,
    low: float,
    high: float,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    numerator = (left * right).sum(-1, keepdim=True)
    denominator = (
        torch.linalg.vector_norm(left, dim=-1, keepdim=True)
        * torch.linalg.vector_norm(right, dim=-1, keepdim=True)
    ).clamp_min(1e-12)
    cosine = (numerator / denominator).clamp(-1, 1)
    if not enabled:
        return left_gate * left + right_gate * right, cosine
    alpha = ((cosine - float(low)) / (float(high) - float(low))).clamp(0, 1)
    both = (left_gate > 0) & (right_gate > 0)
    alpha = torch.where(both, alpha, torch.ones_like(alpha))
    common = 0.5 * (left + right)
    safe_left = common + alpha * (left - common)
    safe_right = common + alpha * (right - common)
    return left_gate * safe_left + right_gate * safe_right, cosine


def _temporal_derivative_cap(
    candidate: torch.Tensor,
    semantic: torch.Tensor,
    runtime: RoutingRuntimeV3,
) -> torch.Tensor:
    ratio = runtime.config.temporal_derivative_ratio
    if ratio <= 0:
        return candidate
    valid_length = runtime.frames * runtime.height * runtime.width
    value = candidate[:, :valid_length].reshape(
        candidate.size(0), runtime.frames, runtime.height * runtime.width, candidate.size(2)
    ).clone()
    sem = semantic[:, :valid_length].reshape_as(value).float()
    active = runtime.active_support[:, :valid_length].reshape(
        candidate.size(0), runtime.frames, runtime.height * runtime.width, 1
    )
    boundary_frames = sorted({f for boundary in runtime.boundaries for f in boundary.temporal_frames})
    for frame in boundary_frames:
        if frame <= 0:
            continue
        difference = value[:, frame] - value[:, frame - 1]
        difference_norm = torch.linalg.vector_norm(difference.float(), dim=-1, keepdim=True)
        reference = 0.5 * (
            torch.linalg.vector_norm(sem[:, frame], dim=-1, keepdim=True)
            + torch.linalg.vector_norm(sem[:, frame - 1], dim=-1, keepdim=True)
        )
        coefficient = (float(ratio) * reference / difference_norm.clamp_min(1e-12)).clamp(max=1)
        value[:, frame] = (
            value[:, frame - 1] + coefficient * difference
        ) * active[:, frame]
    result = candidate.clone()
    result[:, :valid_length] = value.reshape(candidate.size(0), valid_length, candidate.size(2))
    return result


def _token_cap(
    candidate: torch.Tensor,
    semantic: torch.Tensor,
    support: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, float]:
    candidate_norm = torch.linalg.vector_norm(candidate.float(), dim=-1, keepdim=True)
    semantic_norm = torch.linalg.vector_norm(semantic.float(), dim=-1, keepdim=True)
    coefficient = (float(ratio) * semantic_norm / candidate_norm.clamp_min(1e-12)).clamp(max=1)
    coefficient = torch.where(candidate_norm > 0, coefficient, torch.ones_like(coefficient))
    active = support > 0
    fraction = float(((coefficient < 1) & active).sum().item()) / max(1, int(active.sum().item()))
    return (candidate.float() * coefficient).to(candidate.dtype), fraction


def _layer_cap(
    candidate: torch.Tensor,
    semantic: torch.Tensor,
    support: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, float]:
    current = torch.sqrt(masked_energy(candidate, support).clamp_min(1e-24))
    reference = torch.sqrt(masked_energy(semantic, support).clamp_min(1e-24))
    coefficient = (float(ratio) * reference / current.clamp_min(1e-12)).clamp(max=1)
    coefficient = torch.where(current > 0, coefficient, torch.ones_like(coefficient))
    return (candidate.float() * coefficient).to(candidate.dtype), float(coefficient.min().item())


def build_trace_residual_v3(
    block: nn.Module,
    query: torch.Tensor,
    semantic: torch.Tensor,
    runtime: RoutingRuntimeV3,
    *,
    context_lens: torch.Tensor | None,
    scale: float,
    step_index: int,
    layer_id: int,
) -> TraceResidualResult:
    if query.shape != semantic.shape or query.ndim != 3:
        raise ValueError("query and semantic must share [B,L,C]")
    candidate = torch.zeros_like(semantic, dtype=torch.float32)

    for stage in runtime.stages:
        indices = stage.core_indices.to(query.device)
        if indices.numel() == 0:
            continue
        positive, negative, difference = _operator(
            block, query.index_select(1, indices), stage, context_lens
        )
        gate = stage.core_gate.to(query.device).index_select(1, indices)
        gate = gate * _saliency_gate(runtime, stage, positive, indices)
        candidate.index_add_(1, indices, gate.float() * difference)
        if runtime.reader is not None:
            runtime.reader.observe_operator(
                step_index=step_index,
                layer_id=layer_id,
                stage_id=stage.stage_id,
                region="core",
                positive=positive,
                negative=negative,
                difference=difference,
            )

    for boundary in runtime.boundaries:
        indices = boundary.indices.to(query.device)
        if indices.numel() == 0:
            continue
        left_stage = runtime.stages[boundary.left_index]
        right_stage = runtime.stages[boundary.right_index]
        sliced = query.index_select(1, indices)
        lp, ln, left = _operator(block, sliced, left_stage, context_lens)
        rp, rn, right = _operator(block, sliced, right_stage, context_lens)
        left_gate = boundary.left_gate.to(query.device).index_select(1, indices)
        right_gate = boundary.right_gate.to(query.device).index_select(1, indices)
        left_gate *= _saliency_gate(runtime, left_stage, lp, indices)
        right_gate *= _saliency_gate(runtime, right_stage, rp, indices)
        local, cosine = _cosine_morph(
            left,
            right,
            left_gate,
            right_gate,
            low=runtime.config.cosine_low,
            high=runtime.config.cosine_high,
            enabled=runtime.config.boundary_morph == "cosine_trust",
        )
        candidate.index_add_(1, indices, local)
        if runtime.reader is not None:
            runtime.reader.observe_boundary(
                step_index=int(step_index),
                layer_id=int(layer_id),
                left_stage=left_stage.stage_id,
                right_stage=right_stage.stage_id,
                cosine_mean=float(cosine.mean().item()),
                cosine_min=float(cosine.min().item()),
                token_count=int(indices.numel()),
            )

    candidate = _temporal_derivative_cap(candidate, semantic, runtime)
    candidate.mul_(float(scale))
    support = (torch.linalg.vector_norm(candidate, dim=-1, keepdim=True) > 0).to(candidate.dtype)
    token_capped, token_fraction = _token_cap(
        candidate, semantic, support, runtime.config.token_cap_ratio
    )
    layer_capped, layer_coefficient = _layer_cap(
        token_capped, semantic, support, runtime.config.layer_cap_ratio
    )
    if runtime.config.cap_mode == "legacy":
        applied = layer_capped
        group_coefficient = 1.0
    else:
        applied, group_coefficient = runtime.cap_ledger.consume(
            layer_capped, semantic, support, layer_id=layer_id
        )
    if not bool(torch.isfinite(applied).all().item()):
        raise FloatingPointError("TRACE v3 produced a non-finite residual")
    if runtime.reader is not None:
        runtime.reader.observe_layer(
            step_index=int(step_index),
            layer_id=int(layer_id),
            semantic_norm=float(torch.linalg.vector_norm(semantic.float()).item()),
            candidate_norm=float(torch.linalg.vector_norm(candidate).item()),
            applied_norm=float(torch.linalg.vector_norm(applied.float()).item()),
            token_cap_fraction=token_fraction,
            layer_cap_coefficient=layer_coefficient,
            group_cap_coefficient=group_coefficient,
        )
    return TraceResidualResult(
        candidate=candidate,
        applied=applied.to(semantic.dtype),
        token_cap_fraction=token_fraction,
        layer_cap_coefficient=layer_coefficient,
        group_cap_coefficient=group_coefficient,
    )


def trace_block_forward_v3(
    block: nn.Module,
    x: torch.Tensor,
    *,
    e: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    semantic_context: torch.Tensor,
    runtime: RoutingRuntimeV3,
    context_lens: torch.Tensor | None,
    layer_id: int,
    layer_gate: float,
    step_gate: float,
    lambda0: float,
    step_index: int,
    **_: Any,
) -> torch.Tensor:
    if e.dtype != torch.float32:
        raise TypeError(f"Wan modulation must be float32, got {e.dtype}")

    def fp32_context():
        return torch.amp.autocast("cuda", dtype=torch.float32) if x.device.type == "cuda" else nullcontext()

    with fp32_context():
        modulation = (block.modulation.unsqueeze(0) + e).chunk(6, dim=2)
    self_attention = block.self_attn(
        block.norm1(x).float() * (1 + modulation[1].squeeze(2)) + modulation[0].squeeze(2),
        seq_lens,
        grid_sizes,
        freqs,
    )
    with fp32_context():
        x = x + self_attention * modulation[2].squeeze(2)
    query = block.norm3(x)
    semantic = block.cross_attn(query, semantic_context, context_lens)
    result = build_trace_residual_v3(
        block,
        query,
        semantic,
        runtime,
        context_lens=context_lens,
        scale=float(lambda0) * float(layer_gate) * float(step_gate),
        step_index=step_index,
        layer_id=layer_id,
    )
    x = x + semantic + result.applied
    feed_forward = block.ffn(
        block.norm2(x).float() * (1 + modulation[4].squeeze(2)) + modulation[3].squeeze(2)
    )
    with fp32_context():
        return x + feed_forward * modulation[5].squeeze(2)
