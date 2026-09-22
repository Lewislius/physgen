from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .trace_runtime import RoutingRuntime


@dataclass(frozen=True)
class TraceResidualResult:
    candidate: torch.Tensor
    token_capped: torch.Tensor
    applied: torch.Tensor
    token_cap_fraction: float
    global_cap_coefficient: float


def _positive_ratio(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def cap_per_token(
    candidate: torch.Tensor,
    semantic: torch.Tensor,
    *,
    ratio: float = 0.10,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap each token feature norm relative to the semantic token norm."""

    cap_ratio = _positive_ratio(ratio, "token cap ratio")
    epsilon = _positive_ratio(eps, "token cap epsilon")
    if candidate.shape != semantic.shape or candidate.ndim != 3:
        raise ValueError("candidate and semantic must share shape [B,L,C]")
    candidate_fp32 = candidate.float()
    semantic_fp32 = semantic.float()
    candidate_norm = torch.linalg.vector_norm(candidate_fp32, dim=-1, keepdim=True)
    semantic_norm = torch.linalg.vector_norm(semantic_fp32, dim=-1, keepdim=True)
    coefficient = torch.clamp(
        cap_ratio * semantic_norm / (candidate_norm + epsilon),
        max=1.0,
    )
    coefficient = torch.where(
        semantic_norm > epsilon,
        coefficient,
        torch.zeros_like(coefficient),
    )
    return candidate_fp32 * coefficient, coefficient


def cap_global(
    candidate: torch.Tensor,
    semantic: torch.Tensor,
    *,
    ratio: float = 0.02,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap the full sequence residual per sample without amplification."""

    cap_ratio = _positive_ratio(ratio, "global cap ratio")
    epsilon = _positive_ratio(eps, "global cap epsilon")
    if candidate.shape != semantic.shape or candidate.ndim != 3:
        raise ValueError("candidate and semantic must share shape [B,L,C]")
    candidate_fp32 = candidate.float()
    semantic_fp32 = semantic.float()
    candidate_norm = torch.linalg.vector_norm(candidate_fp32, dim=(1, 2), keepdim=True)
    semantic_norm = torch.linalg.vector_norm(semantic_fp32, dim=(1, 2), keepdim=True)
    coefficient = torch.clamp(
        cap_ratio * semantic_norm / (candidate_norm + epsilon),
        max=1.0,
    )
    coefficient = torch.where(
        semantic_norm > epsilon,
        coefficient,
        torch.zeros_like(coefficient),
    )
    return candidate_fp32 * coefficient, coefficient


def _assert_finite(tensor: torch.Tensor, message: str) -> None:
    condition = torch.isfinite(tensor).all()
    if tensor.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    elif not bool(condition.item()):
        raise FloatingPointError(message)


def build_trace_residual(
    block: nn.Module,
    query: torch.Tensor,
    semantic: torch.Tensor,
    runtime: RoutingRuntime,
    *,
    context_lens: torch.Tensor | None = None,
    scale: float,
    token_cap_ratio: float = 0.10,
    global_cap_ratio: float = 0.02,
    eps: float = 1e-12,
) -> TraceResidualResult:
    """Apply stage-local cross-attention and two convex routing layers."""

    scale_value = float(scale)
    if not math.isfinite(scale_value) or scale_value < 0.0:
        raise ValueError("scale must be finite and non-negative")
    if query.shape != semantic.shape or query.ndim != 3:
        raise ValueError("query and semantic must share shape [B,L,C]")
    if query.size(1) != runtime.seq_len:
        raise ValueError(
            f"runtime seq_len={runtime.seq_len} differs from query length={query.size(1)}"
        )

    candidate = torch.zeros_like(semantic, dtype=torch.float32)
    for stage_runtime in runtime.stages:
        indices = stage_runtime.active_indices.to(device=query.device)
        if indices.numel() == 0:
            continue
        query_slice = query.index_select(1, indices)
        positive = block.cross_attn(
            query_slice,
            stage_runtime.contexts.positive,
            context_lens,
        ).float()
        negative = torch.zeros_like(positive)
        for weight, context in zip(
            stage_runtime.contexts.violation_weights,
            stage_runtime.contexts.violations,
        ):
            negative.add_(
                float(weight)
                * block.cross_attn(query_slice, context, context_lens).float()
            )
        gate = stage_runtime.token_gate.to(device=query.device).index_select(1, indices)
        local = gate.float() * (positive - negative)
        candidate.index_add_(1, indices, local)

    candidate.mul_(scale_value)
    token_capped, token_coefficients = cap_per_token(
        candidate,
        semantic,
        ratio=token_cap_ratio,
        eps=eps,
    )
    applied, global_coefficient = cap_global(
        token_capped,
        semantic,
        ratio=global_cap_ratio,
        eps=eps,
    )
    _assert_finite(applied, "TRACE produced a non-finite residual")
    capped_fraction = float((token_coefficients < 1.0).float().mean().item())
    return TraceResidualResult(
        candidate=candidate,
        token_capped=token_capped,
        applied=applied.to(dtype=semantic.dtype),
        token_cap_fraction=capped_fraction,
        global_cap_coefficient=float(global_coefficient.min().item()),
    )


def trace_block_forward(
    block: nn.Module,
    x: torch.Tensor,
    *,
    e: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    semantic_context: torch.Tensor,
    runtime: RoutingRuntime,
    context_lens: torch.Tensor | None,
    layer_id: int,
    layer_gate: float,
    step_gate: float,
    lambda0: float,
    step_index: int,
    token_cap_ratio: float = 0.10,
    global_cap_ratio: float = 0.02,
    cap_eps: float = 1e-12,
    **_: Any,
) -> torch.Tensor:
    """Mirror one frozen Wan block and add a training-free TRACE residual."""

    if e.dtype != torch.float32:
        raise TypeError(f"Wan modulation must be float32, got {e.dtype}")
    def fp32_context():
        return (
            torch.amp.autocast("cuda", dtype=torch.float32)
            if x.device.type == "cuda"
            else nullcontext()
        )

    with fp32_context():
        modulation = (block.modulation.unsqueeze(0) + e).chunk(6, dim=2)
    if modulation[0].dtype != torch.float32:
        raise TypeError("Wan modulation unexpectedly left float32")
    self_attention = block.self_attn(
        block.norm1(x).float() * (1 + modulation[1].squeeze(2))
        + modulation[0].squeeze(2),
        seq_lens,
        grid_sizes,
        freqs,
    )
    with fp32_context():
        x = x + self_attention * modulation[2].squeeze(2)

    query = block.norm3(x)
    semantic = block.cross_attn(query, semantic_context, context_lens)
    scale = float(lambda0) * float(layer_gate) * float(step_gate)
    result = build_trace_residual(
        block,
        query,
        semantic,
        runtime,
        context_lens=context_lens,
        scale=scale,
        token_cap_ratio=token_cap_ratio,
        global_cap_ratio=global_cap_ratio,
        eps=cap_eps,
    )
    if runtime.reader is not None:
        runtime.reader.observe(
            step_index=step_index,
            layer_id=layer_id,
            semantic=semantic,
            candidate=result.candidate,
            applied=result.applied,
            token_cap_fraction=result.token_cap_fraction,
            global_cap_coefficient=result.global_cap_coefficient,
        )
    x = x + semantic + result.applied
    feed_forward = block.ffn(
        block.norm2(x).float() * (1 + modulation[4].squeeze(2))
        + modulation[3].squeeze(2)
    )
    with fp32_context():
        return x + feed_forward * modulation[5].squeeze(2)
