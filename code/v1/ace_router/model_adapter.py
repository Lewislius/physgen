from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from .diagnostics import ResidualDiagnostics


def _sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Exact local equivalent of the pinned Wan helper, avoiding import-time CUDA."""

    if dim % 2 != 0:
        raise ValueError("sinusoidal embedding dimension must be even")
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half)),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def _pad_and_project_context(base_model: nn.Module, context: Sequence[torch.Tensor]) -> torch.Tensor:
    if not context:
        raise ValueError("text context batch must be non-empty")
    padded = []
    for item in context:
        if item.ndim != 2:
            raise ValueError(f"each context must have shape [L, C], got {item.shape}")
        if item.size(0) > base_model.text_len:
            raise ValueError(
                f"context length {item.size(0)} exceeds Wan text_len={base_model.text_len}"
            )
        padded.append(
            torch.cat(
                [
                    item,
                    item.new_zeros(
                        base_model.text_len - item.size(0), item.size(1)
                    ),
                ]
            )
        )
    return base_model.text_embedding(torch.stack(padded))


def _assert_finite(tensor: torch.Tensor, message: str) -> None:
    condition = torch.isfinite(tensor).all()
    if tensor.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    elif not bool(condition.item()):
        raise FloatingPointError(message)


def cap_residual_by_semantic_norm(
    candidate_delta: torch.Tensor,
    semantic: torch.Tensor,
    cap_ratio: float | None,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a per-sample, non-amplifying relative L2 cap in FP32."""

    if cap_ratio is not None and (
        not math.isfinite(float(cap_ratio)) or float(cap_ratio) <= 0.0
    ):
        raise ValueError("residual cap ratio must be finite and positive")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("residual cap epsilon must be finite and positive")
    candidate_fp32 = candidate_delta.float()
    semantic_fp32 = semantic.float()
    if candidate_fp32.shape != semantic_fp32.shape:
        raise ValueError("candidate delta and semantic residual shapes must match")
    if candidate_fp32.ndim < 2:
        raise ValueError("ACE residual tensors must include a batch dimension")

    coefficient_shape = (candidate_fp32.shape[0],) + (1,) * (
        candidate_fp32.ndim - 1
    )
    if cap_ratio is None:
        coefficient = candidate_fp32.new_ones(coefficient_shape)
        return candidate_fp32.to(dtype=semantic.dtype), coefficient

    reduce_dims = tuple(range(1, candidate_fp32.ndim))
    candidate_norm = torch.linalg.vector_norm(
        candidate_fp32, dim=reduce_dims, keepdim=True
    )
    semantic_norm = torch.linalg.vector_norm(
        semantic_fp32, dim=reduce_dims, keepdim=True
    )
    coefficient = torch.clamp(
        float(cap_ratio) * semantic_norm / (candidate_norm + float(eps)),
        max=1.0,
    )
    coefficient = torch.where(
        semantic_norm > float(eps), coefficient, torch.zeros_like(coefficient)
    )
    applied = candidate_fp32 * coefficient
    return applied.to(dtype=semantic.dtype), coefficient


def ace_block_forward(
    block: nn.Module,
    x: torch.Tensor,
    *,
    e: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    semantic_context: torch.Tensor,
    positive_context: torch.Tensor,
    counterfactual_context: torch.Tensor | None,
    residual_mode: str,
    context_lens: torch.Tensor | None,
    layer_id: int,
    layer_gate: float,
    step_gate: float,
    lambda0: float,
    step_index: int,
    total_steps: int,
    diagnostic_sink: ResidualDiagnostics | None,
    active_layer_ids: tuple[int, ...],
    residual_cap_ratio: float | None = None,
    residual_cap_eps: float = 1e-12,
) -> torch.Tensor:
    """Run one existing Wan block with ACE at its cross-attention residual.

    This mirrors the local WanAttentionBlock implementation.  It uses the block's
    existing norm/attention/FFN modules and introduces no parameters.
    """

    if e.dtype != torch.float32:
        raise TypeError(f"Wan modulation must be float32, got {e.dtype}")
    with torch.amp.autocast("cuda", dtype=torch.float32):
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
    with torch.amp.autocast("cuda", dtype=torch.float32):
        x = x + self_attention * modulation[2].squeeze(2)

    query = block.norm3(x)
    semantic = block.cross_attn(query, semantic_context, context_lens)
    positive = block.cross_attn(query, positive_context, context_lens)
    if residual_mode == "positive_only":
        counterfactual = None
        residual_reference = semantic
    elif residual_mode == "positive_minus_counterfactual":
        if counterfactual_context is None:
            raise ValueError(
                "positive_minus_counterfactual requires counterfactual context"
            )
        counterfactual = block.cross_attn(
            query, counterfactual_context, context_lens
        )
        residual_reference = counterfactual
    else:
        raise ValueError(f"unsupported residual mode: {residual_mode!r}")
    scale = float(lambda0) * float(layer_gate) * float(step_gate)
    candidate_delta = scale * (positive.float() - residual_reference.float())
    applied_delta, clip_coefficient = cap_residual_by_semantic_norm(
        candidate_delta,
        semantic,
        residual_cap_ratio,
        eps=residual_cap_eps,
    )
    _assert_finite(
        applied_delta,
        f"non-finite ACE residual at step={step_index}, layer={layer_id}",
    )

    if diagnostic_sink is not None and diagnostic_sink.should_collect(
        step_index=step_index,
        total_steps=total_steps,
        layer_id=layer_id,
        active_layer_ids=active_layer_ids,
    ):
        diagnostic_sink.record(
            residual_mode=residual_mode,
            step_index=step_index,
            total_steps=total_steps,
            layer_id=layer_id,
            layer_gate=layer_gate,
            step_gate=step_gate,
            lambda0=lambda0,
            semantic=semantic,
            positive=positive,
            counterfactual=counterfactual,
            candidate_delta=candidate_delta,
            applied_delta=applied_delta,
            clip_coefficient=clip_coefficient,
            residual_cap_ratio=residual_cap_ratio,
        )

    x = x + semantic + applied_delta
    feed_forward = block.ffn(
        block.norm2(x).float() * (1 + modulation[4].squeeze(2))
        + modulation[3].squeeze(2)
    )
    with torch.amp.autocast("cuda", dtype=torch.float32):
        x = x + feed_forward * modulation[5].squeeze(2)
    return x


def ace_model_forward(
    base_model: nn.Module,
    x: Sequence[torch.Tensor],
    t: torch.Tensor,
    context: Sequence[torch.Tensor],
    seq_len: int,
    *,
    causal_pos_context: Sequence[torch.Tensor],
    causal_neg_context: Sequence[torch.Tensor] | None,
    residual_mode: str,
    layer_gates: Sequence[float],
    step_gate: float,
    lambda0: float,
    step_index: int,
    total_steps: int,
    diagnostic_sink: ResidualDiagnostics | None = None,
    y: Sequence[torch.Tensor] | None = None,
    residual_cap_ratio: float | None = None,
    residual_cap_eps: float = 1e-12,
) -> list[torch.Tensor]:
    if base_model.model_type == "i2v" and y is None:
        raise ValueError("Wan i2v model_type requires y")
    batch_size = len(x)
    if len(context) != batch_size or len(causal_pos_context) != batch_size:
        raise ValueError("latent, semantic, and positive batches must have equal size")
    if residual_mode == "positive_minus_counterfactual":
        if causal_neg_context is None or len(causal_neg_context) != batch_size:
            raise ValueError(
                "contrastive residual requires an equal-size counterfactual batch"
            )
    elif residual_mode != "positive_only":
        raise ValueError(f"unsupported residual mode: {residual_mode!r}")
    if len(layer_gates) != len(base_model.blocks):
        raise ValueError(
            f"expected {len(base_model.blocks)} layer gates, got {len(layer_gates)}"
        )

    device = base_model.patch_embedding.weight.device
    if base_model.freqs.device != device:
        base_model.freqs = base_model.freqs.to(device)

    if y is not None:
        if len(y) != batch_size:
            raise ValueError("x and y batches must have equal size")
        x = [torch.cat([item, condition], dim=0) for item, condition in zip(x, y)]

    x = [base_model.patch_embedding(item.unsqueeze(0)) for item in x]
    grid_sizes = torch.stack(
        [torch.tensor(item.shape[2:], dtype=torch.long) for item in x]
    )
    x = [item.flatten(2).transpose(1, 2) for item in x]
    seq_lens = torch.tensor([item.size(1) for item in x], dtype=torch.long)
    if int(seq_lens.max().item()) > seq_len:
        raise ValueError(
            f"patch sequence length {int(seq_lens.max().item())} exceeds seq_len={seq_len}"
        )
    x = torch.cat(
        [
            torch.cat(
                [
                    item,
                    item.new_zeros(1, seq_len - item.size(1), item.size(2)),
                ],
                dim=1,
            )
            for item in x
        ]
    )

    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        batch_timesteps = t.size(0)
        t = t.flatten()
        time_embedding = base_model.time_embedding(
            _sinusoidal_embedding_1d(base_model.freq_dim, t)
            .unflatten(0, (batch_timesteps, seq_len))
            .float()
        )
        modulation = base_model.time_projection(time_embedding).unflatten(
            2, (6, base_model.dim)
        )
        if time_embedding.dtype != torch.float32 or modulation.dtype != torch.float32:
            raise TypeError("Wan time embeddings must remain float32")

    semantic_context = _pad_and_project_context(base_model, context)
    positive_context = _pad_and_project_context(base_model, causal_pos_context)
    counterfactual_context = (
        _pad_and_project_context(base_model, causal_neg_context)
        if causal_neg_context is not None
        else None
    )
    context_lens = None
    shared_arguments: dict[str, Any] = {
        "e": modulation,
        "seq_lens": seq_lens,
        "grid_sizes": grid_sizes,
        "freqs": base_model.freqs,
        "context_lens": context_lens,
    }
    active_layer_ids = tuple(
        index for index, gate in enumerate(layer_gates) if float(gate) > 0.0
    )

    for layer_id, (block, layer_gate_value) in enumerate(
        zip(base_model.blocks, layer_gates)
    ):
        layer_gate_value = float(layer_gate_value)
        if layer_gate_value == 0.0:
            x = block(x, context=semantic_context, **shared_arguments)
        else:
            x = ace_block_forward(
                block,
                x,
                semantic_context=semantic_context,
                positive_context=positive_context,
                counterfactual_context=counterfactual_context,
                residual_mode=residual_mode,
                layer_id=layer_id,
                layer_gate=layer_gate_value,
                step_gate=step_gate,
                lambda0=lambda0,
                step_index=step_index,
                total_steps=total_steps,
                diagnostic_sink=diagnostic_sink,
                active_layer_ids=active_layer_ids,
                residual_cap_ratio=residual_cap_ratio,
                residual_cap_eps=residual_cap_eps,
                **shared_arguments,
            )

    x = base_model.head(x, time_embedding)
    x = base_model.unpatchify(x, grid_sizes)
    return [item.float() for item in x]


class AceModelProxy(nn.Module):
    """One-copy wrapper around an already loaded, frozen local WanModel."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        if getattr(base_model, "model_type", None) != "ti2v":
            raise ValueError(
                f"ACE V1 expects model_type='ti2v', got {base_model.model_type!r}"
            )
        if getattr(base_model, "num_layers", None) != 30:
            raise ValueError(
                f"ACE V1 expects 30 layers, got {getattr(base_model, 'num_layers', None)}"
            )
        if len(base_model.blocks) != 30:
            raise ValueError(f"ACE V1 expects 30 blocks, got {len(base_model.blocks)}")
        if getattr(base_model, "text_len", None) != 512:
            raise ValueError(
                f"ACE V1 expects text_len=512, got {getattr(base_model, 'text_len', None)}"
            )
        if any(parameter.requires_grad for parameter in base_model.parameters()):
            raise ValueError("ACE V1 requires a fully frozen base model")

    @property
    def num_layers(self) -> int:
        return self.base_model.num_layers

    @property
    def model_type(self) -> str:
        return self.base_model.model_type

    def forward(
        self,
        x: Sequence[torch.Tensor],
        t: torch.Tensor,
        context: Sequence[torch.Tensor],
        seq_len: int,
        y: Sequence[torch.Tensor] | None = None,
        *,
        causal_pos_context: Sequence[torch.Tensor] | None = None,
        causal_neg_context: Sequence[torch.Tensor] | None = None,
        residual_mode: str = "positive_minus_counterfactual",
        layer_gates: Sequence[float] | None = None,
        step_gate: float = 0.0,
        lambda0: float = 0.0,
        residual_cap_ratio: float | None = None,
        residual_cap_eps: float = 1e-12,
        step_index: int = 0,
        total_steps: int = 1,
        diagnostic_sink: ResidualDiagnostics | None = None,
    ) -> list[torch.Tensor]:
        gates = list(layer_gates) if layer_gates is not None else [0.0] * 30
        if len(gates) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} layer gates, got {len(gates)}"
            )
        numeric_gates = [float(gate) for gate in gates]
        if any(not math.isfinite(gate) or gate < 0.0 for gate in numeric_gates):
            raise ValueError("layer gates must be finite and non-negative")
        lambda_value = float(lambda0)
        step_value = float(step_gate)
        if not math.isfinite(lambda_value) or lambda_value < 0.0:
            raise ValueError("lambda0 must be finite and non-negative")
        if not math.isfinite(step_value) or step_value < 0.0:
            raise ValueError("step_gate must be finite and non-negative")
        if residual_cap_ratio is not None and (
            not math.isfinite(float(residual_cap_ratio))
            or float(residual_cap_ratio) <= 0.0
        ):
            raise ValueError("residual_cap_ratio must be finite and positive")
        if not math.isfinite(float(residual_cap_eps)) or float(residual_cap_eps) <= 0.0:
            raise ValueError("residual_cap_eps must be finite and positive")
        active = (
            lambda_value != 0.0
            and step_value != 0.0
            and any(gate != 0.0 for gate in numeric_gates)
        )
        if not active:
            return self.base_model(x=x, t=t, context=context, seq_len=seq_len, y=y)
        if residual_mode not in {"positive_only", "positive_minus_counterfactual"}:
            raise ValueError(f"unsupported residual mode: {residual_mode!r}")
        if causal_pos_context is None:
            raise ValueError("active ACE forward requires a positive causal context")
        if residual_mode == "positive_minus_counterfactual" and causal_neg_context is None:
            raise ValueError("contrastive ACE forward requires counterfactual context")
        if total_steps <= 0 or step_index < 0 or step_index >= total_steps:
            raise ValueError("invalid ACE step index/total steps")
        return ace_model_forward(
            self.base_model,
            x,
            t,
            context,
            seq_len,
            y=y,
            causal_pos_context=causal_pos_context,
            causal_neg_context=causal_neg_context,
            residual_mode=residual_mode,
            layer_gates=numeric_gates,
            step_gate=step_value,
            lambda0=lambda_value,
            residual_cap_ratio=residual_cap_ratio,
            residual_cap_eps=residual_cap_eps,
            step_index=step_index,
            total_steps=total_steps,
            diagnostic_sink=diagnostic_sink,
        )
