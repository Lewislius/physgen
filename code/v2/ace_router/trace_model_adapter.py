from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext

import torch
import torch.nn as nn

from .trace_runtime import RoutingRuntime
from .trace_writer import trace_block_forward


def _sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("sinusoidal embedding dimension must be even")
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half)),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def pad_and_project_context(
    base_model: nn.Module,
    context: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Pad one raw T5 context batch and apply Wan's frozen text projection."""

    if not context:
        raise ValueError("text context batch must be non-empty")
    padded = []
    for index, item in enumerate(context):
        if item.ndim != 2:
            raise ValueError(
                f"context[{index}] must have shape [T,C], got {tuple(item.shape)}"
            )
        if item.size(0) > base_model.text_len:
            raise ValueError(
                f"context length {item.size(0)} exceeds Wan text_len={base_model.text_len}"
            )
        padded.append(
            torch.cat(
                [
                    item,
                    item.new_zeros(base_model.text_len - item.size(0), item.size(1)),
                ]
            )
        )
    return base_model.text_embedding(torch.stack(padded))


def project_trace_context_batches(
    base_model: nn.Module,
    raw_context_batches: Mapping[str, Sequence[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Project all stage contexts without adding modules or parameters."""

    return {
        name: pad_and_project_context(base_model, batch)
        for name, batch in raw_context_batches.items()
    }


def trace_model_forward(
    base_model: nn.Module,
    x: Sequence[torch.Tensor],
    t: torch.Tensor,
    context: Sequence[torch.Tensor],
    seq_len: int,
    *,
    runtime: RoutingRuntime,
    layer_gates: Sequence[float],
    step_gate: float,
    lambda0: float,
    step_index: int,
    total_steps: int,
    y: Sequence[torch.Tensor] | None = None,
    token_cap_ratio: float = 0.10,
    global_cap_ratio: float = 0.02,
    cap_eps: float = 1e-12,
) -> list[torch.Tensor]:
    """Run the frozen Wan model with TRACE on selected blocks."""

    batch_size = len(x)
    if len(context) != batch_size:
        raise ValueError("latent and semantic context batches must match")
    if len(layer_gates) != len(base_model.blocks):
        raise ValueError(
            f"expected {len(base_model.blocks)} layer gates, got {len(layer_gates)}"
        )
    if runtime.seq_len != seq_len:
        raise ValueError("runtime and model seq_len must match")
    if total_steps <= 0 or not 0 <= step_index < total_steps:
        raise ValueError("invalid step_index/total_steps")

    device = base_model.patch_embedding.weight.device
    if base_model.freqs.device != device:
        base_model.freqs = base_model.freqs.to(device)
    if y is not None:
        if len(y) != batch_size:
            raise ValueError("x and y batches must match")
        x = [torch.cat([item, condition], dim=0) for item, condition in zip(x, y)]

    patches = [base_model.patch_embedding(item.unsqueeze(0)) for item in x]
    grid_sizes = torch.stack(
        [torch.tensor(item.shape[2:], dtype=torch.long) for item in patches]
    )
    expected_grid = (runtime.frames, runtime.height, runtime.width)
    for index, grid in enumerate(grid_sizes.tolist()):
        if tuple(grid) != expected_grid:
            raise ValueError(
                f"sample {index} grid {tuple(grid)} differs from runtime {expected_grid}"
            )
    tokens = [item.flatten(2).transpose(1, 2) for item in patches]
    seq_lens = torch.tensor([item.size(1) for item in tokens], dtype=torch.long)
    if int(seq_lens.max().item()) > seq_len:
        raise ValueError("patch sequence exceeds padded seq_len")
    hidden = torch.cat(
        [
            torch.cat(
                [
                    item,
                    item.new_zeros(1, seq_len - item.size(1), item.size(2)),
                ],
                dim=1,
            )
            for item in tokens
        ]
    )

    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    batch_timesteps = t.size(0)
    flattened_t = t.flatten()
    fp32_context = (
        torch.amp.autocast("cuda", dtype=torch.float32)
        if device.type == "cuda"
        else nullcontext()
    )
    with fp32_context:
        time_embedding = base_model.time_embedding(
            _sinusoidal_embedding_1d(base_model.freq_dim, flattened_t)
            .unflatten(0, (batch_timesteps, seq_len))
            .float()
        )
        modulation = base_model.time_projection(time_embedding).unflatten(
            2, (6, base_model.dim)
        )
        if time_embedding.dtype != torch.float32 or modulation.dtype != torch.float32:
            raise TypeError("Wan time embeddings and modulation must remain float32")

    semantic_context = pad_and_project_context(base_model, context)
    shared = {
        "e": modulation,
        "seq_lens": seq_lens,
        "grid_sizes": grid_sizes,
        "freqs": base_model.freqs,
        "context_lens": None,
    }
    for layer_id, (block, layer_gate) in enumerate(zip(base_model.blocks, layer_gates)):
        gate = float(layer_gate)
        if gate == 0.0:
            hidden = block(hidden, context=semantic_context, **shared)
        else:
            hidden = trace_block_forward(
                block,
                hidden,
                semantic_context=semantic_context,
                runtime=runtime,
                layer_id=layer_id,
                layer_gate=gate,
                step_gate=step_gate,
                lambda0=lambda0,
                step_index=step_index,
                token_cap_ratio=token_cap_ratio,
                global_cap_ratio=global_cap_ratio,
                cap_eps=cap_eps,
                **shared,
            )

    output = base_model.head(hidden, time_embedding)
    output = base_model.unpatchify(output, grid_sizes)
    return [item.float() for item in output]


class TraceModelProxy(nn.Module):
    """One-copy wrapper around an already loaded and fully frozen Wan model."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        if getattr(base_model, "model_type", None) != "ti2v":
            raise ValueError("TRACE expects the Wan ti2v model")
        if len(getattr(base_model, "blocks", ())) != 30:
            raise ValueError("TRACE preset expects 30 Wan blocks")
        if getattr(base_model, "num_layers", None) != 30:
            raise ValueError("TRACE preset expects Wan num_layers=30")
        if getattr(base_model, "text_len", None) != 512:
            raise ValueError("TRACE expects Wan text_len=512")
        if any(parameter.requires_grad for parameter in base_model.parameters()):
            raise ValueError("TRACE requires a fully frozen base model")

    @property
    def num_layers(self) -> int:
        return int(self.base_model.num_layers)

    @property
    def model_type(self) -> str:
        return str(self.base_model.model_type)

    def forward(
        self,
        x: Sequence[torch.Tensor],
        t: torch.Tensor,
        context: Sequence[torch.Tensor],
        seq_len: int,
        y: Sequence[torch.Tensor] | None = None,
        *,
        runtime: RoutingRuntime | None = None,
        layer_gates: Sequence[float] | None = None,
        step_gate: float = 0.0,
        lambda0: float = 0.0,
        step_index: int = 0,
        total_steps: int = 1,
        token_cap_ratio: float = 0.10,
        global_cap_ratio: float = 0.02,
        cap_eps: float = 1e-12,
    ) -> list[torch.Tensor]:
        gates = list(layer_gates) if layer_gates is not None else [0.0] * self.num_layers
        if len(gates) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layer gates")
        numeric_gates = [float(value) for value in gates]
        if any(not math.isfinite(value) or value < 0.0 for value in numeric_gates):
            raise ValueError("layer gates must be finite and non-negative")
        lambda_value = float(lambda0)
        step_value = float(step_gate)
        if not math.isfinite(lambda_value) or lambda_value < 0.0:
            raise ValueError("lambda0 must be finite and non-negative")
        if not math.isfinite(step_value) or step_value < 0.0:
            raise ValueError("step_gate must be finite and non-negative")
        active = (
            lambda_value != 0.0
            and step_value != 0.0
            and any(value != 0.0 for value in numeric_gates)
        )
        if not active:
            return self.base_model(x=x, t=t, context=context, seq_len=seq_len, y=y)
        if runtime is None:
            raise ValueError("active TRACE forward requires a RoutingRuntime")
        return trace_model_forward(
            self.base_model,
            x,
            t,
            context,
            seq_len,
            y=y,
            runtime=runtime,
            layer_gates=numeric_gates,
            step_gate=step_value,
            lambda0=lambda_value,
            step_index=step_index,
            total_steps=total_steps,
            token_cap_ratio=token_cap_ratio,
            global_cap_ratio=global_cap_ratio,
            cap_eps=cap_eps,
        )
