from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext

import torch
import torch.nn as nn

from v2.ace_router.trace_model_adapter import (
    _sinusoidal_embedding_1d,
    pad_and_project_context,
    project_trace_context_batches,
)

from .runtime import RoutingRuntimeV3
from .writer import trace_block_forward_v3


def trace_model_forward_v3(
    base_model: nn.Module,
    x: Sequence[torch.Tensor],
    t: torch.Tensor,
    context: Sequence[torch.Tensor],
    seq_len: int,
    *,
    runtime: RoutingRuntimeV3,
    layer_gates: Sequence[float],
    step_gate: float,
    lambda0: float,
    step_index: int,
    total_steps: int,
    y: Sequence[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    batch_size = len(x)
    if len(context) != batch_size:
        raise ValueError("latent and context batch sizes differ")
    if len(layer_gates) != len(base_model.blocks):
        raise ValueError("layer gate count differs from Wan block count")
    if runtime.seq_len != seq_len:
        raise ValueError("runtime and model sequence lengths differ")
    if total_steps <= 0 or not 0 <= step_index < total_steps:
        raise ValueError("invalid step index")
    runtime.cap_ledger.reset()

    device = base_model.patch_embedding.weight.device
    if base_model.freqs.device != device:
        base_model.freqs = base_model.freqs.to(device)
    if y is not None:
        if len(y) != batch_size:
            raise ValueError("latent and image condition batches differ")
        x = [torch.cat([item, condition], dim=0) for item, condition in zip(x, y)]
    patches = [base_model.patch_embedding(item.unsqueeze(0)) for item in x]
    grid_sizes = torch.stack([torch.tensor(item.shape[2:], dtype=torch.long) for item in patches])
    expected = (runtime.frames, runtime.height, runtime.width)
    for index, grid in enumerate(grid_sizes.tolist()):
        if tuple(grid) != expected:
            raise ValueError(f"sample {index} grid {tuple(grid)} differs from runtime {expected}")
    tokens = [item.flatten(2).transpose(1, 2) for item in patches]
    seq_lens = torch.tensor([item.size(1) for item in tokens], dtype=torch.long)
    if int(seq_lens.max().item()) > seq_len:
        raise ValueError("patch sequence exceeds padded sequence")
    hidden = torch.cat(
        [
            torch.cat(
                [item, item.new_zeros(1, seq_len - item.size(1), item.size(2))], dim=1
            )
            for item in tokens
        ]
    )
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    flattened_t = t.flatten()
    fp32 = torch.amp.autocast("cuda", dtype=torch.float32) if device.type == "cuda" else nullcontext()
    with fp32:
        time_embedding = base_model.time_embedding(
            _sinusoidal_embedding_1d(base_model.freq_dim, flattened_t)
            .unflatten(0, (t.size(0), seq_len))
            .float()
        )
        modulation = base_model.time_projection(time_embedding).unflatten(2, (6, base_model.dim))
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
        if float(layer_gate) == 0.0:
            hidden = block(hidden, context=semantic_context, **shared)
        else:
            hidden = trace_block_forward_v3(
                block,
                hidden,
                semantic_context=semantic_context,
                runtime=runtime,
                layer_id=layer_id,
                layer_gate=float(layer_gate),
                step_gate=float(step_gate),
                lambda0=float(lambda0),
                step_index=step_index,
                **shared,
            )
    output = base_model.head(hidden, time_embedding)
    output = base_model.unpatchify(output, grid_sizes)
    return [item.float() for item in output]


class TraceModelProxyV3(nn.Module):
    """One-copy wrapper; the upstream Wan model and weights remain unchanged."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        if getattr(base_model, "model_type", None) != "ti2v":
            raise ValueError("TRACE v3 expects a Wan ti2v model")
        if len(getattr(base_model, "blocks", ())) != 30:
            raise ValueError("TRACE v3 expects 30 Wan blocks")
        if getattr(base_model, "text_len", None) != 512:
            raise ValueError("TRACE v3 expects Wan text_len=512")
        if any(parameter.requires_grad for parameter in base_model.parameters()):
            raise ValueError("TRACE v3 requires frozen Wan parameters")

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
        runtime: RoutingRuntimeV3 | None = None,
        layer_gates: Sequence[float] | None = None,
        step_gate: float = 0.0,
        lambda0: float = 0.0,
        step_index: int = 0,
        total_steps: int = 1,
    ) -> list[torch.Tensor]:
        gates = list(layer_gates) if layer_gates is not None else [0.0] * self.num_layers
        if len(gates) != self.num_layers:
            raise ValueError("invalid layer gate count")
        numeric = [float(value) for value in gates]
        if any(not math.isfinite(x) or x < 0 for x in numeric):
            raise ValueError("layer gates must be finite and non-negative")
        active = float(lambda0) != 0.0 and float(step_gate) != 0.0 and any(x != 0 for x in numeric)
        if not active:
            return self.base_model(x=x, t=t, context=context, seq_len=seq_len, y=y)
        if runtime is None:
            raise ValueError("active TRACE v3 forward requires runtime")
        return trace_model_forward_v3(
            self.base_model,
            x,
            t,
            context,
            seq_len,
            y=y,
            runtime=runtime,
            layer_gates=numeric,
            step_gate=float(step_gate),
            lambda0=float(lambda0),
            step_index=step_index,
            total_steps=total_steps,
        )


__all__ = [
    "TraceModelProxyV3",
    "pad_and_project_context",
    "project_trace_context_batches",
]
