from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from .controllers import AuditReader
from .trace_compile import CompiledTrace
from .trace_model_adapter import project_trace_context_batches
from .trace_runtime import RoutingRuntime, build_routing_runtime


TextEncoder = Callable[[Sequence[str]], Sequence[torch.Tensor]]


@dataclass(frozen=True)
class PreparedTrace:
    semantic_context: tuple[torch.Tensor, ...]
    runtime: RoutingRuntime
    token_lengths: Mapping[str, int]


def prepare_trace_contexts(
    compiled: CompiledTrace,
    *,
    text_encoder: TextEncoder,
    base_model: nn.Module,
    height: int,
    width: int,
    seq_len: int,
    spatial_mode: str = "time",
    spatial_masks: torch.Tensor | None = None,
    reader: AuditReader | None = None,
) -> PreparedTrace:
    """Call T5 once for all semantic, positive, and violation texts."""

    text_map = compiled.context_texts()
    names = tuple(text_map)
    texts = [text_map[name] for name in names]
    encoded = list(text_encoder(texts))
    if len(encoded) != len(texts):
        raise RuntimeError(
            f"text encoder returned {len(encoded)} contexts for {len(texts)} texts"
        )
    for index, tensor in enumerate(encoded):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"encoded context {index} must have shape [T,C]")
    raw_by_name = dict(zip(names, encoded))
    semantic = (raw_by_name.pop("global_semantic"),)
    raw_stage_batches = {name: (tensor,) for name, tensor in raw_by_name.items()}
    projected = project_trace_context_batches(base_model, raw_stage_batches)
    runtime = build_routing_runtime(
        compiled,
        projected,
        height=height,
        width=width,
        seq_len=seq_len,
        spatial_mode=spatial_mode,
        spatial_masks=spatial_masks,
        reader=reader,
    )
    return PreparedTrace(
        semantic_context=semantic,
        runtime=runtime,
        token_lengths={name: int(tensor.size(0)) for name, tensor in zip(names, encoded)},
    )

