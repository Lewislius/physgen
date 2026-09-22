from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as torch_functional
from PIL import Image

from wan.textimage2video import WanTI2V

from v2.ace_router.wan_pipeline import TraceWanTI2V as TraceWanTI2VV2

from .cap_ledger import cap_prediction_delta
from .compile import CompiledTraceV3
from .config import TraceWriterConfigV3
from .controllers import AuditReader
from .model_adapter import TraceModelProxyV3, project_trace_context_batches
from .runtime import RoutingRuntimeV3, build_routing_runtime_v3


@dataclass(frozen=True)
class TraceGenerationOutputV3:
    video: torch.Tensor
    seed: int
    width: int
    height: int
    context_token_lengths: dict[str, int]
    support_manifest: dict[str, Any]
    cfg_audit: tuple[dict[str, Any], ...]


def _finite(tensor: torch.Tensor, message: str) -> None:
    condition = torch.isfinite(tensor).all()
    if tensor.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    elif not bool(condition.item()):
        raise FloatingPointError(message)


class TraceWanTI2VV3(TraceWanTI2VV2):
    """Wan TI2V with v3 contexts, routing, aggregate caps and strict CFG cap."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        base_model = self.model.base_model
        self.model = TraceModelProxyV3(base_model)
        self._v3_cfg_audit: list[dict[str, Any]] = []

    def _prediction_support(
        self,
        runtime: RoutingRuntimeV3,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        source = runtime.prediction_support().to(prediction.device).unsqueeze(1)
        resized = torch_functional.interpolate(
            source,
            size=(prediction.shape[1], prediction.shape[2], prediction.shape[3]),
            mode="trilinear",
            align_corners=False,
        )
        return resized[:, 0].clamp(0, 1)

    def _predict(
        self,
        latent: torch.Tensor,
        token_timestep: torch.Tensor,
        *,
        semantic: torch.Tensor,
        null_context: torch.Tensor,
        runtime: RoutingRuntimeV3,
        cfg: TraceWriterConfigV3,
        step_index: int,
        sampling_steps: int,
        guide_scale: float,
    ) -> torch.Tensor:
        # Estimated mode has no shadow conditional. Scale the injected writer
        # before CFG so its configured cap remains meaningful after guidance.
        effective_lambda = float(cfg.lambda0)
        if cfg.cap_mode == "aggregate_estimated":
            effective_lambda /= max(float(guide_scale), 1.0)
        trace_conditional = self.model(
            [latent],
            t=token_timestep,
            context=[semantic],
            seq_len=runtime.seq_len,
            runtime=runtime,
            layer_gates=cfg.layer_gates(),
            step_gate=cfg.step_gates(sampling_steps)[step_index],
            lambda0=effective_lambda,
            step_index=step_index,
            total_steps=sampling_steps,
        )[0]
        base_conditional = None
        if cfg.cap_mode == "aggregate_strict":
            base_conditional = self.model(
                [latent],
                t=token_timestep,
                context=[semantic],
                seq_len=runtime.seq_len,
                lambda0=0.0,
            )[0]
        unconditional = self.model(
            [latent],
            t=token_timestep,
            context=[null_context],
            seq_len=runtime.seq_len,
            lambda0=0.0,
        )[0]
        _finite(trace_conditional, "non-finite TRACE v3 conditional")
        _finite(unconditional, "non-finite unconditional")
        if base_conditional is None:
            prediction = unconditional + float(guide_scale) * (trace_conditional - unconditional)
            self._v3_cfg_audit.append(
                {
                    "step_index": int(step_index),
                    "mode": cfg.cap_mode,
                    "exact": False,
                    "guide_scale": float(guide_scale),
                    "effective_lambda": effective_lambda,
                }
            )
            return prediction
        _finite(base_conditional, "non-finite base conditional")
        base_cfg = unconditional + float(guide_scale) * (base_conditional - unconditional)
        delta_cfg = float(guide_scale) * (trace_conditional - base_conditional)
        support = self._prediction_support(runtime, delta_cfg)
        capped_delta, record = cap_prediction_delta(
            delta_cfg,
            base_cfg,
            support,
            inside_ratio=cfg.cfg_cap_ratio,
            outside_ratio=cfg.cfg_outside_cap_ratio,
        )
        audit = {
            "step_index": int(step_index),
            "mode": cfg.cap_mode,
            "exact": True,
            "guide_scale": float(guide_scale),
            **record,
        }
        self._v3_cfg_audit.append(audit)
        if runtime.reader is not None:
            runtime.reader.observe_cfg(**audit)
        return base_cfg + capped_delta

    def generate_trace_v3(
        self,
        *,
        compiled: CompiledTraceV3,
        image: Image.Image | None,
        size: tuple[int, int],
        max_area: int,
        frame_num: int = 97,
        sampling_steps: int = 50,
        sample_solver: str = "unipc",
        guide_scale: float = 3.5,
        shift: float = 5.0,
        seed: int = 42,
        writer_config: TraceWriterConfigV3 | None = None,
        reader: AuditReader | None = None,
        offload_model: bool = True,
        show_progress: bool = True,
    ) -> TraceGenerationOutputV3:
        cfg = writer_config or TraceWriterConfigV3()
        if frame_num != 97 or compiled.frames != 25:
            raise ValueError("TRACE v3 requires 97 output frames / 25 latent tokens")
        if compiled.mode == "i2v" and image is None:
            raise ValueError("M3/i2v requires its resolved first frame")
        if compiled.mode == "t2v" and image is not None:
            raise ValueError("M4/t2v must not receive an image")
        if image is None:
            width, height = map(int, size)
            if width * height != max_area:
                raise ValueError("T2V size must match max_area")
        else:
            from wan.utils.utils import best_output_size

            width, height = best_output_size(
                image.width,
                image.height,
                self.vae_stride[2] * self.patch_size[2],
                self.vae_stride[1] * self.patch_size[1],
                max_area,
            )
        if cfg.lambda0 == 0.0:
            token_lengths = self._raw_token_lengths(
                [compiled.contexts.global_anchor, " "], ["global_semantic", "cfg_negative"]
            )
            video = WanTI2V.generate(
                self,
                input_prompt=compiled.contexts.global_anchor,
                img=image,
                size=(width, height),
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=" ",
                seed=seed,
                offload_model=offload_model,
            )
            if video is None:
                raise RuntimeError("base Wan returned no rank-zero video")
            return TraceGenerationOutputV3(video, seed, width, height, token_lengths, {}, ())

        text_map = compiled.context_texts()
        names = list(text_map) + ["cfg_negative"]
        texts = list(text_map.values()) + [""]
        encoded, token_lengths = self._encode_once(texts, names, offload_model=offload_model)
        by_name = dict(zip(names, encoded))
        semantic = by_name.pop("global_semantic")
        null_context = by_name.pop("cfg_negative")
        grid_height = height // (self.vae_stride[1] * self.patch_size[1])
        grid_width = width // (self.vae_stride[2] * self.patch_size[2])
        seq_len = int(math.ceil(compiled.frames * grid_height * grid_width / self.sp_size)) * self.sp_size
        if offload_model or self.init_on_cpu:
            self.model.to(self.device)
            torch.cuda.empty_cache()
        raw_batches = {name: (tensor,) for name, tensor in by_name.items()}
        with torch.amp.autocast("cuda", dtype=self.param_dtype):
            projected = project_trace_context_batches(self.model.base_model, raw_batches)
        runtime = build_routing_runtime_v3(
            compiled,
            projected,
            height=grid_height,
            width=grid_width,
            seq_len=seq_len,
            config=cfg,
            reader=reader,
        )
        self._v3_cfg_audit = []
        common = dict(
            semantic=semantic,
            null_context=null_context,
            runtime=runtime,
            width=width,
            height=height,
            frame_num=frame_num,
            sampling_steps=sampling_steps,
            sample_solver=sample_solver,
            guide_scale=guide_scale,
            shift=shift,
            seed=seed,
            cfg=cfg,
            offload_model=offload_model,
            show_progress=show_progress,
        )
        video = self._trace_t2v(**common) if image is None else self._trace_i2v(image=image, **common)
        return TraceGenerationOutputV3(
            video=video,
            seed=seed,
            width=width,
            height=height,
            context_token_lengths=token_lengths,
            support_manifest=dict(runtime.support_manifest),
            cfg_audit=tuple(self._v3_cfg_audit),
        )
