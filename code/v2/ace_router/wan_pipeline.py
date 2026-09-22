from __future__ import annotations

import gc
import math
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from wan.textimage2video import WanTI2V
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import best_output_size, masks_like

from .trace_compile import CompiledTrace
from .trace_config import TraceWriterConfig
from .controllers import AuditReader
from .trace_masks import build_stage_spatial_masks
from .trace_model_adapter import TraceModelProxy, project_trace_context_batches
from .trace_runtime import build_routing_runtime


@dataclass(frozen=True)
class TraceGenerationOutput:
    video: torch.Tensor
    seed: int
    width: int
    height: int
    context_token_lengths: dict[str, int]


@contextmanager
def _no_sync():
    yield


def _finite(tensor: torch.Tensor, message: str) -> None:
    condition = torch.isfinite(tensor).all()
    if tensor.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    elif not bool(condition.item()):
        raise FloatingPointError(message)


class TraceWanTI2V(WanTI2V):
    """Wan TI2V inference with the training-free fixed-five TRACE writer."""

    def __init__(
        self,
        config: Any,
        checkpoint_dir: str,
        *,
        device_id: int = 0,
        rank: int = 0,
        t5_cpu: bool = True,
        init_on_cpu: bool = True,
        convert_model_dtype: bool = True,
    ) -> None:
        super().__init__(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=t5_cpu,
            init_on_cpu=init_on_cpu,
            convert_model_dtype=convert_model_dtype,
        )
        self.model = TraceModelProxy(self.model)

    def _scheduler(self, solver: str, steps: int, shift: float):
        if solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(steps, device=self.device, shift=shift)
            return scheduler, scheduler.timesteps
        if solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            timesteps, _ = retrieve_timesteps(
                scheduler,
                device=self.device,
                sigmas=get_sampling_sigmas(steps, shift),
            )
            return scheduler, timesteps
        raise ValueError(f"unsupported solver {solver!r}")

    def _encode_once(
        self,
        texts: Sequence[str],
        names: Sequence[str],
        *,
        offload_model: bool,
    ) -> tuple[list[torch.Tensor], dict[str, int]]:
        if len(texts) != len(names):
            raise ValueError("text/name counts differ")
        tokenizer = self.text_encoder.tokenizer
        cleaned = [tokenizer._clean(text) if tokenizer.clean else text for text in texts]
        lengths = [
            len(ids)
            for ids in tokenizer.tokenizer(
                cleaned,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )["input_ids"]
        ]
        if any(length > self.config.text_len for length in lengths):
            raise ValueError(f"TRACE text exceeds Wan text_len={self.config.text_len}")
        if self.t5_cpu:
            encoded = self.text_encoder(list(texts), torch.device("cpu"))
            encoded = [item.to(self.device) for item in encoded]
        else:
            self.text_encoder.model.to(self.device)
            encoded = self.text_encoder(list(texts), self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        return encoded, dict(zip(names, lengths))

    def _raw_token_lengths(
        self,
        texts: Sequence[str],
        names: Sequence[str],
    ) -> dict[str, int]:
        if len(texts) != len(names):
            raise ValueError("text/name counts differ")
        tokenizer = self.text_encoder.tokenizer
        cleaned = [tokenizer._clean(text) if tokenizer.clean else text for text in texts]
        lengths = [
            len(ids)
            for ids in tokenizer.tokenizer(
                cleaned,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )["input_ids"]
        ]
        if any(length > self.config.text_len for length in lengths):
            raise ValueError(f"TRACE text exceeds Wan text_len={self.config.text_len}")
        return dict(zip(names, lengths))

    def generate_trace(
        self,
        *,
        compiled: CompiledTrace,
        image: Image.Image | None,
        size: tuple[int, int],
        max_area: int,
        frame_num: int = 97,
        sampling_steps: int = 50,
        sample_solver: str = "unipc",
        guide_scale: float = 3.5,
        shift: float = 5.0,
        seed: int = 42,
        spatial_mode: str = "time",
        writer_config: TraceWriterConfig | None = None,
        reader: AuditReader | None = None,
        offload_model: bool = True,
        show_progress: bool = True,
    ) -> TraceGenerationOutput:
        if frame_num != 97 or compiled.frames != 25:
            raise ValueError("TRACE v1 requires 97 output frames / 25 latent tokens")
        cfg = writer_config or TraceWriterConfig(
            crossfade_tokens=compiled.crossfade_tokens
        )
        if cfg.crossfade_tokens != compiled.crossfade_tokens:
            raise ValueError("writer config and compiled crossfade differ")
        if sampling_steps <= 0 or seed < 0:
            raise ValueError("sampling_steps must be positive and seed non-negative")
        if spatial_mode not in {"time", "spacetime"}:
            raise ValueError("spatial_mode must be time or spacetime")

        if image is None:
            width, height = int(size[0]), int(size[1])
            if width * height != max_area:
                raise ValueError("T2V size must match max_area")
        else:
            width, height = best_output_size(
                image.width,
                image.height,
                self.vae_stride[2] * self.patch_size[2],
                self.vae_stride[1] * self.patch_size[1],
                max_area,
            )

        if cfg.lambda0 == 0.0:
            # Explicit base-Wan ablation: do not encode, project, or execute any
            # stage/violation context. A whitespace negative bypasses Wan's
            # built-in negative prompt sentinel while tokenizing as empty text.
            token_lengths = self._raw_token_lengths(
                [compiled.plan.global_semantic, " "],
                ["global_semantic", "cfg_negative"],
            )
            video = super().generate(
                input_prompt=compiled.plan.global_semantic,
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
            return TraceGenerationOutput(
                video=video,
                seed=seed,
                width=width,
                height=height,
                context_token_lengths=token_lengths,
            )

        text_map = compiled.context_texts()
        names = list(text_map) + ["cfg_negative"]
        texts = list(text_map.values()) + [""]
        encoded, token_lengths = self._encode_once(
            texts,
            names,
            offload_model=offload_model,
        )
        by_name = dict(zip(names, encoded))
        semantic = by_name.pop("global_semantic")
        null_context = by_name.pop("cfg_negative")
        grid_height = height // (self.vae_stride[1] * self.patch_size[1])
        grid_width = width // (self.vae_stride[2] * self.patch_size[2])
        seq_len = (
            int(
                math.ceil(
                    compiled.frames * grid_height * grid_width / self.sp_size
                )
            )
            * self.sp_size
        )

        if offload_model or self.init_on_cpu:
            self.model.to(self.device)
            torch.cuda.empty_cache()
        raw_stage_batches = {name: (tensor,) for name, tensor in by_name.items()}
        with torch.amp.autocast("cuda", dtype=self.param_dtype):
            projected = project_trace_context_batches(
                self.model.base_model,
                raw_stage_batches,
            )
        spatial_masks = None
        if spatial_mode == "spacetime":
            spatial_masks = build_stage_spatial_masks(
                compiled.plan,
                height=grid_height,
                width=grid_width,
                device=self.device,
            )
        runtime = build_routing_runtime(
            compiled,
            projected,
            height=grid_height,
            width=grid_width,
            seq_len=seq_len,
            spatial_mode=spatial_mode,
            spatial_masks=spatial_masks,
            reader=reader,
        )
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
        if image is None:
            video = self._trace_t2v(**common)
        else:
            video = self._trace_i2v(image=image, **common)
        return TraceGenerationOutput(
            video=video,
            seed=seed,
            width=width,
            height=height,
            context_token_lengths=token_lengths,
        )

    def _predict(
        self,
        latent: torch.Tensor,
        token_timestep: torch.Tensor,
        *,
        semantic: torch.Tensor,
        null_context: torch.Tensor,
        runtime,
        cfg: TraceWriterConfig,
        step_index: int,
        sampling_steps: int,
        guide_scale: float,
    ) -> torch.Tensor:
        conditional = self.model(
            [latent],
            t=token_timestep,
            context=[semantic],
            seq_len=runtime.seq_len,
            runtime=runtime,
            layer_gates=cfg.layer_gates(),
            step_gate=cfg.step_gates(sampling_steps)[step_index],
            lambda0=cfg.lambda0,
            step_index=step_index,
            total_steps=sampling_steps,
            token_cap_ratio=cfg.token_cap_ratio,
            global_cap_ratio=cfg.global_cap_ratio,
        )[0]
        unconditional = self.model(
            [latent],
            t=token_timestep,
            context=[null_context],
            seq_len=runtime.seq_len,
            lambda0=0.0,
        )[0]
        _finite(conditional, "non-finite TRACE conditional prediction")
        _finite(unconditional, "non-finite unconditional prediction")
        return unconditional + guide_scale * (conditional - unconditional)

    def _trace_t2v(
        self,
        *,
        semantic,
        null_context,
        runtime,
        width,
        height,
        frame_num,
        sampling_steps,
        sample_solver,
        guide_scale,
        shift,
        seed,
        cfg,
        offload_model,
        show_progress,
    ) -> torch.Tensor:
        shape = (
            self.vae.model.z_dim,
            (frame_num - 1) // self.vae_stride[0] + 1,
            height // self.vae_stride[1],
            width // self.vae_stride[2],
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latent = torch.randn(*shape, device=self.device, generator=generator)
        _, mask2 = masks_like([latent], zero=False)
        no_sync = getattr(self.model, "no_sync", _no_sync)
        with torch.amp.autocast("cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():
            scheduler, timesteps = self._scheduler(sample_solver, sampling_steps, shift)
            iterator = tqdm(timesteps, disable=not show_progress, desc="TRACE T2V")
            for step_index, value in enumerate(iterator):
                timestep = torch.stack([value]).to(self.device)
                token_timestep = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                token_timestep = torch.cat(
                    [
                        token_timestep,
                        token_timestep.new_ones(runtime.seq_len - token_timestep.numel())
                        * timestep,
                    ]
                ).unsqueeze(0)
                prediction = self._predict(
                    latent,
                    token_timestep,
                    semantic=semantic,
                    null_context=null_context,
                    runtime=runtime,
                    cfg=cfg,
                    step_index=step_index,
                    sampling_steps=sampling_steps,
                    guide_scale=guide_scale,
                )
                latent = scheduler.step(
                    prediction.unsqueeze(0),
                    value,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=generator,
                )[0].squeeze(0)
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            video = self.vae.decode([latent])[0]
        _finite(video, "non-finite decoded video")
        if offload_model:
            gc.collect()
        return video

    def _trace_i2v(
        self,
        *,
        image: Image.Image,
        semantic,
        null_context,
        runtime,
        width,
        height,
        frame_num,
        sampling_steps,
        sample_solver,
        guide_scale,
        shift,
        seed,
        cfg,
        offload_model,
        show_progress,
    ) -> torch.Tensor:
        image = image.convert("RGB")
        scale = max(width / image.width, height / image.height)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
        left, top = (image.width - width) // 2, (image.height - height) // 2
        image = image.crop((left, top, left + width, top + height))
        image_tensor = TF.to_tensor(image).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)
        shape = (
            self.vae.model.z_dim,
            (frame_num - 1) // self.vae_stride[0] + 1,
            height // self.vae_stride[1],
            width // self.vae_stride[2],
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latent = torch.randn(*shape, device=self.device, generator=generator)
        known = self.vae.encode([image_tensor])[0]
        _, mask2 = masks_like([latent], zero=True)
        latent = (1.0 - mask2[0]) * known + mask2[0] * latent
        no_sync = getattr(self.model, "no_sync", _no_sync)
        with torch.amp.autocast("cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():
            scheduler, timesteps = self._scheduler(sample_solver, sampling_steps, shift)
            iterator = tqdm(timesteps, disable=not show_progress, desc="TRACE I2V")
            for step_index, value in enumerate(iterator):
                timestep = torch.stack([value]).to(self.device)
                token_timestep = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                token_timestep = torch.cat(
                    [
                        token_timestep,
                        token_timestep.new_ones(runtime.seq_len - token_timestep.numel())
                        * timestep,
                    ]
                ).unsqueeze(0)
                prediction = self._predict(
                    latent,
                    token_timestep,
                    semantic=semantic,
                    null_context=null_context,
                    runtime=runtime,
                    cfg=cfg,
                    step_index=step_index,
                    sampling_steps=sampling_steps,
                    guide_scale=guide_scale,
                )
                latent = scheduler.step(
                    prediction.unsqueeze(0),
                    value,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=generator,
                )[0].squeeze(0)
                latent = (1.0 - mask2[0]) * known + mask2[0] * latent
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            video = self.vae.decode([latent])[0]
        _finite(video, "non-finite decoded video")
        if offload_model:
            gc.collect()
        return video
