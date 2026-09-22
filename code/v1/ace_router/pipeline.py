from __future__ import annotations

import gc
import math
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.distributed as dist
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

from .diagnostics import ResidualDiagnostics
from .model_adapter import AceModelProxy


# Wan treats an empty n_prompt as a request for its built-in negative prompt.
# Whitespace bypasses that sentinel and is normalized to empty by its tokenizer.
WAN_EMPTY_CFG_PROMPT = " "


@dataclass
class AceGenerationOutput:
    video: torch.Tensor
    seed: int
    width: int
    height: int
    context_token_lengths: dict[str, int]
    step_gates: list[float]
    step_elapsed_seconds: list[float]
    first_frame_reanchor_count: int


@contextmanager
def _noop_no_sync():
    yield


def _assert_finite_async(tensor: torch.Tensor, message: str) -> None:
    condition = torch.isfinite(tensor).all()
    if tensor.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    elif not bool(condition.item()):
        raise FloatingPointError(message)


def _step_timer_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def _resolve_step_timings(
    timer_pairs: Sequence[tuple[torch.cuda.Event, torch.cuda.Event]],
) -> list[float]:
    torch.cuda.synchronize()
    return [start.elapsed_time(end) / 1000.0 for start, end in timer_pairs]


class AceWanTI2V(WanTI2V):
    """WanTI2V with an external, training-free ACE conditional forward."""

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
        t5_fsdp: bool = False,
        dit_fsdp: bool = False,
        use_sp: bool = False,
    ) -> None:
        if t5_fsdp or dit_fsdp or use_sp:
            raise ValueError("ACE V1 does not support T5/DiT FSDP or sequence parallel")
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
        self.model = AceModelProxy(self.model)

    def _raw_token_lengths(self, texts: Sequence[str]) -> list[int]:
        wrapper = self.text_encoder.tokenizer
        cleaned = [wrapper._clean(text) if wrapper.clean else text for text in texts]
        encoded = wrapper.tokenizer(
            cleaned,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        return [len(item) for item in encoded["input_ids"]]

    def _encode_ace_contexts(
        self,
        texts: Sequence[str],
        *,
        names: Sequence[str],
        offload_model: bool,
    ) -> tuple[list[torch.Tensor], dict[str, int]]:
        if len(texts) != len(names):
            raise ValueError("context texts and names must have equal length")
        raw_lengths = self._raw_token_lengths(texts)
        too_long = {
            name: length
            for name, length in zip(names, raw_lengths)
            if length > self.config.text_len
        }
        if too_long:
            raise ValueError(
                f"contexts exceed Wan text_len={self.config.text_len}: {too_long}"
            )

        if self.t5_cpu:
            contexts = self.text_encoder(list(texts), torch.device("cpu"))
            contexts = [context.to(self.device) for context in contexts]
        else:
            self.text_encoder.model.to(self.device)
            contexts = self.text_encoder(list(texts), self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        encoded_lengths = [int(context.size(0)) for context in contexts]
        if encoded_lengths != raw_lengths:
            raise RuntimeError(
                f"unexpected T5 truncation: raw={raw_lengths}, encoded={encoded_lengths}"
            )
        return contexts, dict(zip(names, raw_lengths))

    def _scheduler(
        self,
        sample_solver: str,
        sampling_steps: int,
        shift: float,
    ) -> tuple[Any, torch.Tensor]:
        if sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            return scheduler, scheduler.timesteps
        if sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                scheduler, device=self.device, sigmas=sampling_sigmas
            )
            return scheduler, timesteps
        raise ValueError(f"unsupported sample solver: {sample_solver}")

    def generate_baseline(
        self,
        *,
        input_prompt: str,
        image: Image.Image | None = None,
        size: tuple[int, int] = (1280, 704),
        max_area: int = 1280 * 704,
        frame_num: int = 97,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
        seed: int = 42,
        offload_model: bool = True,
    ) -> AceGenerationOutput:
        """Run the unmodified WanTI2V generation path without ACE contexts."""

        if not isinstance(input_prompt, str) or not input_prompt.strip():
            raise ValueError("input_prompt must be a non-empty string")
        if frame_num <= 0 or (frame_num - 1) % 4 != 0:
            raise ValueError("frame_num must satisfy 4n+1")
        if sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive")
        if seed < 0:
            raise ValueError("baseline batch inference requires a non-negative seed")
        names = ("semantic", "cfg_negative")
        lengths = self._raw_token_lengths([input_prompt, ""])
        too_long = {
            name: length
            for name, length in zip(names, lengths)
            if length > self.config.text_len
        }
        if too_long:
            raise ValueError(
                f"contexts exceed Wan text_len={self.config.text_len}: {too_long}"
            )

        if image is None:
            width, height = int(size[0]), int(size[1])
            if width * height != max_area:
                raise ValueError("T2V max_area must equal width * height")
        else:
            width, height = best_output_size(
                image.width,
                image.height,
                self.vae_stride[2] * self.patch_size[2],
                self.vae_stride[1] * self.patch_size[1],
                max_area,
            )

        video = super().generate(
            input_prompt=input_prompt,
            img=image,
            size=(width, height),
            max_area=max_area,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=WAN_EMPTY_CFG_PROMPT,
            seed=seed,
            offload_model=offload_model,
        )
        if video is None:
            raise RuntimeError("Wan baseline returned no rank-zero video")
        return AceGenerationOutput(
            video=video,
            seed=seed,
            width=width,
            height=height,
            context_token_lengths=dict(zip(names, lengths)),
            step_gates=[],
            step_elapsed_seconds=[],
            first_frame_reanchor_count=(sampling_steps + 1 if image is not None else 0),
        )

    def generate_ace(
        self,
        *,
        input_prompt: str,
        positive_text: str,
        counterfactual_text: str | None,
        residual_mode: str = "positive_minus_counterfactual",
        layer_gates: Sequence[float],
        step_gates: Sequence[float],
        lambda0: float,
        residual_cap_ratio: float | None = None,
        residual_cap_eps: float = 1e-12,
        image: Image.Image | None = None,
        size: tuple[int, int] = (1280, 704),
        max_area: int = 1280 * 704,
        frame_num: int = 97,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
        cfg_negative_mode: str = "empty",
        seed: int = 42,
        offload_model: bool = True,
        diagnostic_sink: ResidualDiagnostics | None = None,
        show_progress: bool = True,
    ) -> AceGenerationOutput:
        if frame_num <= 0 or (frame_num - 1) % 4 != 0:
            raise ValueError("frame_num must satisfy 4n+1")
        if sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive")
        if len(step_gates) != sampling_steps:
            raise ValueError(
                f"expected {sampling_steps} step gates, got {len(step_gates)}"
            )
        if len(layer_gates) != self.model.num_layers:
            raise ValueError(
                f"expected {self.model.num_layers} layer gates, got {len(layer_gates)}"
            )
        numeric_layer_gates = [float(value) for value in layer_gates]
        numeric_step_gates = [float(value) for value in step_gates]
        if any(
            not math.isfinite(value) or value < 0.0
            for value in numeric_layer_gates
        ):
            raise ValueError("layer gates must be finite and non-negative")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in numeric_step_gates
        ):
            raise ValueError("step gates must be finite and non-negative")
        if not math.isfinite(lambda0) or lambda0 < 0.0:
            raise ValueError("lambda0 must be finite and non-negative")
        if residual_cap_ratio is not None and (
            not math.isfinite(residual_cap_ratio) or residual_cap_ratio <= 0.0
        ):
            raise ValueError("residual_cap_ratio must be finite and positive")
        if not math.isfinite(residual_cap_eps) or residual_cap_eps <= 0.0:
            raise ValueError("residual_cap_eps must be finite and positive")
        if not math.isfinite(shift) or shift <= 0.0:
            raise ValueError("shift must be finite and positive")
        if not math.isfinite(guide_scale) or guide_scale < 0.0:
            raise ValueError("guide_scale must be finite and non-negative")
        if cfg_negative_mode not in {"empty", "wan_default"}:
            raise ValueError(
                "cfg_negative_mode must be either 'empty' or 'wan_default'"
            )
        if residual_mode not in {"positive_only", "positive_minus_counterfactual"}:
            raise ValueError(f"unsupported residual mode: {residual_mode!r}")
        required_texts = [
            ("input_prompt", input_prompt),
            ("positive_text", positive_text),
        ]
        if residual_mode == "positive_minus_counterfactual":
            required_texts.append(("counterfactual_text", counterfactual_text))
        for name, text in required_texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if (
            residual_mode == "positive_minus_counterfactual"
            and positive_text.strip() == counterfactual_text.strip()
        ):
            raise ValueError("positive and counterfactual texts must differ")
        if image is None and int(size[0]) * int(size[1]) != max_area:
            raise ValueError("T2V max_area must equal width * height")

        texts = [input_prompt, positive_text]
        names = ["semantic", "positive"]
        if residual_mode == "positive_minus_counterfactual":
            texts.append(counterfactual_text)
            names.append("counterfactual")
        texts.append(
            self.sample_neg_prompt if cfg_negative_mode == "wan_default" else ""
        )
        names.append("cfg_negative")
        contexts, token_lengths = self._encode_ace_contexts(
            texts, names=names, offload_model=offload_model
        )
        semantic_context = contexts[0]
        positive_context = contexts[1]
        counterfactual_context = (
            contexts[2] if residual_mode == "positive_minus_counterfactual" else None
        )
        null_context = contexts[-1]

        if image is None:
            return self._t2v_ace(
                semantic_context=semantic_context,
                positive_context=positive_context,
                counterfactual_context=counterfactual_context,
                residual_mode=residual_mode,
                null_context=null_context,
                token_lengths=token_lengths,
                layer_gates=numeric_layer_gates,
                step_gates=numeric_step_gates,
                lambda0=lambda0,
                residual_cap_ratio=residual_cap_ratio,
                residual_cap_eps=residual_cap_eps,
                size=size,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=seed,
                offload_model=offload_model,
                diagnostic_sink=diagnostic_sink,
                show_progress=show_progress,
            )
        return self._i2v_ace(
            image=image,
            semantic_context=semantic_context,
            positive_context=positive_context,
            counterfactual_context=counterfactual_context,
            residual_mode=residual_mode,
            null_context=null_context,
            token_lengths=token_lengths,
            layer_gates=numeric_layer_gates,
            step_gates=numeric_step_gates,
            lambda0=lambda0,
            residual_cap_ratio=residual_cap_ratio,
            residual_cap_eps=residual_cap_eps,
            max_area=max_area,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=offload_model,
            diagnostic_sink=diagnostic_sink,
            show_progress=show_progress,
        )

    def _t2v_ace(
        self,
        *,
        semantic_context: torch.Tensor,
        positive_context: torch.Tensor,
        counterfactual_context: torch.Tensor | None,
        residual_mode: str,
        null_context: torch.Tensor,
        token_lengths: dict[str, int],
        layer_gates: Sequence[float],
        step_gates: Sequence[float],
        lambda0: float,
        residual_cap_ratio: float | None,
        residual_cap_eps: float,
        size: tuple[int, int],
        frame_num: int,
        shift: float,
        sample_solver: str,
        sampling_steps: int,
        guide_scale: float,
        seed: int,
        offload_model: bool,
        diagnostic_sink: ResidualDiagnostics | None,
        show_progress: bool,
    ) -> AceGenerationOutput:
        width, height = (int(size[0]), int(size[1]))
        divisor_width = self.vae_stride[2] * self.patch_size[2]
        divisor_height = self.vae_stride[1] * self.patch_size[1]
        if width % divisor_width or height % divisor_height:
            raise ValueError(
                f"T2V size must be divisible by ({divisor_width}, {divisor_height})"
            )
        target_shape = (
            self.vae.model.z_dim,
            (frame_num - 1) // self.vae_stride[0] + 1,
            height // self.vae_stride[1],
            width // self.vae_stride[2],
        )
        seq_len = math.ceil(
            target_shape[2]
            * target_shape[3]
            / (self.patch_size[1] * self.patch_size[2])
            * target_shape[1]
            / self.sp_size
        ) * self.sp_size
        resolved_seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=self.device).manual_seed(resolved_seed)
        noise = [
            torch.randn(
                *target_shape,
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )
        ]
        no_sync = getattr(self.model, "no_sync", _noop_no_sync)
        step_timer_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

        with (
            torch.amp.autocast("cuda", dtype=self.param_dtype),
            torch.no_grad(),
            no_sync(),
        ):
            scheduler, timesteps = self._scheduler(
                sample_solver, sampling_steps, shift
            )
            latents = noise
            _, mask2 = masks_like(noise, zero=False)
            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            iterator = tqdm(timesteps, disable=not show_progress, desc="ACE T2V")
            for step_index, timestep_value in enumerate(iterator):
                timer_pair = _step_timer_pair()
                timer_pair[0].record()
                timestep = torch.stack([timestep_value]).to(self.device)
                token_timestep = (
                    mask2[0][0][:, ::2, ::2] * timestep
                ).flatten()
                token_timestep = torch.cat(
                    [
                        token_timestep,
                        token_timestep.new_ones(seq_len - token_timestep.size(0))
                        * timestep,
                    ]
                ).unsqueeze(0)

                conditional = self.model(
                    latents,
                    t=token_timestep,
                    context=[semantic_context],
                    seq_len=seq_len,
                    causal_pos_context=[positive_context],
                    causal_neg_context=(
                        [counterfactual_context]
                        if counterfactual_context is not None
                        else None
                    ),
                    residual_mode=residual_mode,
                    layer_gates=layer_gates,
                    step_gate=float(step_gates[step_index]),
                    lambda0=lambda0,
                    residual_cap_ratio=residual_cap_ratio,
                    residual_cap_eps=residual_cap_eps,
                    step_index=step_index,
                    total_steps=sampling_steps,
                    diagnostic_sink=diagnostic_sink,
                )[0]
                unconditional = self.model(
                    latents,
                    t=token_timestep,
                    context=[null_context],
                    seq_len=seq_len,
                    step_gate=0.0,
                    lambda0=0.0,
                )[0]
                _assert_finite_async(conditional, "non-finite conditional prediction")
                _assert_finite_async(unconditional, "non-finite unconditional prediction")
                prediction = unconditional + guide_scale * (
                    conditional - unconditional
                )
                next_latent = scheduler.step(
                    prediction.unsqueeze(0),
                    timestep_value,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=generator,
                )[0]
                latents = [next_latent.squeeze(0)]
                timer_pair[1].record()
                step_timer_pairs.append(timer_pair)
            step_elapsed_seconds = _resolve_step_timings(step_timer_pairs)
            decoded_latents = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if self.rank != 0:
                raise RuntimeError("ACE V1 only supports rank-zero single-process inference")
            video = self.vae.decode(decoded_latents)[0]

        del noise, latents, decoded_latents, scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        _assert_finite_async(video, "non-finite decoded video")
        return AceGenerationOutput(
            video=video,
            seed=resolved_seed,
            width=width,
            height=height,
            context_token_lengths=token_lengths,
            step_gates=[float(value) for value in step_gates],
            step_elapsed_seconds=step_elapsed_seconds,
            first_frame_reanchor_count=0,
        )

    def _i2v_ace(
        self,
        *,
        image: Image.Image,
        semantic_context: torch.Tensor,
        positive_context: torch.Tensor,
        counterfactual_context: torch.Tensor | None,
        residual_mode: str,
        null_context: torch.Tensor,
        token_lengths: dict[str, int],
        layer_gates: Sequence[float],
        step_gates: Sequence[float],
        lambda0: float,
        residual_cap_ratio: float | None,
        residual_cap_eps: float,
        max_area: int,
        frame_num: int,
        shift: float,
        sample_solver: str,
        sampling_steps: int,
        guide_scale: float,
        seed: int,
        offload_model: bool,
        diagnostic_sink: ResidualDiagnostics | None,
        show_progress: bool,
    ) -> AceGenerationOutput:
        image = image.convert("RGB").copy()
        input_height, input_width = image.height, image.width
        divisor_height = self.patch_size[1] * self.vae_stride[1]
        divisor_width = self.patch_size[2] * self.vae_stride[2]
        width, height = best_output_size(
            input_width,
            input_height,
            divisor_width,
            divisor_height,
            max_area,
        )
        scale = max(width / input_width, height / input_height)
        image = image.resize(
            (round(input_width * scale), round(input_height * scale)),
            Image.Resampling.LANCZOS,
        )
        left = (image.width - width) // 2
        top = (image.height - height) // 2
        image = image.crop((left, top, left + width, top + height))
        if image.size != (width, height):
            raise RuntimeError("I2V image preprocessing produced an unexpected size")
        image_tensor = (
            TF.to_tensor(image).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)
        )

        latent_frames = (frame_num - 1) // self.vae_stride[0] + 1
        seq_len = (
            latent_frames
            * (height // self.vae_stride[1])
            * (width // self.vae_stride[2])
            // (self.patch_size[1] * self.patch_size[2])
        )
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size
        resolved_seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=self.device).manual_seed(resolved_seed)
        noise = torch.randn(
            self.vae.model.z_dim,
            latent_frames,
            height // self.vae_stride[1],
            width // self.vae_stride[2],
            dtype=torch.float32,
            generator=generator,
            device=self.device,
        )
        known_latent = self.vae.encode([image_tensor])
        no_sync = getattr(self.model, "no_sync", _noop_no_sync)
        reanchor_count = 0
        step_timer_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

        with (
            torch.amp.autocast("cuda", dtype=self.param_dtype),
            torch.no_grad(),
            no_sync(),
        ):
            scheduler, timesteps = self._scheduler(
                sample_solver, sampling_steps, shift
            )
            latent = noise
            _, mask2 = masks_like([noise], zero=True)
            latent = (1.0 - mask2[0]) * known_latent[0] + mask2[0] * latent
            reanchor_count += 1
            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            iterator = tqdm(timesteps, disable=not show_progress, desc="ACE I2V")
            for step_index, timestep_value in enumerate(iterator):
                timer_pair = _step_timer_pair()
                timer_pair[0].record()
                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([timestep_value]).to(self.device)
                token_timestep = (
                    mask2[0][0][:, ::2, ::2] * timestep
                ).flatten()
                token_timestep = torch.cat(
                    [
                        token_timestep,
                        token_timestep.new_ones(seq_len - token_timestep.size(0))
                        * timestep,
                    ]
                ).unsqueeze(0)

                conditional = self.model(
                    latent_model_input,
                    t=token_timestep,
                    context=[semantic_context],
                    seq_len=seq_len,
                    causal_pos_context=[positive_context],
                    causal_neg_context=(
                        [counterfactual_context]
                        if counterfactual_context is not None
                        else None
                    ),
                    residual_mode=residual_mode,
                    layer_gates=layer_gates,
                    step_gate=float(step_gates[step_index]),
                    lambda0=lambda0,
                    residual_cap_ratio=residual_cap_ratio,
                    residual_cap_eps=residual_cap_eps,
                    step_index=step_index,
                    total_steps=sampling_steps,
                    diagnostic_sink=diagnostic_sink,
                )[0]
                if offload_model:
                    torch.cuda.empty_cache()
                unconditional = self.model(
                    latent_model_input,
                    t=token_timestep,
                    context=[null_context],
                    seq_len=seq_len,
                    step_gate=0.0,
                    lambda0=0.0,
                )[0]
                if offload_model:
                    torch.cuda.empty_cache()
                _assert_finite_async(conditional, "non-finite conditional prediction")
                _assert_finite_async(unconditional, "non-finite unconditional prediction")
                prediction = unconditional + guide_scale * (
                    conditional - unconditional
                )
                next_latent = scheduler.step(
                    prediction.unsqueeze(0),
                    timestep_value,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=generator,
                )[0]
                latent = next_latent.squeeze(0)
                latent = (
                    (1.0 - mask2[0]) * known_latent[0] + mask2[0] * latent
                )
                reanchor_count += 1
                timer_pair[1].record()
                step_timer_pairs.append(timer_pair)
                del latent_model_input, token_timestep
            step_elapsed_seconds = _resolve_step_timings(step_timer_pairs)
            decoded_latents = [latent]
            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if self.rank != 0:
                raise RuntimeError("ACE V1 only supports rank-zero single-process inference")
            video = self.vae.decode(decoded_latents)[0]

        del noise, latent, decoded_latents, scheduler, known_latent, image_tensor
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        _assert_finite_async(video, "non-finite decoded video")
        return AceGenerationOutput(
            video=video,
            seed=resolved_seed,
            width=width,
            height=height,
            context_token_lengths=token_lengths,
            step_gates=[float(value) for value in step_gates],
            step_elapsed_seconds=step_elapsed_seconds,
            first_frame_reanchor_count=reanchor_count,
        )
