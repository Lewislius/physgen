"""Place the native Wan blocks across devices; preserve its forward and LoRA names."""
from contextlib import nullcontext

import torch

from static_lora.backbone import WanLoRA
from static_lora.lora import lora_named_parameters
from static_lora.runtime import log
from static_lora.native_wan import load_wan


def block_layout(counts, block_count):
    if len(counts) != 4 or any(type(n) is not int or n < 0 for n in counts) or sum(counts) != block_count:
        raise ValueError("Four block counts must cover the native Wan blocks exactly")
    return [stage for stage, count in enumerate(counts) for _ in range(count)]


class ParallelWanLoRA(WanLoRA):
    def __init__(self, wan, spec, devices, counts, recompute=True, activation_offload=True):
        if len(devices) != 4:
            raise ValueError("Expected four stage devices")
        super().__init__(wan, spec, recompute=recompute)
        self.stage_devices = tuple(torch.device(device) for device in devices)
        self.layout = block_layout(counts, len(wan.blocks))
        self.activation_offload = activation_offload
        self._stage_kwargs = {}
        for name in ("patch_embedding", "text_embedding", "time_embedding", "time_projection"):
            getattr(wan, name).to(self.stage_devices[0])
        wan.freqs = wan.freqs.to(self.stage_devices[0])
        for index, block in enumerate(wan.blocks):
            stage = self.layout[index]
            block.to(self.stage_devices[stage])
            block.register_forward_pre_hook(self._block_hook(stage), with_kwargs=True)
        wan.head.to(self.stage_devices[-1])
        wan.head.register_forward_pre_hook(self._head_hook)

    def _block_hook(self, stage):
        def hook(module, args, kwargs):
            device = self.stage_devices[stage]
            if stage not in self._stage_kwargs:
                # Replicate the frozen time/context/rotary tensors once per stage
                # per forward. Keep native seq_lens/grid_sizes on CPU.
                self._stage_kwargs[stage] = {
                    name: value.to(device, non_blocking=True) if name in ("e", "context", "freqs") else value
                    for name, value in kwargs.items()}
            return (args[0].to(device, non_blocking=True), *args[1:]), self._stage_kwargs[stage]
        return hook

    def _head_hook(self, module, args):
        return tuple(value.to(self.stage_devices[-1], non_blocking=True) for value in args)

    def forward(self, x, first, text, sigma):
        if x.device != self.stage_devices[0]:
            raise ValueError("Training latents and conditions must enter on stage 0")
        self._stage_kwargs = {}
        offload = (torch.autograd.graph.save_on_cpu(pin_memory=x.is_cuda)
                   if self.activation_offload and torch.is_grad_enabled() else nullcontext())
        try:
            # Device transfers remain differentiable. Checkpointed blocks capture
            # their already-routed arguments, including on backward recomputation.
            with offload:
                prediction = super().forward(x, first, text, sigma)
                return prediction.to(x.device, non_blocking=True)
        finally:
            # Conditional/unconditional calls and subsequent samples must never
            # reuse an earlier video's timestep or text conditioning.
            self._stage_kwargs = {}

    def placement(self):
        counts = {}
        for name, p in self.named_parameters():
            key = str(p.device)
            row = counts.setdefault(key, dict(parameters=0, bytes=0, trainable_parameters=0))
            row["parameters"] += p.numel()
            row["bytes"] += p.numel() * p.element_size()
            row["trainable_parameters"] += p.numel() if p.requires_grad else 0
        return counts


def load_model(cfg, device, training=True):
    if device != torch.device("cuda", 0):
        raise ValueError("Four-GPU execution must enter on cuda:0")
    # Never materialize the entire 5B model on cuda:0, even while loading.
    wan = load_wan(cfg["paths"]["wan_checkpoint"], torch.device("cpu"))
    execution = cfg["execution"]
    model = ParallelWanLoRA(wan, cfg["lora"], [torch.device("cuda", i) for i in range(4)],
                            execution["block_counts"], recompute=training,
                            activation_offload=training and execution["activation_offload"])
    model.eval()
    if not training:
        model.requires_grad_(False)
    placement = model.placement()
    if set(placement) != {f"cuda:{index}" for index in range(4)}:
        raise RuntimeError(f"Wan was not placed on all four GPUs: {placement}")
    if training and any(row["trainable_parameters"] == 0 for row in placement.values()):
        raise RuntimeError("Each GPU must own LoRA parameters in its assigned blocks")
    log(event="model_parallel_placement", layout=execution["block_counts"], placement=placement,
        activation_checkpointing=training, activation_offload=model.activation_offload,
        lora_parameters=sum(p.numel() for _, p in lora_named_parameters(model)))
    for index in range(4):
        torch.cuda.reset_peak_memory_stats(index)
    return model
