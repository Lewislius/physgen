import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
from safetensors import safe_open

from dataclasses import dataclass

from .corrector import ARCHITECTURE


def shard_blocks(blocks, device, recompute, fp32_norms=False):
    for index, block in enumerate(blocks):
        norms = [module for module in block.modules()
                 if isinstance(module, nn.LayerNorm) and module.elementwise_affine] if fp32_norms else []
        for norm in norms:
            norm.to(device=device, dtype=torch.float32)
        if recompute:
            block = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        blocks[index] = FSDP(
            block, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=False, backward_prefetch=None, forward_prefetch=False,
            limit_all_gathers=True, sync_module_states=False, ignored_modules=norms,
        )


def load_wan(path, device, sharded, recompute):
    from wan.modules.model import WanModel
    model = WanModel.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    # Native time modulation and output projection execute with autocast disabled.
    # Restore their original FP32 checkpoint values, rather than widening rounded BF16 values.
    index = json.loads((Path(path) / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    for name in ("time_embedding", "time_projection", "head"):
        module = getattr(model, name).float()
        values = {}
        for key, filename in index.items():
            if key.startswith(name + "."):
                with safe_open(str(Path(path) / filename), framework="pt", device="cpu") as reader:
                    values[key[len(name) + 1:]] = reader.get_tensor(key).float()
        module.load_state_dict(values)
    for block_index, block in enumerate(model.blocks):
        prefix = f"blocks.{block_index}.norm3."
        block.norm3.float()
        values = {}
        for key, filename in index.items():
            if key.startswith(prefix):
                with safe_open(str(Path(path) / filename), framework="pt", device="cpu") as reader:
                    values[key[len(prefix):]] = reader.get_tensor(key).float()
        block.norm3.load_state_dict(values)
    model.eval().requires_grad_(False)
    if sharded:
        shard_blocks(model.blocks, device, recompute, fp32_norms=True)
        for name in ("patch_embedding", "text_embedding", "time_embedding", "time_projection", "head"):
            getattr(model, name).to(device)
    else:
        model.to(device)
        if recompute:
            for index, block in enumerate(model.blocks):
                model.blocks[index] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    model.freqs = model.freqs.to(device)
    return model


def wan_time_embedding(width, positions):
    """Wan's FP64 cos/sin convention, also usable by CPU test backbones."""
    half = width // 2
    positions = positions.double()
    angles = torch.outer(positions, torch.pow(10000, -torch.arange(half).to(positions) / half))
    return torch.cat((angles.cos(), angles.sin()), dim=1)


@dataclass
class WriteSnapshot:
    before: torch.Tensor
    delta: torch.Tensor
    process: torch.Tensor
    next_block: int
    arguments: dict
    time_embedding: torch.Tensor
    grid_sizes: torch.Tensor
    write_mask: torch.Tensor


class ProcessWan(nn.Module):
    """One Wan pass; oracle replays exactly the suffix AFTER a write site."""
    def __init__(self, wan, corrector, enable_b=False):
        super().__init__()
        if enable_b:
            raise ValueError("V4.3 has no B module")
        self.wan, self.corrector, self.enable_b = wan, corrector, False
        self.diagnostics = None
        if corrector.a_blocks[-1] > len(wan.blocks):
            raise ValueError("Corrector position exceeds Wan depth")

    def correction_sites(self):
        return [(n, "A", f"A@{n}") for n in self.corrector.a_blocks]

    def prepare(self, noisy, first, text, sigma):
        wan = self.wan
        embedded = wan.patch_embedding(noisy)
        grid = torch.tensor([embedded.shape[2:]], dtype=torch.long)
        hidden = embedded.flatten(2).transpose(1, 2)
        if hidden.shape[0] != 1:
            raise ValueError("Wan variable-geometry microbatches currently require batch size 1")
        length = hidden.shape[1]
        if text.shape[1] > wan.text_len:
            raise ValueError("Text exceeds Wan token capacity")
        known = int(grid[0, 1] * grid[0, 2]) if first is not None else 0
        timestep = torch.cat((sigma.new_zeros(known), sigma.reshape(()).expand(length - known) * 1000))
        with torch.autocast(hidden.device.type, enabled=False):
            e = wan.time_embedding(wan_time_embedding(wan.freq_dim, timestep).reshape(1, length, -1).float())
            e0 = wan.time_projection(e).unflatten(2, (6, wan.dim))
        padding = text.new_zeros(1, wan.text_len - text.shape[1], text.shape[2])
        context = wan.text_embedding(torch.cat((text, padding), dim=1))
        arguments = dict(e=e0, seq_lens=torch.tensor([length], dtype=torch.long),
                         grid_sizes=grid, freqs=wan.freqs, context=context, context_lens=None)
        mask = torch.ones(1, length, 1, device=hidden.device, dtype=torch.float32)
        mask[:, :known] = 0
        return hidden, arguments, e, grid, mask

    def prediction(self, hidden, time_embedding, grid):
        return self.wan.unpatchify(self.wan.head(hidden, time_embedding), grid)[0].unsqueeze(0).float()

    def forward(self, noisy, first, text, sigma, coords, process=None,
                disable=(), writer_off=(), capture=(), *, gt_video=None):
        if gt_video is not None:
            raise ValueError("GT is a target, never a V4.3 forward input")
        hidden, arguments, e, grid, mask = self.prepare(noisy, first, text, sigma)
        if hidden.shape[-1] != self.corrector.config["hidden_width"]:
            raise ValueError("P/H interaction width must equal the actual Wan hidden width")
        if process is None:
            process = self.corrector.initialize(first, text, coords)
        states, deltas, snapshots, metrics = {}, {}, {}, {}
        for index, block in enumerate(self.wan.blocks):
            trace = self.diagnostics
            scope = trace.stage("wan_block", block=index + 1) if trace is not None and trace.trace_blocks else nullcontext()
            with scope:
                hidden = block(hidden, **arguments)
            number, label = index + 1, f"A@{index + 1}"
            if number not in self.corrector.a_blocks or "A" in disable or label in disable:
                continue
            before = hidden
            corrected, process, jepa, delta, current = self.corrector(
                hidden, process, text, sigma, coords, number, mask)
            hidden = before if "A" in writer_off or label in writer_off else corrected
            states[label], deltas[label] = jepa, delta
            metrics.update({f"{label}/{k}": v for k, v in current.items()})
            if capture == "all" or label in capture:
                if "A" in writer_off or label in writer_off:
                    raise ValueError("Oracle capture requires the actual enabled writer path")
                snapshots[label] = WriteSnapshot(before.detach().float(), delta.detach().float(),
                    process.detach(), number, {k: v.detach() if torch.is_tensor(v) else v for k, v in arguments.items()},
                    e.detach(), grid, mask)
        return self.prediction(hidden, e, grid), process, dict(states=states, deltas=deltas,
            snapshots=snapshots, metrics=metrics, write_mask=mask)

    def suffix(self, snapshot, corrected_hidden, text, sigma, coords):
        """Current-site P is held fixed; all downstream P/H computations are replayed.

        Parameters are not updated. Keep autograd through frozen Wan operations:
        the oracle requests gradients ONLY with respect to its temporary residual.
        """
        hidden, process = corrected_hidden, snapshot.process
        for index in range(snapshot.next_block, len(self.wan.blocks)):
            hidden = self.wan.blocks[index](hidden, **snapshot.arguments)
            if index + 1 in self.corrector.a_blocks:
                hidden, process, _, _, _ = self.corrector(
                    hidden, process, text, sigma, coords, index + 1, snapshot.write_mask)
        return self.prediction(hidden, snapshot.time_embedding, snapshot.grid_sizes)

