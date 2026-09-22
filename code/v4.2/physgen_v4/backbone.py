import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
from safetensors import safe_open

from .corrector import CONDITION_INITIALIZATIONS


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


class ProcessWan(nn.Module):
    """One ordered Wan pass. Persistent state is an explicit input and output."""

    def __init__(self, wan, corrector, enable_b):
        super().__init__()
        self.wan = wan
        self.corrector = corrector
        self.enable_b = enable_b
        self.diagnostics = None
        self.correction_sites()  # Check the registered layout against the actual Wan depth.

    def correction_sites(self):
        cfg = self.corrector.config
        interval = cfg.get("a_every", 0)
        independent = self.corrector.parameter_sharing == "per_block"
        repeated = independent or interval > 0
        blocks = (self.corrector.a_blocks if independent else
                  range(interval, len(self.wan.blocks) + 1, interval) if repeated else [cfg.get("block_a", 6)])
        sites = [(number, "A", f"A@{number}" if repeated else "A") for number in blocks]
        if self.enable_b:
            number = cfg["block_b"]
            sites.append((number, "B", f"B@{number}" if repeated else "B"))
        if independent and any(type(n) is not int or not 1 <= n <= len(self.wan.blocks) for n, _, _ in sites):
            raise ValueError(f"Corrector block layout {sites} exceeds Wan depth {len(self.wan.blocks)}")
        return sorted(sites, key=lambda item: (item[0], item[1]))

    def call_corrector(self, name, number, hidden, process, text, sigma, coords):
        block = number if (self.corrector.parameter_sharing == "per_block"
                           or self.corrector.config.get("a_every", 0) > 0) else None
        return self.corrector(name, hidden, process, text, sigma, coords, block=block)

    def forward(self, noisy, first, text, sigma, coords, process=None,
                disable=(), writer_off=(), capture=None, *, gt_video=None, capture_write=False):
        if gt_video is not None and process is not None:
            raise ValueError("GT conditions P0 only; do not supply GT when carrying an existing P")
        from wan.modules.model import sinusoidal_embedding_1d
        wan = self.wan
        embedded = wan.patch_embedding(noisy)
        grid_sizes = torch.tensor([embedded.shape[2:]], dtype=torch.long)
        hidden = embedded.flatten(2).transpose(1, 2)
        length = hidden.shape[1]
        seq_lens = torch.tensor([length], dtype=torch.long)
        # TI2V uses a clean first-frame timestep only when an image actually conditions it.
        known_length = grid_sizes[0, 1] * grid_sizes[0, 2] if first is not None else 0
        t = torch.cat((sigma.new_zeros(int(known_length)),
                       sigma.expand(length - int(known_length)) * 1000)).unsqueeze(0)
        with torch.autocast("cuda", enabled=False):
            e = wan.time_embedding(sinusoidal_embedding_1d(wan.freq_dim, t.flatten()).reshape(1, length, -1).float())
            e0 = wan.time_projection(e).unflatten(2, (6, wan.dim))
        padding = text.new_zeros(1, wan.text_len - text.shape[1], text.shape[2])
        context = wan.text_embedding(torch.cat((text, padding), dim=1))
        arguments = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=wan.freqs,
                         context=context, context_lens=None)
        prior = None
        if process is None:
            legacy = {} if self.corrector.initialization in CONDITION_INITIALIZATIONS else dict(noisy=noisy)
            process = self.corrector.initialize(first, text, coords, gt_video=gt_video, **legacy)
            if self.corrector.initialization in CONDITION_INITIALIZATIONS:
                prior = process
        initial = process.detach()
        states, before, metrics = {}, {}, {}
        snapshots = {}
        write_pairs = {}
        cfg = self.corrector.config
        positions = {}
        for number, name, label in self.correction_sites():
            positions.setdefault(number - 1, []).append((name, label))
        for index, block in enumerate(wan.blocks):
            trace = self.diagnostics
            scope = trace.stage('wan_block', block=index + 1) if trace is not None and trace.trace_blocks else nullcontext()
            with scope:
                hidden = block(hidden, **arguments)
            for name, label in positions.get(index, []):
                active = name == "A" or (self.enable_b and sigma.item() < cfg["handoff_sigma"])
                if active and name not in disable and label not in disable:
                    if capture in (name, label, "all"):
                        snapshots[label] = (hidden.clone(), process.clone(), index, arguments, e, grid_sizes)
                    previous_hidden = hidden
                    hidden, process, before[label], current = self.call_corrector(name, index + 1, hidden, process, text, sigma, coords)
                    if capture_write:
                        write_pairs[label] = dict(before=previous_hidden.detach(),
                            after=hidden.detach(), process=process.detach(), block=index + 1, family=name)
                    if name in writer_off or label in writer_off:
                        hidden = previous_hidden
                    states[label] = process
                    metrics.update({f"{label}/{key}": value for key, value in current.items()})
        prediction = wan.unpatchify(wan.head(hidden, e), grid_sizes)[0].unsqueeze(0).float()
        return prediction, process, dict(states=states, before=before, initial=initial, prior=prior,
            prior_gt_conditioned=gt_video is not None,
            prior_time_tokens=len(coords.process[0]), metrics=metrics, snapshots=snapshots, write_pairs=write_pairs)

    def suffix(self, snapshot, text, sigma, coords, policy, site, downstream=True):
        """Offline attribution from exactly the captured H/P; no scheduler is involved."""
        hidden, process, index, arguments, e, grid_sizes = snapshot
        name = site.split("@", 1)[0]
        if policy != "off":
            changed, process, _, _ = self.call_corrector(name, index + 1, hidden, process, text, sigma, coords)
            if policy == "on":
                hidden = changed
        # Also replay a second correction at the captured block, if A/B share a location.
        sites = self.correction_sites()
        captured = next(i for i, (number, family, label) in enumerate(sites)
                        if number == index + 1 and family == name)
        def apply_downstream(block_index, hidden, process):
            if downstream:
                for number, family, _ in sites[captured + 1:]:
                    active = family == "A" or sigma.item() < self.corrector.config["handoff_sigma"]
                    if number == block_index + 1 and active:
                        hidden, process, _, _ = self.call_corrector(family, number, hidden, process, text, sigma, coords)
            return hidden, process
        hidden, process = apply_downstream(index, hidden, process)
        for block_index in range(index + 1, len(self.wan.blocks)):
            hidden = self.wan.blocks[block_index](hidden, **arguments)
            hidden, process = apply_downstream(block_index, hidden, process)
        return self.wan.unpatchify(self.wan.head(hidden, e), grid_sizes)[0].unsqueeze(0).float()
