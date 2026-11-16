"""Frozen official Wan operators referenced by the new experts.

The bank is created *before* main video LoRA is installed.  It therefore points
to the original frozen ``nn.Linear`` objects, never to the main LoRA wrappers.
No official tensor is copied into a checkpoint or optimizer group.
"""
import hashlib
import json
from pathlib import Path

import torch


class OfficialBaseBank:
    def __init__(self, wan):
        self.wan = wan
        self.blocks = list(wan.blocks)
        self._linears = {}
        self._norms = {}
        self._modulations = {}
        self._fingerprints = {}
        self._checkpoint_provenance_cache = None
        self._build()
        self.assert_frozen()

    def _build(self):
        for index, block in enumerate(self.blocks):
            for attention_name in ('self_attn', 'cross_attn'):
                attention = getattr(block, attention_name)
                for target in ('q', 'k', 'v', 'o'):
                    name = f'blocks.{index}.{attention_name}.{target}'
                    module = getattr(attention, target)
                    if not isinstance(module, torch.nn.Linear):
                        raise TypeError(f'{name} must still be the official Linear before LoRA installation')
                    self._linears[name] = module
                for norm in ('norm_q', 'norm_k'):
                    self._norms[f'blocks.{index}.{attention_name}.{norm}'] = getattr(attention, norm)
            for target in ('0', '2'):
                name = f'blocks.{index}.ffn.{target}'
                module = block.ffn[int(target)]
                if not isinstance(module, torch.nn.Linear):
                    raise TypeError(f'{name} must be an official Linear')
                self._linears[name] = module
            self._modulations[index] = block.modulation

    def linear(self, layer, target):
        return self._linears[f'blocks.{layer}.{target}']

    def attention(self, layer, name):
        return getattr(self.blocks[layer], name)

    def ffn(self, layer):
        return self.blocks[layer].ffn

    def norm(self, layer, attention, norm):
        return self._norms[f'blocks.{layer}.{attention}.{norm}']

    def modulation(self, layer):
        return self._modulations[layer]

    def block_norm(self, layer, name):
        return getattr(self.blocks[layer], name)

    def assert_frozen(self):
        for name, module in {**self._linears, **self._norms}.items():
            for parameter_name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    raise ValueError(f'Official base is trainable: {name}.{parameter_name}')
        for index, modulation in self._modulations.items():
            if modulation.requires_grad:
                raise ValueError(f'Official modulation {index} is trainable')

    @torch.no_grad()
    def hashes(self, full=True):
        """Content hashes for the no-update audit.

        ``full=True`` reads every base tensor.  The default training audit uses
        this once before and once after the run; the official index hash is the
        cheap provenance record stored in every checkpoint.
        """
        result = {}
        for name, module in self._linears.items():
            for parameter_name, parameter in module.named_parameters():
                result[f'{name}.{parameter_name}'] = self._tensor_hash(parameter, full)
        for name, norm in self._norms.items():
            for parameter_name, parameter in norm.named_parameters():
                result[f'{name}.{parameter_name}'] = self._tensor_hash(parameter, full)
        for index, modulation in self._modulations.items():
            result[f'blocks.{index}.modulation'] = self._tensor_hash(modulation, full)
        return result

    @staticmethod
    def _tensor_hash(tensor, full):
        value = tensor.detach()
        if full:
            value = value.float().cpu().contiguous()
        else:
            flat = value.flatten()
            value = torch.cat((flat[:4096], flat[-4096:])).float().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def checkpoint_provenance(self, checkpoint_root):
        root = Path(checkpoint_root)
        if (self._checkpoint_provenance_cache is not None
                and self._checkpoint_provenance_cache['official_checkpoint'] == str(root)):
            return dict(self._checkpoint_provenance_cache)
        index = root / 'diffusion_pytorch_model.safetensors.index.json'
        single = root / 'diffusion_pytorch_model.safetensors'
        config = root / 'config.json'
        weight_file = index if index.is_file() else single
        if not weight_file.is_file() or not config.is_file():
            raise FileNotFoundError('Official Wan2.1 checkpoint/config is incomplete')
        def file_sha(path):
            value = hashlib.sha256()
            with Path(path).open('rb') as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    value.update(block)
            return value.hexdigest()

        weight_sha, config_sha = file_sha(weight_file), file_sha(config)
        digest = hashlib.sha256()
        for path, value in ((weight_file, weight_sha), (config, config_sha)):
            digest.update(path.name.encode())
            digest.update(bytes.fromhex(value))
        result = dict(
            official_model='Wan-AI/Wan2.1-T2V-1.3B',
            official_checkpoint=str(root),
            checkpoint_kind=('indexed' if index.is_file() else 'single_file'),
            official_index_sha256=file_sha(index) if index.is_file() else None,
            official_weight_sha256=weight_sha,
            official_config_sha256=config_sha,
            combined_sha256=digest.hexdigest(),
            base_keys=len(self._linears) + len(self._norms) + len(self._modulations),
        )
        self._checkpoint_provenance_cache = result
        return dict(result)


def assert_no_aliasing(main_lora, expert_lora):
    """Main and expert adapters must never be the same Parameter objects."""
    main = {id(parameter) for parameter in main_lora.parameters() if parameter.requires_grad}
    expert = {id(parameter) for parameter in expert_lora.parameters() if parameter.requires_grad}
    overlap = main & expert
    if overlap:
        raise AssertionError(f'{len(overlap)} trainable parameters are aliased')
    return False
