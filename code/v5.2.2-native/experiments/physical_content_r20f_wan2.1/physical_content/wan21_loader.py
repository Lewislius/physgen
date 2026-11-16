"""Official Wan2.1-T2V-1.3B loader with single-file or indexed safetensors.

Wan2.1 ships one ``diffusion_pytorch_model.safetensors`` file, whereas Wan2.2
TI2V-5B ships a sharded index.  The loader accepts both and verifies the loaded
backbone dimensions against the Wan2.1 configuration before use.
"""
from pathlib import Path
import hashlib
import json

import torch

from .wan21_runtime import assert_wan21_source, install_wan21_namespace


def _checkpoint_file(root, name):
    root = Path(root)
    index = root / 'diffusion_pytorch_model.safetensors.index.json'
    if index.is_file():
        mapping = json.loads(index.read_text())['weight_map']
        if name not in mapping:
            raise KeyError(f'{name} is absent from the official checkpoint index')
        return root / mapping[name]
    single = root / 'diffusion_pytorch_model.safetensors'
    if not single.is_file():
        raise FileNotFoundError(f'No official Wan checkpoint at {single}')
    return single


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hash(root):
    """Provenance hash for either the single-file or sharded official model."""
    root = Path(root)
    index = root / 'diffusion_pytorch_model.safetensors.index.json'
    single = root / 'diffusion_pytorch_model.safetensors'
    digest = hashlib.sha256()
    for path in (root / 'config.json', index if index.is_file() else single):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def checkpoint_kind(root):
    root = Path(root)
    return ('indexed' if (root / 'diffusion_pytorch_model.safetensors.index.json').is_file()
            else 'single_file')


def load_wan(cfg, device='cpu'):
    """Load the frozen official Wan2.1-T2V-1.3B backbone.

    The selected parameters are restored in float32 exactly as in the established
    R20 loader.  No project checkpoint or LoRA is loaded here.
    """
    root = Path(cfg['paths']['wan_checkpoint'])
    install_wan21_namespace(cfg['paths']['wan_code'])
    assert_wan21_source(cfg['paths']['wan_code'])
    from wan.modules.model import WanModel
    from safetensors import safe_open

    wan = WanModel.from_pretrained(
        str(root), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True)
    spec = cfg['model']
    actual = (wan.dim, wan.ffn_dim, wan.num_heads, wan.num_layers,
              wan.in_dim, wan.out_dim, wan.text_len)
    expected = (int(spec['width']), int(spec['ffn_width']),
                int(spec['native_heads']), int(spec['native_layers']),
                16, 16, 512)
    if actual != expected:
        raise ValueError(f'Wan2.1 backbone dimensions differ: {actual} != {expected}')
    prefixes = ['time_embedding.', 'time_projection.', 'head.']
    for index in range(len(wan.blocks)):
        prefixes += [
            f'blocks.{index}.{suffix}' for suffix in
            ('modulation', 'norm3.', 'self_attn.norm_q.', 'self_attn.norm_k.',
             'cross_attn.norm_q.', 'cross_attn.norm_k.')]
    for name, parameter in wan.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            with safe_open(str(_checkpoint_file(root, name)), framework='pt',
                           device='cpu') as reader:
                parameter.data = reader.get_tensor(name).float().clone()
    return wan.eval().requires_grad_(False).to(device)
