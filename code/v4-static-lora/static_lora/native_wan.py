"""Original Wan loading and fixed I2V first slice, owned by this LoRA experiment."""
import json
from pathlib import Path

import torch


def fp32_islands(model):
    # Native norms multiply BF16 activations by FP32 scales; BF16 scales silently
    # introduce an extra rounding before RoPE. Modulation additions also stay FP32.
    modules = ["time_embedding", "time_projection", "head"]
    parameters = []
    for i in range(len(model.blocks)):
        modules.extend(f"blocks.{i}.{name}" for name in (
            "norm3", "self_attn.norm_q", "self_attn.norm_k", "cross_attn.norm_q", "cross_attn.norm_k"))
        parameters.append(f"blocks.{i}.modulation")
    return modules, parameters


def load_wan(path, device):
    from safetensors import safe_open
    from wan.modules.model import WanModel
    model = WanModel.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if ((len(model.blocks), model.dim, model.in_dim, model.out_dim, model.text_len) != (30, 3072, 48, 48, 512)
            or tuple(model.patch_size) != (1, 2, 2)):
        raise ValueError("Expected original Wan2.2-TI2V-5B")
    index = json.loads((Path(path) / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    # Restore native FP32 islands from ORIGINAL weights, not from BF16 rounded tensors.
    modules, parameters = fp32_islands(model)
    for name in modules:
        module = model.get_submodule(name).float()
        values = {}
        for key, filename in index.items():
            if key.startswith(name + "."):
                with safe_open(str(Path(path) / filename), framework="pt", device="cpu") as reader:
                    values[key[len(name) + 1:]] = reader.get_tensor(key).float()
        module.load_state_dict(values, strict=True)
    for name in parameters:
        with safe_open(str(Path(path) / index[name]), framework="pt", device="cpu") as reader:
            model.get_parameter(name).data = reader.get_tensor(name).float().clone()
    model.eval().requires_grad_(False).to(device)
    model.freqs = model.freqs.to(device)
    return model


def fix_first(x, first):
    return x.float() if first is None else torch.cat((first.float(), x[:, :, 1:].float()), dim=2)
