"""AdamW and EMA contain only LoRA matrices, never a copy of the 5B backbone."""
import torch

from physgen_v42.optim import SCHEDULE_FORMULA, lr_factor, rng_state, restore_rng
from .lora import assert_lora_only, lora_named_parameters


def optimizer_for(model, cfg):
    assert_lora_only(model)
    params = [p for _, p in lora_named_parameters(model)]
    train = cfg["train"]
    return torch.optim.AdamW([dict(params=params, name="lora", lr=train["peak_lr"])],
                             betas=tuple(train["betas"]), eps=train["eps"], weight_decay=train["weight_decay"])


def set_schedule(optimizer, cfg, step):
    for group in optimizer.param_groups:
        group["lr"] = cfg["train"]["peak_lr"] * lr_factor(step, "lora")


class EMA:
    def __init__(self, model, decay=.995):
        self.decay, self.updates = decay, 0
        self.values = {name: p.detach().float().clone() for name, p in lora_named_parameters(model)}

    @torch.no_grad()
    def update(self, model):
        for name, param in lora_named_parameters(model):
            self.values[name].lerp_(param.float(), 1 - self.decay)
        self.updates += 1

    def state_dict(self):
        return dict(decay=self.decay, updates=self.updates,
                    values={k: v.detach().cpu().clone() for k, v in self.values.items()})

    @torch.no_grad()
    def load_state_dict(self, state):
        if self.decay != state["decay"] or set(self.values) != set(state["values"]):
            raise ValueError("EMA identity differs")
        for name, value in state["values"].items():
            if value.dtype != torch.float32 or value.shape != self.values[name].shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid EMA tensor: {name}")
            self.values[name].copy_(value)
        self.updates = state["updates"]
