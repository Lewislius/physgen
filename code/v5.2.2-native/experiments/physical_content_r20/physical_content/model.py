"""R20 fresh physical adapter and video-LoRA training ownership."""
import torch

from physgen_native.generator import load_wan
from .architecture import NativePhysicalVelocity
from .base_bank import OfficialBaseBank
from .lora import assert_native_text_frozen, install_video_lora
from .wan_loader import ensure_wan_namespace


class PhysicalModel(NativePhysicalVelocity):
    def __init__(self, cfg, device='cuda', wan=None, interface=None):
        if wan is None:
            ensure_wan_namespace(cfg)
            wan = load_wan(cfg, device)
        wan.requires_grad_(False).eval()
        bank = OfficialBaseBank(wan)
        if interface is None:
            interface = torch.load(cfg['paths']['interface'], map_location='cpu', weights_only=True)
            if 'native_scale' not in interface or interface.get('task') != cfg['task']:
                raise ValueError('Formal R20 requires its own task-specific training normalization')
        devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg['initialization']['new_adapter_seed'])
            if devices:
                torch.cuda.manual_seed_all(cfg['initialization']['new_adapter_seed'])
            spec = cfg['model']
            super().__init__(wan, task=cfg['task'], interface=interface,
                             query_width=spec['qwen_width'], q_heads=spec['q_heads'],
                             grounded_heads=spec['grounded_heads'], slots=spec['q_tokens'],
                             rounds=spec['content_rounds'],
                             injection_layers=spec['injection_layers'],
                             checkpointing=cfg['train']['gradient_checkpointing'])
            self.to(device)
            # Physical copies above are made before installing main-video LoRA.
            self.video_lora_targets = install_video_lora(self.wan, cfg)
        self.cfg, self.plan, self.mechanism = cfg, cfg['plan'], cfg['mechanism']
        self.base_bank = bank
        assert_native_text_frozen(self.wan)
        self.learned_names = tuple(n for n, p in self.named_parameters() if p.requires_grad)

    def _role(self, name):
        if name.startswith('wan.'):
            if 'cross_attn' in name:
                raise ValueError('Native text projections cannot be adapted')
            return 'wan_video_lora'
        if name.startswith('enhancement.'):
            return 'enhancement'
        if name.startswith('noise_modulation.') or '.state.' in name or name.endswith('.modulation'):
            return 'physics_state'
        if name.startswith('physics.'):
            return ('physics_write' if '.direct_output.' in name or '.associated_output.' in name
                    else 'physics_read')
        raise ValueError(f'Unassigned parameter: {name}')

    def optim_groups(self):
        named, grouped = dict(self.named_parameters()), {}
        for name in self.learned_names:
            role = self._role(name)
            decay = named[name].ndim >= 2
            grouped.setdefault((role, decay), []).append(named[name])
        return [dict(name=f'{role}:{"matrix" if decay else "no_decay"}', role=role,
                     params=parameters, lr=self.cfg['train']['peak_lrs'][role],
                     weight_decay=(self.cfg['train']['matrix_weight_decay'] if decay else 0.0))
                for (role, decay), parameters in grouped.items()]

    def set_phase(self, step):
        self.wan.eval()
        joint = step >= self.cfg['train']['interface_updates']
        for name, p in self.wan.named_parameters():
            if 'wan.'+name in self.learned_names:
                p.requires_grad_(joint)
        return 'joint' if joint else 'physical_generation'

    def trainable_count(self):
        named = dict(self.named_parameters())
        return {name: named[name].numel() for name in self.learned_names}

    def constant_state(self):
        return {name: value.detach().cpu() for name, value in self.named_buffers()
                if name in ('condition_mean', 'condition_scale', 'enhancement.source_scale')}

    def assert_base_unchanged(self, before, full=False):
        after = self.base_bank.hashes(full=full)
        if after != before:
            raise AssertionError('Frozen official foundation changed')
        return True
