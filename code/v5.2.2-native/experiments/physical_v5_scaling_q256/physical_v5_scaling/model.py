"""Exact original V5 architecture with separate learning-rate roles."""
from physical_training.model import (
    PhysicalModel as OriginalPhysicalModel,
    semantic_ce,
    enable_frozen_decoder_checkpointing,
)
from .schedule import GROUPS


class PhysicalModel(OriginalPhysicalModel):
    def optim_groups(self):
        groups={name:[] for name in GROUPS}
        parameters=dict(self.named_parameters())
        for name in self.learned_names:
            if name.startswith('wan.'):
                role='wan_lora'
            elif name.startswith('enhancement.'):
                role='enhancement'
            elif name.startswith('bridge.'):
                role='semantic_bridge'
            elif name.startswith('fusions.'):
                role='fusion_write' if '.output.' in name else 'fusion_read'
            else:
                raise ValueError(f'Unassigned trainable V5 parameter: {name}')
            groups[role].append(parameters[name])
        return [dict(name=role,params=groups[role],lr=self.cfg['train']['peak_lrs'][role])
                for role in GROUPS]
