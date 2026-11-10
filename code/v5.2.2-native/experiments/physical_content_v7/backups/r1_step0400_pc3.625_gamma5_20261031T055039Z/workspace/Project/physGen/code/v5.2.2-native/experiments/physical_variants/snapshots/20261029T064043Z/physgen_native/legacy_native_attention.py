"""v12 checkpoint compatibility only; the active v13 reader has no intensity a."""
import torch
from torch import nn
from .attention import RMSNorm
from .native_attention import NativeAttentionInterface


class LegacyIntensityNativeAttentionInterface(NativeAttentionInterface):
    def __init__(self, cfg, depth):
        super().__init__(cfg, depth)
        width = self.width
        self.intensity = nn.Sequential(RMSNorm(width), nn.Linear(width, width//16),
                                       nn.SiLU(), nn.Linear(width//16, 1))
        self.intensity_query = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.intensity_query, std=.001)

    def attend(self, attention, x, packet, context_lens):
        t, h, w = packet.geometry
        a = self.intensity(x[:, h*w:]).sigmoid()
        packet.intensity = a.squeeze(-1).reshape(x.shape[0], t-1, h, w)
        return super().attend(attention, x, packet, context_lens)

    def condition_query(self, query, packet):
        a = packet.intensity.flatten(1)[..., None]
        return query + ((a-.5)*self.intensity_query.to(a)).reshape_as(query)
