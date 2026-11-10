"""One knowledge representation; evolving states remain inside native Wan."""
import torch
from torch import nn
from torch.nn import functional as F
from .attention import RMSNorm, fourier


class CompactAttention(nn.Module):
    def __init__(self, width, inner, heads):
        super().__init__()
        self.heads, self.dim = heads, inner//heads
        self.q, self.k, self.v = (nn.Linear(width, inner, bias=False) for _ in range(3))
        self.o = nn.Linear(inner, width, bias=False)
        self.qnorm, self.knorm = RMSNorm(inner), RMSNorm(inner)

    def project(self, context):
        shape = lambda x: x.reshape(x.shape[0], -1, self.heads, self.dim).transpose(1, 2)
        return shape(self.knorm(self.k(context))), shape(self.v(context))

    def read(self, query, kv):
        q = self.qnorm(self.q(query)).reshape(query.shape[0], -1, self.heads, self.dim).transpose(1, 2)
        value = F.scaled_dot_product_attention(q, kv[0].to(q), kv[1].to(q))
        return self.o(value.transpose(1, 2).flatten(2))


class KnowledgeCondition(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg['model']; d = m['width']
        self.norm = RMSNorm(m['qwen_width'])
        self.project = nn.Linear(m['qwen_width'], d)
        self.kind = nn.Embedding(4, d)
        nn.init.normal_(self.kind.weight, std=.02)

    def forward(self, evidence):
        # These are multimodal Qwen states: each readout already attends to the
        # image and all clauses. Native Wan independently receives the first-frame
        # spatial grid; duplicating hundreds of raw image keys would dilute C.
        return self.project(self.norm(evidence['contextual']))+self.kind(evidence['roles'])[None]


class InjectionSite(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.norm = RMSNorm(width)
        self.down, self.up = nn.Linear(width, rank, bias=False), nn.Linear(rank, width, bias=False)
        self.gate = nn.Linear(width+2, 1)
        nn.init.normal_(self.up.weight, std=.01)
        nn.init.normal_(self.gate.weight, std=.001)
        nn.init.zeros_(self.gate.bias)

    def forward(self, query, message, intensity, sigma):
        noise = sigma.to(query).expand(*query.shape[:-1], 1)
        gate = self.gate(torch.cat((self.norm(query), noise, intensity.to(query)), -1)).sigmoid()
        return gate*self.up(F.silu(self.down(message)))


class NativeInterface(nn.Module):
    def __init__(self, cfg, depth):
        super().__init__()
        m = cfg['model']; d = m['width']
        self.norm = RMSNorm(d)
        self.attention = CompactAttention(d, m['attention_width'], m['attention_heads'])
        self.intensity = nn.Sequential(RMSNorm(d), nn.Linear(d, d//16), nn.SiLU(), nn.Linear(d//16, 1))
        self.sites = nn.ModuleDict({str(i): InjectionSite(d, m['interface_rank'])
                                   for i in range(depth) if (i+1) % m['condition_every'] == 0})

    def prepare(self, tokens):
        return self.attention.project(self.norm(tokens))

    def forward(self, x, index, kv, sigma, times, geometry, strength=1.):
        if str(index) not in self.sites: return x, None
        t, h, w = geometry
        future = x.reshape(x.shape[0], t, h*w, -1)[:, 1:]
        # Video time is independent of diffusion noise. Reintroduced at each read.
        position = .1*fourier(torch.stack((times[1:], times[1:]/times[-1].clamp_min(1e-6)), -1), x.shape[-1])
        query = self.norm(future)+position[None, :, None]
        intensity = self.intensity(query).sigmoid()
        if strength == 0: return x, intensity.squeeze(-1).reshape(x.shape[0], t-1, h, w)
        message = self.attention.read(query.flatten(1, 2), kv).reshape_as(query)
        residual = self.sites[str(index)](query, message, intensity, sigma)
        full = x.reshape(x.shape[0], t, h*w, -1)
        output = torch.cat((full[:, :1], future+strength*residual), 1).reshape_as(x)
        return output, intensity.squeeze(-1).reshape(x.shape[0], t-1, h, w)
