"""Small CPU backbones exercising the actual conditioner, Wan adapter and FM loop."""
import copy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from physgen_native.attention import RMSNorm
from physgen_native.config import read_config, validate
from physgen_native.system import PhysGenNative

ROOT = Path(__file__).resolve().parents[1]


def small_config(variant):
    cfg = copy.deepcopy(read_config(ROOT/'configs'/f'{variant.replace("-", "_")}.yaml'))
    cfg['model'].update(width=32, qwen_width=32, attention_width=32, attention_heads=4, condition_every=1)
    cfg['message'].update(workspace_width=32, heads=4)
    cfg['knowledge'].update(image_size=[32, 64], video_size=[32, 64])
    cfg['geometry']['rgb_size'] = [32, 64]
    cfg['data'].update(frames=21, latent_shape=[4, 6, 2, 4])
    cfg['sequence'].update(size=[32, 64])
    cfg['wan_lora'] = dict(rank=2, alpha=2)
    cfg['train'].update(text_dropout=0., native_text_only_dropout=0.)
    return validate(cfg)


def flash_attention(q, k, v, k_lens=None):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2)).transpose(1, 2)


class NativeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = 32, 4, 8
        self.q, self.k, self.v, self.o = (nn.Linear(32, 32) for _ in range(4))
        self.norm_q, self.norm_k = RMSNorm(32), RMSNorm(32)

    def forward(self, x, context, context_lens):
        shape = lambda a: a.reshape(a.shape[0], -1, 4, 8)
        q = shape(self.norm_q(self.q(x)))
        k = shape(self.norm_k(self.k(context)))
        v = shape(self.v(context))
        return self.o(flash_attention(q, k, v, context_lens).flatten(2))


class NativeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = NativeAttention()
        self.cross_attn = NativeAttention()
        self.norm1, self.norm3, self.norm2 = nn.LayerNorm(32), nn.LayerNorm(32), nn.LayerNorm(32)
        self.ffn = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))

    def forward(self, x, e, lengths, grid, freqs, context, context_lens):
        h = self.norm1(x)
        x = x + .1*self.self_attn(h, h, None)
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        return x + self.ffn(self.norm2(x))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 16)

    def forward(self, x, e):
        return self.linear(x+e)


class SmallWan(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim, self.freq_dim, self.text_len = 32, 16, 8
        self.patch_embedding = nn.Conv3d(4, 32, (1, 2, 2), stride=(1, 2, 2))
        self.time_embedding = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 32))
        self.time_projection = nn.Linear(32, 192)
        self.text_embedding = nn.Linear(12, 32)
        self.blocks = nn.ModuleList([NativeBlock() for _ in range(3)])
        self.head = Head()
        self.freqs = torch.zeros(1)
        self.requires_grad_(False)

    def unpatchify(self, x, grid):
        t, h, w = grid[0].tolist()
        return list(x.reshape(1, t, h, w, 4, 2, 2).permute(0, 4, 1, 2, 5, 3, 6)
                    .reshape(1, 4, t, h*2, w*2))


class FrozenKnowledge(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, 140, 32), requires_grad=False)

    def encode(self, first_rgb, caption, clauses, **kwargs):
        memory = self.tokens + first_rgb.mean()*torch.linspace(-1, 1, 32)
        return dict(memory=memory, memory_mask=torch.ones(1, 140, dtype=torch.bool),
                    content_positions=torch.arange(140), reasoning='CPU integration fixture')


class Decoder(nn.Module):
    def forward(self, z, frame_indices=None, resize=None):
        size = resize or [16*z.shape[-2], 16*z.shape[-1]]
        video = F.interpolate(z[:, :3], size=(1+4*(z.shape[2]-1), *size), mode='trilinear', align_corners=True)
        frames = video[0].permute(1, 0, 2, 3).clamp(-1, 1)
        return frames if frame_indices is None else frames[frame_indices]


def model_and_sample(variant):
    cfg = small_config(variant)
    model = PhysGenNative(cfg, SmallWan(), knowledge=FrozenKnowledge(), decoder=Decoder(), device='cpu')
    z = torch.randn(1, 4, 6, 2, 4)*.2
    sample = dict(id='cpu_fixture', latent=z, first=z[:, :, :1].clone(), text=torch.randn(1, 5, 12),
                  times=torch.arange(6).float(), valid=torch.ones(2, 4),
                  caption='Steam spreads through the visible opening.', clauses=[])
    return model, sample
