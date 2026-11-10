from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim,self.num_heads,self.head_dim = 16,4,4
        self.q,self.k,self.v,self.o = [nn.Linear(16,16) for _ in range(4)]
        self.norm_q,self.norm_k = nn.LayerNorm(16),nn.LayerNorm(16)

    def forward(self,x,context):
        split = lambda v: v.reshape(1,-1,4,4).transpose(1,2)
        q,k,v = split(self.norm_q(self.q(x))),split(self.norm_k(self.k(context))),split(self.v(context))
        return self.o(F.scaled_dot_product_attention(q,k.to(q),v.to(q)).transpose(1,2).flatten(2))


class FakeLanguage(nn.Module):
    def __init__(self):
        super().__init__();self.embed_tokens = nn.Embedding(30,24);self.layer=nn.Linear(24,24)
    def forward(self,inputs_embeds,**kwargs):
        return SimpleNamespace(last_hidden_state=self.layer(inputs_embeds.cumsum(1)))


class FakeTokenizer:
    eos_token='!'
    def apply_chat_template(self,*a,**kw):return 'question'
    def encode(self,text,**kwargs):return [ord(c)%30 for c in text]


class SelfAttention(Attention):
    def forward(self,x,*args):
        return super().forward(x,x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn=SelfAttention();self.cross_attn=Attention()
        self.norm1=nn.LayerNorm(16);self.norm2=nn.LayerNorm(16);self.norm3=nn.LayerNorm(16)
        self.ffn=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,16))
        self.modulation=nn.Parameter(torch.randn(1,6,16)*.1)
    def forward(self,x,mod,lengths,grid,freqs,context,*args):
        e=(self.modulation.unsqueeze(0)+mod).chunk(6,dim=2)
        y=self.self_attn(self.norm1(x).float()*(1+e[1].squeeze(2))+e[0].squeeze(2))
        x=x+y*e[2].squeeze(2)
        x=x+self.cross_attn(self.norm3(x),context)
        return x+self.ffn(self.norm2(x).float()*(1+e[4].squeeze(2))+e[3].squeeze(2))*e[5].squeeze(2)


class FakeHead(nn.Linear):
    def forward(self,x,e):return super().forward(x)


class SmallWan(nn.Module):
    def __init__(self):
        super().__init__();self.dim=16;self.freq_dim=8;self.text_len=5
        self.freqs=torch.zeros(4)
        self.patch_embedding=nn.Conv3d(2,16,(1,2,2),stride=(1,2,2))
        self.time_embedding=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,16))
        self.time_projection=nn.Linear(16,96)
        self.text_embedding=nn.Linear(24,16)
        self.blocks=nn.ModuleList([Block() for _ in range(3)])
        self.head=FakeHead(16,8)
    def unpatchify(self,x,grid):
        t,h,w=grid[0].tolist()
        return [x[0].reshape(t,h,w,1,2,2,2).permute(6,0,3,1,4,2,5).reshape(2,t,h*2,w*2)]


