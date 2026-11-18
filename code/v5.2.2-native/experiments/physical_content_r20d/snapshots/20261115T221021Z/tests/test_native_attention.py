"""Actual small Wan: native initialization, value contract, gradients and stages."""
import copy
import importlib
from pathlib import Path
import sys
import warnings

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
from physical_content.common import load_config
from physical_content.model import PhysicalModel
from physical_content.wan_loader import ensure_wan_namespace
from physical_content.objectives import backward_micro


def main():
    torch.set_num_threads(2)
    torch.manual_seed(73)
    warnings.filterwarnings('ignore', message='User provided device_type of.*')
    ensure_wan_namespace({'paths': {'wan_code': '/home/liuzhirui/model/Wan2.2'}})
    native = importlib.import_module('wan.modules.model')
    native.flash_attention = lambda q,k,v,**kw: F.scaled_dot_product_attention(
        q.transpose(1,2).float(),k.transpose(1,2).float(),v.transpose(1,2).float()).transpose(1,2)
    base = native.WanModel(model_type='t2v', dim=128, ffn_dim=256, num_heads=4,
        num_layers=30, text_len=12, text_dim=64, freq_dim=16,
        in_dim=4, out_dim=4, patch_size=(1,2,2)).eval()
    nn.init.normal_(base.head.head.weight,std=.04)
    cfg = load_config(ROOT/'configs/r20dt.yaml')
    cfg['model'].update(width=128,ffn_width=256,native_heads=4,qwen_width=32,
        q_tokens=8,q_heads=4,grounded_heads=2)
    interface = dict(mean=torch.zeros(32),scale=torch.tensor(1.),native_scale=1.3)
    model = PhysicalModel(cfg,device='cpu',wan=copy.deepcopy(base),interface=interface)
    evidence = dict(words=torch.randn(6,32),request_words=torch.randn(5,32),
        native_condition=torch.randn(6,128)*.1,word_groups=torch.tensor([0,0,0,1,1,1]))
    sample = dict(latent=torch.randn(1,4,3,4,4),valid=torch.ones(4,4),
                  times=torch.arange(3)[None]/6,text=torch.randn(1,7,64))
    sigma, noise = torch.tensor(.7),torch.randn_like(sample['latent'])
    noisy = (1-sigma)*sample['latent']+sigma*noise
    with torch.no_grad():
        c,g = model.prepare_content(evidence,sample['text'])
        native_y,_ = model.velocity(noisy,None,sample['text'],sigma,c,g,sample['times'],physical_mode='off')
        initial_y,_ = model.velocity(noisy,None,sample['text'],sigma,c,g,sample['times'])
        torch.testing.assert_close(initial_y,native_y,atol=0,rtol=0)
    before = model.base_bank.hashes(full=True)
    opt = torch.optim.AdamW(model.optim_groups(),lr=1e-3)
    for group in opt.param_groups:group['lr']=1e-3
    norms = {}
    for step in (0,1,2,128):
        phase=model.set_phase(step);opt.zero_grad(set_to_none=True)
        backward_micro(model,evidence,sample,sample['text'],sigma,noise,0.,1)
        if step<128:assert all(p.grad is None for p in model.wan.parameters())
        else:
            assert phase=='joint'
            assert model.wan.blocks[0].self_attn.q.lora_b.grad is not None
        for name,p in model.named_parameters():
            if p.grad is not None:norms[name]=max(norms.get(name,0),float(p.grad.norm()))
        opt.step()
    for name in ['enhancement.seed','enhancement.execute_update.weight',
        'enhancement.route_update.weight','physical_blocks.0.reader.query_base.weight',
        'physical_blocks.0.reader.key.weight','physical_blocks.0.reader.value.weight',
        'physical_blocks.0.read_output.weight','physical_blocks.0.query_update.weight',
        'physical_blocks.4.key_update.weight','noise_modulation.project.1.weight']:
        assert norms.get(name,0)>0,name
    model.assert_base_unchanged(before,full=True)
    assert all(not p.requires_grad for n,p in model.wan.named_parameters() if '.cross_attn.' in n)
    # At fixed layer input, no C value enters the native video-value path.
    block=copy.deepcopy(model.wan.blocks[2]);physical=model.physical_blocks[0]
    h=torch.randn(1,12,128);mod=torch.randn(1,12,6,128)*.05
    grid=torch.tensor([[3,2,2]]);lengths=torch.tensor([12])
    pmod=model.noise_modulation(torch.randn(1,3,128))
    c,g=model.prepare_content(evidence,sample['text'])
    captured=[]
    hook=block.self_attn.v.register_forward_hook(lambda module,args,value:captured.append(value.detach().clone()))
    args=(block,h,mod,lengths,grid,model.wan.freqs,c['context'],c['routing'],
          c['execution'],g,sample['times'],pmod,4)
    normal,_=physical(*args,'normal');bypass,_=physical(*args,'physical_condition_off')
    torch.testing.assert_close(captured[0],captured[1],atol=0,rtol=0);hook.remove()
    assert (normal-bypass).abs().max()>1e-7
    # If native V is spatially constant, physical Q/K cannot create an image
    # value of its own. This checks the execution mechanism, not video quality.
    with torch.no_grad():
        block.self_attn.v.base.weight.zero_();block.self_attn.v.lora_b.zero_()
        normal,_=physical(*args,'normal');bypass,_=physical(*args,'physical_condition_off')
        torch.testing.assert_close(normal,bypass,atol=2e-6,rtol=2e-6)
    print('PASS: native initialization; real generation gradients; 128/272 stages; frozen text; native video values; nonzero conditional routing')


if __name__=='__main__':main()
