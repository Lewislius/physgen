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
from physical_content.grounded_physics import native_block_with_read_state


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
    cfg = load_config(ROOT/'configs/r20ft.yaml')
    cfg['model'].update(width=128,ffn_width=256,native_heads=4,qwen_width=32,
        q_tokens=8,q_heads=4,grounded_heads=2)
    interface = dict(mean=torch.zeros(32),scale=torch.tensor(1.),native_scale=1.3)
    model = PhysicalModel(cfg,device='cpu',wan=copy.deepcopy(base),interface=interface)
    # The exposed read state must not change the official Wan computation.
    bs=torch.randn(1,12,128);bm=torch.randn(1,12,6,128)*.05
    bl=torch.tensor([12]);bg=torch.tensor([[3,2,2]]);bc=torch.randn(1,7,128)
    with torch.no_grad():
        for site in model.injection_layers:
            original=base.blocks[site](bs,bm,bl,bg,base.freqs,bc,None)
            exposed,_=native_block_with_read_state(base.blocks[site],bs,bm,bl,bg,base.freqs,bc)
            torch.testing.assert_close(original,exposed,atol=0,rtol=0)
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
    for site,physical in zip(model.injection_layers,model.physical_blocks):
        torch.testing.assert_close(physical.native.self_attn.q.weight,
                                   base.blocks[site].self_attn.q.weight,atol=0,rtol=0)
    before=model.base_bank.hashes(full=True)
    opt=torch.optim.AdamW(model.optim_groups(),lr=1e-3)
    for group in opt.param_groups:group['lr']=1e-3
    norms={}
    for step in (0,1,2,128):
        phase=model.set_phase(step);opt.zero_grad(set_to_none=True)
        values=backward_micro(model,evidence,sample,sample['text'],sigma,noise,.03,1)
        if step<128:assert all(p.grad is None for p in model.wan.parameters())
        else:assert model.wan.blocks[0].self_attn.q.lora_b.grad is not None
        for name,p in model.named_parameters():
            if p.grad is not None:norms[name]=max(norms.get(name,0),float(p.grad.norm()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
    required=['enhancement.seed','enhancement.route_update.weight','enhancement.execute_update.weight',
      'physical_blocks.0.conditioning.query.weight','physical_blocks.0.conditioning.key.weight',
      'physical_blocks.0.conditioning.value.weight','physical_blocks.0.conditioning.video_to_content.weight',
      'physical_blocks.0.query_update.weight','physical_blocks.4.key_update.weight',
      'physical_blocks.0.native.self_attn.v.weight','physical_blocks.0.native.cross_attn.k.weight',
      'physical_blocks.0.native.ffn.0.weight','physical_blocks.4.output.weight']
    for name in required:assert norms.get(name,0)>0,(name,norms.get(name))
    model.assert_base_unchanged(before,full=True)
    assert all(not p.requires_grad for n,p in model.wan.named_parameters() if '.cross_attn.' in n)
    with torch.no_grad():
        c,g=model.prepare_content(evidence,sample['text'])
        plain,_=model.velocity(noisy,None,sample['text'],sigma,c,g,sample['times'],physical_mode='off')
        empty,eg=model.prepare_content(dict(evidence,relations=['']),sample['text'])
        no_condition,_=model.velocity(noisy,None,sample['text'],sigma,empty,eg,sample['times'])
        torch.testing.assert_close(plain,no_condition,atol=0,rtol=0)
        active,_=model.velocity(noisy,None,sample['text'],sigma,c,g,sample['times'])
        assert (active-plain).abs().max()>1e-6
    # C changes interaction, not the underlying video V at fixed block input.
    physical=model.physical_blocks[0]
    state=torch.randn(1,12,128);mod=torch.randn(1,12,6,128)*.05
    grid=torch.tensor([[3,2,2]]);lengths=torch.tensor([12]);seen=[]
    handle=physical.native.self_attn.v.register_forward_hook(lambda m,a,y:seen.append(y.detach().clone()))
    with torch.no_grad():
        hint=physical(state,state,mod,lengths,grid,model.wan.freqs,c['context'],c['routing'],c['execution'])
        torch.testing.assert_close(seen[0],seen[1],atol=0,rtol=0)
        assert hint.abs().max()>1e-7
        physical.native.self_attn.v.weight.zero_()
        uniform=physical(state,state,mod,lengths,grid,model.wan.freqs,c['context'],c['routing'],c['execution'])
        torch.testing.assert_close(uniform,torch.zeros_like(uniform),atol=5e-6,rtol=5e-6)
    handle.remove()
    reader=physical.conditioning
    reader.video_to_content.weight.data.zero_()
    with torch.no_grad():
        original=reader(state,c['routing'],c['execution'])
        weak=reader(state,c['routing'],c['execution']*.01)
        torch.testing.assert_close(weak,original*.01,rtol=1e-4,atol=1e-7)
        fixed_state=state[:,:1].expand_as(state)
        reader.video_to_content.weight.data.normal_(std=.1)
        pooled=reader(fixed_state,c['routing'],c['execution'])
        unbound=reader(fixed_state,c['routing'],c['execution'],video_binding=False)
        torch.testing.assert_close(pooled,unbound,rtol=1e-4,atol=1e-7)
    print('PASS actual small Wan: native start; correct depth copies; physical/video and C gradients;128/272 ownership; original text frozen; exact blank-condition response; video-value contract. Quality is unproven.')

if __name__=='__main__':main()
