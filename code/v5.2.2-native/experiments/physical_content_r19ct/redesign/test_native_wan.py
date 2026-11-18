"""Native Wan operators, shortened dimensions, through the new complete path."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import bootstrap
from physical_content.common import load_config
from physical_content.wan_loader import ensure_wan_namespace
from physical_content.model import PhysicalModel


class NativeWanIntegration(unittest.TestCase):
    def test_native_operators_forward_backward_and_initial_neutrality(self):
        torch.manual_seed(31);torch.set_num_threads(2)
        cfg=load_config(ROOT/'configs/r19ct.yaml')
        ensure_wan_namespace(cfg)
        from wan.modules.model import WanModel
        cfg['model'].update(width=32,ffn_width=64,native_heads=4,injection_layers=[0,1],
                            q_tokens=8,q_heads=4,qwen_width=16,state_initialization_layer=1)
        cfg['train']['gradient_checkpointing']=True
        wan=WanModel(dim=32,ffn_dim=64,freq_dim=16,text_dim=16,text_len=8,
                     in_dim=48,out_dim=48,num_heads=4,num_layers=30)
        with torch.no_grad():wan.head.head.weight.normal_(0.,.02)
        model=PhysicalModel(cfg,device='cpu',wan=wan,
                            interface=dict(mean=torch.zeros(16),scale=torch.ones(())))
        ev=dict(words=torch.randn(12,16),word_groups=torch.arange(12)//3,
                native_condition=torch.randn(12,32))
        text=torch.randn(1,7,16);noisy=torch.randn(1,48,3,4,4)
        c,g=model.prepare_content(ev,text)
        args=(noisy,None,text,torch.tensor(.4),c,g,torch.tensor([[0.,.5,1.]]))
        def cpu_attention(q,k,v,**kwargs):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
        # Only the CUDA-only attention kernel is replaced by its CPU reference.
        # Native projection, normalization, RoPE, modulation and head code run.
        with patch('wan.modules.model.flash_attention',cpu_attention):
            normal=model.velocity(*args)[0];off=model.velocity(*args,physical_mode='off')[0]
            torch.testing.assert_close(normal,off,atol=0,rtol=0)
            normal.square().mean().backward()
        self.assertGreater(model.mechanisms['0'].feature_output.weight.grad.norm().item(),0.)
        self.assertEqual(normal.shape,noisy.shape)
        self.assertTrue(normal.isfinite().all())


if __name__=='__main__':unittest.main()
