"""Draft architecture behavior; these checks do not measure video quality."""
import unittest
import torch
from grounded_experts import SemanticPhysicalEncoder, PhysicalStateExpert, GroundedPhysicalFusion


class GroundedExpertTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        self.encoder=SemanticPhysicalEncoder(16,12,4,8,2)
        self.expert=PhysicalStateExpert(12,3,24)
        self.fusion=GroundedPhysicalFusion(12,24,3)
        self.anchor=torch.randn(1,8,16)
        self.reasoning=torch.randn(1,11,16)
        self.caption=torch.randn(1,6,16)
        self.native=torch.randn(1,6,12)
        self.native_physics=torch.randn(1,8,12)
        self.valid=torch.tensor([[True,True,True,True,False,False]])
        self.video=torch.randn(1,3,9,12)
        self.times=torch.tensor([[0.,.5,1.]])
        self.noise_embedding=torch.randn(1,1,12).expand(1,3,12)
        self.groups=torch.arange(8)//2

    def content(self, reasoning=None, caption=None):
        return self.encoder(self.anchor,self.reasoning if reasoning is None else reasoning,
                            self.caption if caption is None else caption,self.native,self.valid,
                            self.native_physics)

    def activate(self):
        with torch.no_grad():
            self.fusion.feature_output.weight.normal_(0.,.1)
            self.expert.temporal_attention.out_proj.weight.normal_(0.,.1)
            self.expert.noise_modulation.weight.normal_(0.,.1)
            self.encoder.route_delta.weight.normal_(0.,.1)
            self.encoder.execute_delta.weight.normal_(0.,.1)

    def test_native_start_and_caption_mask(self):
        c=self.content()
        y=self.fusion(self.video,c,self.times,self.noise_embedding,self.groups,self.expert)
        torch.testing.assert_close(y,torch.zeros_like(y),atol=0,rtol=0)
        self.assertEqual(c['caption_binding'][...,-2:].count_nonzero().item(),0)
        self.assertGreater(c['caption_binding'][...,:4].sum().item(),0)

    def test_semantic_and_physical_inputs_both_affect_content(self):
        self.activate()
        c=self.content()
        for changed in [self.content(reasoning=self.reasoning.flip(1)*-1),
                        self.content(caption=self.caption*-1)]:
            self.assertGreater((c['execute']-changed['execute']).norm().item(),1e-6)

    def test_video_feedback_and_temporal_context_are_used(self):
        self.activate();c=self.content()
        y,info=self.fusion(self.video,c,self.times,self.noise_embedding,self.groups,self.expert,return_details=True)
        changed=self.video.clone();changed[:,0]*=-1
        y2,info2=self.fusion(changed,c,self.times,self.noise_embedding,self.groups,self.expert,return_details=True)
        self.assertGreater((info['state'][:,0]-info2['state'][:,0]).norm().item(),1e-6)
        self.assertGreater((y[:,2]-y2[:,2]).norm().item(),1e-6)
        off=self.fusion(self.video,c,self.times,self.noise_embedding,self.groups,self.expert,video_feedback=False)
        self.assertGreater((off-y).norm().item(),1e-6)

    def test_denoising_time_is_distinct_from_video_time(self):
        self.activate();c=self.content()
        y=self.fusion(self.video,c,self.times,self.noise_embedding,self.groups,self.expert)
        changed=self.fusion(self.video,c,self.times,-self.noise_embedding,self.groups,self.expert)
        self.assertGreater((y-changed).norm().item(),1e-6)

    def test_spatial_assignment_follows_video_positions(self):
        self.activate();c=self.content();perm=torch.randperm(9)
        y=self.fusion(self.video,c,self.times,self.noise_embedding,self.groups,self.expert)
        permuted=self.fusion(self.video[:,:,perm],c,self.times,self.noise_embedding,self.groups,self.expert)
        torch.testing.assert_close(permuted,y[:,:,perm],atol=2e-6,rtol=2e-5)

    def test_real_gradient_paths_and_unused_write_mass(self):
        self.activate();c=self.content()
        y,info=self.fusion(self.video,c,self.times,self.noise_embedding,self.groups,self.expert,return_details=True)
        loss=(y-torch.randn_like(y)).square().mean();loss.backward()
        modules=[self.encoder.semantic_attention.attention.out_proj,
                 self.encoder.physical_read.attention.out_proj,
                 self.encoder.request_read.attention.out_proj,
                 self.encoder.execute_delta,self.fusion.video_query,
                 self.fusion.content_key,self.fusion.video_value,
                 self.expert.relation_attention.out_proj,
                 self.expert.temporal_attention.out_proj,
                 self.fusion.feature_output,self.expert.noise_modulation]
        for m in modules:
            self.assertIsNotNone(m.weight.grad)
            self.assertGreater(m.weight.grad.norm().item(),0)
        self.assertTrue((info['write_weights'].sum(-1)<1).all())

    def test_autocast_and_caption_padding(self):
        self.activate()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            c=self.content()
            y=self.fusion(self.video,c,self.times,self.noise_embedding,self.groups,self.expert)
        self.assertTrue(y.isfinite().all())
        altered=self.caption.clone();altered[:,4:]=999
        torch.testing.assert_close(self.content()['route'],self.content(caption=altered)['route'])


if __name__=='__main__':unittest.main()
