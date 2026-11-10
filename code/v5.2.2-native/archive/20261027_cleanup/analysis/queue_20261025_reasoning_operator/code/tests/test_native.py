"""Behavioral tests for actual process objectives, gradient ownership and native operators."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import weakref
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from physgen_native.config import read_config
from physgen_native.system import PhysGenNative
from physgen_native.knowledge import visual_patches, QwenKnowledge
from physgen_native.lora import LoRALinear, knowledge_role, role_checkpoint
from physgen_native.sequence import sequence_error,pair_scale
from physgen_native.verifier import ProcessCritic, omitted_process
from physgen_native.runtime import parameter_groups, schedule, save_checkpoint, load_checkpoint, restore

torch.set_num_threads(2)


def small_config():
    cfg = read_config(ROOT/'configs/native_physical_wisa3857.yaml')
    cfg['model'].update(width=32, qwen_width=32, attention_width=16, attention_heads=4,
                        interface_rank=4, condition_every=1, critic_width=16)
    cfg['knowledge'].update(image_size=[32,64], video_size=[32,64], lora_last_layers=2,
                            lora_rank=2, lora_alpha=2, evidence_pairs=3,reasoning_tokens=4,comparison_tokens=4)
    cfg['stages'] = dict(interface=1, joint=2)
    cfg['wan_lora'] = dict(rank=2, alpha=2)
    cfg['train'].update(accumulation=1, warmup=1, text_dropout=0., requirements_dropout=0.)
    cfg['sequence'].update(size=[8,12], scales=[1,2], lags=[1,4,8,16])
    cfg['verifier'].update(every=1, warmup=1, perturb_lengths=[2,3])
    cfg['rollout'].update(steps=5, gradient_steps=2, start=0, every=1)
    cfg['inference'].update(steps=5)
    cfg['data']['latent_shape'] = [4,6,4,6]
    return cfg


class FakeAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
    def forward(self, x, context):
        q, k, v = self.q(x), self.k(context), self.v(context)
        return self.o((q@k.transpose(-1, -2)/x.shape[-1]**.5).softmax(-1)@v)


class FakeBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn, self.cross_attn = FakeAttention(d), FakeAttention(d)
        self.norm = nn.LayerNorm(d)
    def forward(self, x, e, lengths, grid, freqs, context, context_lens):
        x = x+.1*self.self_attn(self.norm(x), self.norm(x))
        return x+.1*self.cross_attn(self.norm(x), context)


class FakeHead(nn.Module):
    def __init__(self, d): super().__init__(); self.linear = nn.Linear(d, 16)
    def forward(self, x, e): return self.linear(x+e)


class FakeWan(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim=32; self.freq_dim=16; self.text_len=8
        self.patch_embedding = nn.Conv3d(4, 32, (1, 2, 2), stride=(1, 2, 2))
        self.time_embedding = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 32))
        self.time_projection = nn.Linear(32, 192)
        self.text_embedding = nn.Linear(12, 32)
        self.blocks = nn.ModuleList([FakeBlock(32) for _ in range(3)])
        self.head = FakeHead(32)
        self.freqs = torch.zeros(1)
        self.requires_grad_(False)
    def unpatchify(self, x, grid):
        t,h,w = grid[0].tolist()
        y = x.reshape(1,t,h,w,4,2,2).permute(0,4,1,2,5,3,6).reshape(1,4,t,h*2,w*2)
        return list(y)


class FakeKnowledge(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = LoRALinear(nn.Linear(3,32),2,2,knowledge=True)
        self.fixed_features = nn.Linear(6,32).requires_grad_(False)
    def encode(self, first_rgb, caption, clauses, *, adapt=True, readout=None):
        with knowledge_role(adapt): vector = self.projection(first_rgb.mean((-2,-1)))
        n=len(clauses)+3
        return dict(contextual=vector[:,None].expand(-1,n,-1)+torch.arange(n)[None,:,None]*.05,
                    roles=torch.tensor([1]*len(clauses)+[3]*3))
    def evidence(self, frames, times, caption, clauses):
        pool=frames.mean((-2,-1))
        summary=torch.cat((pool.mean(0), (pool[1:]-pool[:-1]).mean(0)))
        return self.fixed_features(summary)[None,None].expand(1,3,-1)
    def process_features(self,frames):
        return frames[::4].permute(0,2,3,1)
    @torch.no_grad()
    def preference(self, first, second, times, caption, clauses):
        p=first.new_tensor([[.75,.1,.15]])
        return p[:,0]+.5*p[:,2],p


class FakeDecoder(nn.Module):
    def forward(self, z, frame_indices=None, resize=None):
        t = 1+4*(z.shape[2]-1)
        video = F.interpolate(z[:, :3], size=(t, 8, 12), mode='trilinear', align_corners=True)
        frames = video[0].permute(1,0,2,3)
        return frames if frame_indices is None else frames[frame_indices]


def sample():
    z = torch.randn(1,4,6,4,6)*.2
    return dict(id='test', latent=z, first=z[:, :, :1].clone(), times=torch.arange(6).float(),
                text=torch.randn(1,6,12), caption='A body changes after an interaction.',
                clauses=[dict(text='An interaction occurs.', basis='prompt', phase='evolution'),
                         dict(text='The earlier state evolves.', basis='prompt', phase='evolution')],
                valid=torch.ones(4,6), intensity=torch.rand(1,5,4,6), change_valid=torch.ones(1,5,4,6))


def small_model():
    cfg=small_config()
    return PhysGenNative(cfg,FakeWan(cfg),knowledge=FakeKnowledge(),decoder=FakeDecoder(),device='cpu')


def grad_size(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


class NativeProcessTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(21)

    def test_precomputed_positive_and_null_conditions_preserve_sampling(self):
        from unittest.mock import patch
        model=small_model().eval();data=sample();condition=model.conditions(data)
        null=torch.zeros_like(data['text']);policy=dict(model.cfg['rollout'],guidance=5.)
        with torch.no_grad():
            positive=model.encode_condition(condition)
            negative=model.encode_condition(dict(condition,caption='',clauses=[]))
            expected=model._sample_tokens(condition,data['text'],null,42,policy)
            with patch.object(model.knowledge,'encode',side_effect=AssertionError('Unexpected Qwen recomputation')), \
                 patch.object(model.decoder,'forward',side_effect=AssertionError('Unexpected decoding')):
                actual=model._sample_tokens(condition,data['text'],null,42,policy,
                    token_override=positive,negative_token_override=negative)
        torch.testing.assert_close(expected,actual,atol=0,rtol=0)
        del model.knowledge,model.conditioner,model.critic
        with torch.no_grad():
            released=model._sample_tokens(condition,data['text'],null,42,policy,
                token_override=positive,negative_token_override=negative)
        torch.testing.assert_close(expected,released,atol=0,rtol=0)

    def test_euler_step_recomputation_preserves_full_condition_derivative(self):
        model=small_model().eval();data=sample();condition=model.conditions(data);null=torch.zeros_like(data['text'])
        c=model.encode_condition(condition).detach().requires_grad_(True)
        policy=dict(model.cfg['rollout'],gradient_steps=5)
        direct=model._sample_tokens(condition,data['text'],null,42,policy,differentiable=True,token_override=c)
        grad=torch.autograd.grad(direct.square().mean(),c)[0]
        recompute=model._sample_tokens(condition,data['text'],null,42,dict(policy,step_checkpointing=True),differentiable=True,token_override=c)
        new_grad=torch.autograd.grad(recompute.square().mean(),c)[0]
        torch.testing.assert_close(direct,recompute,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(grad,new_grad,atol=1e-7,rtol=1e-5)

    def test_condition_refinement_window_matches_exact_complete_sampler_gradient(self):
        model=small_model().eval().requires_grad_(False);data=sample();condition=model.conditions(data)
        null=torch.zeros_like(data['text']);base=model.encode_condition(condition).detach()
        c=(base+.02).requires_grad_(True)
        policy=dict(model.cfg['rollout'],steps=5,gradient_steps=5)
        kwargs=dict(token_override=c,prefix_token_override=base,condition_update_start=3)
        full=model._sample_tokens(condition,data['text'],null,42,policy,differentiable=True,**kwargs)
        gfull=torch.autograd.grad(full.square().mean(),c)[0]
        suffix=model._sample_tokens(condition,data['text'],null,42,dict(policy,gradient_steps=2,discrete_adjoint=True),
                                    differentiable=True,**kwargs)
        gsuffix=torch.autograd.grad(suffix.square().mean(),c)[0]
        torch.testing.assert_close(full,suffix,atol=0,rtol=0)
        torch.testing.assert_close(gfull,gsuffix,atol=1e-7,rtol=1e-5)

    def test_discrete_adjoint_preserves_native_cfg_condition_and_parameter_gradients(self):
        model=small_model().eval();model.set_phase(1)
        data=sample();condition=model.conditions(data);null=torch.zeros_like(data['text'])
        c=model.encode_condition(condition).detach().requires_grad_(True)
        u=model.encode_condition(dict(condition,caption='',clauses=[])).detach().requires_grad_(True)
        policy=dict(model.cfg['rollout'],gradient_steps=5,guidance=5.)
        inputs=(c,u,*[p for p in model.generator.parameters() if p.requires_grad])
        direct=model._sample_tokens(condition,data['text'],null,42,policy,differentiable=True,
            token_override=c,negative_token_override=u)
        expected=torch.autograd.grad(direct.square().mean(),inputs,allow_unused=True)
        actual=model._sample_tokens(condition,data['text'],null,42,dict(policy,discrete_adjoint=True),
            differentiable=True,token_override=c,negative_token_override=u)
        obtained=torch.autograd.grad(actual.square().mean(),inputs,allow_unused=True)
        torch.testing.assert_close(direct,actual,atol=0,rtol=0)
        for a,b in zip(expected,obtained):
            if a is None:self.assertIsNone(b)
            else:torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-4)

    def test_ordered_process_updates_condition_without_a_trainable_verifier(self):
        class Observer(nn.Module):
            def __init__(self):
                super().__init__();self.projection=nn.Linear(3,5)
            def forward(self,frames):return self.projection(frames[::4].permute(0,2,3,1))
        cfg=small_config();cfg['verifier'].update(kind='ordered_process',size=[8,12],lags=[1,2,4,5],scale_floor=.03)
        model=PhysGenNative(cfg,FakeWan(cfg),knowledge=FakeKnowledge(),decoder=FakeDecoder(),device='cpu',process_encoder=Observer())
        data=sample();data['latent'].requires_grad_(True)
        losses,weights,metrics=model.rollout_losses(data,torch.zeros_like(data['text']),1,seed=42)
        model.backward_losses(losses,weights)
        self.assertGreater(grad_size(model.conditioner),0)
        self.assertGreater(grad_size(model.knowledge),0)
        self.assertEqual(sum(p.numel() for p in model.critic.parameters()),0)
        self.assertEqual(grad_size(model.process_encoder),0)
        self.assertIsNone(data['latent'].grad)
        self.assertTrue(metrics['target_uses_reference_future'])
        self.assertFalse(metrics['condition_uses_reference_future'])

    def test_calibrated_verifier_retains_full_generation_condition_feedback(self):
        self.check_calibrated_feedback('regression')

    def test_paired_verifier_retains_full_generation_condition_feedback(self):
        self.check_calibrated_feedback('paired')

    def check_calibrated_feedback(self,method):
        class Observer(nn.Module):
            def __init__(self):
                super().__init__();self.projection=nn.Linear(3,5)
            def forward(self,frames):return self.projection(frames[::4].permute(0,2,3,1))
        cfg=small_config()
        cfg['verifier'].update(kind='calibrated_process',size=[8,12],feature_width=5,width=16,
                              lags=[1,2,4,5],scale_floor=.03,calibration=method)
        cfg['rollout'].update(gradient_steps=5,discrete_adjoint=True)
        model=PhysGenNative(cfg,FakeWan(cfg),knowledge=FakeKnowledge(),decoder=FakeDecoder(),
                           device='cpu',process_encoder=Observer())
        data=sample();data['latent'].requires_grad_(True)
        losses,weights,metrics=model.rollout_losses(data,torch.zeros_like(data['text']),1,seed=42)
        losses['rollout'].backward(retain_graph=True)
        self.assertGreater(grad_size(model.knowledge),0)
        self.assertGreater(grad_size(model.conditioner),0)
        self.assertGreater(grad_size(model.generator.interface),0)
        self.assertEqual(grad_size(model.critic),0)
        self.assertIsNone(data['latent'].grad)
        model.zero_grad(set_to_none=True);losses['critic'].backward()
        self.assertGreater(grad_size(model.critic),0)
        self.assertEqual(grad_size(model.knowledge),0)
        self.assertEqual(grad_size(model.generator),0)
        self.assertFalse(metrics['verifier_scoring_uses_reference_future'])
        self.assertFalse(metrics['condition_uses_reference_future'])
        self.assertEqual(metrics['verifier_paired_calibration'],int(method=='paired'))

    def test_native_text_can_drop_without_erasing_knowledge_condition(self):
        from unittest.mock import patch
        model=small_model();data=sample();null=torch.zeros_like(data['text'])
        model.cfg['train']['native_text_only_dropout']=1.
        seen=[]
        hook=model.generator.register_forward_pre_hook(lambda module,args:seen.append(args[2].detach().clone()))
        try:
            with patch.object(model.knowledge,'encode',wraps=model.knowledge.encode) as encode:
                _,_,metrics=model(data,null,0,sigma_override=.5)
                self.assertEqual(encode.call_args.args[1],data['caption'])
            self.assertTrue(torch.equal(seen[0],null))
            self.assertEqual(metrics['knowledge_text_dropped'],0)
            self.assertEqual(metrics['native_text_dropped'],1)
        finally:hook.remove()

    def test_temporal_verifier_reaches_generated_process_without_condition_shortcut(self):
        cfg=small_config();cfg['verifier'].update(kind='temporal_distribution',feature_width=3,width=16,warmup=0)
        model=PhysGenNative(cfg,FakeWan(cfg),knowledge=FakeKnowledge(),decoder=FakeDecoder(),device='cpu')
        data=sample();null=torch.zeros_like(data['text'])
        losses,weights,metrics=model.rollout_losses(data,null,1,seed=42)
        losses['rollout'].backward()
        self.assertGreater(grad_size(model.conditioner),0)
        self.assertGreater(grad_size(model.knowledge),0)
        self.assertEqual(grad_size(model.critic),0)
        self.assertTrue(metrics['sampled_from_noise'])
        self.assertEqual(metrics['generator_feedback_scale'],1.)

    def test_both_phases_train_generation_and_knowledge(self):
        model=small_model(); data=sample()
        for step in (0,1):
            model.zero_grad(set_to_none=True)
            losses,weights,_=model(data,torch.zeros_like(data['text']),step,sigma_override=.5)
            model.backward_losses(losses,weights)
            for module in (model.conditioner,model.generator.interface,model.knowledge,model.critic):
                self.assertGreater(grad_size(module),0)
            lora=sum(float(p.grad.abs().sum()) for n,p in model.generator.wan.named_parameters() if 'lora_' in n and p.grad is not None)
            self.assertEqual(lora>0,step==1)
            self.assertTrue(all(p.grad is None for n,p in model.generator.wan.named_parameters() if 'lora_' not in n))

    def test_each_process_objective_reaches_knowledge(self):
        for objective in ('sequence','physical'):
            model=small_model(); data=sample()
            losses,_,_=model(data,torch.zeros_like(data['text']),1,sigma_override=.5)
            losses[objective].backward()
            for module in (model.conditioner,model.generator.interface,model.knowledge):
                self.assertGreater(grad_size(module),0,objective)
            self.assertEqual(grad_size(model.critic),0)

    def test_calibration_cannot_update_generation(self):
        model=small_model(); data=sample()
        losses,_,_=model(data,torch.zeros_like(data['text']),1,sigma_override=.5)
        losses['critic'].backward()
        self.assertGreater(grad_size(model.critic),0)
        self.assertEqual(grad_size(model.generator)+grad_size(model.conditioner)+grad_size(model.knowledge),0)

    def test_intensity_is_used_by_generation(self):
        model=small_model(); data=sample()
        losses,_,_=model(data,torch.zeros_like(data['text']),0,sigma_override=.5)
        losses['fm'].backward()
        self.assertGreater(grad_size(model.generator.interface.intensity),0)

    def test_future_targets_do_not_enter_conditions(self):
        model=small_model(); data=sample()
        before=model.encode_condition(model.conditions(data))
        data['latent']=torch.randn_like(data['latent'])*100
        data['intensity'].zero_(); data['evidence_activity']=torch.ones(5)*100
        after=model.encode_condition(model.conditions(data))
        torch.testing.assert_close(before,after)
        self.assertEqual(set(model.conditions(data)),{'first','times','valid','caption','clauses'})

    def test_sampling_suffix_matches_full_sampler_and_backpropagates(self):
        model=small_model(); data=sample(); c=model.conditions(data); null=torch.zeros_like(data['text'])
        model.set_phase(1)
        seen=[]
        hook=model.generator.register_forward_pre_hook(lambda m,a: seen.append(torch.is_grad_enabled()))
        generated=model._sample_tokens(c,data['text'],null,13,model.cfg['rollout'],differentiable=True)
        hook.remove()
        self.assertEqual(seen,[False]*6+[True]*4)
        with torch.no_grad(): expected=model._sample_tokens(c,data['text'],null,13,model.cfg['rollout'])
        torch.testing.assert_close(generated,expected)
        torch.testing.assert_close(generated[:,:,:1],data['first'])
        generated[:,:,-1].square().mean().backward()
        self.assertGreater(grad_size(model.knowledge),0)
        self.assertGreater(grad_size(model.generator.interface),0)

    def test_signed_process_rejects_late_stagnation_and_transient_omission(self):
        cfg=small_config()['sequence']; cfg.update(lags=[1,4,8,16,32,64])
        y,x=torch.meshgrid(torch.linspace(-1,1,8),torch.linspace(-1,1,12),indexing='ij')
        blob=torch.exp(-(x*x+y*y)/.2)
        truth=torch.zeros(81,3,8,12)
        for t in range(1,80):
            truth[t]=torch.sin(torch.tensor(t*torch.pi/80))*blob
        static=torch.zeros_like(truth,requires_grad=True)
        loss,_=sequence_error(static,truth,torch.ones(4,6),cfg)
        loss.backward()
        self.assertGreater(float(static.grad[65:80].abs().sum()),0)
        self.assertGreater(float(loss),.01)
        better=static.detach()-static.grad*100
        self.assertLess(float(sequence_error(better,truth,torch.ones(4,6),cfg)[0]),float(loss))
        self.assertEqual(float(sequence_error(truth,truth,torch.ones(4,6),cfg)[0]),0)
        wrong=-truth
        self.assertGreater(float(sequence_error(wrong,truth,torch.ones(4,6),cfg)[0]),float(loss))
        # A genuinely stationary sequence needs neither invented dynamics nor a floor on activity.
        self.assertEqual(float(sequence_error(static.detach(),static.detach(),torch.ones(4,6),cfg)[0]),0)

    def test_early_energy_does_not_rescale_late_supervision(self):
        energy=torch.ones(20,4,6)*.04
        scale=pair_scale(energy,torch.ones(4,6),.03)
        energy[0]=100
        changed=pair_scale(energy,torch.ones(4,6),.03)
        torch.testing.assert_close(changed[1:],scale[1:])

    def test_case_bootstrap_preserves_paired_seed_controls(self):
        from tools.compare_evaluations import compare
        rows={f'c{i}':dict(sa=3,pc=3) for i in range(8)}
        same=compare([rows,rows],[rows,rows],draws=100)
        self.assertEqual(same['metrics']['pc']['case_bootstrap_95'],[0.,0.])
        up={k:dict(sa=4,pc=4) for k in rows};down={k:dict(sa=2,pc=2) for k in rows}
        balanced=compare([rows,rows],[up,down],draws=100)
        self.assertEqual(balanced['metrics']['pc']['delta'],0.)
        self.assertEqual(balanced['metrics']['pc']['seed_deltas'],[1.,-1.])

    def test_critic_learns_pair_order_and_input_gradient_improves_score(self):
        critic=ProcessCritic(small_config()); opt=torch.optim.Adam(critic.parameters(),lr=.02)
        a=torch.randn(8,3,32); b=a+torch.arange(32)/16
        for _ in range(80):
            opt.zero_grad(); loss,_=critic.calibration_loss(a,b,torch.ones(8)*.9); loss.backward(); opt.step()
        with torch.no_grad(): self.assertGreater(float((critic(a)-critic(b)).mean()),1)
        critic.zero_grad(set_to_none=True); candidate=b.clone().requires_grad_()
        loss,_=critic.generation_loss(candidate,a); loss.backward()
        self.assertEqual(grad_size(critic),0)
        improved=candidate.detach()-candidate.grad*10
        self.assertLess(float(critic.generation_loss(improved,a)[0]),float(loss))

    def test_resume_preserves_optimizer_and_phase(self):
        model=small_model(); cfg=model.cfg; data=sample()
        opt=torch.optim.AdamW(parameter_groups(model,cfg),foreach=False); sched=schedule(opt,cfg)
        losses,weights,_=model(data,torch.zeros_like(data['text']),0,sigma_override=.5)
        model.backward_losses(losses,weights); opt.step(); sched.step()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'latest.pt'; save_checkpoint(model,opt,sched,cfg,1,8,path)
            new=small_model(); optimizer=torch.optim.AdamW(parameter_groups(new,cfg),foreach=False); scheduler=schedule(optimizer,cfg)
            self.assertEqual(restore(new,optimizer,scheduler,load_checkpoint(path),cfg),(1,8))
            for x,y in zip(model.conditioner.parameters(),new.conditioner.parameters()): torch.testing.assert_close(x,y)
            self.assertEqual(scheduler.last_epoch,sched.last_epoch)
            self.assertEqual(new.set_phase(1),'joint')

    def test_checkpoint_captures_knowledge_role(self):
        linear=LoRALinear(nn.Linear(4,4),2,2,knowledge=True)
        with torch.no_grad(): linear.lora_b.fill_(.1)
        x=torch.randn(2,4,requires_grad=True)
        with knowledge_role(False): teacher=role_checkpoint(linear,x)
        with knowledge_role(True): knowledge=role_checkpoint(linear,x)
        teacher.sum().backward(retain_graph=True)
        self.assertIsNone(linear.lora_b.grad)
        self.assertGreater(float(x.grad.abs().sum()),0)
        knowledge.sum().backward()
        self.assertGreater(float(linear.lora_b.grad.abs().sum()),0)


class NativeQwenTests(unittest.TestCase):
    def test_video_packing_preserves_adjacent_frames(self):
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        processor=Qwen3VLVideoProcessor(do_resize=False,do_rescale=False,do_sample_frames=False,
                                       image_mean=[.5]*3,image_std=[.5]*3,
                                       patch_size=16,temporal_patch_size=2,merge_size=2)
        rgb=(torch.rand(4,3,32,64)*2-1).requires_grad_()
        patches,grid=visual_patches(rgb,[32,64],video=True)
        official=processor(videos=[(rgb.detach()+1)/2],return_tensors='pt')
        torch.testing.assert_close(patches,official['pixel_values_videos'],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(grid,official['video_grid_thw'])
        self.assertGreater(float(torch.autograd.grad(patches.square().mean(),rgb)[0].abs().sum()),0)

    def test_native_video_features_have_input_gradients_but_no_teacher_lora(self):
        from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration,AutoTokenizer
        cfg=small_config()
        config=Qwen3VLConfig(text_config=dict(vocab_size=151936,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            rope_scaling={'rope_type':'default','mrope_section':[1,1,2],'mrope_interleaved':True}),
            vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,
                out_hidden_size=32,num_position_embeddings=64,deepstack_visual_indexes=[0,1,2],
                patch_size=16,temporal_patch_size=2,spatial_merge_size=2))
        qwen=QwenKnowledge(cfg,device='cpu',backbone=Qwen3VLForConditionalGeneration(config),
                           tokenizer=AutoTokenizer.from_pretrained(cfg['paths']['qwen_checkpoint'],local_files_only=True)).train()
        for name,p in qwen.named_parameters(): p.requires_grad_(name.endswith(('lora_a','lora_b')))
        image=torch.rand(1,3,32,64)*2-1
        condition=qwen.encode(image,'A wave propagates.',[])
        self.assertEqual(condition['contextual'].shape,(1,4,32))
        self.assertTrue(condition['reasoning'])
        qwen.cfg['max_condition_tokens']=16
        content=qwen.encode(image,'A wave propagates.',[
            dict(text='The initial water surface is visible.',basis='image'),
            dict(text='The disturbance spreads across the surface over time.',basis='prompt')],readout='content')
        self.assertGreater(content['contextual'].shape[1],4)
        self.assertLessEqual(content['contextual'].shape[1],16)
        self.assertEqual(content['roles'].numel(),content['contextual'].shape[1])
        self.assertEqual(set(content['roles'].tolist()),{0,1,3})
        content['contextual'].square().mean().backward()
        self.assertGreater(grad_size(qwen),0)
        qwen.zero_grad(set_to_none=True)
        frames=torch.rand(6,3,32,64,requires_grad=True); times=torch.arange(6).float()
        evidence=qwen.evidence(frames,times,'A wave propagates.',[])
        self.assertEqual(evidence.shape,(1,3,32))
        evidence.square().mean().backward()
        self.assertGreater(float(frames.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for n,p in qwen.named_parameters() if 'lora_' in n))
        condition['contextual'].square().mean().backward()
        self.assertGreater(grad_size(qwen),0)
        pref,prob=qwen.preference(frames.detach(),frames.flip(0).detach(),times,'A wave propagates.',[])
        self.assertTrue(bool(torch.isfinite(pref).all())); self.assertEqual(prob.shape,(1,3))
        clip=torch.rand(21,3,32,64,requires_grad=True)
        features=qwen.process_features(clip)
        self.assertEqual(features.shape,(6,1,2,32))
        features[-1].square().mean().backward()
        self.assertGreater(float(clip.grad[-1].abs().sum()),0)


class NativeVAETests(unittest.TestCase):
    def test_temporal_checkpoint_matches_native_decode_and_gradient(self):
        from physgen_native.decoder import VideoDecoder
        cfg=small_config()
        spec=importlib.util.spec_from_file_location('native_vae_for_test',
            Path(cfg['paths']['wan_code'])/'wan/modules/vae2_2.py')
        native=importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
        vae=native.WanVAE_(dim=8,dec_dim=8,z_dim=4,num_res_blocks=1,
                          temperal_downsample=[False,True,True]).requires_grad_(False).eval()
        decoder=VideoDecoder.__new__(VideoDecoder); nn.Module.__init__(decoder)
        decoder.vae=vae
        decoder.register_buffer('mean',torch.tensor([.1,-.2,.3,-.4]))
        decoder.register_buffer('inverse_std',torch.tensor([1.2,.9,1.1,.8]))
        decoder.cache_count=native.count_conv3d(vae.decoder)
        decoder.unpatchify=native.unpatchify; decoder.checkpointing=True
        z=torch.randn(1,4,6,2,2,requires_grad=True)
        selected=[0,2,4,5,8,17,20]
        reference=vae.decode(z,[decoder.mean,decoder.inverse_std]).float().clamp(-1,1)
        reference=reference[0].permute(1,0,2,3)[selected]
        caches=[]
        def observe_cache(module, inputs, kwargs, output):
            caches.append([weakref.ref(x) for x in kwargs['feat_cache'] if torch.is_tensor(x)])
        handle=vae.decoder.register_forward_hook(observe_cache,with_kwargs=True)
        target=decoder(z,selected)
        handle.remove()
        # A within-segment cache must be released after forward; numerical
        # equivalence alone would not catch a nested-input checkpoint memory leak.
        self.assertGreater(len(caches[1]),0)
        self.assertLess(sum(ref() is not None for ref in caches[1]),len(caches[1])//2)
        torch.testing.assert_close(target,reference)
        expected=torch.autograd.grad(reference.square().mean(),z)[0]
        # A second decode must not change the first call's temporal caches.
        decoder(z.detach()+.1,[0,3,7])
        actual=torch.autograd.grad(target.square().mean(),z)[0]
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-6)
        self.assertGreater(float(actual[:,:,:1].abs().sum()),0)


if __name__=='__main__': unittest.main(verbosity=2)
