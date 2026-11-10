"""No aspect distortion in reasoning, calibrated judges and feedback sampling."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests'))
from test_criterion_verifier import criterion_model, teacher_labels
from test_native import FakeWan, grad_size
from test_fullwidth_attention import NativeBlock
from physgen_native.config import read_config, validate, NATIVE_GEOMETRY_VARIANTS
from physgen_native.geometry import require_same_aspect
from physgen_native.knowledge import QwenKnowledge, visual_patches
from physgen_native.decoder import VideoDecoder
from physgen_native.sequence import sequence_error
from physgen_native.system import PhysGenNative
from physgen_native.runtime import parameter_groups, schedule, save_checkpoint, initialize_generator

torch.set_num_threads(2)


class GeometryDecoder(nn.Module):
    """Small differentiable VAE substitute with the declared 16x RGB geometry."""
    def forward(self, z, frame_indices=None, resize=None):
        size = [16*x for x in z.shape[-2:]]
        video = F.interpolate(z[:, :3], size=(1+4*(z.shape[2]-1), *size), mode='trilinear', align_corners=True)
        frames = video[0].permute(1,0,2,3)
        if frame_indices is not None: frames = frames[frame_indices]
        if resize is not None:
            require_same_aspect(size, resize)
            frames = F.interpolate(frames, size=resize, mode='bilinear', align_corners=False)
        return frames.clamp(-1,1)


def geometry_model(query=False, lora=False):
    from transformers import Qwen3VLForConditionalGeneration
    old, data = criterion_model(query, lora)
    cfg = copy.deepcopy(old.cfg)
    mode = cfg['knowledge']['reasoning_mode']
    cfg['revision'] = next(r for r, pair in NATIVE_GEOMETRY_VARIANTS.items() if pair == (mode, lora))
    cfg['geometry'] = dict(mode='native', rgb_size=[32,64])
    cfg['data'].update(latent_shape=[4,6,2,4], frames=21)
    cfg['knowledge'].update(image_size=[32,64], video_size=[32,64])
    cfg['verifier']['size'] = [32,64]
    cfg['sequence']['size'] = [32,64]
    cfg['sequence']['scales'] = [1]
    validate(cfg)
    knowledge = QwenKnowledge(cfg, 'cpu', backbone=Qwen3VLForConditionalGeneration(old.knowledge.backbone.config),
                              tokenizer=old.knowledge.tokenizer)
    wan = FakeWan(cfg); wan.blocks = nn.ModuleList([NativeBlock() for _ in range(3)])
    model = PhysGenNative(cfg, wan, knowledge=knowledge, decoder=GeometryDecoder(), device='cpu')
    data['latent'] = F.interpolate(data['latent'], size=(6,2,4), mode='trilinear', align_corners=False)
    data['first'] = data['latent'][:, :, :1].clone()
    data['valid'] = torch.ones(2,4)
    return model, data


class NativeGeometryTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(261031)

    def test_production_geometry_validation_rejects_silent_stretch(self):
        for revision,(mode,lora) in NATIVE_GEOMETRY_VARIANTS.items():
            cfg = read_config(ROOT/'configs'/f'native_{mode}_verifier_{"lora" if lora else "frozen"}_geometry_wisa3857.yaml')
            self.assertEqual(cfg['revision'], revision)
            self.assertEqual(cfg['geometry']['rgb_size'], [480,832])
            for section,key in (('knowledge','image_size'), ('knowledge','video_size'), ('verifier','size')):
                bad = copy.deepcopy(cfg); bad[section][key] = [288,512]
                with self.assertRaisesRegex(ValueError, 'native RGB'): validate(bad)
            bad = copy.deepcopy(cfg); bad['sequence']['size'] = [120,192]
            with self.assertRaisesRegex(ValueError, 'Aspect ratio'): validate(bad)
            bad = copy.deepcopy(cfg); bad['sequence']['size'] = [120,208]
            with self.assertRaisesRegex(ValueError, 'native-resolution RGB'): validate(bad)
            bad = copy.deepcopy(cfg); bad['sequence']['scales'] = [1,2,4]
            with self.assertRaisesRegex(ValueError, 'scale 1 only'): validate(bad)
            bad = copy.deepcopy(cfg); bad['geometry']['rgb_size'] = [832,480]
            with self.assertRaisesRegex(ValueError, 'stride 16'): validate(bad)
        require_same_aspect([480,832], [120,208])
        with self.assertRaises(ValueError): require_same_aspect([480,832], [192,320])

    def test_native_832x480_pixels_and_patch_coordinates_are_exact(self):
        h,w = 480,832
        y,x = torch.meshgrid(torch.linspace(-1,1,h), torch.linspace(-1,1,w), indexing='ij')
        image = torch.stack((x,y,((x*.6)**2+y**2 < .25).float()))[None].requires_grad_()
        with patch('physgen_native.knowledge.F.interpolate', side_effect=AssertionError('unexpected resampling')):
            patches, grid = visual_patches(image, [h,w], native_geometry=True)
        self.assertEqual(grid.tolist(), [[1,30,52]])
        self.assertEqual(len(patches)//4, 390)
        restored = patches.reshape(1,15,26,2,2,3,2,16,16).permute(0,6,5,1,3,7,2,4,8).reshape(1,2,3,h,w)
        torch.testing.assert_close(restored[:,0], image, rtol=0, atol=0)
        torch.testing.assert_close(restored[:,1], image, rtol=0, atol=0)
        patches.square().mean().backward(); self.assertGreater(float(image.grad.norm()), 0)
        with self.assertRaisesRegex(ValueError, 'Native observation size mismatch'):
            visual_patches(image, [288,512], native_geometry=True)

    def test_selected_native_decode_matches_full_frames_and_input_gradients(self):
        model,data = geometry_model()
        latent = data['latent'].clone().requires_grad_()
        with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as decode:
            frames,times = model.decode_qwen_view(latent, data)
        self.assertNotIn('resize', decode.call_args.kwargs)
        self.assertEqual(decode.call_args.kwargs['frame_indices'], [0,10,20])
        self.assertEqual(frames.shape, (3,3,32,64))
        torch.testing.assert_close(times, torch.tensor([0.,2.5,5.]))
        full = model.decoder(latent)[[0,10,20]]
        torch.testing.assert_close(frames, full, rtol=0, atol=0)
        probe = torch.randn_like(frames)
        a, = torch.autograd.grad((frames*probe).sum(), latent)
        b, = torch.autograd.grad((full*probe).sum(), latent)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_native_decoder_rejects_even_proportional_smaller_requests(self):
        decoder = VideoDecoder.__new__(VideoDecoder); nn.Module.__init__(decoder)
        decoder.preserve_aspect = True; decoder.native_size = (480,832)
        latent = torch.empty(1,48,1,30,52)
        with self.assertRaisesRegex(ValueError, 'Native observation size mismatch'):
            decoder(latent, resize=[120,208])
        with self.assertRaisesRegex(ValueError, 'Native observation size mismatch'):
            decoder(torch.empty(1,48,1,15,26))

    def test_native_loss_never_resamples_RGB_and_rejects_wrong_shapes(self):
        prediction = torch.rand(3,3,32,64,requires_grad=True)
        reference = torch.rand_like(prediction)
        cfg = dict(size=[32,64],scales=[1],lags=[1,2],scale_floor=.03,state_weight=0.)
        original = F.interpolate
        def masks_only(value,*args,**kwargs):
            if value.shape[1] == 3: raise AssertionError('RGB interpolation should not execute')
            return original(value,*args,**kwargs)
        with patch('physgen_native.sequence.F.interpolate',side_effect=masks_only):
            loss,_ = sequence_error(prediction,reference,torch.ones(2,4),cfg,native_geometry=True)
        loss.backward(); self.assertGreater(float(prediction.grad.norm()),0)
        with self.assertRaisesRegex(ValueError,'Native observation size mismatch'):
            sequence_error(prediction[:,:,:16,:32],reference,torch.ones(2,4),cfg,native_geometry=True)
        with self.assertRaisesRegex(ValueError,'must not pool'):
            sequence_error(prediction,reference,torch.ones(2,4),dict(cfg,scales=[1,2]),native_geometry=True)

    def test_four_variants_native_views_calibration_feedback_and_inference(self):
        for query in (False,True):
            for lora in (False,True):
                with self.subTest(query=query,lora=lora):
                    model,data = geometry_model(query,lora)
                    seen = []
                    original = model.knowledge._inputs
                    def record(clips, *args, **kwargs):
                        seen.extend(tuple(frames.shape[-2:]) for frames,_ in clips)
                        return original(clips, *args, **kwargs)
                    with patch.object(model.knowledge, '_inputs', side_effect=record):
                        losses,weights,_ = model(data, torch.zeros_like(data['text']), 0, sigma_override=.2)
                        model.backward_losses(losses,weights)
                        self.assertGreater(grad_size(model.conditioner),0)
                        model.zero_grad(set_to_none=True)
                        if lora:
                            model.set_phase(1)
                            with patch.object(model.knowledge, 'criterion_teacher', side_effect=teacher_labels):
                                losses,weights,_ = model.qwen_calibration(data,1,0,capture=True)
                            model.backward_losses(losses,weights)
                            self.assertGreater(grad_size(model.knowledge),0)
                            self.assertEqual(model.last_verifier_comparison['metadata']['observation_size_hw'],[32,64])
                            model.zero_grad(set_to_none=True)
                        step = 2 if lora else 1
                        losses,weights,metrics = model.feedback_losses(data,step)
                        model.backward_losses(losses,weights)
                        self.assertGreater(metrics['C_gradient_rms'],0)
                        self.assertEqual(metrics['full_video_frames'],21)
                        self.assertEqual(metrics['observer_frames'],3)
                        self.assertEqual(metrics['observer_native_geometry'],1)
                        self.assertEqual(grad_size(model.knowledge),0)
                        model.zero_grad(set_to_none=True); model.eval().requires_grad_(False)
                        model.cfg['inference']['latent_feedback'].update(enabled=True,sigmas=[.6],step_size=.01)
                        with torch.no_grad():
                            output = model.sample(model.conditions(data),data['text'],torch.zeros_like(data['text']),seed=7)
                        self.assertTrue(bool(torch.isfinite(output).all()))
                    self.assertTrue(seen)
                    self.assertEqual(set(seen),{(32,64)})

    def test_existing_message_parameters_can_initialize_new_geometry(self):
        old,_ = criterion_model(True,True)
        new,_ = geometry_model(True,True)
        optimizer = torch.optim.AdamW(parameter_groups(old,old.cfg)); scheduler = schedule(optimizer,old.cfg)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old.pt'
            save_checkpoint(old,optimizer,scheduler,old.cfg,0,0,path)
            result = initialize_generator(new,new.cfg,path)
        self.assertTrue(result['message_migration'])
        for (name,p),(other,q) in zip(old.conditioner.named_parameters(),new.conditioner.named_parameters()):
            self.assertEqual(name,other); torch.testing.assert_close(p,q)
        self.assertEqual(new.verifier_state['accepted'],0)


if __name__ == '__main__': unittest.main()
