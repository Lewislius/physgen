"""Control ownership and actual removal of physical contributions."""
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import torch

from test_tiny_model import ROOT, W, FF, FakeWan
from physical_content.common import load_config
from physical_content.model import PhysicalModel
from physical_content.interventions import physics_evidence, fixed_content, velocity_mode
from condition_controls import prepared_i2v_control, validate_control_input


class InterventionTests(unittest.TestCase):
    def test_empty_is_actually_empty_and_own_image_is_preserved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'videophy40_VP004/controls/empty_reasoning.pt'
            path.parent.mkdir(parents=True)
            value = dict(sample='videophy40_VP004', intervention='empty_reasoning',
                         scope='first_image_and_request_only', caption='own request',
                         request_word_strings=['own', 'request'], first_image=str(root / 'own.png'),
                         duration=3.0, relations=['', '', '', ''])
            torch.save(value, path)
            loaded = prepared_i2v_control(root, 'VP004', 'empty_reasoning')
            item = dict(caption='own request', image=str(root / 'own.png'), duration=3.0)
            validate_control_input(loaded, item)
            with self.assertRaises(ValueError):
                validate_control_input(loaded, dict(item, image=str(root / 'donor.png')))
            value['relations'] = ['Image-only physics is still a nonempty description.']
            torch.save(value, path)
            with self.assertRaises(ValueError):
                prepared_i2v_control(root, 'VP004', 'empty_reasoning')

    def test_condition_replacement_preserves_native_request(self):
        own = dict(words=torch.ones(4, 16), word_groups=torch.arange(4),
                   request_words=torch.ones(2, 16), caption='own request', sample='own',
                   analysis='own analysis')
        donor = dict(words=torch.zeros(4, 16), word_groups=torch.arange(4),
                     request_words=torch.zeros(2, 16), caption='wrong request',
                     sample='donor', analysis='donor analysis')
        replaced = physics_evidence(own, donor)
        self.assertIs(replaced['request_words'], own['request_words'])
        self.assertIs(replaced['words'], donor['words'])
        self.assertEqual(replaced['caption'], own['caption'])
        self.assertEqual(replaced['analysis'], donor['analysis'])
        context = dict(context=torch.ones(1, 8, W), text_mask=torch.ones(1, 8),
                       routing=torch.ones(1, 8, W), execution=torch.ones(1, 8, W))
        mean = dict(routing=torch.zeros(1, 8, W), execution=torch.zeros(1, 8, W))
        replaced = fixed_content(context, mean)
        self.assertIs(replaced['context'], context['context'])
        self.assertIs(replaced['text_mask'], context['text_mask'])
        self.assertIs(replaced['routing'], mean['routing'])
        self.assertEqual(velocity_mode('no_c'), 'off')

    @torch.no_grad()
    def test_physical_off_and_ffn_off_have_distinct_effects(self):
        torch.manual_seed(123)
        cfg = load_config(ROOT / 'configs/r18g.yaml')
        cfg['model'].update(width=W, ffn_width=FF, injection_layers=[0], q_tokens=8,
                            qwen_width=16, q_heads=4, grounded_heads=2)
        cfg['train']['gradient_checkpointing'] = False
        interface = dict(mean=torch.zeros(16), scale=torch.ones(16),
                         weight=torch.randn(W, 16) * .01, bias=torch.zeros(W),
                         target_scale=torch.tensor(1.))
        model = PhysicalModel(cfg, device='cpu', wan=FakeWan(), interface=interface).eval()
        names_before = model.learned_names
        text = torch.randn(1, 6, 4096)
        context, mask = model._context(text)
        c = dict(routing=torch.randn(1, 8, W), execution=torch.randn(1, 8, W),
                 context=context, text_mask=mask)
        groups = torch.arange(8) // 2
        noise = torch.randn(1, 48, 2, 4, 4)
        times = torch.tensor([[0., .5]])
        module = model.mechanisms['0']
        args = (noise, None, text, torch.tensor(.5), c, groups, times)
        normal = model.velocity(*args)[0]
        explicit = model.velocity(*args, physical_mode='normal')[0]
        off = model.velocity(*args, physical_mode='off')[0]
        torch.testing.assert_close(normal, explicit, rtol=0, atol=0)
        torch.testing.assert_close(normal, off, rtol=2e-5, atol=2e-5)
        torch.nn.init.normal_(module.writer.weight, std=.1)
        torch.nn.init.normal_(module.conditioned.b.weight, std=.1)
        active = model.velocity(*args)[0]
        with patch.object(module.conditioned, 'forward', side_effect=AssertionError('FFN invoked')):
            ffn_off = model.velocity(*args, physical_mode='ffn_off')[0]
            complete_off = model.velocity(*args, physical_mode='off')[0]
        self.assertGreater(float((active - ffn_off).abs().max()), 1e-5)
        self.assertGreater(float((ffn_off - complete_off).abs().max()), 1e-5)
        changed = dict(c, routing=c['routing'] * 50, execution=c['execution'] * -50)
        altered_off = model.velocity(noise, None, text, torch.tensor(.5), changed,
                                     groups, times, physical_mode='off')[0]
        torch.testing.assert_close(complete_off, altered_off, rtol=0, atol=0)
        self.assertEqual(model.learned_names, names_before)


if __name__ == '__main__':
    unittest.main()
