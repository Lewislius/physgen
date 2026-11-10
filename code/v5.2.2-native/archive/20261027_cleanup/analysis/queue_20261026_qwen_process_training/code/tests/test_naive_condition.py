"""The naive comparison must actually bypass Qwen and retain the same reader."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_causal_response import response_model
from test_operator import TextKnowledge, Observer
from test_native import FakeWan, FakeDecoder, grad_size
from physgen_native.system import PhysGenNative
from physgen_native.config import validate, read_config


def naive_model():
    model, sample = response_model()
    cfg = copy.deepcopy(model.cfg)
    cfg['model']['qwen_width'] = sample['text'].shape[-1]
    cfg['knowledge'].update(condition_source='cached_t5', latent_reasoning_steps=0)
    return PhysGenNative(validate(cfg), FakeWan(cfg), knowledge=TextKnowledge(), decoder=FakeDecoder(),
                         process_encoder=Observer(), device='cpu'), sample


class NaiveConditionTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(31)

    def test_generator_training_bypasses_Qwen_but_updates_shared_projection_and_reader(self):
        model, sample = naive_model()
        with patch.object(model.knowledge, 'encode', side_effect=AssertionError('Qwen must not encode naive C')):
            losses, weights, metrics = model(sample, torch.zeros_like(sample['text']), 0, sigma_override=.65)
            model.backward_losses(losses, weights)
        self.assertEqual(set(losses), {'fm', 'process', 'intensity'})
        self.assertEqual(metrics['qwen_condition_active'], 0)
        self.assertEqual(metrics['wan_forward_count'], 1)
        self.assertEqual(grad_size(model.knowledge), 0)
        self.assertGreater(grad_size(model.conditioner), 0)
        self.assertGreater(grad_size(model.generator.interface), 0)

    def test_naive_C_uses_cached_text_and_has_no_image_or_future_read(self):
        model, sample = naive_model()
        original = model.encode_condition(model.conditions(sample))
        changed = dict(sample, first=torch.randn_like(sample['first']), latent=torch.randn_like(sample['latent']))
        torch.testing.assert_close(model.encode_condition(model.conditions(changed)), original, atol=0, rtol=0)
        changed['text'] = torch.randn_like(sample['text'])
        self.assertGreater(float((model.encode_condition(model.conditions(changed))-original).norm()), 0)
        null = model.encode_condition(model.conditions(sample, dropped=True))
        torch.testing.assert_close(null, torch.zeros_like(null), atol=0, rtol=0)

    def test_alternate_text_changes_only_C_text_source(self):
        model, sample = naive_model()
        sample['alternative'] = dict(id='other', caption='Another process.', clauses=[])
        alternate = torch.randn_like(sample['text'][0])
        with patch('physgen_native.data.load_tensor', return_value=alternate):
            result = model._alternative_condition(sample)
        torch.testing.assert_close(result['native_text'], alternate[None])
        torch.testing.assert_close(result['first'], sample['first'])
        self.assertFalse(torch.equal(result['native_text'], sample['text']))

    def test_training_recipe_changes_only_the_generation_condition_source(self):
        baseline = read_config(ROOT/'configs/ablation_native_text_causal_response.yaml')
        main = read_config(ROOT/'configs/native_causal_response_wisa3857.yaml')
        for key in ('data', 'model', 'wan_lora', 'stages', 'loss', 'sequence', 'verifier', 'completion', 'train', 'validation'):
            self.assertEqual(baseline[key], main[key])


if __name__ == '__main__': unittest.main()
