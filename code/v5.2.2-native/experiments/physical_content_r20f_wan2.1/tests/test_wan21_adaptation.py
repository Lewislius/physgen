"""CPU checks for the R20FT -> Wan2.1-T2V-1.3B adaptation.

These tests verify configuration, shapes and the content/physical interfaces.
They do not claim generation quality and do not replace the full-size GPU check.
"""
from pathlib import Path
import copy
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: F401

import torch
import yaml

from physical_content.common import SCHEMA, WAN21_LATENT, load_config, validate_config
from physical_content.grounded_physics import GroundedCondition
from physical_content.native_residual import NativeAnchoredContent
from physgen_native.data import valid_grid


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.raw = yaml.safe_load((ROOT / 'configs/r20ft_wan21.yaml').read_text())

    def test_config_is_wan21(self):
        cfg = copy.deepcopy(self.raw)
        validate_config(cfg)
        self.assertEqual(cfg['schema_version'], SCHEMA)
        self.assertEqual(cfg['model']['width'], 1536)
        self.assertEqual(cfg['model']['ffn_width'], 8960)
        self.assertEqual(cfg['data']['latent_shape'], [16, 21, 60, 104])

    def test_wan22_dimensions_are_rejected(self):
        cfg = copy.deepcopy(self.raw)
        cfg['model']['width'] = 3072
        with self.assertRaises(ValueError):
            validate_config(cfg)
        cfg = copy.deepcopy(self.raw)
        cfg['data']['latent_shape'] = [48, 21, 30, 52]
        with self.assertRaises(ValueError):
            validate_config(cfg)
        cfg = copy.deepcopy(self.raw)
        cfg['base']['model'] = 'Wan2.2-TI2V-5B'
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_load_config_adds_stage_controls(self):
        cfg = load_config(ROOT / 'configs/r20ft_wan21.yaml')
        self.assertEqual(cfg['stages'], {'interface': 128, 'joint': 272})
        self.assertEqual(cfg['variant_id'], 'R20FT-Wan21')


class ValidGridTest(unittest.TestCase):
    def test_wan21_latent_grid(self):
        row = dict(geometry=dict(h=480, w=832, rh=480, rw=832,
                                 top=0, left=0))
        grid = valid_grid(row, 60, 104)
        self.assertEqual(tuple(grid.shape), (60, 104))
        self.assertTrue(bool((grid > 0).all()))


class ContentTest(unittest.TestCase):
    def test_native_anchored_content_wan21_width(self):
        torch.manual_seed(0)
        module = NativeAnchoredContent(query_width=4096, width=1536,
                                       heads=32, grounded_heads=16,
                                       slots=256, rounds=2)
        words = torch.randn(10, 4096)
        request = torch.randn(5, 4096)
        native = torch.randn(10, 1536)
        groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        out = module(words, request, native, groups)
        self.assertEqual(tuple(out['route'].shape), (1, 256, 1536))
        self.assertEqual(tuple(out['execute'].shape), (1, 256, 1536))
        out['route'].square().mean().backward()
        self.assertIsNotNone(module.route_update.weight.grad)

    def test_wan22_width_is_rejected(self):
        with self.assertRaises(ValueError):
            NativeAnchoredContent(query_width=4096, width=3072)


class GroundedConditionTest(unittest.TestCase):
    def test_condition_read_wan21_heads(self):
        from wan.modules.model import WanAttentionBlock
        block = WanAttentionBlock(
            't2v_cross_attn', dim=1536, ffn_dim=8960, num_heads=12,
            window_size=(-1, -1), qk_norm=True, cross_attn_norm=True,
            eps=1e-6)
        condition = GroundedCondition(block.cross_attn)
        self.assertEqual(condition.width, 1536)
        self.assertEqual(condition.heads, 12)
        state = torch.randn(1, 8, 1536)
        route = torch.randn(1, 4, 1536)
        execute = torch.randn(1, 4, 1536)
        value = condition(state, route, execute)
        # The conditioning is read back at the current video positions.
        self.assertEqual(tuple(value.shape), (1, 8, 1536))


if __name__ == '__main__':
    unittest.main()
