"""Real BF16 warm-forward/checkpoint gradient parity, including the default P prior."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.checkpoint import checkpoint
import yaml

from physgen_v4.coordinates import Coordinates
from physgen_v4.corrector import CellBlock, ConditionPrior

torch.set_num_threads(2)


class CheckpointAutocastTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61001)

    def compare_warm_gradients(self, model, inputs, forward):
        reference = None
        for recompute in (False, True):
            current = copy.deepcopy(model)
            leaves = [x.detach().clone().requires_grad_(True) for x in inputs]
            # Match training: no_grad warm-up and supervised forward in one AMP
            # context, followed by backward outside autocast with default checks.
            with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
                with torch.no_grad():
                    forward(current, leaves, recompute)
                result = forward(current, leaves, recompute)
                loss = result.float().square().mean()
            loss.backward()
            values = {'output': result.detach()}
            for name, parameter in current.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                values[name] = parameter.grad
            for index, leaf in enumerate(leaves):
                self.assertIsNotNone(leaf.grad)
                self.assertTrue(bool(torch.isfinite(leaf.grad).all()))
                self.assertGreater(float(leaf.grad.norm()), 0)
                values[f'input_{index}'] = leaf.grad
            if reference is None:
                reference = {name: value.clone() for name, value in values.items()}
            else:
                for name, value in values.items():
                    torch.testing.assert_close(value, reference[name], rtol=1e-5, atol=1e-7, msg=name)

    def test_repeated_cell_bf16_warm_recompute_matches_all_gradients(self):
        model = CellBlock(64, 4)
        inputs = [torch.randn(1, 16, 64), torch.randn(1, 7, 64)]
        def forward(current, values, recompute):
            state, context = values
            for _ in range(2):
                if recompute and torch.is_grad_enabled():
                    state = checkpoint(current, state, context, use_reentrant=False)
                else:
                    state = current(state, context)
            return state
        self.compare_warm_gradients(model, inputs, forward)

    def test_default_512_width_prior_bf16_warm_recompute_matches_all_gradients(self):
        config = yaml.safe_load((ROOT / 'configs/wisa_native_p.yaml').read_text())['corrector']
        model = ConditionPrior(config)
        self.assertEqual(model.query.shape[-1], 512)
        self.assertEqual(len(model.blocks), 2)
        coords = Coordinates.build(torch.linspace(0, 2, 9), 32, 64, 4, 1, teacher_size=64)
        inputs = [torch.randn(1, 48, 1, 2, 4), torch.randn(1, 3, 1664)]
        gt_video = torch.randn(1, 48, 3, 2, 4)
        self.compare_warm_gradients(model, inputs,
            lambda current, values, recompute: current(*values, coords, recompute, gt_video=gt_video))


if __name__ == '__main__':
    unittest.main(verbosity=2)
