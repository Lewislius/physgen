"""Properties required by the diagnosed reference-allocation failure."""
from pathlib import Path
import json
import math
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch
from physical_content.model import Fusion
from completed_suite import completion


class MassRetentionTests(unittest.TestCase):
    def inputs(self):
        logits = torch.tensor([2., 0., -2., 1.], requires_grad=True)
        prior = torch.tensor([math.log(.5), math.log(.5), 0., 0.])
        semantic = torch.tensor([math.log(1.98), math.log(.01), math.log(.01), 0.], requires_grad=True)
        return logits, prior, semantic

    def test_preserves_posterior_allocation_despite_non_neutral_semantics(self):
        logits, prior, semantic = self.inputs()
        original = (logits + prior).softmax(-1)
        old_semantic = (logits + semantic).softmax(-1)
        repaired = Fusion.reference_preserving_attention(logits, prior, semantic)
        self.assertTrue(torch.allclose(prior[:-1].exp().sum(), semantic[:-1].exp().sum()))
        self.assertGreater(float((old_semantic[-1] - original[-1]).abs()), .1)
        torch.testing.assert_close(repaired[-1], original[-1])
        torch.testing.assert_close(repaired[:-1].sum(), original[:-1].sum())
        torch.testing.assert_close(repaired.sum(), torch.tensor(1.))
        self.assertGreater(float((repaired[:-1] - original[:-1]).abs().max()), .01)

    def test_shared_value_component_is_not_amplified_by_semantics(self):
        logits, prior, semantic = self.inputs()
        original = (logits + prior).softmax(-1)
        repaired = Fusion.reference_preserving_attention(logits, prior, semantic)
        values = torch.tensor([[2., -3.], [2., -3.], [2., -3.], [0., 0.]])
        torch.testing.assert_close(repaired @ values, original @ values)

    def test_semantic_signal_and_base_allocation_receive_gradients(self):
        logits, prior, semantic = self.inputs()
        weights = Fusion.reference_preserving_attention(logits, prior, semantic)
        (weights * torch.tensor([1., -2., 3., 0.])).sum().backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.)
        self.assertGreater(float(semantic.grad[:-1].abs().sum()), 0.)
        self.assertEqual(float(semantic.grad[-1]), 0.)

    def test_no_semantics_exactly_restores_original_attention(self):
        logits, prior, _ = self.inputs()
        torch.testing.assert_close(Fusion.reference_preserving_attention(logits, prior),
                                   (logits + prior).softmax(-1), rtol=0., atol=0.)
        torch.testing.assert_close(Fusion.reference_preserving_attention(logits, prior, prior),
                                   (logits + prior).softmax(-1))

    def test_resume_distinguishes_training_and_inference_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); case = root / 'cases/VP010'; case.mkdir(parents=True)
            checkpoint, training, inference = Path('/fixed/own400.pt'), Path('/fixed/training'), Path('/fixed/repair')
            manifest = dict(cases=[{'id':'VP010'}], checkpoint=str(checkpoint), training_source=str(training),
                            inference_source=str(inference), checkpoint_step=400, mode='real', generation_seed=42)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'summary.json').write_text(json.dumps(dict(status='completed', completed_cases=1)))
            (case / 'planned.mp4').write_bytes(b'fixture')
            (case / 'planned.json').write_text(json.dumps(dict(checkpoint=str(checkpoint), step=400,
                mode='real', case='VP010', seed=42, inference_source=str(inference))))
            self.assertEqual(completion(root, checkpoint, training, ['VP010'], 'real', inference_source=inference), (True, False))
            with self.assertRaises(ValueError):
                completion(root, checkpoint, training, ['VP010'], 'real', inference_source=Path('/wrong/source'))


if __name__ == '__main__':
    unittest.main()
