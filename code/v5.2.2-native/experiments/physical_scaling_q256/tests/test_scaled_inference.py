"""Check Q256 final weights, exact sampler replay and the 40-case score totals."""
from pathlib import Path
import contextlib
import io
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'inference'))
import generate
import protocol
import compare
import settings
import torch
from torch import nn
from physical_scaling.train import sample_video


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = {'inference':dict(steps=3,shift=5.,guidance=5.)}
    def enhancement(self,evidence):
        return evidence['c'],torch.arange(256)//64
    def velocity(self,x,first,text,sigma,c,times,groups):
        return x*.03+text.mean()*.2+c.mean()*.1+sigma*.05,torch.tensor(0.)


class Encoder:
    def input(self,sample,drop=False):
        return {'c':torch.full((1,256,2),-.4 if drop else .6)}


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_checkpoint_source_and_readonly_input_cache(self):
        state = protocol.checkpoint_state(settings.JOBS['v6_assoc']['checkpoint'],'v6_assoc')
        self.assertEqual(state['step'],1000)
        self.assertEqual(tuple(state['model']['enhancement.seed'].shape),(1,256,4096))
        self.assertNotIn('optimizer',state)
        self.assertTrue(Path(sys.modules['physical_scaling.model'].__file__).resolve().is_relative_to(settings.TRAIN_SOURCE))
        self.assertFalse(hasattr(generate.InputEncoder,'local_target'))
        with self.assertRaises(ValueError):
            protocol.checkpoint_state(settings.JOBS['v6_assoc']['checkpoint'],'v6_write')
        key='demo21_reference_text'
        source=settings.INPUT_CACHE_READONLY/'input'/f'{key}.pt'
        before=source.stat().st_mtime_ns
        with tempfile.TemporaryDirectory() as tmp:
            encoder=generate.InputEncoder.__new__(generate.InputEncoder)
            encoder.root=Path(tmp)
            encoder.readonly_root=settings.INPUT_CACHE_READONLY
            with encoder.cache('input',key) as (path,value):
                self.assertIsNotNone(value)
                self.assertEqual(value['scope'],'first_image_and_request_only')
                self.assertFalse(path.exists())
        self.assertEqual(source.stat().st_mtime_ns,before)

    def test_sampler_matches_scaled_training_and_saved_noise(self):
        model,encoder=Model(),Encoder()
        sample=dict(first=torch.ones(1,48,1,30,52)*.2,text=torch.tensor([2.]),
                    null_text=torch.tensor([-.5]),times=torch.arange(21)[None]/6)
        expected=sample_video(model,encoder,sample,sample['null_text'],42)
        noise=torch.randn(protocol.LATENT_SHAPE,generator=torch.Generator().manual_seed(42))
        before=noise.clone()
        actual=generate.sample_latent(model,sample,encoder.input(sample),encoder.input(sample,True),noise,42)
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(torch.equal(noise,before))
        self.assertTrue(torch.equal(actual[:,:,:1],sample['first']))

    def test_comparison_arithmetic_and_16_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            manifest=protocol.manifest_for('v6_assoc','videophy40')
            self.assertEqual(manifest['checkpoint_step'],1000)
            self.assertEqual(manifest['architecture']['c_tokens'],256)
            (path/'manifest.json').write_text(json.dumps(manifest))
            for name in ['scores.json','summary.json']:
                (path/name).write_bytes((settings.Q_REFERENCE/name).read_bytes())
            (path/'cases').symlink_to(settings.Q_REFERENCE/'cases',target_is_directory=True)
            with contextlib.redirect_stdout(io.StringIO()):
                result=compare.compare(path,'v6_assoc')
            self.assertEqual(len(result['summaries']),17)
            score=result['summaries'][0]
            self.assertEqual((score['sa_mean'],score['pc_mean'],score['mean_sa_pc'],score['joint_percent']),
                             (3.025,3.45,3.2375,22.5))
            self.assertEqual(score['scalar_scores'],80)
            for variant in ['v5_ce','v6_write','v6_assoc']:
                self.assertTrue(any(r['arm']==f'physical_{variant}_q128_s224' for r in result['summaries']))


if __name__=='__main__':
    unittest.main()
