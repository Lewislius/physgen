"""Inference parity, strict final-weight restore, input isolation and scoring."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'inference'))
import generate
import protocol
import compare
from settings import JOBS, Q_REFERENCE
from physical_training.train import sample_video


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(.7))
        self.learned_names=('weight',)
        self.cfg={'inference':dict(steps=3,shift=5.,guidance=5.)}
    def enhancement(self,evidence):
        return evidence['c'],torch.tensor([0,1])
    def velocity(self,x,first,text,sigma,c,times,groups):
        return x*.03+text.mean()*.2+c.mean()*.1+sigma*.05,torch.tensor(0.)


class ToyEncoder:
    def input(self,sample,drop=False):
        return {'c':torch.tensor([-.4 if drop else .6])}


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_sampler_exactly_matches_training_validation(self):
        model=ToyModel();encoder=ToyEncoder()
        sample={'first':torch.ones(1,48,1,30,52)*.2,'text':torch.tensor([2.]),
                'null_text':torch.tensor([-.5]),'times':torch.arange(21)[None]/6}
        old=sample_video(model,encoder,sample,sample['null_text'],42)
        new=generate.sample_latent(model,sample,encoder.input(sample),encoder.input(sample,True),seed=42)
        self.assertTrue(torch.equal(old,new))
        noise=torch.randn(protocol.LATENT_SHAPE,generator=torch.Generator().manual_seed(42))
        original=noise.clone()
        replay=generate.sample_latent(model,sample,encoder.input(sample),encoder.input(sample,True),noise,42)
        self.assertTrue(torch.equal(replay,old))
        self.assertTrue(torch.equal(noise,original))
        self.assertTrue(torch.equal(replay[:,:,:1],sample['first']))

    def test_restore_all_learned_parameters_without_optimizer(self):
        model=ToyModel()
        protocol.restore_parameters(model,{'model':{'weight':torch.tensor(1.5)}})
        self.assertEqual(float(model.weight),1.5)
        self.assertFalse(model.weight.requires_grad)
        for state in ({'model':{}},{'model':{'weight':torch.ones(2)}}):
            with self.assertRaises(ValueError):protocol.restore_parameters(model,state)

    def test_inputs_have_no_future_or_historical_reasoning(self):
        case=protocol.manifest_for('v6_assoc','videophy40')['cases'][0]
        inputs,canvas=protocol.load_inputs(case)
        self.assertEqual(set(inputs),{'first','text','null_text','times','valid'})
        self.assertTrue(canvas.is_file())
        self.assertFalse(hasattr(generate.InputEncoder,'local_target'))
        self.assertFalse(hasattr(generate.InputEncoder,'semantic_target'))

    def test_each_variant_pins_its_own_final_checkpoint(self):
        for variant,job in JOBS.items():
            state=protocol.checkpoint_state(job['checkpoint'],variant)
            self.assertEqual(state['step'],224)
            self.assertNotIn('optimizer',state)
            self.assertNotIn('rng',state)
            other=next(x for x in JOBS if x!=variant)
            with self.assertRaises(ValueError):protocol.checkpoint_state(job['checkpoint'],other)

    def test_comparison_replay_reproduces_old_q_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)
            manifest=protocol.manifest_for('v6_assoc','videophy40')
            (path/'manifest.json').write_text(json.dumps(manifest))
            for name in ['scores.json','summary.json']:
                (path/name).write_text((Q_REFERENCE/name).read_text())
            (path/'cases').symlink_to(Q_REFERENCE/'cases',target_is_directory=True)
            result=compare.compare(path,'v6_assoc')
            row=result['summaries'][0]
            self.assertEqual((row['sa_total'],row['pc_total'],row['joint_percent']),(121,138,22.5))
            q=next(x for x in result['candidate_minus_reference'] if x['reference']=='v522_native_q')
            self.assertEqual((q['sa_delta'],q['pc_delta'],q['joint_delta_pp']),(0,0,0))

    def test_comparison_rejects_changed_noise_and_sampler(self):
        for field in ['noise','sampling']:
            value=protocol.manifest_for('v5_ce','videophy40')
            if field=='noise':value['cases'][0]['initial_noise']='/tmp/wrong.pt'
            else:value['sampling']['guidance']=7
            with self.assertRaises(ValueError):compare.validate_manifest(value)


if __name__=='__main__':unittest.main()
