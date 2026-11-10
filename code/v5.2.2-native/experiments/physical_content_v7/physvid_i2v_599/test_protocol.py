"""CPU checks for paired coverage, score thresholds and frozen checkpoint identity."""
from pathlib import Path
import ast
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import MODELS, plan, owned_cases, aggregate, physvid_score, case_seed


class ProtocolTests(unittest.TestCase):
    def test_complete_disjoint_inputs(self):
        p=plan();self.assertEqual(len(p['cases']),599)
        owned=[c['id'] for model in MODELS for c in owned_cases(model)]
        self.assertEqual(len(owned),599);self.assertEqual(set(owned),{c['id'] for c in p['cases']})
        self.assertEqual(len({c['caption'] for c in p['cases']}),599)
        self.assertEqual(p['generation_seeds'],[9898539])
        self.assertFalse(p['image_generation']['reuse_physvid_video_frames'])

    def test_threshold_is_not_normalized_mean(self):
        rows=[dict(sa=5,pc=3),dict(sa=3,pc=5),dict(sa=4,pc=4),dict(sa=0,pc=0)]
        x=aggregate(rows)
        self.assertEqual(x['sa_mean'],3);self.assertEqual(x['pc_mean'],3)
        self.assertEqual(x['sa_pass_rate'],.5);self.assertEqual(x['pc_pass_rate'],.5)
        self.assertEqual(x['joint_pass_rate'],.25)

    def test_parser_matches_local_physvid(self):
        # Execute only the author's existing parser AST, without loading a model.
        path=Path('/home/liuzhirui/model/PhysVid/physvid/evaluation/videophy2_eval.py')
        tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef))
        node=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='post_process_outputs')
        space=dict(tqdm=type('Quiet',(),{'write':staticmethod(lambda *args:None)}),
                   dist=type('Rank',(),{'get_rank':staticmethod(lambda:0)}))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),space)
        tester=type('Evaluator',(),{})();tester.tokenizer=type('Tokenizer',(),{'decode':lambda self,x,**kw:x})()
        tester.num_map=dict(zip(('zero','one','two','three','four','five'),range(6)))
        tester.num_map.update({str(i):i for i in range(6)})
        examples=['4','five',' Score: 3. ','four (4)','cannot determine','zero','2/5']
        self.assertEqual([physvid_score(x) for x in examples],space['post_process_outputs'](tester,examples))

    def test_scheduler_matches_physvid(self):
        import importlib.util
        import torch
        path='/home/liuzhirui/model/PhysVid/physvid/models/wan/flow_match.py'
        spec=importlib.util.spec_from_file_location('reference_scheduler',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        reference=m.FlowMatchScheduler(shift=8.,sigma_min=0.,extra_one_step=True);reference.set_timesteps(50)
        grid=torch.linspace(1,0,51);grid=8*grid/(1+7*grid)
        torch.testing.assert_close(grid[:-1],reference.sigmas,rtol=0,atol=0)
        x=torch.ones(1,1,1,1);velocity=x*.3
        for i in (0,24,49):
            torch.testing.assert_close(x+(grid[i+1]-grid[i])*velocity,
                reference.step(velocity,reference.timesteps[i].reshape(1),x),rtol=0,atol=0)

    def test_saved400_identity_and_noise_mapping(self):
        import torch
        for model,spec in plan()['models'].items():
            d=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
            self.assertEqual(d['step'],400);self.assertEqual(d['config']['source_snapshot'],spec['source'])
            self.assertEqual(spec['resource_pool'],'amp-80g' if model in ('wan_lora','r1') else 'blk-96g')
        seeds=[case_seed(9898539,c['index']) for c in plan()['cases']]
        self.assertEqual(len(set(seeds)),599)


if __name__=='__main__':unittest.main()
