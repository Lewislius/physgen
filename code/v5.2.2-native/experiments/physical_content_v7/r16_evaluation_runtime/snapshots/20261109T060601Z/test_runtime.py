"""Verify the actual evaluation launcher, CFG forwarding and pool allocation guard."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import contextlib,io,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime import activate,stage_command,HERE
from gpu_runtime import gpu_guard

SOURCE=HERE.parent/'native_joint_read_r16a/snapshots/20261109T040426Z'

class RuntimeTests(unittest.TestCase):
    def test_original_source_gets_fixed_cfg_in_both_arms(self):
        activate(SOURCE,'amp-80g')
        import guidance
        self.assertEqual(Path(guidance.__file__),HERE/'guidance_fix.py')
        class Model:
            def __init__(self):self.calls=[]
            def velocity(self,*args,**kwargs):
                self.calls.append(kwargs)
                return (2. if len(self.calls)==1 else 1.),None
        for shared in [False,True]:
            model=Model()
            result=guidance.guided_velocity(model,*([None]*10),composition=guidance.LEGACY,
                shared_physical_query=shared,memory_mode='direct')
            self.assertEqual(result,6.)
            self.assertEqual(len(model.calls),2)
            self.assertTrue(all(c['shared_physical_query']==shared and c['memory_mode']=='direct'
                                for c in model.calls))

    def test_stage_rewrite_preserves_script_and_arguments(self):
        command=[sys.executable,'-I','-B','-u',str(SOURCE/'infer.py'),'--cases','VP001']
        rewritten=stage_command(command,SOURCE,'amp-80g')
        self.assertEqual(rewritten[:4],command[:4])
        self.assertEqual(rewritten[4:8],[str(HERE/'execute_stage.py'),str(SOURCE),'amp-80g',command[4]])
        self.assertEqual(rewritten[8:],command[5:])
        evaluator=['/other/existing/python','-I','evaluator.py','input']
        self.assertEqual(stage_command(evaluator,SOURCE,'amp-80g'),evaluator)

    def test_training_cannot_be_launched(self):
        for filename in ['train.py','train_only.py','gpu_preflight.py']:
            with self.assertRaises(ValueError):
                stage_command([sys.executable,'-I',str(SOURCE/filename)],SOURCE,'amp-80g')

    def test_allowed_cards_and_h100_rejection(self):
        for pool,name,gib,accepted in [('amp-80g','NVIDIA A100-SXM4-80GB',80,True),
                ('amp-80g','NVIDIA A800-SXM4-80GB',80,True),
                ('amp-48g','NVIDIA RTX A6000',48,True),
                ('amp-48g','NVIDIA A100-SXM4-40GB',40,True),
                ('ada-48g','NVIDIA L40S',48,True),('amp-80g','NVIDIA H100',80,False)]:
            with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
                 patch('torch.cuda.device_count',return_value=1), \
                 patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(name=name,total_memory=gib*1024**3)):
                if accepted:self.assertTrue(gpu_guard(folder,pool=pool)['accepted'])
                else:
                    with self.assertRaises(SystemExit):gpu_guard(folder,pool=pool)

if __name__=='__main__':unittest.main()
