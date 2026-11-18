"""Exercise inference decomposition on the actual small Wan fixture."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from test_source_residual_fusion import SourceResidualTest
from guidance import guided_velocity,NATIVE_PLUS_PHYSICS,LEGACY


class GuidanceTests(unittest.TestCase):
    setUp=SourceResidualTest.setUp
    model=SourceResidualTest.model
    activate=SourceResidualTest.activate

    @torch.no_grad()
    def test_native_cfg_is_preserved_and_null_physics_cannot_cancel_it(self):
        model=self.model();self.activate(model);model.checkpointing=False
        c,g=model.enhancement(self.evidence);nc={k:v*.3 for k,v in c.items()}
        noisy=(1-self.sigma)*self.clean+self.sigma*self.noise;null=torch.zeros_like(self.text)
        args=(model,noisy,self.sample['first'],self.text,null,self.sigma,c,nc,self.sample['times'],g,g)
        full=guided_velocity(*args,composition=NATIVE_PLUS_PHYSICS)
        base=guided_velocity(*args,composition=NATIVE_PLUS_PHYSICS,physical_enabled=False)
        positive,_=model.velocity(noisy,self.sample['first'],self.text,self.sigma,c,self.sample['times'],g)
        native,_=model.velocity(noisy,self.sample['first'],self.text,self.sigma,c,self.sample['times'],g,physical_enabled=False)
        torch.testing.assert_close(full-base,positive-native,rtol=2e-5,atol=2e-6)
        for value in nc.values():value.add_(7.)
        torch.testing.assert_close(full,guided_velocity(*args,composition=NATIVE_PLUS_PHYSICS),rtol=0,atol=0)

    @torch.no_grad()
    def test_legacy_composition_and_zero_physics_reduction(self):
        model=self.model();model.checkpointing=False
        c,g=model.enhancement(self.evidence);noisy=self.clean;null=torch.zeros_like(self.text)
        args=(model,noisy,self.sample['first'],self.text,null,self.sigma,c,c,self.sample['times'],g,g)
        torch.testing.assert_close(guided_velocity(*args,composition=NATIVE_PLUS_PHYSICS),
            guided_velocity(*args,composition=LEGACY),rtol=2e-5,atol=2e-6)
        self.activate(model);c,g=model.enhancement(self.evidence)
        pos,_=model.velocity(noisy,self.sample['first'],self.text,self.sigma,c,self.sample['times'],g)
        neg,_=model.velocity(noisy,self.sample['first'],null,self.sigma,c,self.sample['times'],g)
        actual=guided_velocity(model,noisy,self.sample['first'],self.text,null,self.sigma,c,c,
            self.sample['times'],g,g,composition=LEGACY)
        torch.testing.assert_close(actual,neg+5*(pos-neg),rtol=0,atol=0)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(GuidanceTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
