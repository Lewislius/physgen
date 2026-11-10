from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from guidance import combine


class GuidanceTests(unittest.TestCase):
    def test_original_scale_is_exact_original_arithmetic(self):
        torch.manual_seed(42)
        for dtype in [torch.float32,torch.bfloat16]:
            p=torch.randn(2,3,4,dtype=dtype);n=torch.randn_like(p)
            self.assertTrue(torch.equal(combine(p,n,None,5.,5.),n+5.*(p-n)))

    def test_physical_component_is_separate_from_native_request(self):
        neutral=torch.tensor([2.,4.]);negative=torch.tensor([1.,1.]);positive=neutral+torch.tensor([.5,-.25])
        base=negative+5*(neutral-negative)
        torch.testing.assert_close(combine(positive,negative,neutral,5.,0.),base)
        torch.testing.assert_close(combine(positive,negative,neutral,5.,1.)-base,positive-neutral)
        # An ineffective C must not gain an effect merely by changing its scale.
        torch.testing.assert_close(combine(neutral,negative,neutral,5.,2.),base)


if __name__=='__main__':unittest.main()
