"""A real pretrained-feature derivative, not an implementation-mirroring test."""
from pathlib import Path
import sys,json,importlib.util
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from bootstrap import ROOT
import torch,numpy as np,yaml
from transformers.utils import logging as transformer_logging
from process_features import FrozenProcessFeatures,feature_distance
from physical_scaling.data import Data

def main():
    torch.set_num_threads(16);transformer_logging.disable_progress_bar()
    cfg=yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text());model=FrozenProcessFeatures(cfg)
    data=Data(cfg,'validation');row=data.by_id['source0000278']
    p=Path('/home/liuzhirui/Project/physGen/code/v5.2/physgen_v52/preparation.py');spec=importlib.util.spec_from_file_location('_existing_read',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    images=m.read_frames(row,[0,24,27]);rgb=torch.from_numpy(np.stack([np.asarray(im) for im in images])).permute(0,3,1,2).float()/127.5-1
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):target=model(rgb[1:])
    prediction=rgb[:1].expand(2,-1,-1,-1).clone().requires_grad_()
    with torch.autocast('cpu',dtype=torch.bfloat16):loss,_=feature_distance(model(prediction),target)
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all() and prediction.grad.norm()>0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    result=dict(loss=float(loss.detach()),input_gradient_norm=float(prediction.grad.norm()),
        nonzero_gradient_fraction=float((prediction.grad!=0).float().mean()),encoder_parameters_frozen=True,
        scope='Derivative to observed frames verified; full decoder-to-C training derivative still to validate if adopted')
    (ROOT/'reports/process_gradient.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
