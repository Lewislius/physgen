"""Fit a fresh full-width semantic interface using only the existing train split."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from bootstrap import ROOT
import torch
from physical_scaling.common import atomic_json,atomic_torch

@torch.no_grad()
def main():
    torch.set_num_threads(16);begin=time.monotonic()
    inventory=json.loads((ROOT/'cache/inventory.json').read_text())
    xs=[];ys=[];counts={}
    for item in inventory:
        if item['split']!='train':continue
        source=torch.load(ROOT/'cache/words'/f'{item["name"]}.pt',map_location='cpu',weights_only=True)
        target=torch.load(ROOT/'cache/native_words'/f'{item["name"]}.pt',map_location='cpu',weights_only=True)
        xs.append(source['words'].float());ys.append(target['native_words'].float())
        counts[item['sample']]=counts.get(item['sample'],0)+1
    x=torch.cat(xs);y=torch.cat(ys);del xs,ys
    mean=x.mean(0);scale=(x-mean).square().mean().sqrt()
    bias=y.mean(0);target_scale=(y-bias).square().mean().sqrt()
    x.sub_(mean).div_(scale);y.sub_(bias)
    gram=x.T@x/len(x);cross=x.T@y/len(x)
    ridge=.01*gram.diag().mean()
    gram.diagonal().add_(ridge)
    weight=torch.linalg.solve(gram,cross).T.contiguous()
    train_error=float((x@weight.T-y).square().mean()/target_scale.square())
    del x,y,gram,cross
    heldout=[]
    for item in inventory:
        if item['split']!='validation' or item['name'].endswith('_null'):continue
        source=torch.load(ROOT/'cache/words'/f'{item["name"]}.pt',map_location='cpu',weights_only=True)
        target=torch.load(ROOT/'cache/native_words'/f'{item["name"]}.pt',map_location='cpu',weights_only=True)['native_words'].float()
        predicted=(source['words'].float()-mean)/scale@weight.T+bias
        error=float((predicted-target).square().mean())
        constant=float((bias-target).square().mean())
        heldout.append(dict(sample=item['sample'],mse=error,constant_mse=constant,
                            prediction_centered_energy=float((predicted-predicted.mean(0)).square().mean())))
    report=dict(training_words=sum(torch.load(ROOT/'cache/words'/f'{item["name"]}.pt',map_location='cpu',weights_only=True)['words'].shape[0] for item in inventory if item['split']=='train'),
        training_samples=len(counts),train_normalized_mse=train_error,ridge=float(ridge),
        heldout=heldout,heldout_relative_mse=sum(r['mse'] for r in heldout)/max(1e-20,sum(r['constant_mse'] for r in heldout)),
        seconds=time.monotonic()-begin,initialization='fresh train-split fit between official foundation features; no project checkpoint')
    atomic_json(report,ROOT/'cache/interface_fit.json')
    if not heldout or report['heldout_relative_mse']>=.95:
        raise RuntimeError('Semantic initializer does not materially outperform a constant native context on held-out data')
    atomic_torch(dict(mean=mean,scale=scale,weight=weight,bias=bias,target_scale=target_scale,
                      training_samples=len(counts),revision='physical_content_v7_r1'),ROOT/'cache/interface.pt')
    print(json.dumps({k:v for k,v in report.items() if k!='heldout'}),flush=True)

if __name__=='__main__':main()
