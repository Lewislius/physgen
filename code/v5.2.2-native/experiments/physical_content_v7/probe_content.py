"""Probe grounded content before/after training without making quality claims."""
from pathlib import Path
import sys,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from bootstrap import ROOT
import torch,yaml
from physical_content.model import Enhancement
from physical_content.evidence import find_input
sys.path.insert(0,str(bootstrap.BASE/'analysis/physical_scaling_audit_20261030'))
from audit_representations import representation

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint');args=parser.parse_args()
    torch.set_num_threads(8)
    cfg=yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text())
    model=Enhancement(cfg).eval()
    if args.checkpoint:
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
        values={k.removeprefix('enhancement.'):v for k,v in saved['model'].items() if k.startswith('enhancement.')}
        values.update({k.removeprefix('enhancement.'):v for k,v in saved['constants'].items()})
        model.load_state_dict(values);tag=f'step{saved["step"]:04d}';del saved
    else:tag='initial'
    names=[f'videophy40_VP{i:03d}_text' for i in [1,6,7,12,14,15,25,30,32,33,37,40]]
    inventory=json.loads((ROOT/'cache/inventory.json').read_text())
    names += [x['name'] for x in inventory if x['split']=='validation' and x['name'].endswith('_text')]
    records=[];means=[];conditions={}
    for name in names:
        evidence=torch.load(find_input(name),map_location='cpu',weights_only=True)
        words=torch.load(ROOT/'cache/words'/f'{name}.pt',map_location='cpu',weights_only=True)
        native=torch.load(ROOT/'cache/native_words'/f'{name}.pt',map_location='cpu',weights_only=True)
        evidence.update({key:words[key] for key in ['words','word_groups','word_positions']});evidence.update(native)
        with torch.autocast('cpu',dtype=torch.bfloat16):c,g=model(evidence)
        row=dict(sample=name,c=representation(c[0],g),semantic_loss=float(model.content_loss(c,evidence)))
        records.append(row);means.append(c[0].mean(0));conditions[name]=c[0].cpu()
    result=dict(tag=tag,checkpoint=args.checkpoint,records=records,
        cross_sample=representation(torch.stack(means)),
        mean_token_cosine=sum(r['c']['pair_cosine'] for r in records)/len(records),
        mean_centered_energy=sum(r['c']['centered_energy_fraction'] for r in records)/len(records),
        mean_semantic_loss=sum(r['semantic_loss'] for r in records)/len(records))
    (ROOT/'reports'/f'content_{tag}.json').write_text(json.dumps(result,indent=2))
    torch.save(conditions,ROOT/'reports'/f'content_{tag}.pt')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

if __name__=='__main__':main()
