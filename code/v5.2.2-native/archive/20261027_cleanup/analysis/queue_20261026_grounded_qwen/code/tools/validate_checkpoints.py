"""Read-only full-process validation of completed training checkpoints."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from physgen_native.data import VideoDataset,load_tensor
from physgen_native.generator import load_wan
from physgen_native.runtime import load_checkpoint,restore,Logger
from physgen_native.system import PhysGenNative
from train.train import validate


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.cuda.set_device(0)
    model=None;architecture=None
    for path in args.checkpoint:
        state=load_checkpoint(path);cfg=state['config']
        current=(cfg['model'],cfg['paths']['wan_checkpoint'],cfg['paths']['qwen_checkpoint'])
        if model is None:
            model=PhysGenNative(cfg,load_wan(cfg),device='cuda').cuda()
            architecture=current
        elif architecture!=current:raise ValueError('Batched validation requires identical model dimensions and base checkpoints')
        model.cfg=cfg;model.knowledge.cfg=cfg['knowledge']
        completed,_=restore(model,None,None,state,cfg);del state
        model.eval().requires_grad_(False)
        null=load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
        run=args.output/Path(path).parent.name
        logger=Logger(run/'metrics.jsonl')
        metrics=validate(model,VideoDataset(cfg,'validation'),null,completed,run,'cuda')
        logger.write('validation',completed,checkpoint=str(Path(path).resolve()),**metrics)
        logger.write('complete',completed);logger.close()


if __name__=='__main__':main()
