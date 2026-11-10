"""Exercise the actual 81-frame paired decoder backward before loading Wan/Qwen."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physgen_native.config import read_config
from physgen_native.data import VideoDataset, move
from physgen_native.decoder import VideoDecoder
from physgen_native.sequence import sequence_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = read_config(args.config)
    torch.cuda.set_device(0)
    torch.manual_seed(cfg['train']['seed'])
    data = VideoDataset(cfg)
    index = next(i for i, row in enumerate(data.rows) if row['id']=='source0000527')
    sample = move(data[index], 'cuda')
    decoder = VideoDecoder(cfg).eval()
    shape = sample['latent'].shape
    a = (sample['latent']+.1*torch.randn_like(sample['latent'])).detach().requires_grad_()
    b = (sample['latent']+.1*torch.randn_like(sample['latent'])).detach().requires_grad_()
    begin = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        with torch.no_grad():
            reference = decoder(sample['latent'], resize=cfg['sequence']['size'])
        def objective(z):
            video = decoder(z, resize=cfg['sequence']['size'])
            return sequence_error(video, reference, sample['valid'], cfg['sequence'])[0]
        ea = checkpoint(objective, a, use_reentrant=False)
        eb = checkpoint(objective, b, use_reentrant=False)
        loss = .1*ea+.1*F.softplus((ea-eb)/cfg['condition_use']['temperature'])
    print(json.dumps(dict(kind='decoder_backward_begin',frames=len(reference),
        allocated_gib=torch.cuda.memory_allocated()/2**30)),flush=True)
    loss.backward()
    torch.cuda.synchronize()
    for value in (a,b):
        if value.grad is None or not torch.isfinite(value.grad).all() or not (value.grad.abs().sum()>0):
            raise RuntimeError('Both complete video branches must have finite nonzero derivatives')
    result = dict(kind='decoder_backward_complete',sample=sample['id'],frames=len(reference),
        latent_shape=list(shape),seconds=time.monotonic()-begin,loss=float(loss.detach()),
        peak_gib=torch.cuda.max_memory_allocated()/2**30,
        reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        gradient_rms=[float(x.grad.square().mean().sqrt()) for x in (a,b)],
        scope='actual pretrained decoder backward and memory; not generation quality or full-model fit')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
