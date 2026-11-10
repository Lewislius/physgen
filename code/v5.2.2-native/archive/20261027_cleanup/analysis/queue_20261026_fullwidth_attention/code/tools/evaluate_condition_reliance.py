"""Separate C competence from its actual use when native text is also present."""
import argparse
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from physgen_native.data import VideoDataset,move,load_tensor
from physgen_native.generator import load_wan
from physgen_native.runtime import load_checkpoint,restore,Logger
from physgen_native.sequence import weighted_mean
from physgen_native.system import PhysGenNative


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',type=int,default=4)
    parser.add_argument('--noise-directory',type=Path)
    args=parser.parse_args();torch.cuda.set_device(0)
    state=load_checkpoint(args.checkpoint);cfg=state['config']
    model=PhysGenNative(cfg,load_wan(cfg),device='cuda').cuda()
    step,_=restore(model,None,None,state,cfg);del state
    model.eval().requires_grad_(False)
    null=load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
    log=Logger(args.output/'metrics.jsonl')
    log.write('evaluation_protocol',step,device=torch.cuda.get_device_name(0),
              noise_source=str(args.noise_directory.resolve()) if args.noise_directory else 'legacy CUDA seeded noise',
              wan_lora_b_max=max(float(p.abs().max()) for n,p in model.generator.wan.named_parameters() if n.endswith('lora_b')))
    for split in ['train','validation']:
        dataset=VideoDataset(cfg,split);conditions=[]
        for index in range(args.cases):
            sample=move(dataset[index],'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                c=model.encode_condition(model.conditions(sample))
                budget=cfg['knowledge']['reasoning_tokens'];cfg['knowledge']['reasoning_tokens']=0
                direct=model.encode_condition(model.conditions(sample));cfg['knowledge']['reasoning_tokens']=budget
            conditions.append((c,direct))
        for index in range(args.cases):
            sample=move(dataset[index],'cuda');c,direct=conditions[index]
            variants=dict(correct=(c,1.),off=(c,0.),other_case=(conditions[(index+1)%args.cases][0],1.),
                          mean_slots=(c.mean(1,keepdim=True).expand_as(c),1.),no_analysis=(direct,1.))
            for seed in [42+index,314159+index]:
                if args.noise_directory:
                    noise=load_tensor(args.noise_directory/f'seed{seed}.pt').cuda()
                    if noise.shape!=sample['latent'].shape:raise ValueError('Shared-noise geometry differs')
                else:
                    noise=torch.randn(sample['latent'].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
                noise_hash=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
                target=(noise-sample['latent'])[:,:,1:]
                for sigma in [.65,.95]:
                    level=torch.tensor(sigma,device='cuda')
                    noisy=(1-level)*sample['latent']+level*noise;noisy[:,:,:1]=sample['first']
                    for text_mode,text in [('full',sample['text']),('without_native_text',null)]:
                        for variant,(tokens,strength) in variants.items():
                            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                                velocity=model.generator(noisy,sample['first'],text,level,tokens,sample['times'],strength)
                            errors=(velocity[:,:,1:]-target).square().mean(1)
                            by_time=(errors[0]*sample['valid']).sum((-2,-1))/sample['valid'].sum()
                            log.write('reliance',step,sample=sample['id'],split=split,seed=seed,sigma=sigma,
                                      native_text=text_mode,variant=variant,fm=weighted_mean(errors,sample['valid']),
                                      fm_by_future_position=by_time.tolist(),noise_sha256=noise_hash)
    log.write('complete',step,training_seed=cfg['train']['seed'],
              scope='FM conditional-use test; denoising velocity is not physical motion',
              peak_gib=torch.cuda.max_memory_allocated()/2**30)
    log.close()


if __name__=='__main__':main()
