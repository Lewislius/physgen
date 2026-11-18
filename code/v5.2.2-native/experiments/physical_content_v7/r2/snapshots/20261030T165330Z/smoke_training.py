"""Exercise fresh R2 interface/joint full backward and Adam state on one GPU.

These three engineering updates are discarded. They never initialize the
experiment; the subsequent training process loads official foundations afresh.
"""
from pathlib import Path
import sys,argparse,os,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from physical_content.common import load_config,gpu_guard,atomic_json
from physical_content.model import PhysicalModel
from physical_content.teacher import Teacher
from physical_content.supervision import generation_loss
from physical_content.process_supervision import ProcessSupervisor
from physical_scaling.data import Data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=load_config(args.config)
    folder=Path(cfg['paths']['runs']).parent/'reports/smoke'/os.getenv('DET_EXPERIMENT_ID','local')
    gpu_guard(folder);torch.set_num_threads(4);torch.manual_seed(cfg['train']['seed'])
    torch.cuda.manual_seed_all(cfg['train']['seed'])
    model=PhysicalModel(cfg).train();model.wan.eval()
    teacher=Teacher(cfg);supervisor=ProcessSupervisor(cfg);data=Data(cfg)
    optimizer=torch.optim.AdamW(model.optim_groups(),betas=(.9,.999),weight_decay=.01,foreach=False)
    records=[]
    for index,(name,sigma_value) in enumerate([('source0000000',.35),('source0001638',.65),('source0001836',.5)]):
        phase=model.set_phase(0 if index==0 else cfg['stages']['interface'])
        optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        sample=data.get(data.rows.index(data.by_id[name]));sigma=torch.tensor(sigma_value,device='cuda')
        noise=torch.randn(sample['latent'].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(42871+index))
        noisy=(1-sigma)*sample['latent']+sigma*noise
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            evidence=teacher.input(sample);c,groups=model.enhancement(evidence)
            prediction,_=model.velocity(noisy,sample['first'],sample['text'],sigma,c,sample['times'],groups)
            generation,_=generation_loss(prediction,noise,sample['latent'],sample['valid'],**cfg['loss']['generation'])
            semantic=model.enhancement.content_loss(c,evidence) if cfg['variant']=='physical_content' else generation.new_zeros(())
            process,details=supervisor(sample,noisy,sigma,prediction,index)
            objective=generation+cfg['loss']['peak_weight']*semantic+cfg['loss']['process_features']['weight']*process
        process_c_gradient=None
        if index==2 and c.numel():
            gradient,=torch.autograd.grad(process,c,retain_graph=True)
            process_c_gradient=float(gradient.float().norm())
            if not torch.isfinite(gradient).all() or process_c_gradient<=0:
                raise ValueError('Real observed-state feedback failed to reach C after fresh interface updates')
            del gradient
        objective.backward()
        norms={}
        for group in optimizer.param_groups:
            values=[p for p in group['params'] if p.grad is not None]
            norm=torch.nn.utils.clip_grad_norm_(values,1.,error_if_nonfinite=True) if values else 0.
            norms[group['name']]=float(norm)
        if phase=='joint' and norms['wan_lora']<=0:raise ValueError('Joint LoRA gradient is missing')
        if cfg['variant']=='physical_content' and norms['fusion_write']<=0:raise ValueError('Physical write gradient is missing')
        if any(p.grad is not None for p in supervisor.features.parameters()):raise ValueError('Frozen feature encoder received parameter gradients')
        if any(p.grad is not None for p in supervisor.decoder.parameters()):raise ValueError('Frozen VAE received parameter gradients')
        optimizer.step();torch.cuda.synchronize()
        row=dict(index=index,phase=phase,sample=name,sigma=sigma_value,generation=float(generation.detach()),
            semantic=float(semantic.detach()),process=float(process.detach()),process_c_gradient=process_c_gradient,
            gradients=norms,details=details,seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        records.append(row);print(row,flush=True)
        atomic_json(dict(config=args.config,status='running',discarded_engineering_updates=len(records),records=records),folder/'result.json')
        del sample,evidence,c,prediction,generation,semantic,process,objective,noisy,noise
    atomic_json(dict(config=args.config,status='passed',discarded_engineering_updates=3,records=records,
        initializes_training=False),folder/'result.json')


if __name__=='__main__':main()
