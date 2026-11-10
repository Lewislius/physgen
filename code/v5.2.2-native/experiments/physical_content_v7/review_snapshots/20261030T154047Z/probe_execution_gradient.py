"""Test real-video feature feedback through decoder/Wan into C, without updates."""
from pathlib import Path
import sys,argparse,os,json,importlib.util,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
ROOT=bootstrap.BASE/'experiments/physical_content_v7'
import torch,numpy as np
from physical_scaling.common import gpu_guard,atomic_json
from physical_scaling.data import Data
from process_features import FrozenProcessFeatures,feature_distance

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);args=parser.parse_args()
    torch.set_num_threads(4);gpu_guard(ROOT/'reports/execution_gradient_allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True);cfg=state['config']
    sys.path.insert(0,cfg['source_snapshot'])
    from physical_content.model import PhysicalModel
    from physical_content.teacher import Teacher
    from physical_content.supervision import generation_loss
    from physgen_native.decoder import VideoDecoder
    model=PhysicalModel(cfg).train();model.wan.eval();model.set_phase(state['step'])
    parameters=dict(model.named_parameters());buffers=dict(model.named_buffers())
    with torch.no_grad():
        for name,value in state['model'].items():parameters[name].copy_(value)
        for name,value in state['constants'].items():buffers[name].copy_(value)
    del state
    teacher=Teacher(cfg);decoder=VideoDecoder(cfg);features=FrozenProcessFeatures(cfg,'cuda');data=Data(cfg)
    p=Path('/home/liuzhirui/Project/physGen/code/v5.2/physgen_v52/preparation.py');spec=importlib.util.spec_from_file_location('_original_read',p);preparation=importlib.util.module_from_spec(spec);spec.loader.exec_module(preparation)
    indices=[24,27,51,54,78,80];rows=[]
    for name,sigma_value in [('source0000000',.35),('source0001638',.65)]:
        sample=data.get(data.rows.index(data.by_id[name]));start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        images=preparation.read_frames(sample['row'],indices)
        real=torch.from_numpy(np.stack([np.asarray(im) for im in images])).permute(0,3,1,2).cuda().float()/127.5-1
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):target=features(real)
        noise=torch.randn(sample['latent'].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(42871))
        sigma=torch.tensor(sigma_value,device='cuda');noisy=(1-sigma)*sample['latent']+sigma*noise
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            evidence=teacher.input(sample);c,groups=model.enhancement(evidence)
            prediction,_=model.velocity(noisy,sample['first'],sample['text'],sigma,c,sample['times'],groups)
            fm,_=generation_loss(prediction,noise,sample['latent'],sample['valid'],**cfg['loss']['generation'])
            predicted_clean=noisy-sigma*prediction
            predicted_clean=torch.cat((sample['first'],predicted_clean[:,:,1:]),2)
            predicted_rgb=decoder(predicted_clean,frame_indices=indices)
            feature,parts=feature_distance(features(predicted_rgb),target,sample['valid'])
        grad_fm,=torch.autograd.grad(fm,c,retain_graph=True)
        selected=model.fusions['29'].output.weight
        grad_feature,grad_write=torch.autograd.grad(feature,(c,selected))
        if not torch.isfinite(grad_feature).all() or grad_feature.norm()==0:raise ValueError('Feature supervision does not reach C')
        if grad_write.norm()==0:raise ValueError('Feature supervision does not reach physical output')
        row=dict(sample=name,sigma=sigma_value,frame_indices=indices,fm=float(fm.detach()),feature=float(feature.detach()),
            c_fm_gradient=float(grad_fm.float().norm()),c_feature_gradient=float(grad_feature.float().norm()),
            c_gradient_cosine=float(torch.nn.functional.cosine_similarity(grad_fm.float().flatten(),grad_feature.float().flatten(),dim=0)),
            last_physical_write_gradient=float(grad_write.float().norm()),peak_gib=torch.cuda.max_memory_allocated()/2**30,
            seconds=time.monotonic()-start,feature_details=parts)
        rows.append(row);print(json.dumps(row),flush=True)
        atomic_json(dict(checkpoint=args.checkpoint,records=rows,optimizer_updates=0,
            scope='Real training-video target only; derivative validation, not evidence of improved generation'),ROOT/'reports/execution_gradient.json')
        del real,target,evidence,c,prediction,predicted_clean,predicted_rgb,fm,feature,grad_fm,grad_feature,grad_write,sample,noisy,noise

if __name__=='__main__':main()
