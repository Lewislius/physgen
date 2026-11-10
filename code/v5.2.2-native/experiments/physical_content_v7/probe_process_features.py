"""Check feature sensitivity using real held-out clips, before using a loss."""
from pathlib import Path
import sys,argparse,json,importlib.util,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
ROOT=bootstrap.BASE/'experiments/physical_content_v7'
import torch,numpy as np,yaml
from transformers.utils import logging as transformer_logging
from physical_scaling.data import Data
from process_features import FrozenProcessFeatures,feature_distance

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(16 if args.device=='cpu' else 4);transformer_logging.disable_progress_bar()
    if args.device=='cuda':
        from physical_scaling.common import gpu_guard
        import os
        gpu_guard(ROOT/'reports/process_feature_allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    cfg=yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text())
    encoder=FrozenProcessFeatures(cfg,args.device)
    source=Path('/home/liuzhirui/Project/physGen/code/v5.2/physgen_v52/preparation.py')
    spec=importlib.util.spec_from_file_location('_existing_frame_reader',source);preparation=importlib.util.module_from_spec(spec);spec.loader.exec_module(preparation)
    data=Data(cfg,'validation');requested=['source0000042','source0000278','source0000693']
    indices=[0,3,24,27,51,54,78,80];results=[]
    for name in requested:
        sample=data.get(data.rows.index(data.by_id[name]),device=args.device,targets=False)
        frames=preparation.read_frames(sample['row'],indices)
        video=torch.from_numpy(np.stack([np.asarray(im) for im in frames])).permute(0,3,1,2).to(args.device).float()/127.5-1
        start=time.monotonic()
        with torch.no_grad(),torch.autocast(args.device,dtype=torch.bfloat16):target=encoder(video)
        variants={'unchanged':video,'repeat_first':video[:1].expand_as(video),
                  'reverse_pair_order':video.reshape(4,2,*video.shape[1:]).flip(0).flatten(0,1),
                  'reverse_within_pairs':video.reshape(4,2,*video.shape[1:]).flip(1).flatten(0,1)}
        rows={}
        for mode,value in variants.items():
            with torch.no_grad(),torch.autocast(args.device,dtype=torch.bfloat16):
                features=target if mode=='unchanged' else encoder(value)
                loss,details=feature_distance(features,target,sample['valid'])
            rows[mode]=dict(distance=float(loss),layers=details)
        result=dict(sample=name,caption=sample['caption'],indices=indices,distances=rows,seconds=time.monotonic()-start)
        results.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
        (ROOT/'reports/process_feature_probe.json').write_text(json.dumps(dict(records=results,
            interpretation='Sensitivity to missing/reordered observed content only; not a physical correctness score; r1 training unchanged'),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
