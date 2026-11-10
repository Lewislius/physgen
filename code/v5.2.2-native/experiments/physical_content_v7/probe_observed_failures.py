"""Check an available training signal against the observed process omissions.

These distances are diagnostic feature comparisons, never physics scores or
training targets obtained from an evaluator. Running r1 training is unchanged.
"""
from pathlib import Path
import sys,json,importlib.util,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch,numpy as np,yaml,imageio.v2 as imageio
from process_features import FrozenProcessFeatures,feature_distance
from physical_scaling.data import Data

ROOT=bootstrap.BASE/'experiments/physical_content_v7'


@torch.no_grad()
def main():
    torch.set_num_threads(16)
    cfg=yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text())
    encoder=FrozenProcessFeatures(cfg,'cpu')
    source=Path('/home/liuzhirui/Project/physGen/code/v5.2/physgen_v52/preparation.py')
    spec=importlib.util.spec_from_file_location('_existing_frame_reader',source)
    preparation=importlib.util.module_from_spec(spec);spec.loader.exec_module(preparation)
    data=Data(cfg,'validation');indices=[24,27,51,54,78,80];records=[]
    for name in ['source0000681','wisa0028628','source0000693','source0000278']:
        sample=data.get(data.rows.index(data.by_id[name]),device='cpu',targets=False)
        images=preparation.read_frames(sample['row'],indices)
        tensor=lambda x:torch.from_numpy(np.stack([np.asarray(im) for im in x])).permute(0,3,1,2).float()/127.5-1
        real=tensor(images);start=time.monotonic()
        with torch.autocast('cpu',dtype=torch.bfloat16):target=encoder(real)
        views={'repeat_first':sample['first_rgb'].expand(len(indices),-1,-1,-1)}
        for key,run in [('baseline','native_lora_66781'),('candidate','physical_content_66778')]:
            path=ROOT/'inference_outputs'/run/'step0400_real_heldout8_seed42/heldout8/cases'/name/'planned.mp4'
            reader=imageio.get_reader(path)
            try:views[key]=tensor([reader.get_data(i) for i in indices])
            finally:reader.close()
        results={}
        for key,video in views.items():
            with torch.autocast('cpu',dtype=torch.bfloat16):
                features=encoder(video);distance,parts=feature_distance(features,target,sample['valid'])
            results[key]={'distance':float(distance),'layers':parts}
        records.append({'id':name,'views':results,'seconds':time.monotonic()-start})
        print(json.dumps({'id':name,'distances':{k:v['distance'] for k,v in results.items()}},ensure_ascii=False),flush=True)
        (ROOT/'reports/observed_failure_feature_probe.json').write_text(json.dumps({
            'frame_indices':indices,'records':records,'optimizer_updates':0,
            'scope':'Do frozen ordered visual targets detect missing real state changes? No quality score or causal correctness claim.'},indent=2))


if __name__=='__main__':main()
