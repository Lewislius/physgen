"""Freeze fresh R2 candidate/baseline sources and matching generation objectives."""
from pathlib import Path
import sys,copy,datetime,shutil,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import yaml


def main():
    root=bootstrap.ROOT;previous=root.parent
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    snapshot=root/'snapshots'/stamp;snapshot.mkdir(parents=True)
    for name in ['bootstrap.py','train.py','smoke_training.py','train_and_review.py']:
        shutil.copy2(root/name,snapshot/name)
    shutil.copytree(root/'physical_content',snapshot/'physical_content')
    for name in ['infer.py','infer_pipeline.py']:
        shutil.copy2(previous/name,snapshot/name)
    (snapshot/'configs').mkdir();(root/'train').mkdir(exist_ok=True)
    launch=yaml.safe_load((previous/'prepare_blk96g.yaml').read_text());records={}
    for variant in ['physical_content','native_lora']:
        cfg=yaml.safe_load((previous/'snapshots/20261030T141432Z/configs'/f'{variant}.yaml').read_text())
        cfg['revision']='physical_content_v7_process_r2';cfg['paths']['runs']=str(root/'runs')
        cfg['source_snapshot']=str(snapshot)
        cfg['loss']['process_features']=dict(kind='frozen_qwen_visual_observations',weight=.5,warmup=32,
            sigma_min=.2,sigma_max=.8,pairs=3,
            frequency='One rotating microbatch per update when its unmodified noise draw is in the interval',
            target='Existing real training clip; ordered early/middle/final observations; four frozen visual feature levels',
            reduction='0.8 mean + 0.2 top-quarter spatial-temporal feature errors',
            parameter_updates='Only existing enhancement/fusion and phase-appropriate Wan LoRA; frozen vision and VAE')
        path=snapshot/'configs'/f'{variant}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        job=copy.deepcopy(launch);job['name']=f'{variant}-v7-process-r2-fresh-review400-{stamp}'
        job['description']='Fresh official foundations; matched FM plus real observed-state feature supervision. Full interface/joint gradient smoke is discarded; review paired generation at400.'
        job['entrypoint']=f'bash {bootstrap.BASE}/tools/runtime_python.sh {snapshot}/train_and_review.py --config {path}'
        jobpath=root/'train'/f'{variant}_{stamp}.yaml';jobpath.write_text(yaml.safe_dump(job,sort_keys=False))
        records[variant]=dict(config=str(path),launcher=str(jobpath))
    (root/'reports').mkdir(exist_ok=True)
    result=dict(snapshot=str(snapshot),variants=records,submitted=False)
    (root/'reports/prepared.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
