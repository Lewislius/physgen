from pathlib import Path
import sys,copy,datetime,shutil,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from bootstrap import ROOT,SCALING
import yaml

def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    snapshot=ROOT/'snapshots'/stamp;snapshot.mkdir(parents=True)
    for path in [ROOT/'bootstrap.py',ROOT/'train.py',ROOT/'AGENTS.md']:
        shutil.copy2(path,snapshot/path.name)
    shutil.copytree(ROOT/'physical_content',snapshot/'physical_content')
    configs=ROOT/'configs';configs.mkdir(exist_ok=True)
    launches=ROOT/'train';launches.mkdir(exist_ok=True)
    base=yaml.safe_load((SCALING/'configs/v6_assoc_q256_s1000.yaml').read_text())
    base['revision']='physical_content_v7_r1'
    base['paths'].update(runs=str(ROOT/'runs'),supervision_cache=str(ROOT/'input_overlay'),
        content_cache=str(ROOT/'cache'),interface=str(ROOT/'cache/interface.pt'))
    base['train']['peak_lrs']['fusion_write']=3e-5
    base['train']['keep_steps']=[128,250,400,500,750,1000]
    base['train']['validation_steps']=[128,400,1000]
    base['loss']['generation']=dict(tail_weight=.2,tail_fraction=.25)
    base['source_snapshot']=str(snapshot)
    launch=yaml.safe_load((ROOT/'prepare_blk96g.yaml').read_text())
    records={}
    for variant in ['physical_content','native_lora']:
        cfg=copy.deepcopy(base);cfg['variant']=variant
        if variant=='native_lora':
            cfg['stages']={'interface':0,'joint':1000}
            cfg['baseline_budget']='Full 1000 native-LoRA updates; no skipped frozen interface; same global data/noise order and generation objective'
        path=configs/f'{variant}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        (snapshot/'configs').mkdir(exist_ok=True);shutil.copy2(path,snapshot/'configs'/path.name)
        job=copy.deepcopy(launch);job['name']=f'{variant}-v7-q256-fresh-s1000-review400-{stamp}'
        job['description']='Fresh total-budget comparison at400/1000; candidate128interface+joint, native baseline trains LoRA every update; same data/noise and generation objective.'
        job['entrypoint']=f'bash {bootstrap.BASE}/tools/runtime_python.sh {snapshot}/train.py --config {snapshot}/configs/{variant}.yaml --stop-after 400'
        jobpath=launches/f'{variant}_{stamp}.yaml';jobpath.write_text(yaml.safe_dump(job,sort_keys=False))
        records[variant]=dict(config=str(snapshot/'configs'/path.name),launcher=str(jobpath))
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/latest_prepared.json').write_text(json.dumps(dict(snapshot=str(snapshot),variants=records),indent=2))
    print(json.dumps(dict(snapshot=str(snapshot),variants=records),indent=2))

if __name__=='__main__':main()
