"""Reuse the exact R1/R10A WMB70/PhysicsIQ60 inputs; prepare, never submit.

Without --checkpoint this only audits cached input compatibility. After an own400
checkpoint and the main40 review, --checkpoint registers that exact model and
freezes the existing benchmark scripts/protocol. No encoding or model downloads.
"""
from pathlib import Path
import argparse,collections,datetime,json,shutil
import torch

BASE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT=BASE/'experiments/physical_content_v7'
HERE=ROOT/'grounded_semantic_fusion_r15f'
BENCH=ROOT/'cross_benchmark_70_60'
REVISION='physical_content_v7_grounded_semantic_fusion_r15f'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--resource-pool',choices=['ada-48g','amp-80g','blk-96g'],default='ada-48g')
    a=p.parse_args();torch.set_num_threads(4)
    protocol=json.loads((BENCH/'protocol.json').read_text());cases=protocol['cases']
    counts=collections.Counter(c['benchmark'] for c in cases)
    if counts!={'worldmodelbench70':70,'physics_iq60':60}:raise ValueError('Keep the exact established130-case protocol')
    domains=collections.Counter(c['domain'] for c in cases if c['benchmark']=='worldmodelbench70')
    scenes=collections.Counter(c['scene'] for c in cases if c['benchmark']=='physics_iq60')
    if len(domains)!=7 or set(domains.values())!={10} or len(scenes)!=20 or set(scenes.values())!={3}:
        raise ValueError('Keep all7 domains and20 complete three-view scenes')
    rows=[]
    for case in cases:
        folder=BENCH/'shared'/case['id'];ready=json.loads((folder/'ready.json').read_text())
        if ready['case']!=case['id'] or ready['caption']!=case['caption']:
            raise ValueError('Shared first-image/request identity differs')
        if not (folder/'first.png').is_file():raise FileNotFoundError(folder/'first.png')
        lengths={}
        for arm in ('positive','negative'):
            value=torch.load(folder/(arm+'.pt'),map_location='cpu',weights_only=True,mmap=True)
            caption=case['caption'] if arm=='positive' else ''
            if value['caption']!=caption or value['request_word_strings']!=caption.split():
                raise ValueError('Incomplete or mismatched literal request span: '+case['id'])
            if value['words'].shape[-1]!=4096 or value['request_words'].shape[-1]!=4096:
                raise ValueError('Frozen Qwen feature interface changed')
            if len(value['word_groups'])!=len(value['words']):raise ValueError('Physical relation mapping differs')
            if value['memory'].ndim!=3 or value['memory'].shape[-1]!=4096 or value['memory_mask'].shape!=value['memory'].shape[:2]:
                raise ValueError('Full input Qwen memory is required: '+case['id'])
            lengths[arm]=dict(request_words=len(value['request_words']),physical_words=len(value['words']))
        native=torch.load(folder/'conditions.pt',map_location='cpu',weights_only=True,mmap=True)
        rows.append(dict(case=case['id'],benchmark=case['benchmark'],native_text_tokens=native['text'].shape[-2],arms=lengths))
    now=datetime.datetime.now(datetime.timezone.utc)
    result=dict(utc=now.isoformat(),status='shared_inputs_compatible',counts=dict(counts),domains=dict(domains),
        scenes=len(scenes),views_per_scene=3,contexts_checked=2*len(cases),
        protocol_source=str(BENCH/'protocol.json'),generation_seeds=protocol['generation_seeds'],sampling=protocol['sampling'],
        same_inputs_as=['wan_lora','r1','r10a'],full_input_memory_available=True,
        source_data_modified=False,new_api_calls=0,new_model_downloads=0,gpu_used=False,
        evaluation_submitted=False,quality_results_available=False,
        policy='Review own400 main40/controls/videos before supplementary allocation; preserve main training priority. NoH100.',cases=rows)
    if a.checkpoint:
        checkpoint=a.checkpoint.resolve()
        if not checkpoint.is_relative_to(HERE/'runs'):raise ValueError('Use an own R15F training run')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        if state['step']!=400 or state['revision']!=REVISION:raise ValueError('Require this architecture own400 checkpoint')
        source=Path(state['config']['source_snapshot']);spec=dict(checkpoint=str(checkpoint),source=str(source),
            step=400,resource_pool=a.resource_pool,reuse_shared_inputs=True)
        if not (source/'source_manifest.json').is_file():raise FileNotFoundError('Frozen training source manifest missing')
        existing=protocol['models'].get('r15f')
        if existing is not None and existing!=spec:raise ValueError('Preserve already registered R15F provenance')
        protocol['models']['r15f']=spec
        destination=HERE/'cross_benchmark_snapshots'/now.strftime('%Y%m%dT%H%M%SZ')
        destination.mkdir(parents=True,exist_ok=False)
        for name in ['common.py','run.py','generate.py','prepare_inputs.py','evaluate_worldmodel.py',
                     'evaluate_physics_iq.py','validate_physics_iq_csv.py']:
            shutil.copyfile(BENCH/name,destination/name)
        (destination/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
        # Extend only the model registry; historical cases, inputs and models stay identical.
        (BENCH/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
        result.update(status='benchmark_source_prepared_not_submitted',model='r15f',checkpoint=str(checkpoint),
            source=str(destination),resource_pool=a.resource_pool,
            command=['bash',str(BASE/'tools/runtime_python.sh'),str(destination/'run.py'),'--model','r15f'])
    output=HERE/'review/cross_benchmark_readiness.json';output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
