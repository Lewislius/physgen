"""Fixed inputs and two checkpoint-by-reasoning crossover arms."""
from pathlib import Path
import json,sys

BASE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT=BASE/'experiments/physical_content_v7'
WORK=ROOT/'r11_r12/r10a_data_crossover20'
PINNED=BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
OLD=ROOT/'execution_value_r10/snapshots/20261101T034557Z'
API=ROOT/'r10a_api400/snapshots/20261103T140613Z'
OLD_OUTPUT=ROOT/'execution_value_r10/inference_outputs/physical_content_67003/step0400_full40_controls16_physical5_seed42/videophy40'
API_OUTPUT=ROOT/'r10a_api400/inference_outputs/physical_content_67271/step0400_api_fixed60_controls16_seed42/videophy40'
SHARED=ROOT/'data_revision_qwen38/benchmark_inputs/videophy60_first_image_request_v1'
CHECKPOINTS={'old':ROOT/'execution_value_r10/runs/physical_content_67003/step0400.pt',
             'api':ROOT/'r10a_api400/runs/physical_content_67271/step0400.pt'}
ARMS={'old_checkpoint_api_reasoning':('old','api'),'api_checkpoint_old_reasoning':('api','old')}
EXTRA=['VP006','VP019','VP033','VP039']

def setup(source):
    sys.path[:0]=[str(source),str(source/'data_tools'),str(PINNED),
        str(BASE/'experiments/physical_scaling_q256/snapshots/20261029T140406Z'),
        str(BASE/'experiments/physical_variants/snapshots/20261029T064043Z')]

def cases():
    fixed=json.loads((ROOT/'request_memory_r11b/subset16.json').read_text())['cases']
    ids={v['id'] for v in fixed}|set(EXTRA)
    old=json.loads((OLD_OUTPUT/'manifest.json').read_text())
    api=json.loads((API_OUTPUT/'manifest.json').read_text());by={v['id']:v for v in api['cases']}
    selected=[v for v in old['cases'] if v['id'] in ids]
    if len(selected)!=20:raise ValueError('Expected existing16 plus four declared diagnostic failures')
    for item in selected:
        other=by[item['id']]
        for key in ['caption','prepared_inputs','initial_noise','geometry','canvas']:
            if item[key]!=other[key]:raise ValueError(f'Different paired input {item["id"]}/{key}')
    return old,selected
