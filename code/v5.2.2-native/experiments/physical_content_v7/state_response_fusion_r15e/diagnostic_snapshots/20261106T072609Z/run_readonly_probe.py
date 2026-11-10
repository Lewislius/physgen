"""Read-only R10A16 generation/scoring. No training or next-candidate launch."""
from pathlib import Path
import os,sys,json
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE))
import bootstrap
from physical_content.common import gpu_guard,atomic_json
from pipeline import run_stage,evaluator_environment
from allocated_gpu import bind_allocated_gpu


def main():
    root=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/state_response_fusion_r15e')
    experiment=os.environ['DET_EXPERIMENT_ID'];folder=root/'reports/probe_jobs'/experiment
    gpu_guard(folder);bind_allocated_gpu(folder,evaluator_environment)
    checkpoint=bootstrap.BASE/'experiments/physical_content_v7/execution_value_r10/runs/physical_content_67003/step0400.pt'
    output=root/'diagnostic_outputs/r10a400_native_video_response_fixed16'
    cases=[c['id'] for c in json.loads((SOURCE/'subset16.json').read_text())['cases']]
    atomic_json(dict(status='readonly_generation',cases=cases,checkpoint=str(checkpoint),source=str(SOURCE),
        training_updates=0,formal_R15E_training=False),folder/'status.json')
    run_stage([sys.executable,'-I','-B','-u',SOURCE/'infer_readonly_r10a.py','--checkpoint',checkpoint,
        '--output',output,'--suite','videophy40','--modes','real','--cases',*cases],folder/'generation.log')
    atomic_json(dict(status='readonly_scoring',training_updates=0),folder/'status.json')
    run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',
        bootstrap.BASE/'tools/evaluate_videophy2.py',output/'videophy40'],folder/'scoring.log',evaluator_environment())
    actual=json.loads((output/'videophy40/scores.json').read_text())
    assert len(actual['cases'])==16 and {x['id'] for x in actual['cases']}==set(cases)
    atomic_json(dict(status='completed',training_updates=0,formal_R15E_training=False,
        source=str(SOURCE),scores=str(output/'videophy40/scores.json')),folder/'status.json')


if __name__=='__main__':main()
