"""Exactly16 R10A post-text-addressing intervention videos; no training."""
from pathlib import Path
import argparse,json,os,sys
HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'plan.json').read_text())
sys.path.insert(0,PLAN['model_source'])
import bootstrap
from benchmark_inputs import PINNED
from pipeline import run_stage,evaluator_environment
from physical_content.common import gpu_guard,atomic_json
from report_results import paired,summarize

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--describe',action='store_true');a=p.parse_args()
    if a.describe:print(json.dumps(PLAN,indent=2));return
    experiment=os.getenv('DET_EXPERIMENT_ID','local');root=HERE/'outputs'/experiment
    gpu_guard(root);atomic_json(PLAN,root/'plan.json')
    py=[sys.executable,'-I','-B','-u']
    atomic_json(dict(status='generation'),root/'status.json')
    run_stage([*py,HERE/'infer_reference.py','--checkpoint',PLAN['checkpoint'],'--output',root,
        '--suite','videophy40','--modes','native_text_value_off','--cases',*PLAN['cases']],root/'generation.log')
    folder=root/'controls/native_text_value_off/videophy40'
    atomic_json(dict(status='scoring'),root/'status.json')
    run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',
        bootstrap.BASE/'tools/evaluate_videophy2.py',folder],root/'scoring.log',evaluator_environment())
    result=json.loads((folder/'scores.json').read_text())
    if len(result['cases'])!=16 or {v['id'] for v in result['cases']}!=set(PLAN['cases']):raise ValueError('Incomplete16')
    refs=json.loads((bootstrap.BASE/'experiments/physical_content_v7/reports/r10_final_evidence.json').read_text())['source_paths']
    original=json.loads(Path(refs['R10-A/original40']).read_text())
    original={'cases':[v for v in original['cases'] if v['id'] in PLAN['cases']]}
    atomic_json(dict(**PLAN,summary=summarize(result['cases']),original_summary=summarize(original['cases']),
        intervention_minus_original=paired(result,original),score_file=str(folder/'scores.json'),
        interpretation='Inference-only change on R10A trained400; not a trained R12B result or proof of visual quality.'),root/'comparison_report.json')
    atomic_json(dict(status='completed',videos=16,training_updates=0),root/'status.json')

if __name__=='__main__':main()
