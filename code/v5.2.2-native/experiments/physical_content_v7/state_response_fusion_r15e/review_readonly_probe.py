"""Paired fixed16 mechanism evidence, explicitly not trained R15E results."""
from pathlib import Path
from datetime import datetime,timezone
import json,importlib.util

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
spec=importlib.util.spec_from_file_location('score_utils',BASE/'denoising_matched_fusion_r15d/report_results.py')
score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
output=ROOT/'diagnostic_outputs/r10a400_native_video_response_fixed16/videophy40'
actual=json.loads((output/'scores.json').read_text());manifest=json.loads((output/'manifest.json').read_text())
ids={c['id'] for c in json.loads((ROOT/'subset16.json').read_text())['cases']}
assert len(actual['cases'])==16 and {c['id'] for c in actual['cases']}==ids
assert manifest['diagnostic_only'] and manifest['training_updates']==0
sources=json.loads((BASE/'reports/r10_final_evidence.json').read_text())['source_paths']
paths={name:Path(sources[name+'/original40']) for name in ['R10-A','R1','native400']}
paths['R15B']=BASE/'request_bound_fusion_r15b/inference_outputs/physical_content_67829/step0400_original_fixed40_controls16_seed42/videophy40/scores.json'
report=dict(utc=datetime.now(timezone.utc).isoformat(),kind='R10A_readonly_execution_probe',
    not_trained_R15E=True,training_updates=0,summary=score.summarize(actual['cases']),comparisons={})
for name,path in paths.items():
    full=json.loads(path.read_text());reference=dict(cases=[c for c in full['cases'] if c['id'] in ids]);old={c['id']:c for c in reference['cases']}
    report['comparisons'][name]=dict(summary=score.summarize(reference['cases']),probe_minus_reference=score.paired(actual,reference),
        per_case=[dict(id=c['id'],sa=c['sa']-old[c['id']]['sa'],pc=c['pc']-old[c['id']]['pc']) for c in actual['cases']],source=str(path))
report['limits']=['This changes execution on an existing R10A checkpoint without training. It is not the proposed fresh400 model or a test of newly trained independent Q.',
    'Fixed16 does not replace full40 targets or cross-benchmark/real-video acceptance.',
    'Review recovery of lost semantics/Joint and actual video quality while retaining PC; do not use per-case winner selection.']
(ROOT/'reports/readonly_probe_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(summary=report['summary'],versus_R10A=report['comparisons']['R10-A']['probe_minus_reference'],not_trained_R15E=True)))
