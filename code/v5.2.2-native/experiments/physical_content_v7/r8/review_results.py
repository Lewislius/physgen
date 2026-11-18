"""Review the full40 candidate and preselected16 controls without changing source scores."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import importlib.util
import json

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
HERE=ROOT/'content_reader_r8'
spec=importlib.util.spec_from_file_location('short_run_review',ROOT/'early_execution_r3/review_results.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
policy_spec=importlib.util.spec_from_file_location('pc_reference_policy',ROOT/'build_results_tables.py')
policy=importlib.util.module_from_spec(policy_spec);policy_spec.loader.exec_module(policy)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True,help='Successful candidate run name, physical_content_EXPERIMENT')
    parser.add_argument('--baseline',help='Successful matched reference run name, native_lora128_EXPERIMENT')
    args=parser.parse_args()
    folder=HERE/'inference_outputs'/args.candidate/'step0128_full40_controls16_physical5_seed42'
    paths=dict(normal=folder/'videophy40',
        r1_reference=ROOT/'inference_outputs/physical_content_66778/step0400_seed42_r1/videophy40',
        native400=ROOT/'inference_outputs/native_lora_66781/step0400_seed42_r1/videophy40',
        r4_64=ROOT/'grounded_read_r4/inference_outputs/physical_content_66871/step0064_full40_physical5_seed42/videophy40',
        r6_128=ROOT/'content_modulation_r6/inference_outputs/physical_content_66910/step0128_full40_controls16_physical5_seed42/videophy40')
    controls=['mean_query_update','global_train_mean','anchor_only','reasoning_swapped','no_c']
    paths.update({mode:folder/'controls'/mode/'videophy40' for mode in controls})
    if args.baseline:paths['native128']=ROOT/'content_modulation_r6/inference_outputs'/args.baseline/'step0128_full40_text5_seed42/videophy40'
    suites={key:review.summarize(value) for key,value in paths.items()}
    cases=[row['id'] for row in json.loads((HERE/'subset16.json').read_text())['cases']]
    comparisons={}
    for name,path in paths.items():
        if name=='normal' or any(suites[key]['status']!='completed' for key in [name,'normal']):continue
        if suites['normal']['count']!=40:raise ValueError('Candidate primary suite must contain all40 cases')
        is_control=name in controls
        if suites[name]['count']!=(16 if is_control else 40):raise ValueError('Unexpected comparison scope')
        comparisons['normal_vs_'+name]=review.comparison.compare(path,paths['normal'],case_ids=cases if is_control else None)
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),gamma=5,seed=42,
        candidate=args.candidate,matched_native128=args.baseline,suites=suites,paired_comparisons=comparisons,
        pc_reference_preservation=policy.assess_pc_reference(suites['r1_reference'],suites['normal']),
        matching='Native128 is the equal-update no-physical-branch baseline:128 LoRA versus16 interface+112 joint. R8 versus R6 preserves parameter initialization, objective and budget while selecting request/reasoning words and aligning global-reader input coordinates. R1-400/native400/R4-64 are different-budget quality references.',
        mean_update_control='Training-derived raw DeltaH averages replace both CFG updates; each input own anchors/native T5/first image remain. Compare the identical preselected16 cases, not normal40 against control16 averages.',
        mean_content_control='Separate complete C averages from the same training inputs replace both CFG conditions. This tests sample-specific C information and is distinct from averaging only the Q update.',
        interpretation='Low cosine of A*DeltaH can follow algebraically from A. Require generation and process benefit and conditional-update benefit; do not equate differentiated output with learned physical reasoning.',
        automatic_adoption=False,automatic_extension=False)
    output=HERE/'reports'/args.candidate/'generation_review.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),suites=suites,completed_comparisons=list(comparisons)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
