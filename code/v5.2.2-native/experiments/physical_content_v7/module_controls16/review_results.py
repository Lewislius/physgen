"""Summarize fixed-gamma R1 module controls on the preselected sixteen inputs."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import csv

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__file__).resolve().parent/'r1_step0400_guidance5_seed42'
spec = importlib.util.spec_from_file_location('short_review', ROOT/'early_execution_r3/review_results.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

SCOPES = dict(
    no_c='Both physical outputs disabled in the same jointly trained checkpoint; LoRA retained.',
    reasoning_swapped='Unrelated reasoning re-encoded with the own image/request; own native T5 and negative condition retained.',
    empty_reasoning='No generated reasoning text; own image/request, native T5 and negative condition retained.',
    mean='Positive C replaced by its own token mean; sample-specific information remains.',
    global_train_mean='Both positive and negative C fixed to their training-derived means; no sample-specific Qwen condition, while native T5/first image remain.',
    anchor_only='Learned C projection disabled in both CFG branches; each input-derived semantic anchor remains.')


def main():
    paths = dict(real=WORK/'references/real', native400=WORK/'references/native400')
    paths.update({mode:WORK/'controls'/mode/'videophy40' for mode in SCOPES})
    suites = {name:review.summarize(path) for name, path in paths.items()}
    comparisons = {}
    for name in ['native400', *SCOPES]:
        if suites[name]['status'] == suites['real']['status'] == 'completed':
            if suites[name]['count'] != 16 or suites['real']['count'] != 16:
                raise ValueError('Only completed sixteen-case suites enter this module comparison')
            comparisons['correct_vs_'+name] = review.comparison.compare(paths[name], paths['real'])
    effects = {}
    for mode in SCOPES:
        pair = comparisons.get('correct_vs_'+mode)
        if pair is None:
            effects[mode] = dict(status='awaiting_complete16', generated=suites[mode]['generated'])
            continue
        effects[mode] = dict(status='completed', intervention=SCOPES[mode],
            pc=pair['metrics']['pc'], sa=pair['metrics']['sa'],
            pc_pass_difference=suites['real']['pc_pass_rate']-suites[mode]['pc_pass_rate'],
            joint_difference=suites['real']['joint_pass_rate']-suites[mode]['joint_pass_rate'])
    report = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), gamma=5, seed=42,
        preserved_reference='R1-400 full40 PC3.625; this same16 normal reference PC3.8125.',
        scopes=SCOPES, suites=suites, effects=effects, paired_comparisons=comparisons,
        interpretation='Inference interventions in the same jointly trained checkpoint. Differences establish dependence, not automatically useful physics or the value of training a module. Judge correct C against the matched native baseline and actual generated processes. Four reused cases are part of these sixteen, not additional independent evidence.')
    output = WORK/'reports/module_effects.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    fields=['mode','status','count','sa_mean','pc_mean','sa_pass_rate','pc_pass_rate','joint_pass_rate']
    with output.with_name('module_summary16.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for name,value in suites.items():
            writer.writerow(dict(mode=name,**{key:value.get(key) for key in fields if key!='mode'}))
    print(json.dumps(dict(output=str(output), modes={name:value['status'] for name,value in effects.items()}),ensure_ascii=False))


if __name__ == '__main__':
    main()
