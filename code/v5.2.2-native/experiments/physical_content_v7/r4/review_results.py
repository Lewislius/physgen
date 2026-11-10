"""Review R4 against matched-budget R3/native and the preserved R1 reference."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('short_run_review',ROOT/'early_execution_r3/review_results.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


def training_match(run):
    reference=ROOT/'early_execution_r3/runs/physical_content_66870'
    if not (run/'config.json').exists():return dict(status='awaiting_training')
    cfg=json.loads((run/'config.json').read_text());old=json.loads((reference/'config.json').read_text())
    config_sections=['stages','train','loss','geometry','knowledge','inference']
    differences=[k for k in config_sections if cfg[k]!=old[k]]
    def read(folder):
        rows=[json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()]
        return {(x['step'],x['micro']):x for x in rows if x['kind']=='micro'},[x for x in rows if x['kind']=='train']
    new,updates=read(run);previous,_=read(reference)
    fields=['sample','sigma','text_dropped','process_active','phase','aux_coefficient','process_coefficient']
    mismatches=[]
    for key,row in new.items():
        if key not in previous:
            mismatches.append(dict(step=key,reason='outside_comparison_budget'));continue
        changed=[field for field in fields if row[field]!=previous[key][field]]
        if row['process_details'].get('frame_indices')!=previous[key]['process_details'].get('frame_indices'):
            changed.append('process_frame_indices')
        if changed:mismatches.append(dict(step=key,fields=changed))
    return dict(status='complete64' if len(updates)==64 else 'training_in_progress',
        completed_updates=len(updates),observed_microsteps=len(new),
        unchanged_training_config_sections=config_sections,config_differences=differences,
        matched_fields=fields+['process_frame_indices'],mismatches=mismatches,
        exact_match_so_far=not differences and not mismatches,
        note='R3 and R4 start fresh with the same training recipe; R4 changes the Q reader, its word coordinates and Q-to-C initializer. This does not isolate these three changes from each other.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',default='physical_content_66871')
    args=parser.parse_args()
    base=ROOT/'early_execution_r3/inference_outputs'
    candidate=HERE/'inference_outputs'/args.candidate/'step0064_subset16_physical5_seed42'
    paths=dict(native64=base/'native_lora_66866/step0064_subset16_physical1_seed42/videophy40',
        r3_64=base/'physical_content_66870/step0064_subset16_physical5_seed42/videophy40',
        r1_reference16=ROOT/'module_controls16/r1_step0400_guidance5_seed42/references/real',
        r1_reference40=ROOT/'inference_outputs/physical_content_66778/step0400_seed42_r1/videophy40',
        r4_64=candidate/'videophy40',
        r4_full40=HERE/'inference_outputs'/args.candidate/'step0064_full40_physical5_seed42/videophy40',
        native64_full40=base/'native_lora_66866/step0064_full40_text5_seed42/videophy40',
        r4_anchor_only=candidate/'controls/anchor_only/videophy40',
        r4_unrelated=candidate/'controls/reasoning_swapped/videophy40',
        r4_no_c=candidate/'controls/no_c/videophy40')
    suites={k:review.summarize(v) for k,v in paths.items()}
    comparisons={}
    for name,baseline in [('matched_native64','native64'),('matched_r3_64','r3_64'),
        ('preserved_r1_16','r1_reference16'),('learned_correction','r4_anchor_only'),
        ('correct_vs_unrelated','r4_unrelated'),('correct_vs_no_c','r4_no_c')]:
        if all(suites[k]['status']=='completed' for k in [baseline,'r4_64']):
            comparisons[name]=review.comparison.compare(paths[baseline],paths['r4_64'])
    for name,baseline in [('full40_matched_native64','native64_full40'),('full40_preserved_r1_400','r1_reference40')]:
        if all(suites[k]['status']=='completed' for k in [baseline,'r4_full40']):
            comparisons[name]=review.comparison.compare(paths[baseline],paths['r4_full40'])
    screen=(review.assess_reference(suites['r1_reference16'],suites['r4_64'])
            if suites['r4_64']['status']=='completed' else dict(status='pending_generation'))
    if 'eligible_for_full40_confirmation' in screen:
        screen['meets_reference_screen_on_subset']=screen.pop('eligible_for_full40_confirmation')
    visual=HERE/'reports/visual_process64.json'
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),gamma=5,
        policy='Preserve/improve PC and improve SA with physical-process and visual evidence; no automatic extension or replacement.',
        training=training_match(HERE/'runs'/args.candidate),suites=suites,paired_comparisons=comparisons,
        reference_screen=screen,
        full40_execution_policy='Explicitly requested by the user; complete the full40 irrespective of the interim16 screening result. No automatic training extension or adoption.',
        actual_process_review=(dict(status='two_selected_cases_reviewed',path=str(visual),frame_count_per_case=6)
            if visual.exists() else dict(status='pending_R4_generation')),
        interpretation='Subset and full40 results are named separately. R1-400 is the preserved quality reference at a different budget; native64 is the matched-budget baseline. No result here establishes cross-seed repeatability.')
    output=HERE/'reports/generation_review.json';output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['suites','paired_comparisons']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
