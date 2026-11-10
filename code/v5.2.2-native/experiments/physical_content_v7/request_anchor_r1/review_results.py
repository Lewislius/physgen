"""Review the complete40 request-anchor inference change on preserved R1-400."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('short_run_review',ROOT/'early_execution_r3/review_results.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


def main():
    paths=dict(original=ROOT/'inference_outputs/physical_content_66778/step0400_seed42_r1/videophy40',
        native400=ROOT/'inference_outputs/native_lora_66781/step0400_seed42_r1/videophy40',
        request_anchored=HERE/'inference_outputs/r1_step0400_full40_gamma5_seed42/controls/request_anchored/videophy40')
    suites={key:review.summarize(value) for key,value in paths.items()}
    comparisons={}
    coverage_groups={}
    if suites['request_anchored']['status']=='completed':
        if suites['request_anchored']['count']!=40:raise ValueError('Main input-revision comparison requires complete40')
        for key in ['original','native400']:
            comparisons['request_anchored_vs_'+key]=review.comparison.compare(paths[key],paths['request_anchored'])
        coverage_file=ROOT/'reports/reasoning_request_coverage40.json'
        if coverage_file.exists():
            coverage=json.loads(coverage_file.read_text())
            if len(coverage['records'])!=40:raise ValueError('Text coverage review must include the complete40')
            for label in ['covered','partial','absent_or_replaced']:
                selected=[row['id'] for row in coverage['records'] if row['coverage']==label]
                coverage_groups[label]=review.comparison.compare(paths['original'],paths['request_anchored'],case_ids=selected)
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),gamma=5,seed=42,suites=suites,
        paired_comparisons=comparisons,
        input_text_coverage_groups=coverage_groups,
        coverage_group_scope='Descriptive strata assigned by the assistant before candidate scores were available. Coverage is not correctness, all40 remain in the main comparison, and subgroup effects do not establish independent statistical validation.',
        intervention='Same R1-400 checkpoint, append verbatim request to full Qwen memory; anchor relations1/2/3/5, preserving original four reasoning sentences in memory, Q256, native T5, own negative C and sampling.',
        interpretation='An input-organization experiment, not new training or an isolated word edit. Confirm PC preservation and semantic/process benefit on the same40 before adoption.',
        automatic_adoption=False)
    output=HERE/'reports/generation_review.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),suites=suites,completed_comparisons=list(comparisons)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
