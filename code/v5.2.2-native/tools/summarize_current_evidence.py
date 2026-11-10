"""Rebuild the completed-evidence summary without mixing interrupted runs."""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def records(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status='Problems remain unresolved; active controlled experiments',
                protected_version='v5.2.1-mechanism',training_sources_added=0)
    protected=json.loads((ROOT/'archive/20261027_cleanup/analysis/protected_source_hashes.json').read_text())
    result['protected_source_check']=dict(count=len(protected),changed=[p for p,h in protected.items()
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h])
    reliance=[]
    for directory in ['reliance_fixed_content_seed42_recovered_65999',
                      'reliance_fixed_content_seed42_recovered_66002',
                      'reliance_fixed_content_seed314159_20261024']:
        path=ROOT/'archive/20261027_cleanup/analysis'/directory/'metrics.jsonl';rows=records(path)
        values=[r for r in rows if r['kind']=='reliance']
        keys=[tuple(r[k] for k in ['sample','split','seed','sigma','native_text','variant']) for r in values]
        if len(values)!=320 or len(set(keys))!=320 or rows[-1]['kind']!='complete':
            raise ValueError(f'Incomplete or duplicate evidence: {path}')
        selected=[r for r in values if r['split']=='validation' and r['native_text']=='full']
        by=defaultdict(list)
        for r in selected:by[r['variant']].append(r['fm'])
        means={k:statistics.mean(v) for k,v in by.items()};baseline=means['correct']
        reliance.append(dict(source=str(path.relative_to(ROOT)),device=rows[0]['device'],
            cases=len({r['sample'] for r in selected}),paired_observations_per_variant=len(by['correct']),
            fm=means,other_condition_relative_error_percent=100*(means['other_case']/baseline-1),
            no_analysis_relative_error_percent=100*(means['no_analysis']/baseline-1)))
    result['content_readout_reliance']=reliance
    rows=records(ROOT/'archive/20261027_cleanup/analysis/pretrained_rgb_temporal_views/metrics.jsonl')
    grouped=defaultdict(dict)
    for r in rows:
        if r['kind']=='view':grouped[(r['context'],r['seed'],r['sample'])][r['view']]=r['score']
    counts=[]
    for context,seed in sorted({(k[0],k[1]) for k in grouped}):
        cases=[v for (c,s,_),v in grouped.items() if c==context and s==seed]
        if len(cases)!=8 or any(set(v)!=set(['original','freeze','late_freeze']) for v in cases):raise ValueError('Incomplete RGB control')
        counts.append(dict(context=context,seed=seed,cases=len(cases),
            original_preferred_to_freeze=sum(v['original']>v['freeze'] for v in cases),
            original_preferred_to_late_freeze=sum(v['original']>v['late_freeze'] for v in cases)))
    result['source_label_verifier_rgb_control']=counts
    result['independent_videophy2_sensitivity']={view:json.loads((ROOT/'archive/20261027_cleanup/analysis/pretrained_rgb_temporal_views'/view/'scores.json').read_text())['summary']
        for view in ['original','freeze','late_freeze']}
    generation=[]
    for run in sorted((ROOT/'archive/20261027_cleanup/analysis/independent_video_evaluation_v2').glob('v522-diagnostic-*')):
        files=sorted(run.glob('step0032*/scores.json'))
        if len(files)!=2:raise ValueError(f'Incomplete paired seeds: {run}')
        cases=[c for p in files for c in json.loads(p.read_text())['cases']]
        generation.append(dict(run=run.name,sources=[str(p.relative_to(ROOT)) for p in files],
            videos=len(cases),sa_mean=statistics.mean(c['sa'] for c in cases),
            pc_mean=statistics.mean(c['pc'] for c in cases),
            joint_pass_rate=statistics.mean(c['sa']>=4 and c['pc']>=4 for c in cases)))
    result['generation_comparison_32_updates']=dict(training_cases=8,training_seeds=1,
        held_out_cases=4,generation_seeds=2,results=generation,
        interpretation='Small development study; discrete-score ties do not establish population equivalence')
    result['recovery_jobs']=json.loads((ROOT/'archive/20261027_cleanup/analysis/queue_20261024_recovery/jobs.json').read_text())
    result['verifier_recovery_job']=json.loads((ROOT/'archive/20261027_cleanup/analysis/queue_20261024_verifier_recovery/job.json').read_text())
    result['interpretation']=[
        'Tiny C-swap effects do not establish meaningful physical condition use.',
        'The source-label verifier learned a static/appearance shortcut. Repair its calibration, retain the verifier.',
        'VideoPhy2 alone does not detect the temporal failures in these controls.',
        'Oracle C updates using reference futures test gradient capacity, not ordinary inference or generalization.',
        'Full-chain differentiation and learned-verifier feedback are pending real generation tests.',
        'The existing development cases cannot also support an untouched final evaluation claim.']
    path=ROOT/'archive/20261027_cleanup/analysis/evidence_20261024.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(path)


if __name__=='__main__':main()
