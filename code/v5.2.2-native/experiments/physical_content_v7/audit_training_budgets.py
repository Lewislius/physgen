"""Compare existing logged update budgets; no models, new scores or training."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent
SPECS = [
    ('R1_128', ROOT / 'runs/physical_content_66778', 128),
    ('R1_250', ROOT / 'runs/physical_content_66778', 250),
    ('R1_400', ROOT / 'runs/physical_content_66778', 400),
    ('native400', ROOT / 'runs/native_lora_66781', 400),
    ('R6_128', ROOT / 'content_modulation_r6/runs/physical_content_66910', 128),
    ('R7_128', ROOT / 'native_coordinate_r7/runs/physical_content_66920', 128),
    ('R8_128', ROOT / 'content_reader_r8/runs/physical_content_66926', 128),
]


def main():
    logs, records, sequences = {}, [], {}
    for label, run, stop in SPECS:
        if run not in logs:
            logs[run] = [json.loads(line) for line in (run / 'train.jsonl').open()]
        cfg = json.loads((run / 'config.json').read_text())
        steps = [r for r in logs[run] if r.get('kind') == 'train' and r['step'] <= stop]
        if [r['step'] for r in steps] != list(range(1, stop + 1)):
            raise ValueError((label, 'Need one complete logged update sequence'))
        micros = [r for r in logs[run] if r.get('kind') == 'micro' and r['step'] <= stop]
        if len(micros) != stop * cfg['train']['accumulation']:
            raise ValueError((label, 'Incomplete microbatch records'))
        sequences[label] = [(r['sample'], r['sigma'], r['text_dropped']) for r in micros]
        ready = next(r for r in logs[run] if r.get('kind') == 'model_ready')
        active_groups = [g for g, n in ready['groups'].items() if n > 0]
        lr_sums = {g: sum(r['learning_rates'][g] for r in steps) for g in active_groups}
        records.append(dict(
            version=label, total_updates=stop,
            interface_updates=sum(r['phase'] == 'interface' for r in steps),
            joint_updates=sum(r['phase'] == 'joint' for r in steps),
            wan_updates=sum(r['learning_rates']['wan_lora'] > 0 for r in steps),
            microbatches=len(micros),
            unique_training_samples=len({r['sample'] for r in micros}),
            module_warmup=cfg['train']['warmup'], wan_warmup=cfg['train']['wan_warmup'],
            decay_start=cfg['train']['decay_start'], auxiliary_warmup=cfg['loss']['warmup'],
            visual_auxiliary='process_features' in cfg['loss'],
            active_parameter_groups=ready['groups'], learning_rate_sums=lr_sums,
            learning_rate_sums_divided_by_peak={
                g: round(value / cfg['train']['peak_lrs'][g], 8) for g, value in lr_sums.items()
            },
            logged_training_minutes=round(sum(r['seconds'] for r in steps) / 60, 3),
            source_config=str(run / 'config.json'), source_log=str(run / 'train.jsonl'),
        ))
    matching = {name: sequences[name] == sequences['R1_128']
                for name in ['R6_128', 'R7_128', 'R8_128']}
    cases = [r['id'] for r in json.loads((ROOT / 'content_reader_r8/subset16.json').read_text())['cases']]
    intermediate = ROOT / 'runs/physical_content_66778/step0250.pt'
    plan = dict(
        maximum_revision=10, status='Not implemented or submitted', main_updates=400,
        primary_reference='R1-400 full40 and native400 full40',
        default_recipe='R1 128-interface +272-joint, module warmup32, Wan warmup128, no decay before400; disclose any evidence-driven deviation',
        intermediate_updates=250, intermediate_cases=cases,
        intermediate_reference_checkpoint=str(intermediate),
        intermediate_reference_available=intermediate.is_file(),
        intermediate_scope='One normal-condition fixed16 result per candidate at250, plus one reusable R1-250 fixed16 result. Each final400 full40 subset supplies its400 comparator. No per-step ablation matrix or baseline training.',
        final_scope='Normal40 and informative fixed16 module controls, gamma5 seed42. Preserve/improve PC3.625 before accepting SA gains.',
        initialization='Fresh official foundations plus training-input interface; R1 project checkpoints are evaluation references only',
        scaling_policy='400 is a declared matched comparison budget, not automatic extension of failed R7/R8 or a prediction that longer training fixes collapse',
    )
    report = dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(), records=records,
        first512_sample_sigma_dropout_match_R1=matching,
        interpretation='LR sums describe logged schedules only. Different Adam histories, gradients, architectures and losses remain different; these sums are not quality measures or equivalent learning amounts.',
        planned_final_revisions=plan,
    )
    output = ROOT / 'reports/training_budget_comparison.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    fields = ['version', 'total_updates', 'interface_updates', 'joint_updates', 'wan_updates',
              'microbatches', 'unique_training_samples', 'module_warmup', 'wan_warmup',
              'decay_start', 'visual_auxiliary']
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in records)
    for row in records:
        print(json.dumps({k: row[k] for k in fields + ['learning_rate_sums_divided_by_peak']}))
    print(json.dumps(dict(first512_sample_sigma_dropout_match_R1=matching,
                         intermediate_reference_available=intermediate.is_file())))


if __name__ == '__main__':
    main()
