"""Compare completed official subset scores without generating or judging videos."""
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parent


def paired(after, before):
    a = {v['id']: v for v in after['cases']}
    b = {v['id']: v for v in before['cases']}
    if set(a) != set(b) or len(a) != 70:
        raise ValueError('Use the exact same complete WMB70 subset')
    values = {
        'instruction': lambda row: row['instruction'],
        'physics': lambda row: sum(row['physical_laws']),
        'framewise_quality': lambda row: int(row['common_sense'][0]),
        'temporal_consistency': lambda row: int(row['common_sense'][1]),
        'total': lambda row: row['instruction'] + sum(row['physical_laws']) + sum(row['common_sense']),
    }
    result = {}
    for name, value in values.items():
        differences = [value(a[key]) - value(b[key]) for key in a]
        result[name] = dict(mean_change=statistics.mean(differences),
            wins=sum(d > 0 for d in differences), ties=sum(d == 0 for d in differences),
            losses=sum(d < 0 for d in differences),
            improved=[key for key in a if value(a[key]) > value(b[key])],
            regressed=[key for key in a if value(a[key]) < value(b[key])])
    return result


def paired_physics(after, before):
    """Pair the same20 scenes; retain all3 views without treating them as60 independent scenes."""
    a={v['scene']:v for v in after['per_scene']}
    b={v['scene']:v for v in before['per_scene']}
    av={(v['scene'],v['perspective']):v for v in after['per_view']}
    bv={(v['scene'],v['perspective']):v for v in before['per_view']}
    if len(a)!=20 or set(a)!=set(b) or len(av)!=60 or set(av)!=set(bv):
        raise ValueError('Use the same complete PhysicsIQ20 scenes and60 views')
    differences={scene:a[scene]['verified_score']-b[scene]['verified_score'] for scene in a}
    tolerance=1e-12
    domains={domain:[s for s in a if a[s]['domain']==domain] for domain in {v['domain'] for v in a.values()}}
    return dict(scene_count=20,view_count=60,
        final_score_view_change=after['scores']['final_score_view']-before['scores']['final_score_view'],
        scene_wins=sum(v>tolerance for v in differences.values()),
        scene_ties=sum(abs(v)<=tolerance for v in differences.values()),
        scene_losses=sum(v < -tolerance for v in differences.values()),
        categories={domain:dict(scene_count=len(scenes),mean_change=statistics.mean(differences[s] for s in scenes))
                    for domain,scenes in sorted(domains.items())},
        component_changes={k:after['scores'][k]-before['scores'][k] for k in after['scores']
                           if k.startswith('score_') and k.endswith('_view')},
        scenes=[dict(scene=s,domain=a[s]['domain'],after=a[s]['verified_score'],
                     before=b[s]['verified_score'],difference=differences[s]) for s in a],
        views=[dict(scene=s,perspective=view,id=av[(s,view)]['id'],
                    after=av[(s,view)]['verified_score'],before=bv[(s,view)]['verified_score'],
                    difference=av[(s,view)]['verified_score']-bv[(s,view)]['verified_score']) for s,view in av],
        scope='Descriptive paired differences over20shared scenarios, each retaining all3views; no independent60-scene statistical claim.')


def main():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    models = tuple(protocol['models'])
    result = dict(protocol=str(ROOT / 'protocol.json'), models={}, paired_WMB70={}, paired_PhysicsIQ20={},
        score_scales={'WorldModelBench': 'Official instruction0..3 + physics0..5 + commonsense0..2 = total0..10',
                      'PhysicsIQ': 'Official Verified final_score_view; report0..1 and percent'},
        scope='Fixed shared subsets, images, captions and per-case seed; own400 checkpoints. These are not full-benchmark scores or VideoPhy2 ratings.',
        interpretation='Keep instruction, physics, visual and temporal components visible. A higher total does not cancel a lower instruction score. Automated physics passes are not proof of correct requested processes.')
    world = {}
    physics = {}
    for model in models:
        folder = ROOT / 'outputs' / model / 'seed42'
        row = dict(checkpoint=protocol['models'][model], worldmodelbench='pending', physics_iq='pending')
        for key, filename in [('worldmodelbench', 'worldmodelbench_summary.json'), ('physics_iq', 'physics_iq_summary.json')]:
            path = folder / filename
            if path.exists():
                data = json.loads(path.read_text())
                row[key] = dict(source=str(path), **{k: v for k, v in data.items() if k != 'cases'})
                if key == 'worldmodelbench':
                    if data['overall']['count'] != 70 or len(data['cases']) != 70:
                        raise ValueError('Do not compare partial WMB scoring')
                    world[model] = data
                else:
                    if data['scene_count']!=20 or data['video_count']!=60:
                        raise ValueError('Do not compare partial PhysicsIQ scoring')
                    physics[model]=data
        result['models'][model] = row
    for baseline in ('wan_lora', 'r1'):
        for model in models:
            if model != baseline and model in world and baseline in world:
                result['paired_WMB70'][model + '_minus_' + baseline] = paired(world[model], world[baseline])
            if model != baseline and model in physics and baseline in physics:
                result['paired_PhysicsIQ20'][model + '_minus_' + baseline] = paired_physics(physics[model],physics[baseline])
    # These are additional acceptance requirements, not a replacement for
    # VideoPhy2 targets or inspection of actual processes and visual quality.
    result['acceptance_requirements'] = dict(
        source=str(ROOT.parent / 'r11_r12/goal_requirements.json'),
        reference='r1', comparison='strictly_greater_using_unrounded_values',
        worldmodelbench_metric='overall.total_score',
        physics_iq_metric='scores.final_score_view (Verified)',
        does_not_replace_videophy_or_visual_requirements=True,
        component_and_domain_review_required=True)
    result['cross_benchmark_numeric_acceptance'] = {}
    # WMB consists of discrete item points. Compare their exact totals rather
    # than allowing different floating-point sums of component means to win.
    world_points = {
        model: sum(row['instruction'] + sum(row['physical_laws']) + sum(row['common_sense'])
                   for row in data['cases'])
        for model, data in world.items()}
    for model in models:
        wmb_pass = (world_points[model] > world_points['r1']
                    if model in world and 'r1' in world else None)
        piq_pass = (physics[model]['scores']['final_score_view'] > physics['r1']['scores']['final_score_view']
                    if model in physics and 'r1' in physics else None)
        result['cross_benchmark_numeric_acceptance'][model] = dict(
            wmb70_total_points=world_points.get(model),
            wmb70_r1_total_points=world_points.get('r1'),
            wmb70_strictly_above_r1=wmb_pass,
            physics_iq_verified_strictly_above_r1=piq_pass,
            both_numeric_requirements_met=wmb_pass is True and piq_pass is True,
            overall_quality_goal_achieved=False,
            remaining_review='VideoPhy2-40/60, paired16 and actual videos remain separate requirements; inspect benchmark components and domains for substantial regressions.')
    destination = ROOT / 'reports/current_comparison.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({m: d['overall'] for m, d in world.items()}, ensure_ascii=False))
    print(json.dumps({pair: {metric: {k: v for k, v in values.items() if k not in ('improved', 'regressed')}
                           for metric, values in differences.items()}
                      for pair, differences in result['paired_WMB70'].items()}, ensure_ascii=False))
    print(json.dumps({'PhysicsIQ_verified':{m:d['scores']['final_score_view'] for m,d in physics.items()}},ensure_ascii=False))


if __name__ == '__main__':
    main()
