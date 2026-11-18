"""Exercise the actual CSV adapter on one real scene with all three views.

This is a metric-wiring fixture, not a generated-model quality evaluation.
"""
from pathlib import Path
import json
import math
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,BENCH,plan,save
from evaluate_physics_iq import summarize_csv


def main():
    sys.path.insert(0,str(BENCH/'physics-IQ-benchmark'))
    import cv2
    import pandas as pd
    from physiq.calculate_and_write_metrics_to_csv import process_videos
    from physiq.calculate_iq_score_stable import IQTable
    cv2.setNumThreads(2)
    scene='trimmed-ball-hits-nothing.mp4'
    cases=[c for c in plan()['cases'] if c.get('scene')==scene]
    fixture=ROOT/'reports/physics_iq_csv_roundtrip_fixture'
    folders={name:fixture/name for name in ('testing','masks','generated','generated_masks')}
    for folder in folders.values():folder.mkdir(parents=True,exist_ok=True)
    def link(source,destination):
        if destination.is_symlink() or destination.exists():
            if destination.resolve()!=source.resolve():raise ValueError('Different existing fixture')
        else:destination.symlink_to(source.resolve())
    for name in ('testing','masks'):
        for source in (ROOT/'physics_iq_gt'/name).glob('*_'+scene):
            link(source,folders[name]/source.name)
    for c in cases:
        stem=f"{c['official_id']}_{{kind}}_16FPS_{c['perspective']}_take-1_{scene}"
        video=ROOT/'physics_iq_gt/testing'/stem.format(kind='testing-videos')
        mask=ROOT/'physics_iq_gt/masks'/stem.format(kind='video-masks')
        link(video,folders['generated']/c['output_filename'])
        link(mask,folders['generated_masks']/mask.name)
    csvfile=fixture/'real_self_comparison.csv'
    if not csvfile.exists():
        process_videos(str(folders['testing']),str(folders['generated']),
            str(folders['masks']),str(folders['generated_masks']),str(csvfile),
            fps=16,video_time_selection='first',n_processes=0)
    try:
        IQTable(pd.read_csv(csvfile)).get_output_dict()
    except Exception as error:
        old_error=dict(type=type(error).__name__,message=str(error))
    else:
        raise AssertionError('The original unparsed CSV path unexpectedly succeeded')
    result=summarize_csv(csvfile,cases,'real_self_test_fixture')
    assert result['scene_count']==1 and result['video_count']==3
    assert math.isclose(result['scores']['final_score_view'],1.,abs_tol=1e-12)
    assert len(result['per_view'])==3 and len(result['per_scene'])==1
    assert all(math.isclose(v['verified_score'],1.,abs_tol=1e-12) for v in result['per_view'])
    report=dict(passed=True,fixture_only_not_model_result=True,
        real_scene=scene,original_adapter_error=old_error,
        corrected_reader='Official IQTable.from_csv',
        official_verified_score=result['scores']['final_score_view'],
        views=result['per_view'],scenes=result['per_scene'],raw_csv=str(csvfile))
    save(fixture/'validation.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
