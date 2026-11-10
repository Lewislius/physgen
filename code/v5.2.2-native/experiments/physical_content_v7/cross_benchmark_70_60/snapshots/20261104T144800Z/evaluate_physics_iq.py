"""Fixed20-scene/all3-view adapter around unchanged official Physics-IQ metrics."""
from pathlib import Path
import argparse,json,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,BENCH,plan,save

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);a=p.parse_args()
    sys.path.insert(0,str(BENCH/'physics-IQ-benchmark'))
    import cv2,pandas as pd
    from physiq.binary_mask_generator import generate_mask
    from physiq.calculate_and_write_metrics_to_csv import process_videos
    from physiq.calculate_iq_score_stable import IQTable
    from physiq.calculate_iq_score import calculate_iq_score
    cv2.setNumThreads(2)
    cases=[c for c in plan()['cases'] if c['benchmark']=='physics_iq60']
    out=ROOT/'outputs'/a.model/'seed42';videos=out/'physics_iq60';metrics=out/'physics_iq_metrics';metrics.mkdir(exist_ok=True)
    expected={c['output_filename'] for c in cases}
    if {p.name for p in videos.glob('*.mp4')}!=expected:raise ValueError('Require all60 matched videos with original identifiers')
    masks=metrics/'generated_masks';masks.mkdir(exist_ok=True)
    for case in cases:
        path=videos/case['output_filename'];cap=cv2.VideoCapture(str(path))
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration',
            '-of','default=noprint_wrappers=1:nokey=1',str(path)],text=True))
        if frames!=80 or abs(fps-16)>1e-6 or abs(duration-5)>.001:raise ValueError('PhysicsIQ requires exactly80 frames at16fps,5seconds')
        name=case['output_filename'].split('_',2)
        expected_mask=masks/f'{name[0]}_video-masks_16FPS_{name[1]}_take-1_{name[2]}'
        if not expected_mask.exists():generate_mask(str(path),str(masks/case['output_filename']),False)
        cap=cv2.VideoCapture(str(expected_mask));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        if n!=80:raise ValueError('Incomplete official generated mask')
    csvfile=metrics/(a.model+'.csv')
    if not csvfile.exists():
        process_videos(str(ROOT/'physics_iq_gt/testing'),str(videos),str(ROOT/'physics_iq_gt/masks'),
            str(masks),str(csvfile),fps=16,video_time_selection='first',n_processes=1)
    frame=pd.read_csv(csvfile);scene_category={c['scene']:c['domain'] for c in cases}
    if len(frame)!=20 or set(frame['scenario'])!=set(scene_category):raise ValueError('Metrics must contain every selected scene once')
    scores=IQTable(frame.copy()).get_output_dict();scores['final_score_origround']=calculate_iq_score(str(csvfile))[0]
    categories={category:IQTable(frame[frame['scenario'].isin([s for s,d in scene_category.items() if d==category])].copy()).get_output_dict() for category in sorted(set(scene_category.values()))}
    save(out/'physics_iq_summary.json',dict(model=a.model,subset='20scenes_all3views_60videos',generated_rounds=1,
        primary_metric='final_score_view (Verified)',scores=scores,categories=categories,
        verified_percent=100*scores['final_score_view'],scene_count=20,video_count=60,
        original_score_source=str(BENCH/'physics-IQ-benchmark/physiq/calculate_iq_score_stable.py'),
        official_metric_functions_unchanged=True,full_benchmark_score=False,raw_metrics_csv=str(csvfile),
        input_scope='Only official take1 switch image and prompt condition generation; both real takes used exclusively by official evaluator.'))

if __name__=='__main__':main()
