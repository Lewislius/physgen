"""Compare already-generated B videos with matched references, on CPU only."""
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.with_name('physical_content_r20')))
import review_a_videos as review


reference_sources = review.sources


def sources(task, case, physvid):
    retained = ({'PhysVid', 'R18E', 'R18G', 'R20AT'} if task == 't2v'
                else {'R15B', 'B400+I', 'R20AI'})
    videos = [(label, path) for label, path in reference_sources(task, case, physvid)
              if label in retained]
    tag = 'bt' if task == 't2v' else 'bi'
    suite = 'videophy40' if int(case['id'][2:]) <= 40 else 'videophy60'
    video = (HERE / 'inference_outputs' / f'r20{tag}_video_physics400' /
             'step0400_seed42' / suite / 'cases' / case['id'] / 'planned.mp4')
    return videos + [('R20'+tag.upper(), video)]


if __name__ == '__main__':
    review.sources = sources
    review.main()
