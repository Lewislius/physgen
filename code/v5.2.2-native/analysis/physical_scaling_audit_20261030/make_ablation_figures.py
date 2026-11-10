"""Inspect matched condition interventions; no aggregate RGB-change metric."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from make_figures import frames,NEW
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    for case in ['VP014','VP015']:
        videos={'Real reasoning':NEW/'cases'/case/'planned.mp4'}
        for mode in ['swapped','image_only','no_c']:
            videos[mode]=HERE/'gpu_results'/mode/'cases'/case/'planned.mp4'
        fig,axes=plt.subplots(4,6,figsize=(16,9),constrained_layout=True)
        for j,(mode,path) in enumerate(videos.items()):
            video=frames(path)
            for i,idx in enumerate([0,16,32,48,64,80]):
                axes[j,i].imshow(video[idx]);axes[j,i].set_xticks([]);axes[j,i].set_yticks([])
                if i==0:axes[j,i].set_ylabel(mode)
                if j==0:axes[j,i].set_title(f'{idx/24:.2f}s')
        fig.suptitle(f'{case}: same first frame, text prompt, noise seed and Wan weights')
        fig.savefig(HERE/f'ablation_{case}.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
