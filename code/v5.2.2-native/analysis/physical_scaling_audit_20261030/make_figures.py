"""Static scientific figures and paired frame inspection of recorded videos."""
from pathlib import Path
import json,subprocess,textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;BASE=OUT.parents[1]
NEW=BASE/'experiments/physical_scaling_q256/inference_outputs/v6_assoc_66678/final1000_demo21_videophy40_r1/videophy40'
OLD=BASE/'experiments/physical_variants/inference_outputs/v6_assoc_66642/final224_demo21_videophy40_r1/videophy40'


def frames(path):
    r=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(path),'-vf','scale=208:120','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True,check=True)
    return np.frombuffer(r.stdout,dtype=np.uint8).reshape(-1,120,208,3)


def main():
    names=['assoc128','assoc500','assoc1000','write1000','ce1000']
    data=[json.loads((OUT/f'representations_{n}.json').read_text()) for n in names]
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    for ax,key,title in zip(axes[:2],['centered_energy_fraction','energy_effective_rank'],['C: energy outside token mean','C: effective rank (singular-value energy)']):
        vals=[d['summary'][key]['mean'] for d in data]
        ax.bar(names,vals);ax.tick_params(axis='x',rotation=30);ax.set_title(title)
        if key=='centered_energy_fraction':ax.set_yscale('log')
        for i,v in enumerate(vals):ax.text(i,v,f'{v:.3g}',ha='center',va='bottom',fontsize=9)
    t=json.loads((OUT/'block_trace_fp32.json').read_text())['assoc1000']['stages']
    axes[2].plot(range(len(t)),[max(1e-9,1-r['pair_cosine']) for r in t],'o-')
    axes[2].set_yscale('log');axes[2].set_xticks(range(len(t)),[r['stage'] for r in t],rotation=70,fontsize=8)
    axes[2].set_title('VP015: 1 - pairwise cosine through Q -> C (FP32)')
    fig.savefig(OUT/'c_collapse.png',dpi=170);plt.close(fig)
    target=json.loads((OUT/'association_target_statistics.json').read_text())
    w=target['same_samples_baselines']
    fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
    for key,label in [('actual_kl','Actual trained attention'),('uniform_kl','Always 20% each'),('static_spatial_kl','Teacher time-average (oracle diagnostic)')]:
        ax.plot([r['end'] for r in w],[r[key] for r in w],'o-',label=label)
    ax.set_xlabel('Last optimizer update in window');ax.set_ylabel('Mean KL on same active training examples');ax.legend();ax.grid(alpha=.3)
    fig.savefig(OUT/'association_vs_constant.png',dpi=170);plt.close(fig)
    # Preserve all cases; figure selection illustrates changed and stable outcomes.
    metrics=[]
    for case in ['VP006','VP014','VP015','VP025','VP030','VP032']:
        a=frames(OLD/'cases'/case/'planned.mp4');b=frames(NEW/'cases'/case/'planned.mp4')
        assert len(a)==len(b)==81
        d=np.abs(a.astype(float)-b.astype(float))/255
        metrics.append(dict(case=case,old_vs_new_rgb_mae=float(d[1:].mean()),
                            old_temporal_rgb_change=float(np.abs(np.diff(a.astype(float),axis=0)).mean()/255),
                            new_temporal_rgb_change=float(np.abs(np.diff(b.astype(float),axis=0)).mean()/255)))
        fig,axes=plt.subplots(2,6,figsize=(15,5.5),constrained_layout=True)
        for i,idx in enumerate([0,16,32,48,64,80]):
            for j,video in enumerate([a,b]):
                axes[j,i].imshow(video[idx]);axes[j,i].set_xticks([]);axes[j,i].set_yticks([])
                if j==0:axes[j,i].set_title(f'{idx/24:.2f}s')
                if i==0:axes[j,i].set_ylabel(['Q128 / 224','Q256 / 1000'][j])
        caption=json.loads((NEW/'cases'/case/'planned.json').read_text())['caption']
        fig.suptitle(case+': '+textwrap.fill(caption,110))
        fig.savefig(OUT/f'paired_{case}.png',dpi=130);plt.close(fig)
    (OUT/'video_pixel_diagnostics.json').write_text(json.dumps(metrics,indent=2))
    for case in ['VP014','VP015']:
        allvideos={'real':frames(NEW/'cases'/case/'planned.mp4')}
        for mode in ['swapped','image_only','no_c']:
            p=OUT/'gpu_results'/mode/'cases'/case/'planned.mp4'
            if p.exists():allvideos[mode]=frames(p)
        fig,axes=plt.subplots(len(allvideos),6,figsize=(15,2*len(allvideos)),squeeze=False,constrained_layout=True)
        for j,(mode,vid) in enumerate(allvideos.items()):
            for i,idx in enumerate([0,16,32,48,64,80]):
                axes[j,i].imshow(vid[idx]);axes[j,i].set_xticks([]);axes[j,i].set_yticks([])
                if i==0:axes[j,i].set_ylabel(mode)
                if j==0:axes[j,i].set_title(f'{idx/24:.2f}s')
        fig.suptitle(case+' - identical native text, first image and initial noise')
        fig.savefig(OUT/f'ablation_{case}.png',dpi=130);plt.close(fig)


if __name__=='__main__':main()
