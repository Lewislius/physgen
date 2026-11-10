"""Refresh one comparable table from available scores, clearly marking partial sets."""
from pathlib import Path
import csv
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT, MODELS, plan, read, save


def main():
    p=plan();rows=[];details={}
    ref=p['reference_summary']
    rows.append(dict(model='PhysVid_existing_T2V',status='completed',videos=ref['count'],seed_sets=1,
        sa_mean=ref['sa_mean'],pc_mean=ref['pc_mean'],sa_rate=ref['sa_pass_rate'],pc_rate=ref['pc_pass_rate'],joint_rate=ref['joint_pass_rate']))
    for model in MODELS:
        path=ROOT/'outputs'/model/'summary.json'
        if not path.exists():continue
        d=read(path);details[model]=d;m=d['seed_mean']
        rows.append(dict(model=model,status=d['status'],videos=sum(x['count'] for x in d['seeds']),seed_sets=len(d['seeds']),
            sa_mean=m['sa_mean'],pc_mean=m['pc_mean'],sa_rate=m['sa_pass_rate'],pc_rate=m['pc_pass_rate'],joint_rate=m['joint_pass_rate']))
    save(ROOT/'comparison.json',dict(rows=rows,details=details,scope=p['scope'],
        comparison_note='Five I2V models share images/noise; existing PhysVid T2V is a contextual reference, not identical input conditioning. Partial rows are not full599 results.'))
    path=ROOT/'comparison.csv';temp=path.with_suffix('.csv.tmp')
    # Called only by the submission-side review, not concurrently by GPU workers.
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temp.replace(path)
    print(rows)


if __name__=='__main__':main()
