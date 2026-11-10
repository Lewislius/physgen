"""Recover a complete isolated metric stream from its trial stdout.

This is needed for the 2026-10-24 rescheduling collision: a graceful cancellation
allowed two attempts to append to the same shared file. Keep that damaged file;
recover each attempt separately from the cluster's container-specific log.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=subprocess.run(['det','-m','http://10.130.130.5:4096','experiment','logs',str(args.experiment)],
                          capture_output=True,text=True,check=True)
    rows=[]
    for line in result.stdout.splitlines():
        if ' || {' not in line:continue
        try:row=json.loads(line.split(' || ',1)[1])
        except json.JSONDecodeError:continue
        if row.get('kind') in ['evaluation_protocol','reliance','complete']:rows.append(row)
    keys=[tuple(row[k] for k in ['sample','split','seed','sigma','native_text','variant'])
          for row in rows if row['kind']=='reliance']
    if len(rows)!=322 or len(keys)!=320 or len(set(keys))!=320 or rows[-1]['kind']!='complete':
        raise ValueError(f'Not a complete unique study: {len(rows)} records, {len(set(keys))} cases; no result written')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (args.output/'provenance.json').write_text(json.dumps(dict(experiment=args.experiment,
        source='Independent cluster stdout; only complete JSON metric records retained',records=len(rows)),indent=2))
    print(json.dumps(dict(experiment=args.experiment,rows=len(rows),device=rows[0]['device'],output=str(args.output))))


if __name__=='__main__':main()
