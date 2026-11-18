"""Sequential local text writing and compatible feature rebuild on one GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--encode-only',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    def stage(name,command):
        status=dict(stage=name,source=str(args.source),selection=str(args.selection),
                    seconds=time.monotonic()-started,api_calls=0)
        (args.output/'stage.json').write_text(json.dumps(status,indent=2)+'\n')
        with (args.output/(name+'.log')).open('a') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    python=[sys.executable,'-I','-B','-u']
    if not args.encode_only:
        stage('write_local_text',[*python,str(args.source/'extract_local.py'),
            '--selection',str(args.selection),'--output',str(args.records),
            '--model','/home/liuzhirui/model/Qwen3-30B-A3B-Instruct-2507',
            '--batch-size','16'])
    stage('encode_local_features',[*python,str(args.source/'encode_local.py'),
        '--selection',str(args.selection),'--text-records',str(args.records),
        '--output',str(args.output)])
    (args.output/'stage.json').write_text(json.dumps(dict(stage='complete',
        source=str(args.source),selection=str(args.selection),
        seconds=time.monotonic()-started,api_calls=0),indent=2)+'\n')


if __name__=='__main__':main()
