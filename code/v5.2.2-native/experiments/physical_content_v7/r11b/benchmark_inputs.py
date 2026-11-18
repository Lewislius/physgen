"""The existing fixed60 I2V inputs, with one explicit API reasoning cache."""
from pathlib import Path
import json
import sys
import bootstrap

ROOT=bootstrap.BASE/'experiments/physical_content_v7'
PINNED=bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
sys.path.insert(0,str(PINNED))
from protocol import load_inputs,load_noise,verify_video


def reference():
    return json.loads((ROOT/'videophy60/protocol.json').read_text())


def case_input(case):
    inputs,canvas=load_inputs(case)
    suite='videophy40' if int(case['id'][2:])<=40 else 'videophy60'
    return inputs,dict(id=f'{suite}_{case["id"]}',case=case['id'],caption=case['caption'],
        image=str(canvas.resolve()),duration=float(inputs['times'][-1]))
