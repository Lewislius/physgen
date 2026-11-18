"""One completed400 audit from the saved checkpoint and actual paired training logs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,math
import torch,yaml

ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT.parent/'execution_value_r10/runs/physical_content_67003/train.jsonl'


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]


def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=2e-6,abs_tol=2e-7)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=int,required=True)
    args=parser.parse_args()
    run=ROOT/'runs'/f'physical_content_{args.experiment}'
    path=run/'step0400.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    prepared=json.loads((ROOT/'prepared.json').read_text())
    cfg=saved['config']
    assert saved['step']==400 and saved['cursor']==1600
    assert saved['revision']=='physical_content_v7_semantic_response_fusion_r15e'
    assert cfg==yaml.safe_load(Path(prepared['config']).read_text())
    assert cfg['source_snapshot']==prepared['snapshot']
    assert cfg['stages']==dict(interface=128,joint=272)
    assert cfg['loss']['semantic_target']=='routing_only'
    assert not cfg.get('initialize_generator')
    count=sum(x.numel() for x in saved['model'].values())
    assert count==670010368
    actual=records(run/'train.jsonl');prior=records(REFERENCE)
    micro=[r for r in actual if r.get('kind')=='micro']
    updates=[r for r in actual if r.get('kind')=='train']
    expected={(r['step'],r['micro']):r for r in prior if r.get('kind')=='micro'}
    assert len(micro)==1600 and len(updates)==400
    assert [r['step'] for r in updates]==list(range(1,401))
    matched=['sample','sigma','text_dropped','aux_coefficient']
    for row in micro:
        ref=expected[(row['step'],row['micro'])]
        assert all(row[k]==ref[k] for k in matched),(row['step'],row['micro'])
        assert row['phase']==('interface' if row['step']<=128 else 'joint')
        assert close(row['generation'],.6*row['fm']+.2*row['tail']+.2*row['temporal'])
        assert close(row['weighted_aux'],row['aux_coefficient']*row['aux'])
        assert close(row['route_aux'],row['aux'])
        assert close(row['weighted_route_aux'],row['weighted_aux'])
        assert all(math.isfinite(row[k]) for k in ['fm','tail','temporal','generation','aux','weighted_aux'])
    for row in updates:
        if row['step']<=128:
            assert row['learning_rates']['wan_lora']==0 and row['gradients']['wan_lora']==0
        else:
            assert row['learning_rates']['wan_lora']>0 and row['gradients']['wan_lora']>0
    ready=next(r for r in actual if r.get('kind')=='model_ready')
    assert sum(ready['groups'].values())==count
    report=dict(utc=datetime.now(timezone.utc).isoformat(),passed=True,experiment=args.experiment,
        checkpoint=str(path),revision=saved['revision'],step=400,cursor=1600,
        saved_parameter_count=count,parameter_groups=ready['groups'],stages=cfg['stages'],
        matched_reference_records=len(micro),reference=str(REFERENCE),matched_fields=matched,
        wan_lora_interface_frozen_joint_updated=True,routing_only_alignment_verified=True,
        generation_and_auxiliary_arithmetic_verified=True,finite_logged_losses=True,
        data_revision=cfg['data_revision'],optimizer_step_wall_minutes=sum(r['seconds'] for r in updates)/60,
        scope='Completed training/config/header/ownership audit, not generation-quality evidence. Logged sigma and deterministic noise seeds do not independently verify unstored noise tensors. Step timings exclude model loading/checkpoint saves.')
    (ROOT/'reports/completed400_training_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','experiment','step','cursor','matched_reference_records','saved_parameter_count']}))


if __name__=='__main__':main()
