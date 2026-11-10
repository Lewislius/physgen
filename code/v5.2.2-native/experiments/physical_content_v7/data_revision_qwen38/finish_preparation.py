"""Fresh train-input interface fit and explicit readiness for matched400 only."""
from pathlib import Path
import argparse
import json
import random
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from encode import ROOT,SOURCE,CACHE
from annotate import save
import torch
from physical_scaling.common import atomic_torch
from physical_scaling.data import SampleOrder


def check_schedule(originals,selection):
    schedule=json.loads(Path(selection['schedule']).read_text())['records']
    train=[r for r in originals if r['split']=='train']
    order=iter(SampleOrder(len(train),42))
    expected=[train[next(order)]['id'] for _ in range(1600)]
    if expected!=[r['id'] for r in schedule] or set(expected)!=set(selection['ids']):
        raise ValueError('Prepared data differs from the actual matched400 sampling schedule')
    return schedule


def fit(x,y):
    mean=x.mean(0);scale=(x-mean).square().mean().sqrt()
    bias=y.mean(0);target_scale=(y-bias).square().mean().sqrt()
    normalized=(x-mean)/scale
    centered=y-bias
    gram=normalized.T@normalized/len(x)
    cross=normalized.T@centered/len(x)
    ridge=.01*gram.diag().mean()
    gram.diagonal().add_(ridge)
    weight=torch.linalg.solve(gram,cross).T.contiguous()
    mse=float((normalized@weight.T-centered).square().mean()/target_scale.square())
    return dict(mean=mean,scale=scale,bias=bias,target_scale=target_scale,weight=weight),dict(ridge=float(ridge),train_normalized_mse=mse)


def check_pairs(output,selection):
    root=output/'encoded';cache=root/'content'
    rows=json.loads((root/'base/manifest.json').read_text())['records']
    originals=json.loads((CACHE/'base/manifest.json').read_text())['records']
    if [r['id'] for r in rows]!=[r['id'] for r in originals]:
        raise ValueError('The original full manifest order must be retained')
    chosen=set(selection['ids']);revision=selection['revision'];lookup={r['id']:r for r in rows}
    for row,old in zip(rows,originals):
        for key in ('split','video','indices','pts','geometry'):
            if row[key]!=old[key]:raise ValueError(f'Changed source clip or split: {row["id"]} {key}')
        if row['id'] not in chosen and row['caption']!=old['caption']:
            raise ValueError('Out-of-scope caption changed')
    flags=[]
    for sample in selection['ids']:
        caption=lookup[sample]['caption']
        annotation=json.loads((output/'records'/sample/'caption.json').read_text())
        if annotation['revision']!=revision or annotation['caption']!=caption:
            raise ValueError('Caption provenance mismatch')
        if 'sentence_terminal_missing' in annotation['flags']:
            raise ValueError(f'Unfinished caption requires correction: {sample}')
        if annotation['flags']:flags.append(dict(sample=sample,flags=annotation['flags']))
        t5=json.loads((root/'base/text'/f'{sample}.json').read_text())
        if t5['caption']!=caption or t5['revision']!=revision or t5['tokens']>512:
            raise ValueError('Caption/T5 mismatch or truncation')
        if not (root/'base/text'/f'{sample}.pt').is_file():raise FileNotFoundError(sample)
        for arm in ('text','null'):
            name=f'{sample}_{arm}';expected=caption if arm=='text' else ''
            source=json.loads((output/'reasoning'/f'{name}.json').read_text())
            local=json.loads((cache/'input'/f'{name}.json').read_text())
            if (local['caption']!=expected or source['caption']!=expected
                    or local['relations']!=source['relations'] or local['revision']!=revision
                    or local['scope']!='first_image_and_request_only' or local['memory_shape'][-1]!=4096):
                raise ValueError('First-image reasoning/local feature provenance mismatch')
            for folder in ('input','words','native_words'):
                if not (cache/folder/f'{name}.pt').is_file():raise FileNotFoundError(f'{folder}/{name}')
            words=torch.load(cache/'words'/f'{name}.pt',map_location='cpu',weights_only=True)
            native=torch.load(cache/'native_words'/f'{name}.pt',map_location='cpu',weights_only=True)
            for value in (words,native):
                if (value['caption']!=expected or value['relations']!=source['relations']
                        or value['revision']!=revision or value['sample']!=sample):
                    raise ValueError('Physical-word/native-target mismatch')
            x,y=words['words'],native['native_words']
            if x.shape!=(y.shape[0],4096) or y.shape[1]!=3072 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError('Invalid full-width word target pair')
    return check_schedule(originals,selection),flags


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--required-only',action='store_true',help='Fit the required interface once; skip repeated pair audit and holdout experiment')
    p.add_argument('--describe',action='store_true')
    args=p.parse_args();output=args.output.resolve()
    if args.describe:
        print(json.dumps(dict(scope='1600 actual400 samples only',fitting='Actual 1446 text /154 null schedule, no 50% null reweighting',
            validation='64 deterministic train-input holdouts for interface verification, then fresh fit on all1600',
            new_video_supervision=False,old_cache_writes=False,starts_training=False)))
        return
    torch.set_num_threads(4);begin=time.monotonic()
    selection=json.loads((output/'selection.json').read_text())
    if selection['scope']!='actual400_training_only' or len(selection['ids'])!=1600:
        raise ValueError('Only the explicitly selected 400-update data is authorized')
    status=json.loads((output/'encoding_status.json').read_text())
    if not status['complete']:raise ValueError('Local encoding has not completed')
    if args.required_only:
        originals=json.loads((CACHE/'base/manifest.json').read_text())['records']
        schedule=check_schedule(originals,selection)
        flags=[dict(sample=sample,flags=record['flags']) for sample in selection['ids']
               if (record:=json.loads((output/'records'/sample/'caption.json').read_text()))['flags']]
    else:
        schedule,flags=check_pairs(output,selection)
    cache=output/'encoded/content'
    holdout=set() if args.required_only else set(random.Random(38).sample(selection['ids'],64))
    print(json.dumps(dict(stage='loading_required_training_word_pairs',samples=len(schedule))),flush=True)
    xs=[];ys=[];held=[];start=0
    for item in schedule:
        name=item['id']+('_null' if item['drop'] else '_text')
        x=torch.load(cache/'words'/f'{name}.pt',map_location='cpu',weights_only=True)['words'].float()
        y=torch.load(cache/'native_words'/f'{name}.pt',map_location='cpu',weights_only=True)['native_words'].float()
        xs.append(x);ys.append(y)
        held.extend([item['id'] in holdout]*len(x));start+=len(x)
    x=torch.cat(xs);y=torch.cat(ys);del xs,ys
    mse=constant=relative=None
    if not args.required_only:
        mask=torch.tensor(held,dtype=torch.bool)
        initial,_=fit(x[~mask],y[~mask])
        predicted=(x[mask]-initial['mean'])/initial['scale']@initial['weight'].T+initial['bias']
        mse=float((predicted-y[mask]).square().mean())
        constant=float((initial['bias']-y[mask]).square().mean())
        relative=mse/max(constant,1e-20)
        del initial,predicted
    report=dict(revision=selection['revision'],training_samples=len(schedule),training_words=len(x),
        actual_text_arms=sum(not item['drop'] for item in schedule),actual_null_arms=sum(item['drop'] for item in schedule),
        interface_holdout_samples=sorted(holdout),heldout_relative_mse=relative,
        heldout_mse=mse,heldout_constant_mse=constant,
        repeated_pair_audit_performed=not args.required_only,
        holdout_experiment_performed=not args.required_only,
        scope='Required interface initialization; no video generation-quality claim',
        initialization='Fresh regression of official foundation features from the actual400 training inputs; no project checkpoint')
    save(cache/'interface_fit.json',report)
    if relative is not None and not relative<.95:
        raise RuntimeError('Fresh interface failed the existing improvement-over-constant criterion')
    print(json.dumps(dict(stage='fitting_required_interface',words=len(x))),flush=True)
    parameters,metrics=fit(x,y)
    report.update(metrics,seconds=time.monotonic()-begin)
    atomic_torch(dict(**parameters,training_samples=len(schedule),revision=selection['revision']),cache/'interface.pt')
    save(cache/'interface_fit.json',report)
    import yaml
    overlay=dict(data_revision=selection['revision'],paths=dict(cache=str(output/'encoded'),
        content_cache=str(cache),input_evidence=str(cache/'input'),interface=str(cache/'interface.pt')),
        data_contract=dict(maximum_training_updates=400,seed=42,accumulation=4,
            requires_cache_reader=str(HERE/'cache_reader.py'),preserve_full_manifest_order=True),
        note='Data-path overlay for the next authorized candidate. Not an R11A training config; does not start training.')
    (output/'training_data_overlay.yaml').write_text(yaml.safe_dump(overlay,sort_keys=False,allow_unicode=True))
    ready=dict(training_ready=True,revision=selection['revision'],maximum_training_updates=400,
        unique_training_samples=1600,local_contexts=3200,manifest_records=3857,
        original_sampling_order_preserved=True,unchanged_validation=True,caption_style_flags=flags,
        interface_holdout_relative_mse=relative,active_r11a_changed=False,
        repeated_pair_audit_performed=not args.required_only,holdout_experiment_performed=not args.required_only,
        scope='Ready for the exact seed42/accumulation4 400-update schedule only; no automatic new training')
    manifest_path=output/'encoded/base/manifest.json'
    manifest=json.loads(manifest_path.read_text());manifest.update(training_ready=True,reason=ready['scope'])
    save(manifest_path,manifest);save(output/'encoded/readiness.json',ready)
    print(json.dumps(ready,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
