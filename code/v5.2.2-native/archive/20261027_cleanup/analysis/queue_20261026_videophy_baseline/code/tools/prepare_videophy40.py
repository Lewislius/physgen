"""Fix 40 existing I2V cases and reuse completed Wan-LoRA scores, without sampling."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OCCAM=ROOT.parent/'v5.2-occam'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output):
    source=OCCAM/'inference_outputs/videophy2_joint400_60_pair_seed42'
    original=json.loads((source/'manifest.json').read_text())
    cases=original['cases'][:40]
    assert len(cases)==40 and len({c['id'] for c in cases})==40
    arms={'wan_lora_joint400':source,
          'wan_lora_independent400':OCCAM/'inference_outputs/videophy2_independent_lora400_60_seed42'}
    records={}
    for name,folder in arms.items():
        manifest=json.loads((folder/'manifest.json').read_text())
        if manifest['settings']!=original['settings'] or manifest['evaluation']!=original['evaluation']:
            raise ValueError('Historical sampling/evaluation settings differ')
        other={c['id']:c for c in manifest['cases']};rows=[]
        for case in cases:
            if other[case['id']]!=case:raise ValueError('Historical case inputs differ')
            directory=folder/'cases'/case['id'];video=directory/'wan_lora_only.mp4'
            if not video.is_file():raise FileNotFoundError(video)
            row=dict(id=case['id'],caption=case['caption'],video=str(video))
            for task in ('sa','pc'):
                path=directory/f'wan_lora_only.{task}.json';score=json.loads(path.read_text())
                if score['case_id']!=case['id'] or score['task']!=task or score.get('caption')!=case['caption']:
                    raise ValueError('Historical score/caption mismatch')
                row[task]=score['score'];row[f'{task}_sha256']=digest(path)
            rows.append(row)
        records[name]=dict(directory=str(folder),checkpoint=manifest['checkpoint'],cases=rows)
    cache=Path('/home/liuzhirui/Project/physGen/data_cache/wisa_4000/base/manifest.json')
    train={r['caption'].strip().lower() for r in json.loads(cache.read_text())['records'] if r['split']=='train'}
    overlap=[c['id'] for c in cases if c['caption'].strip().lower() in train]
    protocol=dict(scope=original['scope'],source=str(source),source_manifest_sha256=digest(source/'manifest.json'),
        selection='First 40 existing IDs, fixed before this candidate is scored; no score-based selection',
        cases=cases,settings=original['settings'],evaluation=original['evaluation'],
        primary_reference='wan_lora_joint400',historical=records,
        target='Improve independent VideoPhy2 physical/semantic scores and joint pass rate; inspect full visual demos.',
        exact_train_caption_overlap=overlap,overlap_scope='Exact captions only, not a complete semantic or visual duplicate audit',
        training_evaluator_used=False,new_baseline_generations=0)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol was already fixed differently')
    path.write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n')
    (output/'case_ids.json').write_text(json.dumps([c['id'] for c in cases],indent=2)+'\n')
    for name,record in records.items():
        rows=record['cases'];n=len(rows)
        print(name,json.dumps(dict(n=n,sa=sum(x['sa'] for x in rows)/n,pc=sum(x['pc'] for x in rows)/n,
            joint=sum(x['sa']>=4 and x['pc']>=4 for x in rows)/n)))
    print('exact_train_caption_overlap',overlap)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');prepare(p.parse_args().output)
