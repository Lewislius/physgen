"""Freeze the exact locally evaluated PhysVid caption set, checkpoints and jobs."""
from pathlib import Path
import csv
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE, ROOT, MODELS, read, save, aggregate


def main():
    import yaml
    old = BASE / 'experiments/physical_content_v7'
    reference = Path('/home/liuzhirui/model/PhysVid/res/videophy2/scores.json')
    scores = read(reference)
    csv_path = Path('/home/liuzhirui/benchmark/videophy/videophy2_test.csv')
    rows = list(csv.DictReader(csv_path.open()))
    metadata = {}
    for row in rows:
        caption = row['caption'].strip().strip('\"\'').strip()
        if caption:
            metadata.setdefault(caption, {k: row[k] for k in ('category', 'action', 'is_hard')})
    captions = {x['caption'] for x in scores.values()}
    if captions != set(metadata) or len(captions) != 599:
        raise ValueError('Must use the same599 captions as the completed local PhysVid evaluation')
    versions = dict(wan_lora='native400', r1='R1_400', r9='R9_400', r10a='R10_400', r10b='R10B_400')
    records = {x['version']: x for x in csv.DictReader((old/'reports/completed_full40_summary.csv').open())}
    models = {}
    for model, version in versions.items():
        previous = read(Path(records[version]['source_scores']).parent/'manifest.json')
        ckpt = Path(previous['checkpoint'])
        if not ckpt.is_file() or previous['checkpoint_step'] != 400:
            raise ValueError('Need the preserved own400 checkpoint: '+model)
        models[model] = dict(checkpoint=str(ckpt), source=previous['training_source'],
                             step=400, resource_pool='amp-80g' if model in ('wan_lora','r1') else 'blk-96g')
    config = yaml.safe_load(Path('/home/liuzhirui/model/PhysVid/configs/generate_videophy_samples.yaml').read_text())
    cases = []
    for index, (key, row) in enumerate(scores.items()):
        caption = row['caption']
        cases.append(dict(id=f'PV{index+1:04d}', index=index, caption=caption,
            physvid_key=key, **metadata[caption], preparation_owner=MODELS[index % len(MODELS)],
            image_seed=20261101+index,
            first_prompt=('A single realistic photograph of the initial scene immediately BEFORE the following action begins: '
                          +caption+' Show the participants and objects in their starting positions, with enough room for the action. '
                          'Show one moment only, not a montage, not the completed outcome. No text or diagrams.')))
    protocol = dict(schema=1, scope='Shared independently generated first-image I2V; matches the one completed local PhysVid599 set, not paper five-seed T2V.',
        reference_scores=str(reference), reference_summary=aggregate(list(scores.values())),
        reference_seed_evidence='/home/liuzhirui/model/PhysVid/res/videophy2/logs/all.log',
        reference_generation_seed=9898539, completed_reference_sets=1,
        failed_reference_attempt_seeds=[1025458,8699227,8932740,8989918],
        generation_seeds=[9898539], reference_rng_note='Local PhysVid used distributed CUDA RNG; its seed identifies the run. Our five models use paired per-caption CPU noise, not the same latent noise as the different Wan2.1 architecture.',
        dataset=str(csv_path), paper_caption_count=590, actual_public_unique_count=599,
        models=models, cases=cases,
        sampling=dict(width=832,height=480,frames=81,fps=16,steps=50,shift=8.0,text_guidance=6.0,physical_guidance=5.0,
                      negative_prompt=config['negative_prompt'],sampler='shifted_euler',
                      guidance_note='Native CFG6 matches PhysVid; physical gamma5 retained by explicit user instruction; three-branch existing guidance formula.'),
        image_generation=dict(generator='Official pretrained Wan2.2-TI2V-5B, no project LoRA',steps=50,shift=5.,guidance=5.,solver='unipc',
                              shared_across_models=True,shared_across_seeds=True,reuse_physvid_video_frames=False,reuse_old60=False),
        evidence=dict(scope='Only generated initial image and original caption; no reference video, score, physics labels or future frames',
                      duration_seconds=5.0,reasoning_tokens=128,negative_C='Own image without text request, unchanged model convention'),
        evaluation=dict(code='/home/liuzhirui/model/PhysVid/physvid/evaluation/videophy2',
                        checkpoint='/home/liuzhirui/benchmark/videophy/videophy_2_auto',num_frames=32,batch_size=1,
                        seed=20261009,generation=dict(do_sample=False,top_k=1,temperature=.001,max_length=256),
                        score_parser='Exact local PhysVid parser; preserve raw output and zero fallbacks',threshold=4),
        sources=['https://arxiv.org/html/2603.26285v1','https://github.com/5aurabhpathak/PhysVid/blob/main/physvid/evaluation/videophy2_eval.py'])
    path=ROOT/'protocol.json'
    if path.exists() and read(path)!=protocol:
        raise ValueError('Existing frozen evaluation protocol differs')
    save(path,protocol)
    print(json.dumps(dict(cases=len(cases),seeds=protocol['generation_seeds'],videos=len(cases)*len(models),models=models)))


if __name__ == '__main__':
    main()
