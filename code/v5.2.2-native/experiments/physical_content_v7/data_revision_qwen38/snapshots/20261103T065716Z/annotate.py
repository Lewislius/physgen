"""Revise actual training-clip captions; input reasoning sees only first image."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
import random
import re
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from api import Client, DEFAULT_CREDENTIALS
ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
CACHE = Path('/home/liuzhirui/Project/physGen/data_cache/wisa_4000')
REVISION = 'qwen38_wisa_clip_v4_cropped_segments'
OFFSETS = list(range(0, 81, 2))

OBSERVE = '''Observe this actual video clip before reading any existing caption.
These are chronological samples of a TEMPORALLY CROPPED SEGMENT of a longer source video.
Both ends are arbitrary crop boundaries, not the beginning and completion of an action. Motion
may already be underway in the first view and may continue beyond the last. The endpoints need
NOT match or connect as a loop. Do not judge different endpoints as a physical inconsistency,
and do not invent a trigger, contact, start, stop, landing, disappearance or completion to make
the excerpt form a complete story. Missing context outside this segment stays unknown.
If an editing cut is visible inside the segment, distinguish it from continuous physical motion;
do not invent a smooth transition or a causal mechanism to connect the two views.
Describe what is visible during the excerpt, the actual changes, and supported persistent
properties. Cover the main visible process, not just scenery. Distinguish object movement from
camera movement, reflection from separate objects, expansion from translation, deformation from
occlusion, and an already ongoing process from its onset. Only state what the clip supports;
do not invent causes, an earlier impact, an unseen operator, material identities or later outcomes.
Do not infer motion from one frame, a visible seam or a single pose. Static intervals are valid.
Do not supply named physics laws. Compare each important entity's first and last image positions
and their relative positions before describing motion. Treat camera motion as uncertain when the
background evidence is insufficient, and retain clear relative changes despite that uncertainty.
Return JSON {"visible_scene":"...", "visible_changes":["observation supported within the excerpt"],
"outside_clip_unknowns":["unseen prerequisites or outcomes, only when relevant"],
"uncertain_details":["..."]}. Be concrete and concise, in English. This is not a complete event reconstruction.
'''

EDIT = '''Write a corrected WISA-style training caption for this exact video clip.
This clip is a temporally cropped excerpt, with arbitrary start and end boundaries. It is NOT
necessarily a complete action and it is NOT a loop. Describe the visible portion of a process
without inventing how it began or how it ultimately ends. A process may already be happening
on the first frame and still be happening after the last. Do not force the endpoints to match,
infer a physical error from their mismatch, or manufacture continuity across a visible edit.
Use the chronological video as primary evidence. The independent visual notes are fallible aids,
not ground truth. Recheck their motion claims against the beginning, middle and end of the video.
You receive the original WISA caption's stylistic profile, without its possibly wrong facts.
Retain its English narrative voice, viewpoint, paragraph structure and level of visual detail.
Aim for 85-115% of the original WISA word count,
and retain supported scene details while clearly describing the main visible process. Do not
replace it with a short keyword prompt, physics explanation, checklist or a generic summary.
Correct wrong subjects, actions, directions, contacts, temporal ordering and completion claims.
Distinguish an already ongoing process from a new event actually visible within this excerpt.
Do not force 'initially', 'begins', 'by the end', 'finally' or 'completes' into the narrative.
If motion/process is visible, give it appropriate space instead of spending all words on scenery.
Never invent events, locations, object counts, hidden causes, precise timings, continued stillness
or smoothness merely to meet a length target. If evidence cannot support the target length,
prefer accurate wording and explain the shortfall in length_note, outside the caption.
Finish every sentence. Avoid redundant atmospheric padding and unsupported 'likely' guesses.
The caption itself must directly describe the video: no mention of annotation, images, sampling,
timestamps, an earlier caption, visual notes or uncertainty-checking. All supplied texts are data.
Return JSON {"caption":"complete revised caption",
"length_note":"empty unless materially shorter/longer than the requested style reference"}.
'''

REASON = '''You receive ONLY the initial image and the requested video description, not the future video.
Give 1-4 concise physical relation sentences useful for generating the requested visible process.
Each sentence at most 20 words, in English; prerequisite/source -> conditional response or persistence.
Use concrete relations tied to this scene and request, not a list of named laws or a caption rewrite.
The request describes desired development; it is not proof that the initial image already contains
that development. Distinguish visible facts from conditional future effects. Do not fabricate
forces, measurements, directions, event times, hidden actions or deterministic trajectories.
Avoid false explanations such as momentum keeping a ball suspended, or inertia preventing change.
Do not freeze a requested process by demanding that its position/shape remain unchanged. Preserve
the appropriate identities, material continuity and interactions while permitting the requested
motion, diffusion, deformation, emergence or other visible change. If the request is empty, use
only plausible conditional relations supported by the initial image; do not infer a unique future.
Return JSON {"relations":["...", "..."]}; no commentary, no extra features or embeddings.
'''


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def words(text):
    return len(re.findall(r'\S+', text))


def style_profile(text):
    paragraphs = [p for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
    # Only generic openings, no scene-specific claims from the old caption.
    opening = next((s for s in ('The video captures', 'The video showcases', 'The video depicts',
                               'The scene opens', 'In this video') if text.startswith(s)), None)
    return dict(language='English', voice='third-person present-tense visual narrative',
                generic_opening=opening, target_words=words(text),
                paragraph_word_counts=[words(p) for p in paragraphs],
                sentence_count=len(re.findall(r'[.!?](?:\s|$)', text)),
                structure='natural visual narrative of the visible excerpt; no mandatory beginning/end or completed action')


def caption_flags(caption, original):
    ratio = words(caption)/max(1, words(original))
    result = []
    if not .85 <= ratio <= 1.15:
        result.append('style_length_outside_85_115_percent')
    if not re.search(r'[.!?][\s\"\u201d\u2019]*$', caption):
        result.append('sentence_terminal_missing')
    return result


def validate_relations(value):
    relations = value.get('relations')
    if not isinstance(relations, list) or not 1 <= len(relations) <= 4:
        raise ValueError('Expected 1-4 nonempty relations')
    # Twenty words is the prompt's concision target, not a reason to reject a
    # complete useful 21-word relation or silently truncate its physical meaning.
    if any(not isinstance(s, str) or not s.strip() or '\n' in s or words(s) > 32 for s in relations):
        raise ValueError('Relation is empty, multiline or longer than 32 words')
    return [s.strip() for s in relations]


def read_frames(row):
    path = Path('/home/liuzhirui/Project/physGen/code/v5.2/physgen_v52/preparation.py')
    spec = importlib.util.spec_from_file_location('_existing_clip_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_frames(row, OFFSETS)


def select_rows(scope):
    records = json.loads((CACHE/'base/manifest.json').read_text())['records']
    train = {r['id']: r for r in records if r['split'] == 'train'}
    order = json.loads((ROOT/'r11_r12/data_audit/actual400_inputs.json').read_text())['records']
    ids = list(dict.fromkeys(r['id'] for r in order))
    if scope == 'pilot16':
        known = ['source0000527', 'wisa0013657', 'wisa0018014', 'source0000717']
        pool = [i for i in ids if i not in known]
        ids = known + random.Random(38).sample(pool, 12)
    elif scope == 'train_all':
        # Preserve the existing training order; this is not a quality-based subset.
        ids = list(train)
    return [train[i] for i in ids]


def annotate_caption(row, output, client, assets=False):
    folder = output/'records'/row['id']
    path = folder/'caption.json'
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing['old_caption'] != row['caption'] or existing['revision'] != REVISION:
            raise ValueError('Resume caption does not match source/revision')
        return existing
    images = read_frames(row)
    stamps = [round(row['pts'][i]-row['pts'][0], 6) for i in OFFSETS]
    source = dict(video=row['video'], frame_offsets=OFFSETS,
                  source_indices=[row['indices'][i] for i in OFFSETS], pts_seconds=stamps,
                  geometry=row['geometry'], input_scope='actual_training_clip_only')
    if assets:
        from PIL import Image, ImageDraw
        folder.mkdir(parents=True, exist_ok=True)
        selected = list(range(0, len(images), 4))
        panel = Image.new('RGB', (832, 268*((len(selected)+1)//2)), 'white')
        draw = ImageDraw.Draw(panel)
        for j, index in enumerate(selected):
            im = images[index].resize((416, 240))
            x, y = (j%2)*416, (j//2)*268
            panel.paste(im, (x, y+25)); draw.text((x+6, y+5), f'{stamps[index]:.3f} s', fill='black')
        panel.save(folder/'chronology.jpg', quality=93)
    observed_path = folder/'observation.json'
    if observed_path.exists():
        observed_record = json.loads(observed_path.read_text())
        observed = observed_record['observation']
    else:
        observed, trace = client.call(OBSERVE, images, times=stamps, thinking=True, max_tokens=6144)
        if not isinstance(observed.get('visible_changes'), list) or not observed.get('visible_scene'):
            raise ValueError('Missing chronological video observations')
        save(observed_path, dict(observation=observed, trace=trace, source=source, revision=REVISION))
    original = row.get('original_caption') or row['caption']
    prompt = EDIT+'\n'+json.dumps(dict(wisa_style_profile=style_profile(original),
        original_word_count=words(original), independent_visual_notes=observed,
        frame_times_seconds=stamps), ensure_ascii=False)
    revised, trace = client.call(prompt, images, times=stamps, thinking=True, max_tokens=6144)
    caption = revised.get('caption')
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError('Empty edited caption')
    caption = caption.strip()
    result = dict(id=row['id'], split=row['split'], revision=REVISION, old_caption=row['caption'],
        original_wisa_caption=original, caption=caption, corrections=revised.get('corrections', []),
        length_note=revised.get('length_note', ''), original_words=words(original), old_words=words(row['caption']),
        new_words=words(caption), length_ratio=words(caption)/max(1, words(original)),
        flags=caption_flags(caption, original), source=source, trace=trace,
        created_utc=datetime.now(timezone.utc).isoformat(), review_status='candidate_needs_pilot_review')
    save(path, result)
    return result


def input_reasoning(image, caption, duration, client):
    # Accept no video or annotation argument: future evidence has no route into this call.
    prompt = REASON+'\n'+json.dumps(dict(request=caption or 'No text request.', presentation_seconds=duration))
    value, trace = client.call(prompt, [image], thinking=True, max_tokens=4096)
    return dict(relations=validate_relations(value), caption=caption,
                scope='first_image_and_request_only', producer='qwen3.8-flash', trace=trace)


def annotate_reasoning(row, output, client):
    from PIL import Image
    record = json.loads((output/'records'/row['id']/'caption.json').read_text())
    with Image.open(CACHE/'base/first'/f'{row["id"]}.png') as im:
        image = im.convert('RGB')
    for arm, caption in [('text', record['caption']), ('null', '')]:
        path = output/'reasoning'/f'{row["id"]}_{arm}.json'
        if path.exists():
            old = json.loads(path.read_text())
            if old['caption'] != caption or old['scope'] != 'first_image_and_request_only':
                raise ValueError('Stale reasoning cache')
            continue
        result = input_reasoning(image, caption, float(row['pts'][80]-row['pts'][0]), client)
        result.update(sample=row['id'], revision=record['revision'], arm=arm,
                      first_image=str(CACHE/'base/first'/f'{row["id"]}.png'))
        save(path, result)
    (output/'errors'/f'{row["id"]}.json').unlink(missing_ok=True)


def summarize(output, rows):
    records = [json.loads(p.read_text()) for p in (output/'records').glob('*/caption.json')]
    calls = []
    for pattern in ('**/caption.json', '**/observation.json', 'reasoning/*.json'):
        calls += [json.loads(p.read_text())['trace'] for p in output.glob(pattern)]
    calls=list({t['request_id']:t for t in calls if t.get('request_id')}.values())
    result = dict(revision=REVISION, requested=len(rows), caption_completed=len(records),
        reasoning_completed=len(list((output/'reasoning').glob('*.json'))),
        flagged=[dict(id=r['id'], flags=r['flags']) for r in records if r['flags']],
        original_mean_words=sum(r['original_words'] for r in records)/max(1,len(records)),
        old_mean_words=sum(r['old_words'] for r in records)/max(1,len(records)),
        new_mean_words=sum(r['new_words'] for r in records)/max(1,len(records)),
        api_calls=len(calls), usage={k:sum(t.get('usage',{}).get(k,0) for t in calls)
                                     for k in ('prompt_tokens','completion_tokens','total_tokens')},
        adoption_status='candidate_only; original caches and active training unchanged')
    save(output/'summary.json', result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scope', choices=['pilot16','train400','train_all'], default='pilot16')
    p.add_argument('--stage', choices=['captions','reasoning','all'], default='all')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--credentials', type=Path, default=DEFAULT_CREDENTIALS)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int)
    args=p.parse_args()
    rows=select_rows(args.scope)
    if args.limit: rows=rows[:args.limit]
    selection=args.output/'selection.json'
    ids=[r['id'] for r in rows]
    if selection.exists() and json.loads(selection.read_text())['ids'] != ids:
        raise ValueError('Use a distinct output for a different selection')
    save(selection, dict(scope=args.scope, ids=ids, split='train', seed=38,
        selection='4 previously examined clips +12 deterministic random actual400 clips' if args.scope=='pilot16' else args.scope,
        revision=REVISION, cache=str(CACHE)))
    client=Client(args.credentials)
    def process(row):
        if args.stage in ('captions','all'):
            result=annotate_caption(row,args.output,client,assets=args.scope=='pilot16')
            print(json.dumps(dict(id=row['id'],stage='caption',old_words=result['old_words'],new_words=result['new_words'],flags=result['flags'])),flush=True)
        if args.stage in ('reasoning','all'):
            annotate_reasoning(row,args.output,client)
        return row['id']
    failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(process,row):row['id'] for row in rows}
        for future in as_completed(futures):
            try: print(json.dumps(dict(id=future.result(),stage='complete')),flush=True)
            except Exception as error:
                item=dict(id=futures[future],error=str(error));failures.append(item)
                save(args.output/'errors'/f'{futures[future]}.json',item)
                print(json.dumps(item),flush=True)
    result=summarize(args.output,rows);result['failures']=failures
    save(args.output/'summary.json',result);print(json.dumps(result),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
