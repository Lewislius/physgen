"""Join existing cached clips to released WISA annotations without changing either.

This produces annotation SOURCES, not approved physical targets. Caption-derived
WISA intervals/properties are never promoted to measured event times/properties.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = Path('/run/determined/NAS1/public/Dataset/WISA/data/wisa-80k.json')
CACHE = Path('/home/liuzhirui/Project/physGen/data_cache/wisa_4000')
OUTPUT = ROOT / 'wisa_supervision_sources'


def interval(value):
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        a, b = (float(x) for x in value)
    except (TypeError, ValueError):
        return None
    return (a, b) if math.isfinite(a) and math.isfinite(b) and a <= b else None


def main():
    source = json.loads(SOURCE.read_text())
    clips = json.loads((CACHE / 'base/manifest.json').read_text())['records']
    source_keys, physical_keys, source_labels, names = (Counter() for _ in range(4))
    index = {}
    for number, row in enumerate(source):
        source_keys.update(row.keys())
        annotation = row.get('physical_annotation') or {}
        physical_keys.update(annotation.keys())
        source_labels.update([str(row.get('label'))])
        name = row['video_name']
        names.update([name])
        index[name] = (number, row)

    OUTPUT.mkdir(exist_ok=True)
    selected_labels, present, splits = Counter(), Counter(), Counter()
    unmatched, ambiguous, examples = [], [], []
    counts = Counter()
    durations, matched = [], 0
    playback_examples = []
    playback = re.compile(r'slow[ -]?motion|time[ -]?lapse|timelapse|sped[ -]?up|fast[ -]?forward', re.I)
    with (OUTPUT / 'matched_annotation_sources.jsonl').open('w') as sink:
        for clip in clips:
            name = Path(clip['video']).name
            if name not in index:
                unmatched.append(clip['id'])
                continue
            if names[name] != 1:
                ambiguous.append(dict(sample_id=clip['id'], video_name=name, matches=names[name]))
                continue
            number, row = index[name]
            annotation = row.get('physical_annotation') or {}
            matched += 1
            splits.update([clip['split']])
            selected_labels.update([str(row.get('label'))])
            present.update(k for k, v in annotation.items() if v not in (None, '', [], {}))
            pts = clip['pts']
            relative = [float(t - pts[0]) for t in pts]
            durations.append(relative[-1])
            counts['all_pts_strictly_increasing'] += int(all(b > a for a, b in zip(pts, pts[1:])))
            counts['source_label_present_in_cache'] += int(clip.get('category') == row.get('label'))
            counts['cache_has_nonempty_interactions'] += int(bool(clip.get('interactions')))
            counts['cache_has_nonempty_objects'] += int(bool(clip.get('objects')))
            counts['cached_caption_differs_from_source'] += int(clip['caption'].strip() != row['captions'].strip())
            mentions_playback = bool(playback.search(clip['caption']))
            counts['cached_caption_mentions_playback_time_manipulation'] += int(mentions_playback)
            if mentions_playback and len(playback_examples) < 12:
                playback_examples.append(dict(id=clip['id'], label=row['label'],
                    phrases=playback.findall(clip['caption']), caption=clip['caption']))
            n1 = interval(annotation.get('quantify_n1'))
            if n1 is not None:
                counts['parseable_n1'] += 1
                counts['n1_end_exceeds_source_duration'] += int(n1[1] > float(row['duration']) + 0.01)
                counts['n1_has_no_overlap_with_cached_pts_if_misread_as_source_event'] += int(
                    n1[1] < pts[0] or n1[0] > pts[-1])
            record = dict(
                sample_id=clip['id'], split=clip['split'], video_name=name,
                original_catalog_index=number, source_catalog=str(SOURCE),
                source_video=str(clip['video']), source_duration_seconds=row['duration'],
                source_fps=row['fps'], source_pts_seconds=[pts[0], pts[-1]],
                selected_source_frame_indices=clip['indices'],
                rgb_times_seconds=relative, latent_times_seconds=relative[::4],
                geometry=clip['geometry'], cached_caption=clip['caption'],
                original_wisa_caption=row.get('captions'), source_label=row.get('label'),
                original_physical_annotation=annotation,
                target_status='raw_annotation_source_not_approved_physical_ground_truth',
                annotation_scope='original_full_clip_caption_derived_not_cached_crop_verified',
                event_time_status='n1_is_not_a_verified_frame_level_event_annotation',
                spatial_target_status='no_source_region_or_contact_mask_in_this_catalog',
                inference_condition=False,
            )
            sink.write(json.dumps(record, ensure_ascii=False) + '\n')
            if len(examples) < 4:
                examples.append({k: record[k] for k in (
                    'sample_id', 'video_name', 'source_duration_seconds', 'source_pts_seconds',
                    'source_label', 'original_physical_annotation')})

    summary = dict(
        source_catalog=str(SOURCE), cache_manifest=str(CACHE / 'base/manifest.json'),
        source_records=len(source), cached_records=len(clips), matched_unique=matched,
        unmatched=unmatched, ambiguous=ambiguous, matched_splits=dict(splits),
        source_keys=dict(source_keys), physical_annotation_keys=dict(physical_keys),
        source_label_distribution=dict(source_labels), selected_label_distribution=dict(selected_labels),
        selected_nonempty_physical_fields=dict(present), selected_counts=dict(counts),
        cached_clip_interval_seconds_min=min(durations),
        cached_clip_interval_seconds_max=max(durations), examples=examples,
        playback_time_note='PTS measures presentation time; caption matches are indicators, not calibrated capture-time scaling.',
        note='Counts establish data availability/provenance, not correctness of physical labels.',
    )
    (OUTPUT / 'availability_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUTPUT / 'playback_time_caption_examples.json').write_text(json.dumps(playback_examples, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in (
        'source_records', 'cached_records', 'matched_unique', 'unmatched', 'ambiguous',
        'matched_splits', 'selected_label_distribution', 'selected_nonempty_physical_fields',
        'selected_counts', 'cached_clip_interval_seconds_min', 'cached_clip_interval_seconds_max',
    )}, ensure_ascii=False, indent=2))
    print('saved_sources', OUTPUT / 'matched_annotation_sources.jsonl')
    print('saved_summary', OUTPUT / 'availability_summary.json')


if __name__ == '__main__':
    main()
