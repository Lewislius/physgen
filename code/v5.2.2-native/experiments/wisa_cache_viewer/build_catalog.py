"""Build the browse index for the WISA cache viewer."""
from pathlib import Path
import argparse
import json
import statistics

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
QWEN38 = (BASE / 'experiments/physical_content_v7/data_revision_qwen38/runs/'
          '20261103_dataset')
MANIFEST = QWEN38 / 'encoded/base/manifest.json'
SELECTION = QWEN38 / 'selection.json'
REASONING = QWEN38 / 'reasoning'
DATA_CACHE = Path('/home/liuzhirui/Project/physGen/data_cache/wisa_4000')
FIRST = DATA_CACHE / 'base/first'
OLD_ROOTS = (
    BASE / 'experiments/physical_scaling_q256/supervision_cache/r1/input',
    BASE / 'experiments/physical_variants/supervision_cache/r1/input',
    BASE / 'experiments/physical_v5_scaling_q256/supervision_cache/r1/input',
)
OUTPUT = Path(__file__).resolve().parent / 'data/catalog.json'


def old_input(sample, arm):
    name = f'{sample}_{arm}.pt'
    for root in OLD_ROOTS:
        path = root / name
        if path.is_file():
            return path
    return None


def short(text, limit=320):
    text = (text or '').strip()
    return text if len(text) <= limit else text[:limit].rstrip() + '…'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, help='Optional debug limit')
    args = parser.parse_args(argv)
    rows = json.loads(MANIFEST.read_text())['records']
    selected = set(json.loads(SELECTION.read_text())['ids'])
    if args.limit:
        rows = rows[:args.limit]
    catalog = []
    for row in rows:
        sample = row['id']
        pts = [float(value) for value in row['pts']]
        diffs = [b - a for a, b in zip(pts, pts[1:]) if b > a]
        fps = 1 / statistics.median(diffs) if diffs else 24.0
        old_text = old_input(sample, 'text') is not None
        old_null = old_input(sample, 'null') is not None
        new_text = (REASONING / f'{sample}_text.json').is_file()
        new_null = (REASONING / f'{sample}_null.json').is_file()
        catalog.append(dict(
            id=sample,
            split=row['split'],
            caption=short(row.get('caption')),
            original_caption=short(row.get('original_caption')),
            qwen38=sample in selected,
            old_text=old_text,
            old_null=old_null,
            new_text=new_text,
            new_null=new_null,
            frames=len(row['indices']),
            start_frame=int(row['indices'][0]),
            end_frame=int(row['indices'][-1]),
            fps=round(fps, 6),
            duration=round(pts[-1] - pts[0], 6) if len(pts) > 1 else 0.0,
            first_available=(FIRST / f'{sample}.png').is_file(),
            video_available=Path(row['video']).is_file(),
        ))
    payload = dict(
        generated_from=str(MANIFEST),
        qwen38_revision='qwen38_wisa_reviewed_v1',
        old_source='physical_variants / physical_scaling_q256 supervision caches',
        selected_qwen38=len(selected),
        total=len(catalog),
        counts=dict(
            train=sum(item['split'] == 'train' for item in catalog),
            validation=sum(item['split'] == 'validation' for item in catalog),
            old_text=sum(item['old_text'] for item in catalog),
            old_null=sum(item['old_null'] for item in catalog),
            new_text=sum(item['new_text'] for item in catalog),
            new_null=sum(item['new_null'] for item in catalog),
        ),
        samples=catalog,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in payload.items() if key != 'samples'},
                     ensure_ascii=False, indent=2))
    print(OUTPUT)


if __name__ == '__main__':
    main()
