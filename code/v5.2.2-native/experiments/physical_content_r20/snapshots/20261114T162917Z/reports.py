"""Small local report from the official VideoPhy2 AutoEval JSON."""
from pathlib import Path
import argparse
import json


def summarize(folder):
    path = Path(folder) / 'scores.json'
    value = json.loads(path.read_text())
    summary = value['summary']
    cases = value.get('cases', [])
    return dict(
        source=str(path),
        count=summary['count'],
        sa_mean=summary['sa_mean'],
        pc_mean=summary['pc_mean'],
        sa_pass_rate=summary['sa_pass_rate'],
        pc_pass_rate=summary['pc_pass_rate'],
        joint_pass_rate=summary['joint_pass_rate'],
        joint_count=sum(row['sa'] >= 4 and row['pc'] >= 4 for row in cases),
        cases=cases,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folders', nargs='+', type=Path, required=True)
    args = parser.parse_args(argv)
    report = {str(folder): summarize(folder) for folder in args.folders}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: {k: v for k, v in value.items() if k != 'cases'}
                      for key, value in report.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
