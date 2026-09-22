"""Export fixed validation conditions and empty rating rows; no automatic quality labels."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decord import VideoReader, cpu
from PIL import Image

from physgen_v4.runtime import read_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", nargs="+", default=["wan", "A", "full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    args = parser.parse_args()
    config = read_config(args.config)
    manifest = json.loads((Path(config["paths"]["cache_root"]) / "manifest.json").read_text())
    records = [record for record in manifest["records"]
               if record["split"] == "validation" and not record.get("excluded_reason")][:args.count]
    output = Path(args.output)
    (output / "images").mkdir(parents=True, exist_ok=True)
    cases = []
    for record in records:
        case_id = f"wisa_{record['index']:07d}"
        reader = VideoReader(str(Path(config["paths"]["wisa_videos"]) / record["video_name"]), ctx=cpu(0), num_threads=1)
        image = output / "images" / f"{case_id}.png"
        Image.fromarray(reader[record["view"]["start_frame"]].asnumpy()).save(image)
        cases.append(dict(case_id=case_id, image=str(image.resolve()), prompt=record["caption"],
                          caption_scope=record["caption_scope"], fps=record["view"]["source_fps"],
                          frames=record["view"]["frames"], source_view=record["view"], video_name=record["video_name"]))
    (output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    with (output / "ratings_template.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["case_id", "seed", "variant", "process_correct", "quality", "notes"])
        for case in cases:
            for seed in args.seeds:
                for variant in args.variants:
                    writer.writerow([case["case_id"], seed, variant, "", "", ""])


if __name__ == "__main__":
    main()
