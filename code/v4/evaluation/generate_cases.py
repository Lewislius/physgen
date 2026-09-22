"""Generate every requested paired condition; no scoring, selection, retry, or restart."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.infer_native_p import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", nargs="+", choices=("wan", "A", "full"), default=["wan", "A", "full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--solver", choices=("euler", "unipc", "dpm++"), default="euler")
    parser.add_argument("--guidance", type=float, default=5)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text())
    for case in cases:
        for seed in args.seeds:
            for variant in args.variants:
                generate(argparse.Namespace(checkpoint=args.checkpoint, image=case["image"], prompt=case["prompt"],
                    negative_prompt="", output=str(Path(args.output) / f"{case['case_id']}_{variant}_seed{seed}.mp4"),
                    device=args.device, seed=seed, frames=case["frames"], fps=case["fps"], duration=None,
                    steps=args.steps, shift=5.0, guidance=args.guidance, solver=args.solver,
                    variant=variant, writer_off=[], reset_state=False))


if __name__ == "__main__":
    main()
