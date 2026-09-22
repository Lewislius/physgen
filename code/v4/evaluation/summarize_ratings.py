"""Cluster paired video ratings by source case, then report bootstrap confidence intervals."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return dict(successes=successes, cases=count, rate=p,
                confidence_interval_95=[max(0.0, center - radius), min(1.0, center + radius)])


def summarize(path, baseline, candidate, draws, seed):
    with Path(path).open() as file:
        rows = list(csv.DictReader(file))
    scores = {(row["case_id"], row["seed"], row["variant"]): row for row in rows}
    pairs = [(row["case_id"], row["seed"]) for row in rows if row["variant"] == baseline]
    cases = list(dict.fromkeys(case for case, _ in pairs))
    totals = []
    old_all_pass, new_all_pass, damage_cases, eligible_cases = [], [], [], []
    for case in cases:
        case_pairs = [(c, s) for c, s in pairs if c == case]
        values = []
        for case_id, sample_seed in case_pairs:
            old, new = scores[(case_id, sample_seed, baseline)], scores[(case_id, sample_seed, candidate)]
            b, c = float(old["process_correct"]), float(new["process_correct"])
            values.append([b, c, c - b, b * (1 - c), float(new["quality"]) - float(old["quality"])])
        totals.append(np.mean(values, axis=0))
        old_all_pass.append(all(value[0] == 1 for value in values))
        new_all_pass.append(all(value[1] == 1 for value in values))
        eligible_cases.append(any(value[0] == 1 for value in values))
        damage_cases.append(any(value[3] == 1 for value in values))
    totals = np.asarray(totals)
    generator = np.random.default_rng(seed)
    samples = np.stack([totals[generator.integers(0, len(cases), len(cases))].mean(axis=0) for _ in range(draws)])
    names = ["baseline_success", "candidate_success", "net_success_gain", "harm_among_all_pairs", "quality_delta"]
    estimates = {name: dict(mean=float(totals[:, i].mean()), confidence_interval_95=interval(samples[:, i]))
                 for i, name in enumerate(names)}
    # Unconditional harm is defined even when the baseline has no successful videos.
    # Conditional damage is only estimable when baseline successes exist in the sample.
    base_rate = totals[:, 0].mean()
    conditional = float(totals[:, 3].mean() / base_rate) if base_rate > 0 else None
    case_damage = wilson(sum(damage_cases), sum(eligible_cases)) if any(eligible_cases) else None
    return dict(baseline=baseline, candidate=candidate, independent_source_cases=len(cases),
                video_pairs=len(pairs), bootstrap_draws=draws, bootstrap_seed=seed, estimates=estimates,
                damage_among_baseline_successes=conditional,
                independent_case_intervals=dict(baseline_all_seeds_pass=wilson(sum(old_all_pass), len(cases)),
                    candidate_all_seeds_pass=wilson(sum(new_all_pass), len(cases)),
                    any_damage_given_some_baseline_success=case_damage),
                scope="human process/quality ratings; case-cluster intervals, not a guarantee on new inputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize(args.ratings, args.baseline, args.candidate, args.draws, args.seed)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
