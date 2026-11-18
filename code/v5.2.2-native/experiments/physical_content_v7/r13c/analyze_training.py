"""Compare completed training records on identical samples; never infer video quality."""
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES = {
    "R10A": ROOT.parent / "execution_value_r10/runs/physical_content_67003/train.jsonl",
    "R13A": ROOT.parent / "native_operator_r13a/runs/physical_content_67598/train.jsonl",
    "R13B": ROOT.parent / "interaction_query_r13b/runs/physical_content_67610/train.jsonl",
    "R13C": ROOT / "runs/physical_content_67682/train.jsonl",
}


def summary(values):
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values))


def main():
    records = {name: [json.loads(line) for line in path.read_text().splitlines()]
               for name, path in SOURCES.items()}
    updates, micros = {}, {}
    for name, rows in records.items():
        u = [row for row in rows if row["kind"] == "train"]
        m = [row for row in rows if row["kind"] == "micro"]
        assert len(u) == 400 and {r["step"] for r in u} == set(range(1, 401)), name
        assert len(m) == 1600, name
        updates[name] = {row["step"]: row for row in u}
        micros[name] = {(row["step"], row["micro"]): row for row in m}
        assert len(micros[name]) == 1600, name
    fields = ("sample", "sigma", "text_dropped", "aux_active", "aux_coefficient")
    for name, rows in micros.items():
        for key, row in rows.items():
            assert all(row[f] == micros["R10A"][key][f] for f in fields), (name, key)
    windows = []
    for lo, hi in ((1, 32), (33, 64), (65, 96), (97, 128), (129, 256), (257, 400), (369, 400)):
        models = {}
        for name in SOURCES:
            u = [r for s, r in updates[name].items() if lo <= s <= hi]
            m = [r for (s, _), r in micros[name].items()
                 if lo <= s <= hi and not r["text_dropped"]]
            models[name] = dict(
                positive_microbatches=len(m),
                total_execution_c_token_cosine=summary([r["c_pair_cosine"] for r in m]),
                token_centered_energy_fraction=summary([r["c_centered_energy"] for r in m]),
                cosine_at_least_0_9=sum(r["c_pair_cosine"] >= .9 for r in m),
                losses={k: summary([r[k] for r in u]) for k in ("fm", "tail", "aux", "weighted_aux")},
                gradients={k: summary([r["gradients"][k] for r in u]) for k in u[0]["gradients"]},
                clipped_steps={k: sum(r["clip_scales"][k] < 1 for r in u) for k in u[0]["clip_scales"]},
                phases={phase: sum(r["phase"] == phase for r in u) for phase in ("interface", "joint")},
            )
        windows.append(dict(steps=[lo, hi], models=models))
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources={name: str(path) for name, path in SOURCES.items()},
        matched_microbatches=1600, matched_fields=fields,
        complete_updates_per_model=400, windows=windows,
        scope=[
            "C statistics concern total execution-C tokens within each positive sample, not cross-sample means or isolated learned residuals.",
            "Windows compare the same sample/sigma/dropout records, but representations evolve within a window; they are not a held-out fixed-checkpoint probe.",
            "R13C uses32+368; other three use128+272. Architecture and training schedule are both changed.",
            "fusion_read/write parameter groups have different operators across models; gradient magnitudes do not measure useful contribution.",
            "FM, token cosine and clipping records are mechanism evidence, not generation-quality metrics or acceptance criteria.",
        ],
        quality_established=False,
    )
    out = ROOT / "reports/completed400_training_dynamics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(report=str(out), windows=[dict(
        steps=w["steps"], models={name: dict(
            cosine=v["total_execution_c_token_cosine"]["mean"],
            centered_energy=v["token_centered_energy_fraction"]["mean"],
            positive_microbatches=v["positive_microbatches"],
            high_cosine_count=v["cosine_at_least_0_9"],
        ) for name, v in w["models"].items()}) for w in windows]), ensure_ascii=False))


if __name__ == "__main__":
    main()
