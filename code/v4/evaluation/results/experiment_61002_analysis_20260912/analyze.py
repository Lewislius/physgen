"""Read-only audit of the exact 1--306 step range in the supplied experiment log.

Run with the moviestory Python environment. Writes only artifacts beside this file;
does not import the training code, open CUDA, or load model/optimizer checkpoints.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SESSION = ROOT / "train/train_log/wisa_native_p_A1_4x96g/20260912T125641Z-a822428d"
LOG = ROOT / "logs/experiment_61002_trial_61841_logs.txt"
SITES = [5, 10, 15, 20, 25, 30]
WINDOWS = [(1, 50), (51, 100), (101, 150), (151, 200), (201, 250), (251, 300), (301, 306)]


def read_records(path):
    # Snapshot complete lines from a file that the active job may still append to.
    raw = path.read_bytes()
    complete = raw[:raw.rfind(b"\n") + 1]
    return [json.loads(line) for line in complete.splitlines() if line.strip()]


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(records, key):
    return float(np.mean([row["metrics"][key] for row in records]))


def rolling(values, width=25):
    return np.convolve(values, np.ones(width) / width, mode="valid")


def main():
    raw_log = LOG.read_bytes()
    console = {}
    line_numbers = {}
    for line_number, line in enumerate(raw_log.decode().splitlines(), 1):
        match = re.search(r"\[train\] step=(\d+)/(\d+)\s+(.*)", line)
        if match:
            step = int(match[1])
            console[step] = {key: float(value) for key, value in re.findall(r"([^\s=]+)=([^\s]+)", match[3])}
            line_numbers[step] = line_number
    end = max(console)
    if end != 306:
        raise ValueError(f"The supplied log changed: expected last step 306, got {end}")
    records = read_records(SESSION / "steps.jsonl")
    train = [row for row in records if row["split"] == "train" and row["step"] <= end]
    validation = [row for row in records if row["split"] == "validation" and row["step"] <= end]
    micro = [row for row in read_records(SESSION / "micro_rank0.jsonl") if row["step"] <= end]
    config = json.loads((SESSION / "config.json").read_text())
    if [row["step"] for row in train] != list(range(1, end + 1)):
        raise ValueError("Missing or repeated optimizer steps")
    if len(micro) != end * config["train"]["accumulation_steps"]:
        raise ValueError("Incomplete microbatch coverage")
    max_console_difference = max(abs(value - row["metrics"][key])
                                 for row in train for key, value in console[row["step"]].items())
    if max_console_difference > 1e-5:
        raise ValueError("The detailed session does not match the supplied console log")
    nonfinite = sum(not np.isfinite(value) for row in train for value in row["metrics"].values())
    keys = list(train[0]["metrics"])
    windows = []
    for lower, upper in WINDOWS:
        subset = [row for row in train if lower <= row["step"] <= upper]
        row = dict(start=lower, end=upper, count=len(subset))
        row.update({key: mean(subset, key) for key in keys})
        row["diagnostic/fixed_full_aux_weight"] = float(np.mean([
            item["metrics"]["loss/fm"]
            + (item["metrics"]["loss/weighted_struct"] + item["metrics"]["loss/weighted_prior"])
            / item["metrics"]["loss/struct_warmup"] for item in subset]))
        row["diagnostic/fm_fraction_of_total"] = row["loss/fm"] / row["loss/total"]
        row["diagnostic/initializer_squared_gradient_fraction"] = float(np.mean([
            item["metrics"]["grad/initialization_norm"] ** 2
            / item["metrics"]["grad/global_norm_before_clip"] ** 2 for item in subset]))
        windows.append(row)
    write_csv("window_means.csv", windows)
    write_csv("step_metrics.csv", [dict(step=row["step"], **row["metrics"]) for row in train])
    buckets = []
    for lower, upper in WINDOWS:
        for a, b in zip([0, .2, .4, .6, .8], [.2, .4, .6, .8, 1.0]):
            subset = [row for row in micro if lower <= row["step"] <= upper
                      and a <= row["metrics"]["sigma"] < b]
            buckets.append(dict(start=lower, end=upper, sigma_lower=a, sigma_upper=b, count=len(subset),
                                **{key: mean(subset, key) for key in ["sigma", "loss/fm", "loss/struct"]}))
    write_csv("micro_sigma_bins.csv", buckets)
    gate_rows = []
    recent = [row for row in train if 251 <= row["step"] <= 300]
    for site in SITES:
        prefix = f"A@{site}/"
        permanently_closed = next(row["step"] for i, row in enumerate(train)
                                  if all(other["metrics"][prefix + "state_gate_mean"] < .001
                                         for other in train[i:]))
        gate_rows.append(dict(site=f"A@{site}", permanently_below_0_001_from_step=permanently_closed,
                              **{key: mean(recent, prefix + key) for key in
                                 ["state_gate_mean", "state_gate_low_fraction", "candidate_rms",
                                  "effective_update_rms", "struct_gain", "write_relative_rms",
                                  "write_gate_mean", "read_gate_mean"]}))
    write_csv("gate_audit.csv", gate_rows)
    val_rows = []
    for row in validation:
        for sigma in config["validation"]["sigmas"]:
            prefix = f"sigma_{sigma:.2f}/"
            val_rows.append(dict(step=row["step"], sigma_config=sigma,
                                 **{key[len(prefix):]: value for key, value in row["metrics"].items()
                                    if key.startswith(prefix)}))
    write_csv("validation_metrics.csv", val_rows)
    x = np.array([row["step"] for row in train])
    metrics = {key: np.array([row["metrics"][key] for row in train]) for key in keys}
    closed_start = next(row["step"] for i, row in enumerate(train)
                        if all(all(other["metrics"][f"A@{site}/state_gate_mean"] < .001 for site in SITES)
                               for other in train[i:]))
    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        evidence_scope="Supplied console log only (steps 1--306); matching detailed records from the same session. No new training or inference.",
        console_log=str(LOG), console_log_sha256=hashlib.sha256(raw_log).hexdigest(),
        session=str(SESSION), train_steps=len(train), microbatches=len(micro),
        unique_training_records=len({row["index"] for row in micro}),
        console_jsonl_max_absolute_difference=max_console_difference,
        selected_console_lines={step: line_numbers[step] for step in [1, 100, 200, 250, 300, 306]},
        nonfinite_training_scalars=int(nonfinite), max_global_gradient_norm=float(metrics["grad/global_norm_before_clip"].max()),
        clipped_steps=int((metrics["grad/clip_coefficient"] < 1).sum()),
        sigma_fm_step_correlation=float(np.corrcoef(metrics["sigma"], metrics["loss/fm"])[0, 1]),
        fm_step_standard_deviation=float(metrics["loss/fm"].std(ddof=1)),
        all_six_mean_state_gates_permanently_below_0_001_from_step=closed_start,
        max_abs_initial_vs_struct_mse_after_step_100=float(np.max(np.abs(
            metrics["P/initial_mse"][100:] - metrics["loss/struct"][100:]))),
        gate_audit=gate_rows,
        training_config=config,
        validation_summary=[dict(step=row["step"], validation_loss=row["metrics"]["validation_loss"])
                            for row in validation],
        limits=["Gradient norms are for parameter groups and cannot attribute gradients to individual losses.",
                "Validation covers one video at five sigmas, without a vanilla-Wan baseline or step-zero measurement.",
                "Training windows and noise bins have different sample/noise realizations; they are descriptive, not paired quality estimates.",
                "constant_target_mse uses each ground-truth video's own mean feature; it is an oracle diagnostic, not an available inference baseline.",
                "Current on-disk default configuration differs from this job's loaded configuration."])
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (OUT / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    ax = axes[0, 0]
    for i, (key, label) in enumerate([("loss/total", "Total"), ("loss/fm", "FM"),
                                     ("loss/weighted_struct", "Weighted P structure"),
                                     ("loss/weighted_prior", "Weighted P prior")]):
        ax.plot(x, metrics[key], color=colors[i], alpha=.14, linewidth=.6)
        ax.plot(x[24:], rolling(metrics[key]), color=colors[i], label=label)
    ax.axvline(200, linestyle="--", color="gray", linewidth=1)
    ax.set(title="Loss terms: thin raw, thick trailing 25-step mean", ylabel="Loss")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for i, (key, label) in enumerate([("P/initial_mse", "Initial P MSE"), ("loss/struct", "Mean corrected P MSE"),
                                     ("P/constant_target_mse", "GT-mean oracle MSE")]):
        ax.plot(x[24:], rolling(metrics[key]), label=label, color=colors[i], linestyle=["-", "--", ":"][i])
    ax.axvline(100, linestyle="--", color="gray", linewidth=1)
    ax.set(title="After step 100, corrected P overlaps initial P", ylabel="Unweighted MSE")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for i, site in enumerate(SITES):
        ax.semilogy(x, np.maximum(metrics[f"A@{site}/state_gate_mean"], 1e-10), label=f"A@{site}", color=colors[i])
    ax.axhline(.001, color="gray", linestyle=":", linewidth=1)
    ax.axvline(100, color="gray", linestyle="--", linewidth=1)
    ax.set(title="All six P-update gates saturate near zero", ylabel="Mean state gate (log scale)", ylim=(1e-9, 1.8))
    ax.legend(ncol=3, fontsize=8)
    ax = axes[1, 1]
    for i, site in enumerate(SITES):
        ax.semilogy(x, np.maximum(metrics[f"A@{site}/effective_update_rms"], 1e-11), label=f"A@{site}", color=colors[i])
    ax.axvline(100, color="gray", linestyle="--", linewidth=1)
    ax.set(title="Actual P increments vanish", ylabel="RMS of P_after - P_before (log scale)")
    ax = axes[2, 0]
    for i, site in enumerate(SITES):
        ax.plot(x[24:], rolling(metrics[f"A@{site}/write_relative_rms"]) * 100, label=f"A@{site}", color=colors[i])
    ax.set(title="Writers remain active while P updates disappear", ylabel="Residual RMS / hidden RMS (%)")
    ax.legend(ncol=3, fontsize=8)
    ax = axes[2, 1]
    for i, (a, b) in enumerate(zip([0, .2, .4, .6, .8], [.2, .4, .6, .8, 1])):
        selected = [row for row in buckets if row["sigma_lower"] == a and row["end"] <= 300]
        ax.plot([row["end"] for row in selected], [row["loss/fm"] for row in selected], "o-",
                color=colors[i], label=f"sigma [{a:.1f}, {b:.1f})")
    ax.set(title="FM by per-sample sigma: 50-step descriptive means", ylabel="FM MSE")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=.17)
    fig.suptitle("Experiment 61002 / Trial 61841 | supplied log: steps 1-306\n"
                 "P-state update collapse is directly observed; video-quality gain is not measured", fontsize=14)
    fig.savefig(OUT / "training_diagnosis.png", dpi=170)
    fig.savefig(OUT / "training_diagnosis.pdf")
    plt.close(fig)
    print(json.dumps({"output": str(OUT), "steps": len(train), "microbatches": len(micro),
                      "all_gates_closed_from": closed_start, "console_match_max_error": max_console_difference,
                      "nonfinite": int(nonfinite)}, indent=2))


if __name__ == "__main__":
    main()
