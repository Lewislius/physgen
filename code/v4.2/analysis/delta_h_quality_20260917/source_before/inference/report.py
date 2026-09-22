"""Export fixed-condition JEPA diagnostics and an unfilled human video-quality review table."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from physgen_v42.diagnostics import similarity
from physgen_v42.geometry import Geometry
from physgen_v42.runtime import log, output_path, read_json, read_jsonl, save_json, verify_files, job_lock
from inference.conditions import file_stamp, teacher_identity


QUALITY_FIELDS = ("object_count_failure", "identity_failure", "unmotivated_shape_failure", "face_hand_failure",
                  "contact_response_failure", "occlusion_order_failure", "visible_interpenetration_failure",
                  "color_exposure_failure", "event_incomplete", "static_video_failure")


def report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plan = read_json(root / "cases.json")
    if file_stamp(Path(plan["checkpoint"]) / "ema.pt") != plan["ema_file"]:
        raise ValueError("Report EMA differs from the generation checkpoint")
    weights = torch.load(Path(plan["checkpoint"]) / "ema.pt", map_location="cpu", weights_only=True)
    scale = weights["s_z"].float()
    rows, quality = [], []
    for case in plan["cases"]:
        folder = root / case["key"]
        meta = read_json(folder / "metadata.json")
        if meta["status"] != "completed":
            raise ValueError(f"Unfinished generation: {case['key']}")
        verify_files(folder, meta["files"])
        stamp = read_json(folder / "generated_jepa.meta.json")
        if stamp["identity"] != teacher_identity(plan, case, folder / "generated_teacher_view.pt", "generated"):
            raise ValueError(f"Stale generated-video JEPA feature: {case['key']}")
        verify_files(folder, {"generated_jepa.pt": stamp["file"]})
        saved = torch.load(folder / "final_state.pt", map_location="cpu", weights_only=True)
        generated = torch.load(folder / "generated_jepa.pt", map_location="cpu", weights_only=True).float()
        geo = Geometry.build(saved["geometry"], torch.device("cpu"))
        target = instances = None
        if "gt_record" in case:
            cache = Path(plan["config"]["paths"]["cache"])
            target = torch.load(cache / "teacher" / (case["id"] + ".pt"), map_location="cpu", weights_only=True).float()
            if case["gt_record"]["objects_reliable"]:
                instances = torch.load(cache / "instances" / (case["id"] + ".pt"), map_location="cpu", weights_only=True)
        def add(task, sigma, left, right, metrics, gain=None):
            base = dict(case=case["key"], group=case["group"], mode=case["mode"], task=task,
                        sigma=sigma, left=left, right=right, gain_from_previous=gain)
            rows.append(dict(base, region="global", **{k: metrics[k] for k in
                             ("mse_normalized", "cosine", "dynamic_cosine", "dynamic_status")}))
            for local in metrics.get("instances", []):
                rows.append(dict(base, gain_from_previous=None, region=f'instance_{local["index"]}',
                                 **{k: local[k] for k in ("mse_normalized", "cosine", "dynamic_cosine", "dynamic_status")}))
        if saved["p15"] is not None:
            add("generated", saved["sigma"], "p15_last_call", "z_generated",
                similarity(saved["p15"], generated, geo.p_weight, scale, instances))
        if target is not None:
            if saved["p15"] is not None:
                add("generated", saved["sigma"], "p15_last_call", "z_gt", similarity(saved["p15"], target, geo.p_weight, scale, instances))
            add("generated", None, "z_generated", "z_gt", similarity(generated, target, geo.p_weight, scale, instances))
        traces = read_jsonl(folder / "trajectory.jsonl")
        for task, values in (("solver", traces), ("gt_noised", read_jsonl(folder / "gt_noised.jsonl"))):
            for row in values:
                relations = row.get("state_to_gt")
                if relations:
                    for name in ("p0", "p5", "p15"):
                        gain = relations.get("gain_0_to_5" if name == "p5" else "gain_5_to_15" if name == "p15" else "")
                        add(task, row["sigma"], name, "z_gt", relations[name], gain)
        if len(traces) != plan["recipe"]["steps"]:
            raise ValueError(f"Unexpected solver call count: {case['key']}")
        fig, axes = plt.subplots(4, 1, figsize=(9, 11), constrained_layout=True)
        for name in ("writer5", "writer15", "velocity"):
            if name not in traces[0]["metrics"]:
                continue
            axes[0].plot([r["sigma"] for r in traces], [r["metrics"][name]["mean_ratio"] for r in traces], label=name)
            axes[1].plot([r["sigma"] for r in traces], [r["metrics"][name]["compression"] for r in traces], label=name)
        for name in ("state5", "state15"):
            if name not in traces[0]["metrics"]:
                continue
            axes[2].plot([r["sigma"] for r in traces], [r["metrics"][name]["candidate_rms"] for r in traces],
                         linestyle="--", label=name + " candidate")
            axes[2].plot([r["sigma"] for r in traces], [r["metrics"][name]["actual_rms"] for r in traces], label=name + " effective")
            axes[3].plot([r["sigma"] for r in traces], [r["metrics"][name]["delta_over_before"] for r in traces], label=name)
        for ax in axes:
            ax.invert_xaxis()
            ax.set_xlabel("solver sigma")
            if ax.lines:
                ax.legend()
        axes[0].set_ylabel("Actual residual / reference RMS")
        axes[1].set_ylabel("Effective / candidate RMS")
        axes[2].set_ylabel("State update RMS")
        axes[3].set_ylabel("State update / previous P RMS")
        fig.savefig(folder / "read_write_curves.png", dpi=150)
        plt.close(fig)
        quality.append(dict(case=case["key"], group=case["group"], mode=case["mode"],
                            category=case.get("gt_record", {}).get("category", "prompt"),
                            video=str(folder / "video.mp4"), duration=meta["duration"],
                            reviewer="", reviewed="", **{field: "" for field in QUALITY_FIELDS}, notes=""))
    destination = root / "p_jepa_similarity.csv"
    with destination.open("w", newline="") as stream:
        fields = list(rows[0]) if rows else ["case", "group", "mode", "task", "sigma", "left", "right",
                                            "gain_from_previous", "region", "mse_normalized", "cosine",
                                            "dynamic_cosine", "dynamic_status"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    review_file = root / "quality_review.csv"
    if not review_file.exists():
        with review_file.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(quality[0]))
            writer.writeheader()
            writer.writerows(quality)
    with review_file.open() as stream:
        reviewed = list(csv.DictReader(stream))
    if {r["case"] for r in reviewed} != {r["case"] for r in quality} or len(reviewed) != len(quality):
        raise ValueError("Quality review CSV does not match the fixed case suite")
    summary = {}
    for group in dict.fromkeys(c["group"] for c in plan["cases"]):
        group_rows = [r for r in reviewed if r["group"] == group]
        complete = [r for r in group_rows if r["reviewed"] == "1" and r["reviewer"].strip()
                    and all(r[field] in ("0", "1", "na") for field in QUALITY_FIELDS)]
        summary[group] = dict(total=len(group_rows), reviewed=len(complete), pending=len(group_rows) - len(complete),
                              failure_videos=sum(any(r[field] == "1" for field in QUALITY_FIELDS) for r in complete),
                              failures={field: sum(r[field] == "1" for r in complete) for field in QUALITY_FIELDS},
                              applicable={field: sum(r[field] in ("0", "1") for r in complete) for field in QUALITY_FIELDS},
                              failed_cases=[r["case"] for r in complete if any(r[field] == "1" for field in QUALITY_FIELDS)])
    save_json(dict(status="human_review_pending" if any(s["pending"] for s in summary.values()) else "reviewed",
                   groups=summary, caveat="JEPA similarity and temporal smoothness are not physical correctness",
                   checkpoint=plan["checkpoint"], checkpoint_step=plan["checkpoint_step"],
                   ema_updates=plan["ema_updates"], weights="ema.pt", seed=plan["recipe"]["seed"]), root / "quality_summary.json")
    (root / "REVIEW.md").write_text(
        "# 固定生成验收\n\n逐条播放完整视频，再填写 quality_review.csv：reviewer、reviewed=1；各 failure 列 0=未发现、1=失败、na=不适用。"
        "未填项保持待审，不计为成功。查看遮挡、正常进出画、正常形变后再判断数量/身份异常。二维mask重叠不能证明三维穿模。"
        "静止视频不能以低差分视为成功。修改CSV后重新运行 report.py；已有人工内容不会被覆盖。\n\n"
        "train用于机制/记忆诊断；validation用于留出泛化；demo/reference分别报告，不混合统计。"
        "P15来自metadata所记最后一次solver调用，不能冒充clean-video JEPA。\n")
    log(event="report_complete", cases=len(plan["cases"]), similarity_rows=len(rows), groups=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = output_path(args.output)
    with job_lock(root / ".inference.lock"):
        report(root)


if __name__ == "__main__":
    main()
